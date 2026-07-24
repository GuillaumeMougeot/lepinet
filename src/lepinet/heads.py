"""Cosine classification heads for hierarchical (species / genus / family / ...) classification.

This is the clean, ``mini_trainer``-free reimplementation of the **independent** head that won
the project baseline (test species macro-F1 **0.9148**, run ``20260716-154156``; see
``journal/2026-07-does-longtail-help.md``). Only the independent head is implemented here — the
hierarchical / conditional / autoregressive variants that lived in ``mini_trainer`` are
deliberately dropped (autoregressive never won; the marginalization path lives in
``export.py``). New head types are added via :data:`HEAD_REGISTRY` without touching this module.

Design choices (favouring **clarity over checkpoint-loadability** — the original mini_trainer
checkpoint is not loadable here; parity is established by retraining instead):

* The *math* matches the original cosine head (``cosine_to_zscore(F.linear(F.normalize(x), W)) +
  b`` with unit-norm prototypes), but everything that existed only for the mini_trainer class
  hierarchy is gone: the ``_weight_bias`` cache and ``active_indices`` branching (the source of
  the GPU reference-cycle leak that forced ``GCCallback``), the dead ``BatchNorm``, the
  ``linear``/``layers[0]`` alias, the ``_extra_state`` dict, and the parent-index ``mask``
  buffers (the independent head does not use them).
* **The forward path has no data-dependent control flow** — just ``linear`` / ``normalize`` /
  ``acos`` / affine — so it traces cleanly to ONNX with ``dynamo=False``.

The head is **N-level generic**: pass ``n_classes`` (fine→coarse) of any length. The Lepidoptera
default is the 3 levels ``[species, genus, family]``.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # PyTorch >= 2.1
    from torch.nn.utils.parametrizations import weight_norm
except Exception:  # pragma: no cover - older installs
    from torch.nn.utils import weight_norm


# ---------------------------------------------------------------------------
# Cosine -> z-score (copied verbatim from mini_trainer/utils/_core/math.py)
# ---------------------------------------------------------------------------

def cosine_to_zscore(cosine: torch.Tensor, ndim: int) -> torch.Tensor:
    r"""Map a cosine similarity between unit vectors in ``ndim`` dimensions to a z-score.

    ``Z(x) = sqrt(ndim - 2) * (acos(-x) - pi/2)``. For random unit vectors the cosine is
    tightly concentrated near 0; this transform stretches it to an approximately standard
    normal, which is what makes the cosine logits behave like ordinary pre-softmax scores.
    Kept identical to the original so trained weights reproduce exactly.
    """
    z_var = 1.0 / (float(ndim) - 2.0) ** 0.5
    z_mu = torch.pi / 2.0
    z_rel = torch.acos(-cosine.clamp(-1 + 1e-7, 1 - 1e-7))
    return (z_rel - z_mu) / z_var


# ---------------------------------------------------------------------------
# Taxonomy -> parent-index ("sparse") masks
# ---------------------------------------------------------------------------

def sparse_masks_from_labels(
    labels: OrderedDict[str, tuple[str, ...]],
    cls2idx: dict,
) -> list[torch.Tensor]:
    """Parent-index masks from hierarchical labels and per-level class→index maps.

    ``mask_i[child_idx] = parent_idx`` maps each class at level ``i`` to its parent at level
    ``i+1``. Returns ``N-1`` masks for an ``N``-level hierarchy. Copied from
    ``mini_trainer.hierarchical.integration`` (behaviour preserved, including the conflict /
    missing-parent checks).
    """
    cls2idx = {str(k): v for k, v in cls2idx.items()}
    nlvl = len(cls2idx)
    masks = [[-1 for _ in range(len(cls2idx[str(lvl)]))] for lvl in range(nlvl - 1)]
    for lab in labels.values():
        idx = [cls2idx[str(lvl)][cls] for lvl, cls in enumerate(lab)]
        for mask_i, (child, parent) in enumerate(zip(idx, idx[1:])):
            if masks[mask_i][child] not in (-1, parent):
                raise ValueError(
                    f"Conflicting labels at level {mask_i} class {child}: had parent "
                    f"{masks[mask_i][child]}, now found {parent}."
                )
            masks[mask_i][child] = parent
    invalid = [(mi, ei) for mi, m in enumerate(masks) for ei, e in enumerate(m) if e == -1]
    if invalid:
        raise ValueError(
            f"Unable to construct sparse masks: {len(invalid)} classes have no parent "
            f"(first few: {invalid[:10]}). The hierarchy and class index disagree."
        )
    return [torch.tensor(m, dtype=torch.long) for m in masks]


def build_class_spec(df, vocabs: dict, levels: Sequence[str]):
    """Build ``cls2idx`` (per-level label→index) and ``sparse_masks`` from a dataframe.

    ``cls2idx`` keys are stringified level indices ("0", "1", ...) fine→coarse, matching the
    order of ``levels`` and of the head's outputs, so label indices agree between the fastai
    ``CategoryBlock`` and the head's masks.
    """
    cls2idx = {str(i): {v: idx for idx, v in enumerate(vocabs[level])} for i, level in enumerate(levels)}
    unique = df.drop_duplicates(levels[0])
    labels = OrderedDict(
        (getattr(row, levels[0]), tuple(getattr(row, level) for level in levels))
        for row in unique.itertuples(index=False)
    )
    return cls2idx, sparse_masks_from_labels(labels, cls2idx)


# ---------------------------------------------------------------------------
# Independent cosine head
# ---------------------------------------------------------------------------

@torch.no_grad()
def _init_spherical_repulsion(weight: torch.Tensor, iterations: int = 100, lr: float = 0.5) -> None:
    """Spread ``weight``'s rows apart on the unit sphere (repulsion init for the finest level).

    Copied from ``mini_trainer.modeling.Classifier.init_spherical_repulsion``. Only the finest
    (species) prototype layer gets this; coarser levels use plain normal init. Reproduced so a
    from-scratch retrain matches the original training dynamics.
    """
    nn.init.normal_(weight)
    for _ in range(iterations):
        weight.div_(weight.norm(dim=1, keepdim=True).clamp(min=1e-9))
        grad = weight @ weight.t() @ weight  # d/dW sum((W W^T)^2) = 4 W W^T W
        proj = (grad * weight).sum(dim=1, keepdim=True) * weight
        weight.sub_(lr * (grad - proj))
    weight.div_(weight.norm(dim=1, keepdim=True).clamp(min=1e-9))


class IndependentHead(nn.Module):
    """N-level cosine head: one L2-normalized prototype layer per taxonomic level.

    A shared bottleneck (``hidden`` + LeakyReLU + L2-normalize) produces one unit embedding;
    each level scores it against its own unit-norm prototypes (``weight_norm`` with the row norm
    frozen at 1) and passes the cosine through :func:`cosine_to_zscore`. "Independent" = the
    levels do not interact (no hierarchy propagation); this beat the hierarchical/autoregressive
    variants in the benchmark. Parent/child relationships are **not** part of this head — they
    live in the checkpoint's hierarchy table and ``taxonomy.json`` and are used only at export
    time for marginalization.

    Args:
        in_features: backbone feature width fed to the head (e.g. 1280 for effnetv2_s).
        n_classes: classes per level, **fine→coarse** (e.g. ``[12041, 4333, 102]``).
        hidden: bottleneck width. ``True`` (default) → a ``Linear(in_features, in_features)``
            bottleneck; an ``int`` sets the width explicitly (the size lever); ``False`` → none.
        droprate: dropout before the bottleneck.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: Sequence[int],
        hidden: bool | int = True,
        droprate: float = 0.1,
    ):
        super().__init__()
        if isinstance(hidden, bool):
            self.preclass_size = in_features
            build_hidden = hidden
        elif isinstance(hidden, int) and hidden > 0:
            self.preclass_size = hidden
            build_hidden = True
        else:
            raise TypeError(f"`hidden` must be a bool or positive int, got {hidden!r}.")

        self.in_features = in_features
        self.n_classes = list(n_classes)
        self.n_levels = len(self.n_classes)

        self.hidden = nn.Linear(in_features, self.preclass_size) if build_hidden else None
        self.dropout = nn.Dropout(droprate) if build_hidden else None

        # One cosine prototype layer per level. The finest level gets spherical-repulsion init
        # (rows spread on the sphere); coarser levels use plain normal init.
        self.layers = nn.ModuleList(
            self._normalize_layer(nn.Linear(self.preclass_size, n, bias=True), orthogonal_init=(i == 0))
            for i, n in enumerate(self.n_classes)
        )

    @staticmethod
    @torch.no_grad()
    def _normalize_layer(layer: nn.Linear, orthogonal_init: bool = False) -> nn.Linear:
        """Freeze the bias at 0, apply ``weight_norm`` and freeze each row's norm at 1.

        This is what makes the layer a *cosine* classifier: the effective ``layer.weight`` is
        always unit-norm per row, so ``F.linear(unit_embedding, weight)`` is a cosine similarity.
        """
        if layer.bias is not None:
            layer.bias.fill_(0)
            layer.bias.requires_grad_(False)
        if orthogonal_init:
            _init_spherical_repulsion(layer.weight)
        weight_norm(layer, name="weight", dim=0)
        layer.parametrizations.weight.original0.fill_(1)          # row norm = 1
        layer.parametrizations.weight.original0.requires_grad_(False)  # ...and frozen
        return layer

    def preclassification(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features → one L2-normalized embedding of width ``preclass_size``."""
        if self.hidden is not None:
            x = self.dropout(x)
            x = F.leaky_relu(self.hidden(x))
        return F.normalize(x, 2.0, -1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """``x``: pooled backbone features ``[N, in_features]``. Returns per-level logits, fine→coarse."""
        emb = self.preclassification(x)
        return [cosine_to_zscore(F.linear(emb, layer.weight), self.preclass_size) + layer.bias
                for layer in self.layers]


class PooledHead(nn.Module):
    """Global-average-pool a ``[N, C, H, W]`` backbone map to ``[N, C]``, then run ``head`` in fp32.

    The cosine head normalizes ``hidden(x)``; as prototype norms grow during training that
    ``hidden`` output can overflow fp16 → ``inf`` → ``normalize(inf) = NaN`` (the classic
    ArcFace/cosine-margin instability). Forcing the head to fp32 under autocast avoids it while
    the backbone keeps the AMP speedup. This is why bf16 is the package default and why fp16 is
    still safe here (``journal/2026-07-autoregressive-fp16-instability.md``). Head-agnostic: it
    only pools and hands off, so any head in :data:`HEAD_REGISTRY` works behind it.
    """

    def __init__(self, head: nn.Module):
        super().__init__()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = head

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.head(x.float())


# ---------------------------------------------------------------------------
# Factory / registry (the seam for experimenting with new heads -- see §6 of the proposal)
# ---------------------------------------------------------------------------

HEAD_REGISTRY: dict[str, type[nn.Module]] = {"independent": IndependentHead}


def build_head(
    head_name: str,
    in_features: int,
    n_classes: Sequence[int],
    hidden: bool | int = True,
) -> PooledHead:
    """Construct ``PooledHead(<head>)`` for one of :data:`HEAD_REGISTRY`.

    The baseline registers only ``"independent"``; a ``dev/`` experiment can register another
    head class and reach it here without editing this module. A head that needs the taxonomy
    (e.g. a hierarchical head) should take it via its own constructor and be built directly.
    """
    if head_name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head {head_name!r}; registered: {sorted(HEAD_REGISTRY)}.")
    head = HEAD_REGISTRY[head_name](in_features, n_classes, hidden=hidden)
    return PooledHead(head)


def infer_hidden_from_state_dict(head_state: dict, prefix: str = "head.") -> bool | int:
    """Recover the ``hidden`` setting from a saved head ``state_dict`` (robust to stale metadata).

    The legacy checkpoint stores ``hidden=None`` in its sidecar metadata even though it has a
    real ``Linear`` bottleneck, so we read the geometry from the weights instead: presence and
    output width of ``hidden.weight``. Returns ``True`` when the bottleneck width equals the
    input width (the default), the int width when it differs, or ``False`` when absent.
    """
    w = head_state.get(f"{prefix}hidden.weight")
    if w is None:
        return False
    out_features, in_features = w.shape
    return True if out_features == in_features else int(out_features)
