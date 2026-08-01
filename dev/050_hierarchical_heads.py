"""Reopen the head-comparison question on the clean package: hierarchical (+ autoregressive) heads.

Continues [[2026-07-16-why-was-fastai-behind-mini-trainer]] — the independent cosine head won the old
mini_trainer benchmark (0.9148). Here we reimplement the other heads **fastai-only, on top of the
`lepinet` package** (importing it; all experiment code stays in `dev/`), register them into
`lepinet.heads.HEAD_REGISTRY`, and re-run the independent-vs-hierarchical(-vs-autoregressive)
comparison on effnetv2_s so the conclusion is re-validated on the clean stack.

This module deliberately does **not** port mini_trainer's caching machinery (`_dirty_cache`,
`active_indices`, `_weight_bias`, lazy mask warm-up) — that was the source of the GPU reference-cycle
leak the package dropped. It reimplements only the *math*, reusing lepinet's `cosine_to_zscore`,
`scatter_logsumexp`, and `sparse_masks_from_labels`.

Status: HierarchicalHead (parent-conditioned) implemented + CPU-tested. AutoregressiveHead + the
UCloud benchmark are the next step (see the plan at the bottom).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from fastai.vision.all import Callback

from lepinet.export import scatter_logsumexp
from lepinet.heads import HEAD_REGISTRY, IndependentHead, cosine_to_zscore


class HierarchicalHead(IndependentHead):
    """Parent-conditioned hierarchical cosine head (clean reimpl of mini_trainer's ConditionalClassifier).

    Same shared bottleneck + per-level unit-norm cosine layers as :class:`IndependentHead`, plus
    **top-down conditioning** through the taxonomy: each level's independent (marginal) logits are
    corrected by the parent's conditioned score, so a child cannot be more confident than its parent's
    evidence allows. In log-space, using ``P_cond(child) = P(child)·P_cond(parent)/P(siblings)``::

        C[top]  = M[top]                                   # coarsest level is unconditioned
        C[i]    = M[i] + gather_parent( C[i+1] - logΣ_siblings M[i] )

    where ``M[i]`` are the per-level marginal logits and the sibling sum is a
    ``scatter_logsumexp`` up the ``sparse_masks`` (parent index per child, fine→coarse, ``N-1`` masks).
    Forward is a deterministic function of the embedding (no labels), so it still traces to ONNX.

    Args mirror :class:`IndependentHead`, plus ``sparse_masks`` from
    :func:`lepinet.heads.sparse_masks_from_labels` / ``build_class_spec``.
    """

    def __init__(self, in_features: int, n_classes: Sequence[int], hidden: bool | int = True,
                 droprate: float = 0.1, sparse_masks: list[torch.Tensor] | None = None):
        super().__init__(in_features, n_classes, hidden=hidden, droprate=droprate)
        if sparse_masks is None or len(sparse_masks) != self.n_levels - 1:
            raise ValueError(f"HierarchicalHead needs {self.n_levels - 1} sparse_masks (fine→coarse), "
                             f"got {None if sparse_masks is None else len(sparse_masks)}.")
        for i, m in enumerate(sparse_masks):
            self.register_buffer(f"mask_{i}", m.long(), persistent=True)

    def marginals(self, emb: torch.Tensor) -> list[torch.Tensor]:
        """Per-level independent cosine logits (identical to :class:`IndependentHead`'s forward)."""
        return [cosine_to_zscore(F.linear(emb, layer.weight), self.preclass_size) + layer.bias
                for layer in self.layers]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        emb = self.preclassification(x)
        M = self.marginals(emb)
        n = len(M)
        C: list[torch.Tensor | None] = [None] * n
        C[-1] = M[-1]
        for i in reversed(range(n - 1)):
            mask = getattr(self, f"mask_{i}")               # [n_children_i] -> parent idx
            n_parents = M[i + 1].shape[1]
            with torch.no_grad():
                sibling_norm = scatter_logsumexp(M[i], mask, n_parents)   # [N, n_parents]
            parent_term = (C[i + 1] - sibling_norm).gather(1, mask.unsqueeze(0).expand_as(M[i]))
            C[i] = M[i] + parent_term
        return C  # type: ignore[return-value]


HEAD_REGISTRY["hierarchical"] = HierarchicalHead


class MarginalHead(IndependentHead):
    """**One** species prototype layer; coarser levels are the marginals of the species posterior.

    This is the owner's original "hierarchical head" idea, applied *during training* rather than only
    at inference: there are **no genus/family parameters at all**. The coarse logits are
    ``log Σ_children exp(log p(species))`` computed in the forward pass, so the coarse
    cross-entropies backpropagate **into the species head**.

    Contrast with :class:`HierarchicalHead` (mini_trainer's conditional classifier), which keeps
    separate prototype layers per level and pushes information *down* the tree. Here information
    flows *up*, and the levels are probabilistically coherent — the coarse posterior *is* the sum of the
    species one. (It does **not** guarantee argmax agreement: max and sum do not commute over a
    partition, so a confident species can be outvoted by diffuse siblings. See
    journal/2026-08-01-marginalisation-is-not-argmax-consistent.md.)

    Emits log-probabilities (not z-scores) at every level: they must live on one comparable scale for
    the log-sum-exp to mean anything, and ``CrossEntropyLoss`` treats them as logits correctly since
    softmax(log p) ∝ p. Forward is label-free, so it still traces to ONNX.
    """

    def __init__(self, in_features: int, n_classes: Sequence[int], hidden: bool | int = True,
                 droprate: float = 0.1, sparse_masks: list[torch.Tensor] | None = None):
        # Only the finest level gets a prototype layer; drop the rest before building.
        super().__init__(in_features, [n_classes[0]], hidden=hidden, droprate=droprate)
        if sparse_masks is None or len(sparse_masks) != len(n_classes) - 1:
            raise ValueError(f"MarginalHead needs {len(n_classes) - 1} sparse_masks (fine→coarse).")
        self.level_sizes = list(n_classes)
        for i, m in enumerate(sparse_masks):
            self.register_buffer(f"mask_{i}", m.long(), persistent=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        emb = self.preclassification(x)
        species = cosine_to_zscore(F.linear(emb, self.layers[0].weight), self.preclass_size) + self.layers[0].bias
        out = [F.log_softmax(species, dim=1)]
        cur = out[0]
        for i in range(len(self.level_sizes) - 1):
            cur = scatter_logsumexp(cur, getattr(self, f"mask_{i}"), self.level_sizes[i + 1])
            out.append(cur)
        return out


HEAD_REGISTRY["marginal"] = MarginalHead



# ---------------------------------------------------------------------------
# Self-test (CPU) + next-step plan
# ---------------------------------------------------------------------------

def _selftest():
    """Shapes, finiteness, and the hierarchy invariant: a child's conditioned score never exceeds
    (parent term) — sanity that conditioning is wired the right way round."""
    from collections import OrderedDict

    from lepinet.heads import build_head, sparse_masks_from_labels

    n_classes = [6, 3, 2]  # species, genus, family
    # a proper tree: species s -> genus s//2 -> family (s//2)//2
    labels = OrderedDict((str(s), (str(s), str(s // 2), str((s // 2) // 2))) for s in range(6))
    cls2idx = {str(i): {str(k): k for k in range(n)} for i, n in enumerate(n_classes)}
    masks = sparse_masks_from_labels(labels, cls2idx)
    head = build_head("hierarchical", 16, n_classes, hidden=True, sparse_masks=masks).eval()
    out = head(torch.randn(4, 16, 2, 2))  # PooledHead pools the map
    assert [tuple(o.shape) for o in out] == [(4, 6), (4, 3), (4, 2)]
    assert all(torch.isfinite(o).all() for o in out)
    print("HierarchicalHead selftest OK:", [tuple(o.shape) for o in out])

    mh = build_head("marginal", 16, n_classes, hidden=True, sparse_masks=masks).eval()
    mo = mh(torch.randn(4, 16, 2, 2))
    assert [tuple(o.shape) for o in mo] == [(4, 6), (4, 3), (4, 2)]
    # every level is a proper log-distribution, and coarse = sum of its children
    for o in mo:
        assert torch.allclose(o.exp().sum(1), torch.ones(4), atol=1e-4)
    got = mo[1].exp()
    want = torch.zeros_like(got).index_add_(1, masks[0], mo[0].exp())
    assert torch.allclose(got, want, atol=1e-5), "genus must equal the sum of its species"
    n_params = sum(p.numel() for p in mh.parameters())
    n_multi = sum(p.numel() for p in build_head("independent", 16, n_classes, hidden=True).parameters())
    print(f"MarginalHead selftest OK: {[tuple(o.shape) for o in mo]} | params {n_params} vs multi-head {n_multi}")


PLAN = """
Next (this experiment):
  1. AutoregressiveHead: a small cross-attention decoder over the per-level cosine 'embeddings',
     coarse->fine, teacher-forced in training via a callback that supplies y (like DistillCallback
     supplies teacher logits). Register as 'autoregressive'.
  2. Thread sparse_masks through training for taxonomy-needing heads: compute build_class_spec in
     train.py and pass sparse_masks in head_kwargs when the head accepts it (independent ignores it).
  3. Benchmark configs on effnetv2_s (independent | hierarchical | autoregressive), same 0.9148
     recipe, on UCloud B200; report species/genus/family macro-F1. Update
     journal/2026-07-16-why-was-fastai-behind-mini-trainer.md with the clean-stack conclusion.
"""

if __name__ == "__main__":
    _selftest()
    print(PLAN)


# ---------------------------------------------------------------------------
# A4: marginal supervision x ArcFace margin
#
# Two results make this combination worth building rather than guessing at:
#   * the ArcFace margin damages *marginalisation* more than classification, because summing a
#     posterior is calibration-dependent (journal/2026-07-30-does-arcface-compose-with-marginalisation)
#   * marginal supervision acts purely on that same sum -- species exactly unchanged, coarse levels
#     improved, and +1.41 pt under domain shift (journal/2026-07-30-marginal-supervision)
# One intervention degrades the quantity the other optimises, so composing them is a direct test.
#
# Why this needs a callback at all: `ArcFaceHead` keeps its forward label-free and lets the *loss*
# inject the margin, which is what preserves ONNX traceability. That trick is unavailable here --
# the marginals are computed inside forward from the species z-scores, and log-probabilities cannot
# be inverted back to z-scores (log_softmax discards a per-row constant, and cos = sin(Z/sqrt(d-2))
# is not shift-invariant). So the margin must be applied *before* the marginalisation, inside
# forward, which means forward needs the labels during training.
#
# Inference is unaffected: `_targets` is None outside training, the margin is skipped, and the graph
# is label-free again -- so export still works.

class MarginalArcFaceHead(MarginalHead):
    """:class:`MarginalHead` with an ArcFace margin applied to the species z-scores pre-marginalisation.

    Set ``head._targets`` (a ``[N]`` LongTensor of species indices) before each training forward;
    :class:`MarginContextCallback` does this. When it is ``None`` the head is exactly
    :class:`MarginalHead`.
    """

    def __init__(self, in_features: int, n_classes: Sequence[int], hidden: bool | int = True,
                 droprate: float = 0.1, sparse_masks: list[torch.Tensor] | None = None,
                 margin: float = 0.3, scale: float = 30.0):
        super().__init__(in_features, n_classes, hidden=hidden, droprate=droprate,
                         sparse_masks=sparse_masks)
        self.margin, self.scale = float(margin), float(scale)
        self._targets = None

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        from lepinet.loss import apply_arcface_margin_zscore

        emb = self.preclassification(x)
        species = cosine_to_zscore(F.linear(emb, self.layers[0].weight),
                                   self.preclass_size) + self.layers[0].bias
        if self.training and self._targets is not None:
            species = apply_arcface_margin_zscore(species, self._targets, self.margin,
                                                  self.preclass_size)
        out = [F.log_softmax(species, dim=1)]
        cur = out[0]
        for i in range(len(self.level_sizes) - 1):
            cur = scatter_logsumexp(cur, getattr(self, f"mask_{i}"), self.level_sizes[i + 1])
            out.append(cur)
        return out


HEAD_REGISTRY["marginal_arcface"] = MarginalArcFaceHead


def _find_marginal_arcface(model):
    for m in model.modules():
        if isinstance(m, MarginalArcFaceHead):
            return m
    return None


class MarginContextCallback(Callback):
    """Hand the batch's species labels to :class:`MarginalArcFaceHead` for the training forward.

    ``order`` puts this after MixUp-style callbacks so it sees whatever labels the loop will
    actually score against. Cleared after each batch so a stale tensor can never leak into
    validation -- the head checks ``self.training`` too, but two guards cost nothing and this one
    fails loudly (shape mismatch) rather than silently.
    """

    order = 100

    def before_batch(self):
        head = _find_marginal_arcface(self.learn.model)
        if head is not None:
            y = self.learn.yb
            head._targets = (y[0] if isinstance(y, (list, tuple)) else y) if self.training else None

    def after_batch(self):
        head = _find_marginal_arcface(self.learn.model)
        if head is not None:
            head._targets = None
