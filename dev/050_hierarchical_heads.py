"""Reopen the head-comparison question on the clean package: hierarchical (+ autoregressive) heads.

Continues [[2026-07-why-was-fastai-behind-mini-trainer]] — the independent cosine head won the old
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


PLAN = """
Next (this experiment):
  1. AutoregressiveHead: a small cross-attention decoder over the per-level cosine 'embeddings',
     coarse->fine, teacher-forced in training via a callback that supplies y (like DistillCallback
     supplies teacher logits). Register as 'autoregressive'.
  2. Thread sparse_masks through training for taxonomy-needing heads: compute build_class_spec in
     train.py and pass sparse_masks in head_kwargs when the head accepts it (independent ignores it).
  3. Benchmark configs on effnetv2_s (independent | hierarchical | autoregressive), same 0.9148
     recipe, on UCloud B200; report species/genus/family macro-F1. Update
     journal/2026-07-why-was-fastai-behind-mini-trainer.md with the clean-stack conclusion.
"""

if __name__ == "__main__":
    _selftest()
    print(PLAN)
