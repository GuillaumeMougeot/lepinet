"""Multi-level cross-entropy loss for the hierarchical head.

Clean reimplementation of ``mini_trainer.hierarchical.loss.MultiLevelWeightedCrossEntropyLoss``:
one ``CrossEntropyLoss`` per taxonomic level, summed with optional per-level weights. Generic in
the number of levels.

The one subtlety worth keeping is the **per-level label-smoothing adjustment**. A single
smoothing value ``ls`` applied at every level would push the model to give the correct *parent*
higher confidence than the correct *leaf* (because smoothing spreads mass to leaf siblings, which
share the parent), which is backwards for hierarchical learning. The fix (from mini_trainer) is
to shrink smoothing at coarser levels: ``ls_L = 1 - (1 - ls) ** (1 / (L + 1))`` so the effective
"keep" probability ``(1 - ls_L)`` compounds consistently up the tree.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MultiLevelCELoss:
    """Sum of per-level cross-entropy losses (returns a scalar).

    Deliberately **not** an ``nn.Module``: it holds its own device/dtype tensors and there is
    nothing to register, so wrapping it as a module only invites fastai's ``loss_func.to(device)``
    to fight it. ``__call__`` takes stacked per-level logits and stacked targets.

    Args:
        num_classes: classes per level, fine→coarse.
        weights: per-level loss weights (default all 1).
        label_smoothing: base smoothing at the finest level; ``None`` → ``1/num_classes[0]``
            (mini_trainer's default). Adjusted per level as described in the module docstring.
        device / dtype: where the weight tensor lives.
    """

    def __init__(
        self,
        num_classes: Sequence[int],
        weights: Sequence[float] | None = None,
        label_smoothing: float | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        reduction: str = "mean",
    ):
        self.n_levels = len(num_classes)
        if label_smoothing is None:
            label_smoothing = 1.0 / num_classes[0]
        if weights is None:
            weights = [1.0] * self.n_levels
        self.weights = torch.as_tensor(weights, device=device, dtype=dtype)
        self._reduction = reduction
        # Coarser levels smooth less, so (1 - ls) compounds consistently up the hierarchy.
        per_level_ls = [1 - (1 - label_smoothing) ** (1 / (i + 1)) for i in range(self.n_levels)]
        self._loss_fns = [nn.CrossEntropyLoss(label_smoothing=ls, reduction=reduction) for ls in per_level_ls]

    @property
    def reduction(self) -> str:
        return self._reduction

    @reduction.setter
    def reduction(self, value: str) -> None:
        # fastai's MixUp/CutMix toggle reduction to 'none' to lerp per-sample losses, then restore.
        self._reduction = value
        for fn in self._loss_fns:
            fn.reduction = value

    def per_level(self, preds: Sequence[torch.Tensor], targets: torch.Tensor) -> list[torch.Tensor]:
        """Per-level weighted losses (scalars, or per-sample ``[N]`` when reduction='none')."""
        targets = targets.transpose(0, 1)  # [N, L] -> [L, N]
        return [self._loss_fns[i](preds[i], targets[i]) * self.weights[i] for i in range(self.n_levels)]

    def __call__(self, preds: Sequence[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        return sum(self.per_level(preds, targets))


class FastaiLossWrapper:
    """Adapt :class:`MultiLevelCELoss` to fastai's ``loss_func(preds, *yb)`` contract.

    fastai passes the per-level targets as separate positional args (``*yb``); this stacks them
    into ``[N, L]`` and calls the criterion. Exposes ``reduction`` (delegated to the criterion) so
    fastai's MixUp/CutMix can switch it to 'none' for per-sample loss mixing. Not an ``nn.Module``
    for the same reason as above.
    """

    # Marks targets as integer class indices, so fastai's MixUp/CutMix mix via the *loss*
    # (lerping per-sample losses) rather than mixing the integer labels themselves.
    y_int = True

    def __init__(self, criterion: MultiLevelCELoss):
        self.criterion = criterion

    @property
    def reduction(self) -> str:
        return self.criterion.reduction

    @reduction.setter
    def reduction(self, value: str) -> None:
        self.criterion.reduction = value

    def __call__(self, preds: Sequence[torch.Tensor], *yb: torch.Tensor) -> torch.Tensor:
        return self.criterion(preds, torch.stack(yb, dim=1))
