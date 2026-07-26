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

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def apply_arcface_margin(logit: torch.Tensor, target: torch.Tensor, margin: float, scale: float) -> torch.Tensor:
    """Inject an ArcFace additive angular margin into scaled-cosine logits (``logit = s·cos θ``).

    Penalises the *true* class by ``cos(θ + m)`` before rescaling by ``s`` (Deng et al. 2019), so
    cross-entropy has to push the true-class angle ``m`` radians tighter than a plain cosine head
    would. Non-target logits are untouched. Uses the ``cos(θ+m) = cos θ·cos m − sin θ·sin m``
    identity (no ``acos``), which is numerically clean in fp32 (the head runs fp32 under autocast).
    Applied by the loss, not the head, because only the loss sees the labels — keeping the head
    forward label-free and ONNX-traceable.
    """
    cos = (logit / scale).clamp(-1 + 1e-7, 1 - 1e-7)
    sin = torch.sqrt((1.0 - cos * cos).clamp_min(1e-12))
    phi = cos * math.cos(margin) - sin * math.sin(margin)   # cos(θ + m)
    one_hot = F.one_hot(target, num_classes=logit.size(1)).to(logit.dtype)
    return scale * (one_hot * phi + (1.0 - one_hot) * cos)


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
        arc_scale: float | None = None,
        arc_margins: Sequence[float] | None = None,
    ):
        self.n_levels = len(num_classes)
        if label_smoothing is None:
            label_smoothing = 1.0 / num_classes[0]
        if weights is None:
            weights = [1.0] * self.n_levels
        self.weights = torch.as_tensor(weights, device=device, dtype=dtype)
        self._reduction = reduction
        # ArcFace (optional): a per-level angular margin, applied in training only. arc_margins is
        # None for the plain cosine baseline; a list (fine→coarse, 0 = no margin on that level)
        # when the head is `arcface`. Needs arc_scale (the head's `s`) to recover cos θ = logit/s.
        self.arc_scale = arc_scale
        self.arc_margins = list(arc_margins) if arc_margins is not None else None
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
        """Per-level weighted losses (scalars, or per-sample ``[N]`` when reduction='none').

        When an ArcFace margin is configured for a level, it is injected here — but only when the
        logits carry gradient (``requires_grad``), i.e. during *training*. Validation runs under
        ``no_grad``, so the margin is skipped and the reported val loss / metric use the plain
        cosine logits (the model is selected on the margin-free score, as it should be).
        """
        targets = targets.transpose(0, 1)  # [N, L] -> [L, N]
        losses = []
        for i in range(self.n_levels):
            logit = preds[i]
            m = self.arc_margins[i] if self.arc_margins is not None else 0.0
            if m > 0 and self.arc_scale is not None and logit.requires_grad:
                logit = apply_arcface_margin(logit, targets[i], m, self.arc_scale)
            losses.append(self._loss_fns[i](logit, targets[i]) * self.weights[i])
        return losses

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


class DistillLoss:
    """Knowledge-distillation loss: hard-label CE blended with per-level soft-target KD.

    ``total = (1-α)·CE(student, labels) + α·Σ_level T²·KL(softmax(student/T) ‖ softmax(teacher/T))``.

    Distillation is the *training method* for a small student, not a post-hoc compressor
    ([[2026-07-lepi-app]]): the teacher's soft posterior over 12 k species carries the hierarchy and
    the tail structure implicitly, so the student learns more than the hard labels alone can teach.
    KD is applied per level (fine→coarse), matching the N-level head.

    The teacher's per-level logits for the current batch are injected by
    :class:`~lepinet.callbacks.DistillCallback` (which runs the frozen teacher on the same input);
    when they are absent — e.g. during validation, where the callback clears them — this degrades to
    plain hard-label CE, so the reported valid loss and the metrics are margin/KD-free.

    Top-level ``loss_func`` (does its own ``*yb`` stacking, like :class:`FastaiLossWrapper`).
    ``kd_levels`` restricts KD to a subset of levels (default: all). Temperature ``T`` (~2–6) softens
    both sides; the ``T²`` factor keeps the KD gradient scale comparable to the CE term.
    """

    y_int = True  # incompatible with MixUp for now; the config guards against enabling both

    def __init__(
        self,
        criterion: MultiLevelCELoss,
        alpha: float = 0.5,
        temperature: float = 1.0,  # T=1 for the cosine z-score head; T=4 over-softens and hurt (journal)
        kd_levels: Sequence[int] | None = None,
    ):
        self.criterion = criterion
        self.alpha = float(alpha)
        self.T = float(temperature)
        self.kd_levels = set(kd_levels) if kd_levels is not None else None
        self.teacher_logits: list[torch.Tensor] | None = None  # set per training batch by the callback

    @property
    def reduction(self) -> str:
        return self.criterion.reduction

    @reduction.setter
    def reduction(self, value: str) -> None:
        self.criterion.reduction = value

    def __call__(self, preds: Sequence[torch.Tensor], *yb: torch.Tensor) -> torch.Tensor:
        hard = self.criterion(preds, torch.stack(yb, dim=1))
        if self.alpha == 0 or self.teacher_logits is None:
            return hard
        kd = hard.new_zeros(())
        for i, (ps, pt) in enumerate(zip(preds, self.teacher_logits)):
            if self.kd_levels is not None and i not in self.kd_levels:
                continue
            log_s = F.log_softmax(ps.float() / self.T, dim=1)
            soft_t = F.softmax(pt.float() / self.T, dim=1)
            kd = kd + F.kl_div(log_s, soft_t, reduction="batchmean") * (self.T ** 2)
        return (1.0 - self.alpha) * hard + self.alpha * kd
