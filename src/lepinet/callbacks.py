"""fastai training callbacks: NaN safety and an (optional) GC leak guard.

``HostMemoryGuard`` (host-RAM OOM guard) lives in :mod:`lepinet.memory` and is re-exported here so
callers have one import site for "the training callbacks".
"""
from __future__ import annotations

import gc

import torch
from fastai.callback.core import Callback, CancelBatchException, CancelFitException

from .memory import HostMemoryGuard  # re-export

__all__ = ["NaNGuard", "GCCallback", "HostMemoryGuard", "MixUpMulti", "DistillCallback"]


class NaNGuard(Callback):
    """Skip any training batch whose loss is non-finite, instead of letting it poison the weights.

    A single bad batch — e.g. a degenerate image whose embedding normalizes to NaN — otherwise
    backprops one optimizer step and permanently NaNs the whole model. Because the NaN originates
    in the forward, gradient clipping cannot catch it; only skipping the batch (weights untouched)
    does. Aborts if NaNs are persistent, so genuine divergence fails loudly rather than training on
    nothing. Runs before backward via ``CancelBatchException``.
    """

    order = -5  # before GradientClip / the optimizer step

    def __init__(self, max_consecutive: int = 10):
        self.max_consecutive = max_consecutive
        self.n_consecutive = 0
        self.n_skipped = 0

    def after_loss(self):
        if not self.training:
            return
        if not torch.isfinite(self.loss):
            self.n_consecutive += 1
            self.n_skipped += 1
            if self.n_consecutive >= self.max_consecutive:
                raise CancelFitException(
                    f"NaNGuard: {self.n_consecutive} consecutive non-finite losses -- training diverged."
                )
            # Zero the loss so fastai's smoothed-loss recorder doesn't log NaN forever; the model
            # is untouched because the batch is cancelled before backward.
            self.learn.loss = self.learn.loss_grad = torch.zeros_like(self.loss)
            raise CancelBatchException
        self.n_consecutive = 0

    def after_fit(self):
        if self.n_skipped:
            print(f"NaNGuard: skipped {self.n_skipped} non-finite-loss batch(es) during training.")


class MixUpMulti(Callback):
    """MixUp that supports **multiple targets** (our species/genus/family levels).

    fastai's stock ``MixUp`` reads the batch size from ``self.y.size(0)``, which fails when
    ``self.y`` is a tuple of per-level targets. This is the same handler with the batch size read
    from the input tensor ``self.x`` instead. It relies on the loss carrying ``y_int=True``
    (:class:`~lepinet.loss.FastaiLossWrapper`), so mixing happens through the loss — each level's
    per-sample loss is lerped between the two label sets — not by mixing integer labels.

    A regulariser for longer / bigger runs (``journal/2026-07-24-bigger-everything.md``). CutMix for
    the multi-target case is a further change (its ``before_batch`` also indexes ``self.y``) and is
    not implemented yet.
    """

    run_valid = False

    def __init__(self, alpha: float = 0.4):
        from torch.distributions.beta import Beta

        self.distrib = Beta(torch.tensor(alpha), torch.tensor(alpha))

    def before_train(self):
        self.stack_y = getattr(self.learn.loss_func, "y_int", False)
        if self.stack_y:
            self.old_lf, self.learn.loss_func = self.learn.loss_func, self.lf

    def after_train(self):
        if self.stack_y:
            self.learn.loss_func = self.old_lf

    def after_cancel_train(self):
        self.after_train()

    def after_cancel_fit(self):
        self.after_train()

    def before_batch(self):
        from fastai.torch_core import unsqueeze
        from fastcore.foundation import L

        bs = self.x.size(0)  # batch size from the input tensor, not the target tuple
        lam = self.distrib.sample((bs,)).squeeze().to(self.x.device)
        lam = torch.stack([lam, 1 - lam], 1)
        self.lam = lam.max(1)[0]
        shuffle = torch.randperm(bs).to(self.x.device)
        xb1, self.yb1 = tuple(L(self.xb).itemgot(shuffle)), tuple(L(self.yb).itemgot(shuffle))
        nx_dims = len(self.x.size())
        self.learn.xb = tuple(L(xb1, self.xb).map_zip(torch.lerp, weight=unsqueeze(self.lam, n=nx_dims - 1)))

    def lf(self, pred, *yb):
        from fastai.callback.mixup import reduce_loss
        from fastai.losses import NoneReduce

        if not self.training:
            return self.old_lf(pred, *yb)
        with NoneReduce(self.old_lf) as lf:
            loss = torch.lerp(lf(pred, *self.yb1), lf(pred, *yb), self.lam)
        return reduce_loss(loss, getattr(self.old_lf, "reduction", "mean"))


class DistillCallback(Callback):
    """Run a frozen teacher on each **training** batch and hand its logits to :class:`~lepinet.loss.DistillLoss`.

    The teacher must share the student's exact class vocabulary and level order (checked at build
    time in :func:`lepinet.train.train`), so its per-level logits align index-for-index with the
    student's. Runs the teacher in fp32 with no grad (the input ``xb`` is fp32 even when the student
    trains in bf16), which gives stable soft targets and never touches the student's graph.

    On non-training batches it clears ``teacher_logits`` so the loss falls back to plain hard-label
    CE — validation loss/metrics stay KD-free (and no teacher compute is wasted).
    """

    order = -10  # before MixUp etc.; here we simply never combine the two (config guards it)

    def __init__(self, teacher: torch.nn.Module):
        self.teacher = teacher

    def before_fit(self):
        device = next(self.learn.model.parameters()).device
        self.teacher.to(device).float().eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def before_batch(self):
        if not self.training:
            self.learn.loss_func.teacher_logits = None
            return
        with torch.no_grad():
            self.learn.loss_func.teacher_logits = [o.detach() for o in self.teacher(self.learn.xb[0])]


class GCCallback(Callback):
    """Force a generation-0 garbage collection after every batch.

    **Kept dormant by default** (``journal/2026-07-24-src-lepinet-baseline-port.md``, D3). It existed
    to break the reference cycle that the original head's ``_weight_bias`` cache created by stashing
    a graph-attached weight view into a persistent buffer each forward — without it, GPU memory
    climbed every batch until OOM. The clean :class:`~lepinet.heads.IndependentHead` has no such
    cache, so the cycle should be gone and this should be unnecessary. It is left here, ready to
    switch on (``GCCallback()`` in the callback list), in case a future head reintroduces a
    per-batch cycle — measure GPU memory before relying on its absence.

    ``gc.collect(0)`` (youngest generation only) is deliberate: the cycle, if present, dies within
    one batch, so gen-0 catches it at ~4 ms, whereas a full collection scans long-lived objects
    (params, optimizer state) and cost ~250 ms/batch at this class count.
    """

    def after_batch(self):
        gc.collect(0)
