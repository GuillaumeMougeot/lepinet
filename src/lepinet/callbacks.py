"""fastai training callbacks: NaN safety and an (optional) GC leak guard.

``HostMemoryGuard`` (host-RAM OOM guard) lives in :mod:`lepinet.memory` and is re-exported here so
callers have one import site for "the training callbacks".
"""
from __future__ import annotations

import gc

import torch
from fastai.callback.core import Callback, CancelBatchException, CancelFitException

from .memory import HostMemoryGuard  # re-export

__all__ = ["NaNGuard", "GCCallback", "HostMemoryGuard"]


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


class GCCallback(Callback):
    """Force a generation-0 garbage collection after every batch.

    **Kept dormant by default** (``journal/2026-07-src-lepinet-baseline-port.md``, D3). It existed
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
