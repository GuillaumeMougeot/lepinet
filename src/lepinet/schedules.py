"""LR schedules and crash-recovery, for Muon-safe (LR-only) training.

Muon (MuonAuxAdamW) re-partitions parameter groups and takes tuple betas, so fastai's stock
`fit_one_cycle`/`fit_flat_cos` -- which also schedule *momentum* -- choke on it. These build the
LR curve as a plain function of position in [0, 1] and drive it with a ParamScheduler over LR
only. Split out of the fit call so `fit_resume` can rebuild the exact curve and continue it after
an interruption instead of restarting the anneal (journal/2026-07-gpu-hang.md).

one_cycle >> flat_cos here: flat_cos only anneals its final ~25%, and at this dataset/epoch scale
the model is still converging hard at the end, so it is graded mid-descent. one_cycle anneals
~90% of the run and was the single biggest optimisation lever
(journal/2026-07-why-was-fastai-behind-mini-trainer.md).
"""
from __future__ import annotations

from fastai.callback.schedule import (ParamScheduler, SchedCos, SchedLin, SchedNo,
                                      combine_scheds)


def warmup_cos_schedule(n_epoch, lr, warmup_epochs, schedule,
                        warmup_div=100.0, cos_pct=0.25):
    """LR curve (fn of pos in [0,1] over the whole run): a short warmup ramp, then either a full
    cosine anneal (`one_cycle`) or flat-then-cosine over the final `cos_pct` (`flat_cos`)."""
    warmup_pct = warmup_epochs / n_epoch
    if not 0 < warmup_pct < 1:
        raise ValueError(f"warmup_epochs ({warmup_epochs}) must be in (0, n_epoch={n_epoch}).")
    ramp = SchedLin(lr / warmup_div, lr)
    if schedule == "one_cycle":
        pcts, scheds = [warmup_pct, 1 - warmup_pct], [ramp, SchedCos(lr, 0)]
    else:  # flat_cos
        flat_pct = 1 - warmup_pct - cos_pct
        if flat_pct <= 0:
            raise ValueError(f"warmup_epochs too large: no room for the flat phase before the "
                             f"final {cos_pct:.0%} cosine anneal.")
        pcts, scheds = [warmup_pct, flat_pct, cos_pct], [ramp, SchedNo(lr, lr), SchedCos(lr, 0)]
    return combine_scheds(pcts, scheds)


def front_loaded_schedule(n_epoch, lr, warmup_epochs, fast_decay_epochs, lr_mid_frac,
                          warmup_div=100.0):
    """Warmup, a fast cosine decay to `lr * lr_mid_frac` by `fast_decay_epochs`, then a slow
    cosine to 0 -- spends the learning-rate budget early."""
    warmup_pct = warmup_epochs / n_epoch
    fast_pct = fast_decay_epochs / n_epoch - warmup_pct
    slow_pct = 1 - fast_decay_epochs / n_epoch
    if not (warmup_pct > 0 and fast_pct > 0 and slow_pct > 0):
        raise ValueError(f"Need 0 < warmup_epochs ({warmup_epochs}) < fast_decay_epochs "
                         f"({fast_decay_epochs}) < n_epoch ({n_epoch}).")
    lr_mid = lr * lr_mid_frac
    return combine_scheds([warmup_pct, fast_pct, slow_pct],
                          [SchedLin(lr / warmup_div, lr), SchedCos(lr, lr_mid), SchedCos(lr_mid, 0)])


def fit_scheduled(learn, n_epoch, sched_fn):
    """learn.fit driving LR (only) with `sched_fn` (from the builders above). Model must be
    unfrozen -- Muon re-partitions groups and doesn't round-trip through fastai's freeze."""
    learn.unfreeze()
    learn.fit(n_epoch, cbs=ParamScheduler({"lr": sched_fn}))


def fit_resume(learn, full_sched, n_epoch, epochs_done):
    """Continue an interrupted run on the *same* LR curve. fastai's ParamScheduler position
    restarts at 0 each fit, so a naive resume replays the warmup; this remaps the restarted [0,1]
    range onto [epochs_done/n_epoch, 1] of the original curve, so the LR trajectory across the
    interruption matches an uninterrupted run. Load the last checkpoint's weights first (Muon
    momentum restarts cold -- costs a few batches of re-warming, not correctness)."""
    start = epochs_done / n_epoch
    if n_epoch - epochs_done <= 0:
        raise ValueError(f"epochs_done ({epochs_done}) >= n_epoch ({n_epoch}); nothing to resume.")
    learn.unfreeze()
    resumed = lambda pos: full_sched(start + pos * (1 - start))
    learn.fit(n_epoch - epochs_done, cbs=ParamScheduler({"lr": resumed}))
