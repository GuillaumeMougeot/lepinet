"""An ``lr_find`` equivalent for ArcFace's margin ``m`` — and why the scale ``s`` needs no search.

`lr_find` works because the learning rate's effect on the loss appears within ~100 batches. The two
ArcFace hyperparameters are not alike, so they deserve different treatments:

**`s` (scale) — solved analytically, no search needed.** For the true class to reach probability `p`
among `C` classes (non-target cosines ≈ 0, target ≈ 1), the softmax needs
``s ≥ log((C−1)·p/(1−p))``  (NormFace, Wang et al. 2017).
Below that floor the loss cannot be driven down no matter how good the embedding; far above it, `s`
only sharpens an already-saturated softmax. At C = 12,041 that is **s ≥ 11.6** (p=0.9) or **14.0**
(p=0.99) — so the usual s ∈ [16, 64] is a plateau, not a peak, and the common practice of grid-
searching it is mostly wasted compute.

*And this is exactly why the z-score variant needs no `s` at all*: ``cosine_to_zscore`` multiplies by
``√(ndim−2)``, i.e. **35.7** at hidden=1280 and **15.9** at hidden=256 — already above the floor. The
transform supplies a principled scale that the raw-cosine head has to invent.

**`m` (margin) — a range test, the true `lr_find` analogue.** `m`'s effect *does* appear quickly: too
large and the loss stalls or climbs within a few hundred steps (the model cannot satisfy the rotated
target), too small and nothing changes. So ramp `m` from 0 upward across a short run, record the
smoothed loss, and read off the **knee** — the largest margin the model still absorbs. The best `m`
for open-set is just below it. One short run replaces a grid of full trainings.

    python dev/055_margin_scale_find.py --config configs/<arcface>.yaml --steps 400 --m-max 1.0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def analytic_scale_floor(n_classes: int, p: float = 0.9) -> float:
    """Smallest ``s`` that lets the softmax express confidence ``p`` over ``n_classes``."""
    return math.log((n_classes - 1) * p / (1.0 - p))


def zscore_scale(ndim: int) -> float:
    """The scale ``cosine_to_zscore`` applies implicitly: ``√(ndim − 2)``."""
    return math.sqrt(ndim - 2.0)


def scale_report(n_classes_per_level, hidden_dims=(1280, 256)) -> dict:
    rep = {"floors": {}, "zscore_scale": {}}
    for c in n_classes_per_level:
        rep["floors"][str(c)] = {f"p={p}": round(analytic_scale_floor(c, p), 2) for p in (0.9, 0.99)}
    for d in hidden_dims:
        rep["zscore_scale"][str(d)] = round(zscore_scale(d), 2)
    return rep


class MarginRamp:
    """fastai callback: ramp the ArcFace margin 0 → ``m_max`` across ``n_steps``, logging the loss.

    The direct analogue of ``lr_find``'s LR ramp. Reads/writes ``loss_func.criterion.arc_margins``
    (level 0 only — the species margin is the one that matters) and records a smoothed loss so the
    knee is legible.
    """

    def __init__(self, m_max=1.0, n_steps=400, beta=0.98):
        from fastai.callback.core import Callback  # noqa: F401  (documented dependency)

        self.m_max, self.n_steps, self.beta = m_max, n_steps, beta
        self.step, self.avg, self.hist = 0, 0.0, []

    def _set_margin(self, learn, m):
        crit = getattr(learn.loss_func, "criterion", learn.loss_func)
        if crit.arc_margins is None:
            crit.arc_margins = [0.0] * crit.n_levels
        crit.arc_margins[0] = float(m)

    def before_batch(self, learn):
        self._set_margin(learn, self.m_max * self.step / max(self.n_steps - 1, 1))

    def after_batch(self, learn, loss):
        m = self.m_max * self.step / max(self.n_steps - 1, 1)
        self.avg = self.beta * self.avg + (1 - self.beta) * loss
        smooth = self.avg / (1 - self.beta ** (self.step + 1))
        self.hist.append({"step": self.step, "margin": m, "loss": loss, "smooth": smooth})
        self.step += 1
        return self.step < self.n_steps


def knee(hist, tol: float = 1.05) -> float:
    """Largest margin whose smoothed loss stays within ``tol`` of the running minimum.

    Same reading as ``lr_find``: the point just before the curve turns up is the usable limit; the
    recommended value is a safety factor below it.
    """
    best = min(h["smooth"] for h in hist)
    ok = [h["margin"] for h in hist if h["smooth"] <= tol * best]
    return max(ok) if ok else 0.0


def main(a):
    from lepinet.config import load_config

    cfg, _ = load_config(a.config)
    # Scale first: it is free.
    import pandas as pd
    df = pd.read_parquet(cfg.parquet_path)
    n_species = df[cfg.levels[0]].nunique()
    hidden = cfg.hidden if isinstance(cfg.hidden, int) else 1280
    rep = scale_report([n_species], hidden_dims=(hidden,))
    floor = analytic_scale_floor(n_species)
    zs = zscore_scale(hidden)
    print(f"scale: floor(p=0.9) = {floor:.1f} | z-score gives {zs:.1f} "
          f"-> {'ADEQUATE, no s needed' if zs >= floor else 'TOO SMALL, set s explicitly'}")
    print(f"       configured s = {cfg.arcface_scale} "
          f"({'above floor' if cfg.arcface_scale >= floor else 'BELOW FLOOR — will underfit'})")
    Path(a.out).write_text(json.dumps({"scale": rep, "config_s": cfg.arcface_scale}, indent=2))
    print(f"wrote {a.out}")
    print("\nMargin range test: wire MarginRamp into a short training run "
          "(see the class docstring) and read knee(hist).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="margin_scale.json")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--m-max", type=float, default=1.0)
    main(ap.parse_args())
