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

from fastai.callback.core import Callback, CancelFitException


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


class MarginRamp(Callback):
    """fastai callback: ramp the ArcFace margin 0 → ``m_max`` over ``n_steps``, logging the loss.

    The direct analogue of ``lr_find``'s LR ramp. Writes ``loss_func.criterion.arc_margins[0]``
    (species — the level whose margin matters) before each batch and records an exponentially
    smoothed loss, then stops the fit. Reading it is the same as reading ``lr_find``: the curve is
    flat while the margin is absorbed and turns up at the point the model can no longer satisfy the
    rotated target.
    """

    order = 5  # after the loss function exists

    def __init__(self, m_max=1.0, n_steps=400, beta=0.98):
        self.m_max, self.n_steps, self.beta = m_max, n_steps, beta
        self.step, self.avg, self.hist = 0, 0.0, []

    @property
    def _crit(self):
        return getattr(self.learn.loss_func, "criterion", self.learn.loss_func)

    def before_fit(self):
        crit = self._crit
        if getattr(crit, "arc_margins", None) is None:
            crit.arc_margins = [0.0] * crit.n_levels

    def _margin(self):
        return self.m_max * self.step / max(self.n_steps - 1, 1)

    def before_batch(self):
        if self.training:
            self._crit.arc_margins[0] = float(self._margin())

    def after_batch(self):
        if not self.training:
            return
        loss = float(self.learn.loss.detach().cpu())
        self.avg = self.beta * self.avg + (1 - self.beta) * loss
        smooth = self.avg / (1 - self.beta ** (self.step + 1))
        self.hist.append({"step": self.step, "margin": self._margin(), "loss": loss, "smooth": smooth})
        self.step += 1
        if self.step >= self.n_steps:
            raise CancelFitException


def knee(hist, tol: float = 1.05) -> float:
    """Largest margin whose smoothed loss stays within ``tol`` of the running minimum.

    Same reading as ``lr_find``: the point just before the curve turns up is the usable limit; the
    recommended value is a safety factor below it.
    """
    best = min(h["smooth"] for h in hist)
    ok = [h["margin"] for h in hist if h["smooth"] <= tol * best]
    return max(ok) if ok else 0.0


def run_margin_find(config_path, m_max=1.0, n_steps=400, out="margin_find.json", init_from=None):
    """Build the training setup from a config, ramp the margin, and report the knee.

    ``init_from`` is **not optional in practice**: from a random init the loss is dominated by
    ordinary early-training dynamics (it rises regardless of the margin), so the knee is
    meaningless — measured on a synthetic run, it reads 0. ``lr_find`` gets away with a cold start
    because the LR's effect is immediate and overwhelming; a margin's is not. Warm-starting from a
    **converged, margin-free** checkpoint makes the loss flat, so any rise is attributable to ``m``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lepinet import data as data_mod
    from lepinet.callbacks import NaNGuard
    from lepinet.config import load_config
    from lepinet.heads import build_head
    from lepinet.loss import FastaiLossWrapper, MultiLevelCELoss
    from lepinet.metrics import default_metrics
    from lepinet.model import arch_body_features, arch_is_vit, build_learner, resolve_arch

    cfg, _ = load_config(config_path)
    if cfg.head != "arcface":
        raise SystemExit(f"margin find needs head='arcface', config has {cfg.head!r}")
    data_mod.ensure_fork_start_method()
    levels = list(cfg.levels)
    df, _ = data_mod.gen_df(cfg.parquet_path, Path(cfg.out_dir), cfg.min_img_per_spc, cfg.fold,
                            Path(cfg.parquet_path).parent / "hierarchy.csv", cfg.family_filter, levels=levels)
    vocabs = {lv: sorted(df[lv].unique().tolist()) for lv in levels}
    n_classes = [len(vocabs[lv]) for lv in levels]
    wgts = data_mod.sample_weights(df, level=levels[0], power=cfg.oversample_power, levels=levels)
    dls = data_mod.make_dls(df, vocabs, cfg.img_dir, cfg.aug_img_size, cfg.img_size, cfg.batch_size,
                            cfg.num_workers, aug_kwargs=cfg.aug_kwargs, sample_wgts=wgts, levels=levels)
    arch = resolve_arch(cfg.model_arch_name)
    vit = arch_is_vit(arch, img_size=cfg.img_size)
    nf = arch_body_features(arch, img_size=cfg.img_size)
    head = build_head("arcface", nf, n_classes, hidden=cfg.hidden, pool=not vit,
                      scale=cfg.arcface_scale, margin=0.0, zscore=cfg.arcface_zscore)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = MultiLevelCELoss(n_classes, weights=cfg.level_weights, label_smoothing=cfg.label_smoothing,
                            device=device, arc_scale=cfg.arcface_scale,
                            arc_margins=[0.0] * len(n_classes),
                            arc_zscore=cfg.arcface_zscore, arc_ndim=head.head.preclass_size)
    ramp = MarginRamp(m_max=m_max, n_steps=n_steps)
    learn = build_learner(dls, arch, head, FastaiLossWrapper(crit), default_metrics(levels),
                          Path(cfg.out_dir) / "models", [NaNGuard(), ramp], optimizer=cfg.optimizer, vit=vit)
    if init_from:
        from lepinet.test import resolve_checkpoint_path
        state = torch.load(resolve_checkpoint_path(init_from), map_location="cpu", weights_only=False)
        missing, unexpected = learn.model.load_state_dict(state["model_state_dict"], strict=False)
        print(f"warm start from {init_from} (missing {len(missing)}, unexpected {len(unexpected)})")
    else:
        print("WARNING: no --init-from; from a random init the knee is dominated by early-training "
              "dynamics and will not reflect the margin.")
    if cfg.fp16:
        learn = learn.to_bf16() if cfg.precision == "bf16" else learn.to_fp16()
    learn.unfreeze()
    print(f"Margin range test: 0 -> {m_max} over {n_steps} steps (zscore={cfg.arcface_zscore})")
    learn.fit(1, cfg.base_lr)

    k = knee(ramp.hist)
    rec = round(0.75 * k, 2)
    json.dump({"knee": k, "recommended_m": rec, "history": ramp.hist}, open(out, "w"), indent=2)
    ms = [h["margin"] for h in ramp.hist]
    sm = [h["smooth"] for h in ramp.hist]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ms, sm, color="#1b6b4a")
    plt.axvline(k, ls="--", c="#c2492f", label=f"knee m={k:.2f}")
    plt.axvline(rec, ls=":", c="#333", label=f"recommended m={rec:.2f}")
    plt.xlabel("ArcFace margin m")
    plt.ylabel("smoothed train loss")
    plt.title("Margin range test (the lr_find analogue)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out).with_suffix(".png"), dpi=150)
    print(f"knee m={k:.3f} -> recommended m={rec} | wrote {out} and {Path(out).with_suffix('.png')}")
    return rec


def main(a):
    from lepinet.config import load_config

    cfg, _ = load_config(a.config)
    import pandas as pd
    n_species = pd.read_parquet(cfg.parquet_path)[cfg.levels[0]].nunique()
    hidden = cfg.hidden if isinstance(cfg.hidden, int) else 1280
    floor, zs = analytic_scale_floor(n_species), zscore_scale(hidden)
    print(f"scale: floor(p=0.9)={floor:.1f} | z-score gives {zs:.1f} -> "
          f"{'ADEQUATE, s is inert under zscore' if zs >= floor else 'set s explicitly'}")
    if a.scale_only:
        return
    run_margin_find(a.config, m_max=a.m_max, n_steps=a.steps, out=a.out, init_from=a.init_from)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="margin_find.json")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--m-max", type=float, default=1.0)
    ap.add_argument("--scale-only", action="store_true")
    ap.add_argument("--init-from", help="Converged margin-free checkpoint to warm-start from (strongly recommended).")
    main(ap.parse_args())
