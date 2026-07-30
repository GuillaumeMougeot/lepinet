"""Find ArcFace's margin by *short independent fine-tunes*, scored on embedding geometry.

Replaces the range test in `dev/055`, which was invalid: a margin **mechanically** lowers the
true-class logit, so cross-entropy rises with `m` regardless of model quality, and a ramp also
entangles `m` with training progress (journal 2026-07-story-and-directions.md). The signal we want —
open-set separation — only appears once training has *reshaped the embedding*, so the probe must
include adaptation.

Protocol, per margin value, all from the **same converged margin-free checkpoint**:
  1. warm-start,
  2. fine-tune ``--steps`` batches with that (fixed) margin,
  3. score the **geometry margin** ``intra − inter`` on a held-out sample — cosine to the model's own
     prototypes, the quantity that drives the novelty score.

Geometry margin is a cheap stand-in for OOD AUROC, and we have two calibration points from full
runs: **0.182 → AUROC 0.601** (no margin) and **0.610 → AUROC 0.9115** (m = 0.3). Species accuracy is
also reported, since the useful `m` is the one that buys separation without costing accuracy.

    python dev/057_margin_grid.py --config <arcface.yaml> --init-from <baseline.pt> \\
        --margins 0.1 0.2 0.3 0.5 --steps 200 --out margin_grid.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from fastai.callback.core import Callback, CancelFitException


class StopAfter(Callback):
    """Stop the fit after N training batches (a probe, not a run)."""

    order = 5

    def __init__(self, n_steps: int):
        self.n_steps, self.step = n_steps, 0

    def after_batch(self):
        if not self.training:
            return
        self.step += 1
        if self.step >= self.n_steps:
            raise CancelFitException


@torch.no_grad()
def geometry_margin(model, dls, df, levels, vocabs, device, n_batches=8):
    """``intra − inter`` on a few validation batches: mean cos to own prototype minus to nearest wrong."""
    from lepinet.test import dl_num_workers

    dl = dls.test_dl(df, num_workers=dl_num_workers(dls.train))
    body, head = model[0], model[1].head
    model.to(device).eval()
    w = torch.nn.functional.normalize(head.layers[0].weight.detach(), dim=1).to(device)
    vidx = {str(v): i for i, v in enumerate(vocabs[levels[0]])}
    own_all = np.array([vidx.get(s, -1) for s in df[levels[0]].astype(str)])
    intra, inter, correct, seen = [], [], 0, 0
    for bi, batch in enumerate(dl):
        if bi >= n_batches:
            break
        feats = body(batch[0].to(device))
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1) if feats.ndim == 4 else feats
        sims = (head.preclassification(pooled.float()) @ w.T).cpu().numpy()
        idx = own_all[seen: seen + len(sims)]
        ok = idx >= 0
        if ok.any():
            rows = np.arange(len(sims))[ok]
            own = sims[rows, idx[ok]]
            masked = sims[ok].copy()
            masked[np.arange(ok.sum()), idx[ok]] = -np.inf
            intra.append(own)
            inter.append(masked.max(1))
            correct += int((sims[ok].argmax(1) == idx[ok]).sum())
        seen += len(sims)
    intra, inter = np.concatenate(intra), np.concatenate(inter)
    return {"intra": float(intra.mean()), "inter": float(inter.mean()),
            "geometry_margin": float((intra - inter).mean()),
            "acc": correct / max(len(intra), 1), "n": int(len(intra))}


def build(cfg, margin, init_from, steps):
    """A learner warm-started from ``init_from``, with a fixed ArcFace margin."""
    from lepinet import data as data_mod
    from lepinet.callbacks import NaNGuard
    from lepinet.heads import build_head
    from lepinet.loss import FastaiLossWrapper, MultiLevelCELoss
    from lepinet.metrics import default_metrics
    from lepinet.model import arch_body_features, arch_is_vit, build_learner, resolve_arch
    from lepinet.test import resolve_checkpoint_path

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
                      scale=cfg.arcface_scale, margin=margin, zscore=cfg.arcface_zscore)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crit = MultiLevelCELoss(n_classes, weights=cfg.level_weights, label_smoothing=cfg.label_smoothing,
                            device=device, arc_scale=cfg.arcface_scale,
                            arc_margins=[margin] + [0.0] * (len(n_classes) - 1),
                            arc_zscore=cfg.arcface_zscore, arc_ndim=head.head.preclass_size)
    learn = build_learner(dls, arch, head, FastaiLossWrapper(crit), default_metrics(levels),
                          Path(cfg.out_dir) / "models", [NaNGuard(), StopAfter(steps)],
                          optimizer=cfg.optimizer, vit=vit)
    # A SEPARATE loader for scoring: the training dls use the lowmem (numpy-indexed) getters, whose
    # `test_dl` takes indices, not a DataFrame — feeding it one raises a fastai type assertion.
    # `lepinet.test.evaluate` builds its loaders with lowmem=False for the same reason.
    val_df = df[df["is_valid"]].head(4096).reset_index(drop=True) if "is_valid" in df.columns else df.head(4096)
    eval_dls = data_mod.make_dls(val_df, vocabs, cfg.img_dir, cfg.aug_img_size, cfg.img_size,
                                 cfg.batch_size, cfg.num_workers, lowmem=False, levels=levels)
    state = torch.load(resolve_checkpoint_path(init_from), map_location="cpu", weights_only=False)
    learn.model.load_state_dict(state["model_state_dict"], strict=False)
    if cfg.fp16:
        learn = learn.to_bf16() if cfg.precision == "bf16" else learn.to_fp16()
    learn.unfreeze()
    return learn, eval_dls, val_df, levels, vocabs, device


def main(a):
    from lepinet.config import load_config

    cfg, _ = load_config(a.config)
    if cfg.head != "arcface":
        raise SystemExit(f"needs head='arcface', got {cfg.head!r}")

    results = {}
    for m in a.margins:
        print(f"\n=== margin {m} ===")
        c = copy.deepcopy(cfg)
        learn, eval_dls, val_df, levels, vocabs, device = build(c, m, a.init_from, a.steps)
        val = val_df.head(a.n_eval)
        base = geometry_margin(learn.model, eval_dls, val, levels, vocabs, device)
        learn.fit(1, cfg.base_lr)
        after = geometry_margin(learn.model, eval_dls, val, levels, vocabs, device)
        results[str(m)] = {"before": base, "after": after}
        print(f"  geometry margin {base['geometry_margin']:.3f} -> {after['geometry_margin']:.3f} "
              f"| acc {base['acc']:.3f} -> {after['acc']:.3f}")
        del learn
        torch.cuda.empty_cache()

    # Calibration from full runs: geometry margin 0.182 -> AUROC 0.601, 0.610 -> 0.9115.
    best = max(results, key=lambda k: results[k]["after"]["geometry_margin"])
    payload = {"results": results, "best_margin_by_geometry": float(best),
               "calibration": {"0.182": 0.601, "0.610": 0.9115},
               "note": "geometry margin is a proxy for OOD AUROC; confirm the winner with a full run"}
    Path(a.out).write_text(json.dumps(payload, indent=2))
    print(f"\nbest by geometry margin: m={best} -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", required=True, help="Converged margin-free checkpoint.")
    ap.add_argument("--margins", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=1024)
    ap.add_argument("--out", default="margin_grid.json")
    main(ap.parse_args())
