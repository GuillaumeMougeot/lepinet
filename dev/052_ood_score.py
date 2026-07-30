"""Open-set / OOD scoring for an ArcFace (or any cosine) lepinet model — #8.

Face-recognition insight ([[2026-07-25-teacher-student-app-bridge]] Q5): an ArcFace embedding gives
each class a tight, well-separated angular region, and the **max cosine similarity to any class
prototype** is a meaningful "do I know this?" score even for classes never seen in training. So:

    OOD score(image) = - max_species_logit(image)        # high logit = confident/known, low = novel

We evaluate it as open-set detection on a dataset that mixes known and OOD species (flemming):
label = 1 if the true species is out-of-vocabulary (novel), score = -max_species_logit; report
**AUROC** (how well the score separates known from novel) plus the known/OOD score distributions.
Hierarchy extension: the same against genus/family prototypes says "unknown species, but Noctuidae".

    python dev/052_ood_score.py --model '...arcface...*.pt' --parquet flemming.parquet \\
        --img-dir /path/referenced --out ood.json [--img-size 320]

Runs after the ArcFace model trains. Compares fairly against the plain cosine (independent) model —
the hypothesis is ArcFace's margin makes the known/OOD gap wider (better AUROC).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from lepinet.test import dl_num_workers


@torch.no_grad()
def max_logits(model, dls, test_df, device, level_idx: int = 0, num_workers: int | None = None) -> np.ndarray:
    """Per-image max logit at ``level_idx`` (species) — the (negated) OOD score input.

    ``num_workers`` must be passed explicitly: fastai's ``DataLoader.num_workers`` attribute is a
    hardcoded dummy (always 1), so reading it pinned this to a single worker (~1 img/s on the /work
    mount). See ``lepinet.test.dl_num_workers``.
    """
    import time
    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    test_dl = dls.test_dl(test_df, num_workers=nw)
    print(f"OOD dataloader: num_workers={nw}, batches={len(test_dl)}")
    t0 = time.perf_counter()
    model.to(device).eval()
    out = []
    for batch in test_dl:
        logits = model(batch[0].to(device))[level_idx].float()
        out.append(logits.max(dim=1).values.cpu().numpy())
    res = np.concatenate(out)
    dt = time.perf_counter() - t0
    print(f"OOD inference: {len(res)} images in {dt:.1f}s = {len(res)/max(dt,1e-9):.1f} img/s")
    return res


def ood_auroc(scores_known: np.ndarray, scores_ood: np.ndarray) -> float:
    """AUROC treating OOD as the positive class; ``score`` = -max_logit (higher ⇒ more OOD).

    Threshold-free rank statistic (no sklearn needed): P(score_ood > score_known)."""
    from itertools import product  # noqa: F401  (kept explicit; we use a vectorized rank instead)
    all_s = np.concatenate([scores_known, scores_ood])
    ranks = all_s.argsort().argsort()  # 0..N-1
    r_ood = ranks[len(scores_known):].sum()
    n_k, n_o = len(scores_known), len(scores_ood)
    auc = (r_ood - n_o * (n_o - 1) / 2) / (n_k * n_o)  # Mann–Whitney U / (n_k n_o)
    return float(auc)


def evaluate_ood(model_path, parquet_path, img_dir, out, img_size=256, levels=None, num_workers=32):
    from lepinet.data import DEFAULT_LEVELS, filter_df, make_dls
    from lepinet.test import dl_num_workers, load_model, resolve_checkpoint_path  # noqa: F401

    ckpt = torch.load(resolve_checkpoint_path(model_path), map_location="cpu", weights_only=False)
    levels = levels or ckpt.get("levels", DEFAULT_LEVELS)
    vocabs = ckpt["vocabs"]
    known = {str(v) for v in vocabs[levels[0]]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = filter_df(pd.read_parquet(parquet_path), keep_in=["0"], levels=levels)
    if "image_path" not in df.columns:
        df["image_path"] = df[levels[0]].astype(str) + "/" + df["filename"]
    for lv in levels:
        df[lv] = df[lv].astype(str)
    df["is_valid"] = np.arange(len(df)) % 5 == 0
    is_ood = ~df[levels[0]].isin(known)
    print(f"{len(df)} images | OOD-species images: {int(is_ood.sum())} ({100*is_ood.mean():.1f}%)")

    dls = make_dls(df[["image_path", "is_valid", *levels]].reset_index(drop=True),
                   vocabs, img_dir, int(img_size * 460 / 256), img_size, 128, num_workers, lowmem=False, levels=levels)
    model, _ = load_model(ckpt, img_size=img_size)
    ml = max_logits(model, dls, df, device, num_workers=num_workers)
    score = -ml  # higher ⇒ more OOD
    auroc = ood_auroc(score[~is_ood.values], score[is_ood.values]) if is_ood.any() else float("nan")
    res = {
        "auroc_ood": auroc,
        "n": int(len(df)), "n_ood": int(is_ood.sum()),
        "known_max_logit": {"mean": float(ml[~is_ood.values].mean()), "std": float(ml[~is_ood.values].std())},
        "ood_max_logit": {"mean": float(ml[is_ood.values].mean()) if is_ood.any() else None,
                          "std": float(ml[is_ood.values].std()) if is_ood.any() else None},
        "head": ckpt["head"],
    }
    json.dump(res, open(out, "w"), indent=2)
    print(f"OOD AUROC (known vs novel species): {auroc:.4f}  -> {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", default="ood.json")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=32)
    a = ap.parse_args()
    evaluate_ood(a.model, a.parquet, a.img_dir, a.out, a.img_size, num_workers=a.num_workers)
