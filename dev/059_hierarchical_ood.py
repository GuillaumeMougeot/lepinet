"""Open-set detection **stratified by taxonomic distance** (C3).

Our first OOD benchmark treated novelty as binary — a species is known or it is not. But in a
taxonomy novelty has *degrees*, and they are not equally hard:

  * **near** — unseen species, but its **genus is known** (a sibling of something trained on)
  * **mid**  — unseen genus, but its **family is known**
  * **far**  — unseen **family** entirely

A useful novelty score should find `far` easy and `near` hard, because `near` is genuinely similar to
what the model knows. Reporting one AUROC over all three hides that structure and lets an easy
long-tail dominate the number.

Data comes from the **unfiltered** global_lepi parquet (`..._postprocessed.parquet`) rather than the
quality-filtered one. Training used `min_img_per_spc=50`, so every species below that floor is
genuinely unseen — a held-out-taxa split we get for free, in-domain, with **64,504 novel images
across 38,907 species** instead of the 3,171 the filtered parquet exposed.

    python dev/059_hierarchical_ood.py --model ckpt.pt --parquet unfiltered.parquet \\
        --img-dir images/ --out ood_strat.json

Caveat kept in view: these unseen species are the *rare* ones, so they may be systematically harder
(fewer/worse images) than a random held-out species would be. A deliberate hold-out of *common*
taxa is the controlled version (C3b) and is the follow-up if these numbers matter.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

LEVELS = ["speciesKey", "genusKey", "familyKey"]


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(neg scores above pos) via Mann-Whitney; here pos = novel, neg = known."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    ranks = allv.argsort().argsort()
    r_neg = ranks[: len(neg)].sum()
    return float((r_neg - len(neg) * (len(neg) - 1) / 2) / (len(neg) * len(pos)))


def stratify(df: pd.DataFrame, vocabs: dict) -> pd.Series:
    """Label each row known / near / mid / far by how much of its lineage the model has seen."""
    known = {lv: {str(v) for v in vocabs[lv]} for lv in LEVELS}
    sp_new = ~df[LEVELS[0]].astype(str).isin(known[LEVELS[0]])
    gn_new = ~df[LEVELS[1]].astype(str).isin(known[LEVELS[1]])
    fm_new = ~df[LEVELS[2]].astype(str).isin(known[LEVELS[2]])
    out = pd.Series("known", index=df.index, dtype=object)
    out[sp_new] = "near"          # unseen species, known genus
    out[sp_new & gn_new] = "mid"  # unseen genus, known family
    out[sp_new & gn_new & fm_new] = "far"
    return out


@torch.no_grad()
def max_cosine(model, dls, df, device, num_workers=32):
    from lepinet.test import dl_num_workers

    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    dl = dls.test_dl(df, num_workers=nw)
    print(f"  dataloader: num_workers={nw}, batches={len(dl)}")
    body, head = model[0], model[1].head
    model.to(device).eval()
    w = torch.nn.functional.normalize(head.layers[0].weight.detach(), dim=1).to(device)
    out = []
    for batch in dl:
        feats = body(batch[0].to(device))
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1) if feats.ndim == 4 else feats
        out.append((head.preclassification(pooled.float()) @ w.T).max(1).values.cpu().numpy())
    return np.concatenate(out)


def main(a):
    from lepinet.data import make_dls
    from lepinet.test import _paths_exist, load_model, resolve_checkpoint_path

    ckpt = torch.load(resolve_checkpoint_path(a.model), map_location="cpu", weights_only=False)
    vocabs = ckpt["vocabs"]
    df = pd.read_parquet(a.parquet)
    df = df[df["set"].astype(str) == a.test_set]
    for lv in LEVELS:
        df[lv] = df[lv].astype(str)
    df["image_path"] = df[LEVELS[0]] + "/" + df["filename"]
    df["_strat"] = stratify(df, vocabs)

    # The unfiltered catalogue lists images the mirror may not have.
    from pathlib import Path
    keep = _paths_exist(Path(a.img_dir), df["image_path"])
    print(f"{len(df)} rows | {int((~keep).sum())} missing from disk -> dropped")
    df = df[keep]

    # Cap each stratum: 'far' is tiny, 'known' is huge; equal-ish groups keep the AUROCs comparable.
    parts = []
    for name, g in df.groupby("_strat"):
        n = min(len(g), a.per_stratum)
        parts.append(g.sample(n, random_state=0))
        print(f"  {name:6s} {len(g):7d} available -> {n} sampled")
    df = pd.concat(parts).reset_index(drop=True)
    df["is_valid"] = np.arange(len(df)) % 5 == 0

    # The loader's CategoryBlock can only encode labels present in the training vocabulary — and
    # novel taxa are, by definition, absent ("Label '1861242' was not included in the training
    # dataset"). Scoring never reads y (the novelty score is max cos over prototypes), so feed the
    # loader an in-vocab placeholder and keep the true stratum in `_strat`.
    loader_df = df[["image_path", "is_valid", *LEVELS]].copy()
    for lv in LEVELS:
        loader_df[lv] = str(vocabs[lv][0])
    dls = make_dls(loader_df, vocabs, a.img_dir,
                   int(a.img_size * 460 / 256), a.img_size, 128, a.num_workers,
                   lowmem=False, levels=LEVELS)
    model, _ = load_model(ckpt, img_size=a.img_size)
    score = max_cosine(model, dls, loader_df, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       a.num_workers)

    strat = df["_strat"].to_numpy()
    known = score[strat == "known"]
    res = {"n": {k: int((strat == k).sum()) for k in ("known", "near", "mid", "far")},
           "mean_max_cos": {k: float(score[strat == k].mean()) for k in ("known", "near", "mid", "far")
                            if (strat == k).any()},
           "auroc_vs_known": {}}
    for k in ("near", "mid", "far"):
        if (strat == k).any():
            res["auroc_vs_known"][k] = auroc(score[strat == k], known)
    novel = np.isin(strat, ["near", "mid", "far"])
    res["auroc_vs_known"]["all_novel"] = auroc(score[novel], known)
    res["head"] = ckpt["head"]

    print("\nAUROC (known vs novel) by taxonomic distance:")
    for k, v in res["auroc_vs_known"].items():
        print(f"  {k:10s} {v:.4f}   mean max-cos {res['mean_max_cos'].get(k, float('nan')):.3f}"
              if k in res["mean_max_cos"] else f"  {k:10s} {v:.4f}")
    print(f"  known mean max-cos {res['mean_max_cos']['known']:.3f}")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", default="ood_stratified.json")
    ap.add_argument("--test-set", default="0")
    ap.add_argument("--per-stratum", type=int, default=8000)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=32)
    main(ap.parse_args())
