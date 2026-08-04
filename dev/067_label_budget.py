"""Build labelled target-domain subsets, to price what labels would have bought.

The project's robustness result is label-free: self-training on machine-guessed labels moves the
shifted score by +7.94 pt. The reviewer question that follows is *"how many real labels would that
have taken, and what would they have cost?"* — and it is better answered than deflected.

**The design isolates label quality from everything else.** The self-training arm at 1x replication
used **12,230 images at 0.39 % of training with 98.15 %-accurate machine labels** and scored probe
0.7354. This script builds the same thing with **real** labels, so an arm at N = 12,230 differs from
it in exactly one respect: the labels are correct rather than 98 % correct. Smaller N then traces how
few labels still buy something.

**Sampling spreads across capture groups on purpose.** A user labelling 500 trap images would not
label 500 consecutive frames of one night, and near-duplicates would make a budget look further than
it is. Images are drawn round-robin over (trap, night) groups, so a budget of N covers as many
distinct nights as it can.

**Labels come only from `adapt`.** `probe` is never touched, so the evaluation stays honest — the
same discipline as the pseudo-label path (`journal/2026-08-02-the-shifted-benchmark-is-also-the-
adaptation-set.md`).

    python dev/067_label_budget.py --adapt-parquet ... --groups ... --n 2500 --out labels_2500.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def spread_sample(df: pd.DataFrame, n: int, group_col: str, seed: int = 0) -> pd.DataFrame:
    """Take ``n`` rows, round-robin over groups, so a budget covers as many groups as possible."""
    rng = np.random.default_rng(seed)
    order = {}
    for g, idx in df.groupby(group_col).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        order[g] = list(idx)
    groups = sorted(order)
    rng.shuffle(groups)
    picked, i = [], 0
    while len(picked) < n and any(order.values()):
        g = groups[i % len(groups)]
        if order[g]:
            picked.append(order[g].pop())
        i += 1
    return df.loc[picked[:n]]


def main(a):
    df = pd.read_parquet(a.adapt_parquet).reset_index(drop=True)
    groups = pd.read_parquet(a.groups)[["image_path", "group"]]
    df = df.merge(groups, on="image_path", how="left")
    if df["group"].isna().any():
        raise ValueError(f"{int(df['group'].isna().sum())} rows have no capture group; the join key "
                         f"must match dev/064's image_path exactly")

    sub = spread_sample(df, a.n, "group", a.seed)
    levels = a.levels.split(",")
    out = sub[["image_path", "filename", *levels]].copy()
    # The adapt parquet stores taxon keys as object columns holding a mix of int and str, which
    # pyarrow refuses to serialise ("Expected bytes, got a 'int' object"). Everything downstream
    # compares them as strings anyway, so normalise here rather than at each consumer.
    for c in ["image_path", "filename", *levels]:
        out[c] = out[c].astype(str)
    out["image_path"] = a.path_prefix + out["image_path"]
    out["set"] = "2"          # never validation ('1'), never test ('0') -- same guard as pseudo rows

    out.to_parquet(a.out)
    summary = {"n_requested": a.n, "n_written": len(out),
               "groups_covered": int(sub["group"].nunique()),
               "groups_available": int(df["group"].nunique()),
               "species_covered": int(sub[a.levels.split(",")[0]].nunique()),
               "labels": "REAL (from the trap dataset), drawn from `adapt` only"}
    Path(str(a.out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt-parquet", required=True)
    ap.add_argument("--groups", required=True, help="dev/064's flemming_groups.parquet")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="speciesKey,genusKey,familyKey")
    ap.add_argument("--path-prefix", default="../../flemming/images/")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
