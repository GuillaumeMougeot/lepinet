"""L7 — was cutting the head of the distribution a mistake?

Our dataset construction capped images per species at roughly 2,000. The intent was balance; it was
never tested. Two facts found on 2026-08-27 make it worth testing now:

**ToL-200M did not do it.** For the 12,494 species we share, ToL holds **19.1 M images against our
6.2 M** — 3.1x more — while the *medians* are close (201 vs 260). The whole difference is the head.
And only 1.76 M of ToL's Lepidoptera images belong to species we lack, so the data we are missing is
overwhelmingly **more images of species we already model**, not tail taxa.

**Our own findings argue the cap was the wrong instrument.** Capping is *data-level* rebalancing,
which is exactly what L4/cRT concluded belongs in the classifier instead; and the closest measured
relative, sqrt-oversampling, costs 1.52 pt under shift at 20 M and **2.88 pt at 198 M** -- the one
intervention that got *worse* with scale. Meanwhile P1b established that representation quality is
what dominates under shift, and a representation is what more data buys.

## Why a sweep downward answers a question about going upward

We do not have the discarded images locally, so we cannot simply restore the head. But we can measure
the **slope**. Capping harder uses only data we already hold:

    cap 250  -> 2.13 M train images (37 % of current)
    cap 500  -> 3.18 M (56 %)
    cap 1000 -> 4.49 M (79 %)
    current  -> 5.70 M (~2,000, with leakage to 6,407)

If accuracy is still climbing at the right-hand end, we are on the wrong side of the optimum and
acquiring more head images is justified. If it has flattened by 1,000, the cap costs nothing and the
whole ToL-data direction can be closed cheaply. Either way the answer costs three short runs and no
downloads.

**The test fold is never touched.** Only training rows are capped -- filtering the evaluation set is
the single mistake this project has paid for most ([[2026-07-24-src-lepinet-baseline-port]]), and a
cap applied to the test fold would drop the tail out of a macro average and inflate every number.

    python dev/078_head_cap_sweep.py --cap 500 --out data/head_caps
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(a):
    d = pd.read_parquet(a.parquet)
    is_test = d["set"].astype(str) == a.test_set
    train, test = d[~is_test], d[is_test]

    # Highest-numbered rows are as arbitrary as any other order here, so shuffle before taking the
    # head: a deterministic slice could correlate with upload date, publisher or image quality.
    kept = (train.sample(frac=1.0, random_state=a.seed)
                 .groupby(a.level, sort=False)
                 .head(a.cap))
    out = pd.concat([kept, test], ignore_index=True)

    c_before = train.groupby(a.level).size()
    c_after = kept.groupby(a.level).size()
    print(f"cap {a.cap} images/species on the TRAIN split only")
    print(f"  train {len(train):,} -> {len(kept):,} ({100 * len(kept) / len(train):.1f} %)")
    print(f"  species affected: {int((c_before > a.cap).sum()):,} of {c_before.size:,}")
    print(f"  median imgs/species {c_before.median():.0f} -> {c_after.median():.0f}")
    print(f"  test fold UNTOUCHED: {len(test):,} rows, {test[a.level].nunique():,} species")
    assert len(test) == int(is_test.sum()), "test fold changed -- refusing to write"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out} ({len(out):,} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/global/0032836-250426092105405_processing_metadata_"
                                        "postprocessed_quality_filtered.parquet")
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--level", default="speciesKey")
    p.add_argument("--test-set", default="0")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
