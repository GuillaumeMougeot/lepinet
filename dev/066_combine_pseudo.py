"""B3 stage 2: merge the pseudo-labelled trap rows into the global training parquet.

Two details decide whether the resulting run measures anything.

**The pseudo rows must not be reachable as validation or test.** They carry ``set='2'``, so
``gen_df``'s ``remove_in=['0']`` keeps them and ``prepare_df``'s ``valid_set='1'`` never marks them
valid. The model is therefore selected and scored entirely on real labels; the pseudo-labels only
ever appear in the training split.

**The rare-species floor must not see them.** ``min_img_per_spc`` counts rows per species *after*
the merge, so a species with 30 real images and 30 pseudo ones would cross a floor of 50 and enter
the vocabulary through the back door — changing the label set and making the run
incomparable to B1. The filter is therefore applied to the real rows **before** merging, and the
pseudo rows are restricted to species that already survived it.

    python dev/066_combine_pseudo.py --global-parquet ... --pseudo-parquet ... --out combined.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(a):
    real = pd.read_parquet(a.global_parquet)
    pseudo = pd.read_parquet(a.pseudo_parquet)
    level = a.level

    # Apply the rare-species floor to the REAL rows only, then restrict pseudo rows to that vocab.
    train_real = real[real["set"].astype(str) != "0"]
    counts = train_real[level].astype(str).value_counts()
    keep = set(counts[counts >= a.min_img_per_spc].index)
    print(f"real rows {len(real)} | species surviving min_img_per_spc={a.min_img_per_spc}: {len(keep)}")

    before = len(pseudo)
    pseudo = pseudo[pseudo[level].astype(str).isin(keep)].copy()
    print(f"pseudo rows {before} -> {len(pseudo)} after restricting to the real vocabulary "
          f"({pseudo[level].nunique()} species)")

    if (pseudo["set"].astype(str) != "2").any():
        raise ValueError("pseudo rows must all carry set='2' so they can never be validation/test")

    cols = [c for c in ("image_path", "filename", "set", level) if c in real.columns]
    for c in cols:
        if c not in pseudo.columns:
            raise ValueError(f"pseudo parquet is missing {c!r}; the merge would produce NaNs")
    # `real` has no image_path column -- prepare_df builds it. Give it one explicitly so the two
    # halves are homogeneous and the preserved-path branch applies uniformly.
    real = real.copy()
    if "image_path" not in real.columns:
        real["image_path"] = real[level].astype(str) + "/" + real["filename"]

    keep_cols = ["image_path", "filename", "set", level]

    # Replication, and why it is not optional. The gate retains ~12k trap images against ~3.1M real
    # ones -- **0.39 % of training**. At that share the pseudo rows appear in roughly one batch in
    # 250, and B3 would return a null result that says nothing about self-training, only that 0.4 %
    # more data changes nothing. Repeating them to a stated fraction is the standard fix and makes
    # the intervention's size an explicit, reportable hyperparameter instead of an accident of how
    # many images happened to clear the gate.
    #
    # The cost is honest and must be stated with the result: ~12k *unique* images seen many times,
    # so the model can memorise them. Augmentation differs per epoch, and they are 5 % of batches
    # rather than the whole objective, but if B3 wins, a replication sweep is the first control.
    reps = 1
    if a.target_frac > 0 and len(pseudo):
        reps = max(1, round(a.target_frac * len(real) / ((1 - a.target_frac) * len(pseudo))))
        print(f"replicating pseudo rows {reps}x to reach ~{a.target_frac:.0%} of training "
              f"({len(pseudo)} -> {len(pseudo) * reps})")
    pseudo_rep = pd.concat([pseudo[keep_cols]] * reps, ignore_index=True) if reps > 1 else pseudo[keep_cols]
    combined = pd.concat([real[keep_cols], pseudo_rep], ignore_index=True)
    combined.to_parquet(a.out)

    summary = {"n_real": len(real), "n_pseudo_unique": len(pseudo), "replication": reps,
               "n_pseudo_rows": len(pseudo_rep), "n_combined": len(combined),
               "pseudo_frac": round(len(pseudo_rep) / len(combined), 4),
               "species_real": int(real[level].nunique()),
               "species_combined": int(combined[level].nunique()),
               "min_img_per_spc_applied_to_real_only": a.min_img_per_spc}
    Path(str(a.out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    # The species count must not grow: if it did, the pseudo rows changed the label set.
    if summary["species_combined"] > summary["species_real"]:
        raise SystemExit("pseudo rows introduced new species -- the label set changed, so this run "
                         "is not comparable to the baseline")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-parquet", required=True)
    ap.add_argument("--pseudo-parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="speciesKey")
    ap.add_argument("--min-img-per-spc", type=int, default=50)
    ap.add_argument("--target-frac", type=float, default=0.05,
                    help="Replicate pseudo rows until they are this fraction of training. 0 disables.")
    main(ap.parse_args())
