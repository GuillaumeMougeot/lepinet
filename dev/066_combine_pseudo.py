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

    # Coarse levels for the pseudo rows. The labeller was species-only, but a head like
    # `marginal_arcface` needs genus and family columns or `prepare_df` dies on a KeyError six hours
    # into the queue. They are derived from the REAL parquet's species->genus->family mapping, which
    # is taxonomy rather than label information -- the pseudo row's species key is still the model's
    # own prediction, and looking up that species' genus is a fact about the taxonomy.
    levels = [c.strip() for c in a.levels.split(",")]
    missing = [lv for lv in levels[1:] if lv not in pseudo.columns]
    if missing:
        tax = real.drop_duplicates(subset=[level])[[level, *levels[1:]]].copy()
        for c in tax.columns:
            tax[c] = tax[c].astype("string").astype(str)
        tax = tax.set_index(level)
        for lv in missing:
            pseudo[lv] = pseudo[level].astype(str).map(tax[lv])
        n_unmapped = int(pseudo[missing].isna().any(axis=1).sum())
        if n_unmapped:
            print(f"dropping {n_unmapped} pseudo rows whose species has no taxonomy entry")
            pseudo = pseudo.dropna(subset=missing)
        print(f"derived {missing} for the pseudo rows from the real parquet's taxonomy")

    # `real` has no image_path column -- prepare_df builds it. Give it one explicitly so the two
    # halves are homogeneous and the preserved-path branch applies uniformly.
    real = real.copy()
    if "image_path" not in real.columns:
        real["image_path"] = real[level].astype(str) + "/" + real["filename"]

    keep_cols = ["image_path", "filename", "set", *levels]

    # Replication. NOTE (2026-08-04): the paragraph below was written before the sweep and its
    # central claim is FALSE. 0.39 % of training (no replication) buys +4.42 pt, 97 % of what 13x
    # buys, with better transfer to unseen species. The optimum is ~2 %; above it, replication
    # converts adaptation into memorisation of the pseudo-labelled images. Default `--target-frac`
    # is now 0.02. Kept as written because the reasoning failed in an instructive way --
    # journal/2026-08-04-replication-sweep.md.
    #
    # Original comment, wrong: The gate retains ~12k trap images against ~3.1M real
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
    ap.add_argument("--levels", default="speciesKey",
                    help="Comma-separated levels the training config needs, fine->coarse. A head "
                         "such as marginal_arcface needs all three or prepare_df raises KeyError.")
    ap.add_argument("--min-img-per-spc", type=int, default=50)
    ap.add_argument("--target-frac", type=float, default=0.02,
                    help="Replicate pseudo rows to this fraction of training. 0.02 is the measured optimum; "
                         "above ~0.05 the gain falls and transfer to unseen species halves. 0 disables.")
    main(ap.parse_args())
