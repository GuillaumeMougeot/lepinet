"""Build a lepinet-eval labels parquet for the flemming_helsing `referenced` open-set test.

The dataset ships no parquet, but `datasets/flemming_helsing/restructured/valid/example_pred.csv`
is already in mini_metrics long format (one row per image x level) and carries the ground-truth
`label` per level. We pivot it into the (speciesKey, genusKey, familyKey, filename, set) schema
`lepinet test` expects. Images live at `.../resized/valid/referenced/<speciesKey>/<uuid>.jpg`, so
`evaluate` reconstructs the path as `<speciesKey>/<filename>` against that img_dir.

Run the eval open-set (`--no-drop-unknown-species`) so the ~3% OOD species (not in the model's
12,041 vocab) are kept with `known_label=False`; then `mini_metrics -t 0 0 0` on the predictions.

    python dev/048_flemming_parquet.py example_pred.csv flemming_labels.parquet
"""
from __future__ import annotations

import sys

import pandas as pd

LEVELS = ["speciesKey", "genusKey", "familyKey"]  # level 0, 1, 2 in the CSV


def build(example_pred_csv: str, out_parquet: str) -> pd.DataFrame:
    ex = pd.read_csv(example_pred_csv)
    lab = ex.pivot_table(index=["instance_id", "filename"], columns="level", values="label",
                         aggfunc="first").reset_index()
    lab = lab.rename(columns={0: "speciesKey", 1: "genusKey", 2: "familyKey"})
    for c in LEVELS:
        lab[c] = lab[c].astype("int64")
    lab["set"] = "0"
    out = lab[[*LEVELS, "filename", "set"]]
    out.to_parquet(out_parquet, index=False)
    print(f"{len(out)} images | species {out.speciesKey.nunique()} genus {out.genusKey.nunique()} "
          f"family {out.familyKey.nunique()} -> {out_parquet}")
    return out


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
