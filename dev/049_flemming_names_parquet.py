"""Build a lepinet-eval labels parquet for the *names* flemming dataset (`data/flemming`).

Unlike `flemming_helsing` (folders named by GBIF speciesKey), this dataset's image folders are
named by **species name**, with `name2id.csv` (verbatimScientificName -> family/genus/speciesKey,
resolved via the GBIF API) giving the keys. This tends to have far fewer OOD species than the
key-named set (id reconciliation), so it's the cleaner external test.

We emit the standard schema plus an explicit `image_path = <name>/<file>` column, which
`lepinet.test.evaluate` uses verbatim (it only reconstructs `<speciesKey>/<file>` when no
`image_path` column is present). Run the eval open-set (`--no-drop-unknown-species`) + `--tta`.

    python dev/049_flemming_names_parquet.py data/flemming out.parquet
"""
from __future__ import annotations

import os
import sys

import pandas as pd


def build(root: str, out_parquet: str) -> pd.DataFrame:
    n2i = pd.read_csv(os.path.join(root, "name2id.csv"))
    n2i["verbatimScientificName"] = n2i["verbatimScientificName"].astype(str)
    keys = {r.verbatimScientificName: (int(r.speciesKey), int(r.genusKey), int(r.familyKey))
            for r in n2i.itertuples(index=False)}
    img_root = os.path.join(root, "images")
    rows, unmapped = [], set()
    for name in os.listdir(img_root):
        d = os.path.join(img_root, name)
        if not os.path.isdir(d):
            continue
        if name not in keys:
            unmapped.add(name)
            continue
        sp, gn, fm = keys[name]
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"speciesKey": sp, "genusKey": gn, "familyKey": fm,
                             "filename": f, "image_path": f"{name}/{f}", "set": "0"})
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print(f"{len(df)} images | species {df.speciesKey.nunique()} | unmapped folders {len(unmapped)} "
          f"-> {out_parquet}")
    return df


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
