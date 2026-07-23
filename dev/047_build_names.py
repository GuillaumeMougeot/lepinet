"""Build the app's `names.json` from the dataset parquet + a bundle's `taxonomy.json`.

The model is key-based (GBIF taxon keys), but the app should show readable names. The dataset
parquet carries them (`scientificName`, `genus`, `family`, `specificEpithet`); this maps each
taxon key to a display name and writes them **aligned to the taxonomy vocab order** so the app can
index names by the same logit index it uses for everything else.

Species names are the clean binomial `Genus epithet` (dropping GBIF's authorship), falling back to
`scientificName` when the parts are missing. Genus/family are the plain name strings.

Usage:
    python dev/047_build_names.py --parquet data/global/…_quality_filtered.parquet \
        --taxonomy data/global/bundles/<bundle>/taxonomy.json \
        --out data/global/bundles/<bundle>/names.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def build(parquet_path, taxonomy_path, out_path):
    tax = json.loads(Path(taxonomy_path).read_text())
    df = pd.read_parquet(
        parquet_path,
        columns=["speciesKey", "genusKey", "familyKey", "scientificName", "genus", "family", "specificEpithet"],
    )

    def binomial(row):
        g = str(row["genus"]) if pd.notna(row["genus"]) else ""
        e = str(row["specificEpithet"]) if pd.notna(row["specificEpithet"]) else ""
        b = f"{g} {e}".strip()
        return b or (str(row["scientificName"]) if pd.notna(row["scientificName"]) else "")

    sp = df.dropna(subset=["speciesKey"]).drop_duplicates("speciesKey")
    sp_map = {str(k): binomial(r) for k, r in zip(sp["speciesKey"], sp.to_dict("records"))}
    gn = df.dropna(subset=["genusKey"]).drop_duplicates("genusKey")
    gn_map = {str(k): (str(v) if pd.notna(v) else "") for k, v in zip(gn["genusKey"], gn["genus"])}
    fm = df.dropna(subset=["familyKey"]).drop_duplicates("familyKey")
    fm_map = {str(k): (str(v) if pd.notna(v) else "") for k, v in zip(fm["familyKey"], fm["family"])}

    def align(vocab, m):
        return [m.get(str(k), "") for k in vocab]

    names = {
        "species": align(tax["vocabs"]["species"], sp_map),
        "genus": align(tax["vocabs"]["genus"], gn_map),
        "family": align(tax["vocabs"]["family"], fm_map),
    }
    missing = {lvl: sum(1 for x in v if not x) for lvl, v in names.items()}
    print(f"Names built. Missing (shown as key in the app): {missing}")

    doc = {
        "note": "Display names aligned to taxonomy.json vocab order; empty string = unknown (app shows the key).",
        "names": names,
    }
    Path(out_path).write_text(json.dumps(doc, ensure_ascii=False))
    print(f"Wrote {out_path} ({Path(out_path).stat().st_size/1024:.0f} KB)")


def cli():
    p = argparse.ArgumentParser(description="Build names.json for lepinet-app (Phase D helper).")
    p.add_argument("--parquet", required=True, help="Dataset parquet with name columns.")
    p.add_argument("--taxonomy", required=True, help="Bundle taxonomy.json (defines vocab order).")
    p.add_argument("--out", required=True, help="Output names.json path.")
    a = p.parse_args()
    build(a.parquet, a.taxonomy, a.out)


if __name__ == "__main__":
    cli()
