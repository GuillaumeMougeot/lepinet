"""What lifting the 50-image species floor would add to Global Lepidoptera, if species below it are kept
because their genus or family has enough images (owner's question, 2026-11-17).

Reads the GBIF download before the quality filter (`..._postprocessed.parquet`, every image downloaded) and the
training parquet (`..._quality_filtered.parquet`, 12,632 species with >= 50 images after the filter). Counts, for
the species that are not trained: how many have a genus (or family) with >= 50 images in the download, and their
images; and, with no floor at all, the images per family and per genus. Keys are the download's legacy GBIF keys.
The ToL counts were made by the ToL session (journal entry 2026-11-17, dev/111).

    .venv/bin/python dev/110_species_floor.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROOT = "data/global/0032836-250426092105405_processing_metadata_postprocessed"


def main():
    a = pd.read_parquet(f"{ROOT}.parquet", columns=["speciesKey", "genusKey", "familyKey", "set"])
    q = pd.read_parquet(f"{ROOT}_quality_filtered.parquet", columns=["speciesKey", "familyKey"])
    kept = set(q.speciesKey.unique())
    rate = len(q) / a.speciesKey.isin(kept).sum()  # share of a trained species' rows the quality filter keeps
    sp = a.groupby("speciesKey").agg(n=("set", "size"), genus=("genusKey", "first"), family=("familyKey", "first"))
    g, f = a.groupby("genusKey").size(), a.groupby("familyKey").size()
    sp["gn"], sp["fn"] = sp.genus.map(g), sp.family.map(f)
    cur = sp.index.isin(kept)
    below = ~cur
    print(f"download: {len(a):,} images, {len(sp):,} species, {len(g):,} genera, {len(f)} families")
    print(f"trained: {cur.sum():,} species, {len(q):,} images (the quality filter kept {rate:.1%} of their rows)")
    print(f"not trained: {below.sum():,} species, {sp.n[below].sum():,} images (~{int(sp.n[below].sum() * rate):,} "
          f"after a quality filter at the same rate)")
    cur_g, cur_f = set(sp.genus[cur]), set(sp.family[cur])
    for lab, col in (("genus", "gn"), ("family", "fn")):
        m = below & (sp[col] >= 50)
        print(f"{lab} >= 50 images: +{m.sum():,} species, +{sp.n[m].sum():,} images (~{int(sp.n[m].sum() * rate):,}); "
              f"new genera {len(set(sp.genus[m]) - cur_g):,}, new families {len(set(sp.family[m]) - cur_f)}; "
              f"median images per added species {int(sp.n[m].median())}, single-image species {int((sp.n[m] == 1).sum()):,}")
    print(f"no floor: families {len(f)}, images per family min {f.min()}, p1 {int(np.percentile(f, 1))}, "
          f"median {int(f.median())}, under 50: {(f < 50).sum()}; genera under 50: {(g < 50).sum():,} of {len(g):,}")
    fq = q.familyKey.value_counts()
    print(f"trained set: families {len(fq)}, min images {fq.min()}")


if __name__ == "__main__":
    main()
