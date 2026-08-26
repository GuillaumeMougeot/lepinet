"""How much of global_lepi is already inside TreeOfLife-200M, and therefore inside BioCLIP-2?

The owner's hypothesis, and it decides two things at once:

**Contamination.** BioCLIP-2 was trained on ToL-200M. If our training images are in there, any
*in-distribution* comparison against BioCLIP-2 is contaminated in its favour, and a head fitted on
its embeddings of its own training data reports an inflated number. That is finding 4b's shape --
a corpus quietly serving two roles -- and it has to be recorded in the split before anyone runs the
comparison, not discovered afterwards.

**Interpretation of the 44 %.** If BioCLIP-2 has effectively seen our training set and still scores
44 % on trap images, that is not a weak representation; it is a representation that fits the source
domain and transfers poorly to a different camera. Which is finding 13 at 43x our data, and it would
argue *against* training our own ToL backbone rather than for it.

## Why this is cheap

The embedding cache is parquet, 666 shards, sorted by `source_dataset` then taxonomy. Lepidoptera
(GBIF-sourced) occupies shards 100-161, found from **row-group statistics in the file footers** --
no data downloaded. We then read only the taxonomy and provenance columns and never touch `emb`,
which is 768 x fp16 = 1536 bytes/row and ~99 % of the bytes. Column pruning turns a 350 GB dataset
into a few GB of strings.

## The two joins

- **Taxon level:** our `genus` + `specificEpithet` against ToL `species` (ToL stores the epithet
  alone, so it must be paired with `genus`). Answers "does BioCLIP-2 know our label set?"
- **Image level:** our `gbifID` against ToL `source_id` where `source_dataset == 'gbif'`. This is
  the strong one: it answers "has BioCLIP-2 seen these exact photographs?" -- an occurrence-level
  identity, not a name match.

    python dev/076_tol_overlap.py --out data/tol_overlap
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = "datasets/imageomics/TreeOfLife-200M-Embeddings"
SHARD = REPO + "/bioclip-2_float16/train-{:05d}-of-00666.parquet"
COLS = ["source_dataset", "source_id", "order", "family", "genus", "species", "publisher"]


def scan_tol(shards: range, fs):
    """Stream the shards, accumulating only aggregates.

    21.7 M rows x 7 string columns does not want to live in a single DataFrame on this box, and
    nothing downstream needs the rows -- only per-species counts, the genus/family cardinalities and
    the set of GBIF occurrence ids. So each shard is reduced and dropped.
    """
    counts: dict[str, int] = {}
    genera, families, gbif_ids = set(), set(), set()
    n_rows = 0
    for i in shards:
        with fs.open(SHARD.format(i), "rb") as f:
            d = pq.ParquetFile(f).read(columns=COLS).to_pandas()
        d = d[d["order"] == "Lepidoptera"]
        n_rows += len(d)
        sp = (d["genus"].fillna("") + " " + d["species"].fillna("")).str.strip()
        for k, v in sp.value_counts().items():
            counts[k] = counts.get(k, 0) + int(v)
        genera.update(d["genus"].dropna().unique())
        families.update(d["family"].dropna().unique())
        g = d.loc[d["source_dataset"] == "gbif", "source_id"].dropna().astype(str)
        gbif_ids.update(g)
        print(f"  shard {i:3d}: {len(d):7d} rows | running: {n_rows:9,d} imgs, "
              f"{len(counts):6,d} species, {len(gbif_ids):9,d} gbif ids", flush=True)
    return counts, genera, families, gbif_ids, n_rows


def main(a):
    from huggingface_hub import HfFileSystem

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"reading ToL shards {a.first}-{a.last} (taxonomy columns only, `emb` never touched)")
    counts, genera, families, tol_ids, n_rows = scan_tol(range(a.first, a.last + 1), HfFileSystem())
    tol_counts = pd.Series(counts).drop(labels=[""], errors="ignore")
    print(f"\nToL Lepidoptera: {n_rows:,} images, {len(tol_counts):,} species, "
          f"{len(genera):,} genera, {len(families):,} families")

    ours = pd.read_parquet(a.parquet, columns=["gbifID", "genus", "specificEpithet",
                                               "family", "speciesKey", "set"])
    ours["sp"] = (ours["genus"].fillna("") + " " + ours["specificEpithet"].fillna("")).str.strip()
    # The label set is the training vocabulary, which is what "does it know our classes" means.
    train = ours[ours["set"].astype(str) != "0"]
    keep = train["speciesKey"].value_counts()
    vocab_keys = set(keep[keep >= a.min_img_per_spc].index)
    vocab = ours[ours["speciesKey"].isin(vocab_keys)]
    our_species = set(vocab["sp"]) - {""}
    print(f"ours: {len(ours):,} images, {len(our_species):,} species in the >= "
          f"{a.min_img_per_spc}-image vocabulary")

    tol_species = set(tol_counts.index)
    shared = our_species & tol_species
    covered_imgs = int(tol_counts.reindex(list(shared)).fillna(0).sum())

    # Image-level: GBIF occurrence id identity, not a name match.
    our_ids = set(ours["gbifID"].dropna().astype(str))
    shared_ids = our_ids & tol_ids
    test_ids = set(ours[ours["set"].astype(str) == "0"]["gbifID"].dropna().astype(str))
    leaked_test = test_ids & tol_ids

    res = {
        "tol_lepidoptera_images": int(n_rows),
        "tol_lepidoptera_species": int(len(tol_species)),
        "our_vocab_species": len(our_species),
        "species_shared": len(shared),
        "species_shared_pct_of_ours": round(100 * len(shared) / max(len(our_species), 1), 2),
        "tol_images_in_shared_species": covered_imgs,
        "our_images_total": int(len(ours)),
        "image_ids_shared": len(shared_ids),
        "image_ids_shared_pct_of_ours": round(100 * len(shared_ids) / max(len(our_ids), 1), 2),
        "our_TEST_fold_images_in_tol": len(leaked_test),
        "our_TEST_fold_pct_in_tol": round(100 * len(leaked_test) / max(len(test_ids), 1), 2),
    }
    print("\n" + json.dumps(res, indent=2))
    (out / "overlap.json").write_text(json.dumps(res, indent=2))

    # Persist the id set so a decontaminated split can be built without re-scanning 62 shards.
    pd.DataFrame({"gbifID": sorted(tol_ids)}).to_parquet(out / "tol_gbif_ids.parquet", index=False)
    clean = sorted(test_ids - tol_ids)
    pd.DataFrame({"gbifID": clean}).to_parquet(out / "test_ids_clean_of_tol.parquet", index=False)
    print(f"decontaminated test fold: {len(clean):,} of {len(test_ids):,} images "
          f"({100*len(clean)/max(len(test_ids),1):.1f} % survive)")

    pd.Series(sorted(our_species - tol_species)).to_csv(out / "species_ours_not_in_tol.csv",
                                                        index=False, header=["species"])
    tol_counts.sort_values(ascending=False).to_csv(out / "tol_species_counts.csv")
    print(f"wrote {out}/overlap.json and two CSVs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/global/0032836-250426092105405_processing_metadata_"
                                        "postprocessed_quality_filtered.parquet")
    p.add_argument("--out", default="data/tol_overlap")
    p.add_argument("--first", type=int, default=100)
    p.add_argument("--last", type=int, default=161)
    p.add_argument("--min-img-per-spc", type=int, default=50)
    main(p.parse_args())
