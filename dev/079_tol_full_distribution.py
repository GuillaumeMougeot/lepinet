"""How large is TreeOfLife-200M under OUR data policy, and how many taxa does it hold per rank?

The owner's proposal: download ToL-200M capped at ~2,000 images/species, at reduced resolution, and
train the lepinet recipe on it. Three claimed benefits -- a genuine test of whether the
representation is the problem, **seven taxonomic ranks instead of our three**, and a tractable
compute budget once the head is capped.

Everything turns on one number nobody has: **how many images survive a 2,000/species cap across the
whole corpus.** For Lepidoptera it is 42.8 % (20.9 M -> 8.9 M), but Lepidoptera is unusually
head-heavy -- butterflies are the most photographed insects on earth -- so that fraction cannot be
assumed to generalise to plants, fungi or beetles.

This scans all 666 shards' taxonomy columns and never touches `emb` (768 x fp16 = 1536 bytes/row,
~99 % of the bytes), the same trick `dev/076` used for the contamination scan. A few hours of
streaming, no GPU, no downloads.

Reports, per rank and for the whole corpus:

  * taxa counts at each of the 7 ranks -- what a hierarchical head would actually have to predict
  * images surviving caps of 500 / 1000 / 2000 / 5000 per species
  * the same with a min-images floor applied, since the tail is cheap in images and huge in taxa

    python dev/079_tol_full_distribution.py --out data/tol_overlap/full_distribution.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SHARD = ("datasets/imageomics/TreeOfLife-200M-Embeddings/bioclip-2_float16/"
         "train-{:05d}-of-00666.parquet")
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
CAPS = [250, 500, 1000, 2000, 5000, 20000]


def main(a):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()

    sp_counts: Counter = Counter()
    rank_taxa = {r: set() for r in RANKS}
    n_rows = 0

    for i in range(a.first, a.last + 1):
        with fs.open(SHARD.format(i), "rb") as f:
            d = pq.ParquetFile(f).read(columns=RANKS).to_pandas()
        n_rows += len(d)
        # A bare epithet is not a species: "alba" occurs in hundreds of genera. The key must be
        # genus+epithet or every count below is silently merged across the tree.
        key = (d["genus"].fillna("") + " " + d["species"].fillna("")).str.strip()
        sp_counts.update(key[key != ""].value_counts().to_dict())
        for r in RANKS[:-1]:
            rank_taxa[r].update(x for x in d[r].dropna().unique() if x)
        if i % 25 == 0 or i == a.last:
            print(f"  shard {i:3d}/{a.last}: {n_rows:>12,} rows | {len(sp_counts):>9,} species",
                  flush=True)

    c = np.array(sorted(sp_counts.values(), reverse=True))
    res = {"n_rows_scanned": int(n_rows), "n_species": int(c.size),
           "taxa_per_rank": {r: len(rank_taxa[r]) for r in RANKS[:-1]} | {"species": int(c.size)},
           "top10_share_pct": round(100 * c[:10].sum() / c.sum(), 2),
           "top1000_share_pct": round(100 * c[:1000].sum() / c.sum(), 2),
           "caps": {}, "caps_with_min50": {}}
    for cap in CAPS:
        res["caps"][str(cap)] = int(np.minimum(c, cap).sum())
        c50 = c[c >= 50]
        res["caps_with_min50"][str(cap)] = {"images": int(np.minimum(c50, cap).sum()),
                                            "species": int(c50.size)}

    print("\n" + json.dumps(res, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    np.save(str(Path(a.out).with_suffix(".counts.npy")), c)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=665)
    p.add_argument("--out", default="data/tol_overlap/full_distribution.json")
    main(p.parse_args())
