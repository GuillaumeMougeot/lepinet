"""Is the ToL model's embedding more taxonomically structured than BioCLIP-2's? (owner, 2026-11-19)

Uses the frozen embeddings TL1/Z21 already saved (data/population_probes/tol_e1/emb_{heldout,rare}_*), no GPU.
Per encoder and set, on species means (images L2-normalised, averaged per species, centred over species, renormalised):
- spearman: rank correlation between the cosine of two species means and their taxonomic distance (ranks to the
  lowest shared rank, 1 = same genus ... 7 = different kingdom). More negative = closer in the tree, closer in space.
- cos@k: mean cosine of species pairs whose lowest shared rank is k (genus, family, order, class, other).
- nn: share of species whose nearest species mean is in the same genus / family / order.
- fisher: trace(between-species covariance) / trace(within-species covariance), image level.
Held-out taxa: 1,810 species withheld from ToL training (BioCLIP-2 trained on them). Rare Species: 400 species.

    .venv/bin/python dev/112_embedding_structure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("data/population_probes/tol_e1")
RANKS = ["genus", "family", "order", "class", "phylum", "kingdom"]
ENC = {"ToL (frozen)": "", "BioCLIP-2": "_bioclip2", "DINOv3 raw": "_dinov3", "TL4 (ToL->moths)": "_tl4",
       "VD (DINOv3->moths)": "_vd"}


def taxonomy(kind, paths):
    if kind == "heldout":
        x = pd.read_parquet("data/tol_eval/index_heldout.parquet").set_index("path").loc[paths]
        return x[["species"] + RANKS].reset_index(drop=True)
    parts = [Path(p).parent.name.split("_") for p in paths]  # Kingdom_Phylum_Class_Order_Family_Genus_species
    cols = ["kingdom", "phylum", "class", "order", "family", "genus"]
    df = pd.DataFrame({c: ["|".join(p[:i + 1]) for p in parts] for i, c in enumerate(cols)})
    df["species"] = ["|".join(p) for p in parts]
    return df


def measure(e, tax):
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    sp, inv = np.unique(tax.species.to_numpy(), return_inverse=True)
    n = np.bincount(inv)
    mu = np.zeros((len(sp), e.shape[1]))
    np.add.at(mu, inv, e)
    mu /= n[:, None]
    within = ((e - mu[inv]) ** 2).sum() / len(e)
    between = (n[:, None] * (mu - mu.mean(0)) ** 2).sum() / len(e)
    m = mu - mu.mean(0)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    C = m @ m.T
    first = tax.groupby(inv).first()
    dist = np.full(C.shape, 7)  # 7 = different kingdom
    for k, r in reversed(list(enumerate(RANKS, start=1))):
        v = first[r].to_numpy()
        dist[v[:, None] == v[None, :]] = k
    iu = np.triu_indices(len(sp), 1)
    out = {"species": len(sp), "spearman": spearmanr(C[iu], dist[iu]).statistic, "fisher": between / within}
    for k, r in ((1, "genus"), (2, "family"), (3, "order"), (4, "class")):
        sel = dist[iu] == k
        out[f"cos@{r}"] = C[iu][sel].mean() if sel.any() else np.nan
    out["cos@other"] = C[iu][dist[iu] >= 5].mean()
    np.fill_diagonal(C, -9)
    nn = C.argmax(1)
    for k, r in ((1, "genus"), (2, "family"), (3, "order")):
        has = (dist <= k).sum(1) > 1  # a species with at least one other species in that taxon here
        out[f"nn_{r}"] = (dist[np.arange(len(sp)), nn] <= k)[has].mean()
    return out


def main():
    rows = []
    for kind in ("heldout", "rare"):
        tax = None
        for name, suf in ENC.items():
            d = ROOT / f"emb_{kind}{suf}"
            if not (d / "embeddings.npy").exists():
                continue
            paths = json.load(open(d / "paths.json"))
            if tax is None:
                ref, tax = paths, taxonomy(kind, paths)
            assert paths == ref, (kind, name)
            r = measure(np.load(d / "embeddings.npy").astype(np.float32), tax)
            rows.append({"set": kind, "encoder": name, **r})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = Path("data/population_probes/embedding_structure.csv")
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
