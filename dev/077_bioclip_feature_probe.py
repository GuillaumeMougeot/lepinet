"""Two questions about BioCLIP-2, answered by measurement instead of argument, on the local GPU.

**Q1 — is the published 768-d embedding cache usable?** `imageomics/TreeOfLife-200M-Embeddings`
stores the *projected* 768-d CLIP embedding, not the 1024-d pooled tower output that `dev/075` uses
(`proj = None`). The objection is **not** width: the bottleneck sweep showed narrowing the
representation costs little. It is that the CLIP projection is a *learned* map fitted to pull each
image toward the text embedding of its taxon name — and sibling species have near-identical name
strings, so their text embeddings nearly coincide, so training pulls their image embeddings toward
nearly the same point. That is a specific mechanism for collapsing exactly the fine-grained sibling
separation this task lives on, and it would bite hardest on the `near` stratum.

A PCA or random projection would preserve the geometry; a supervised one need not. So: measure it.

**Q2 — was the owner's 44 % the right way to evaluate the representation?** It used *text-encoder*
embeddings, i.e. zero-shot classification: embed class names, embed the image, take the nearest text
vector. That is a valid zero-shot number, but it measures the text encoder and the image-text
alignment, not the image representation's separability. It is also prompt-sensitive: BioCLIP's own
prompting uses the full taxonomic hierarchy, not a bare binomial. The standard measurement for "how
good is this representation for my task" is a probe on frozen features — which is what P1a is.

This script puts all three on the same images and the same species, so they are comparable:

    zero-shot (binomial prompt)  |  zero-shot (full-hierarchy prompt)  |  nearest-centroid, 768  |  nearest-centroid, 1024

Nearest-centroid is used rather than a trained head because it needs no fitting, is what H1 measured
(centroids cost 0.29 pt against a trained matrix), and keeps the comparison to one variable: the
space the vectors live in.

    python dev/077_bioclip_feature_probe.py --n-species 200 --per-species 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HIER = ["kingdom", "phylum", "class", "order", "family", "genus"]


def sample(parquet: str, img_dir: Path, n_species: int, per_species: int, seed: int,
           taxonomy_from: str | None = None, all_species: bool = False):
    """Species with enough images, sampled so that many genera contribute >1 species.

    The sibling comparison is the whole point, so a sample of unrelated species would make the
    question unanswerable. Genera are drawn first, then species within them.
    """
    tax_cols = [*HIER, "specificEpithet"]
    if taxonomy_from:
        # The trap parquet carries only GBIF keys, so the names the text encoder needs are joined
        # in from the global parquet by speciesKey. Same taxonomy, same strings, no re-derivation.
        d = pd.read_parquet(parquet)
        tx = (pd.read_parquet(taxonomy_from, columns=["speciesKey", *tax_cols])
                .dropna(subset=["speciesKey"]).drop_duplicates("speciesKey"))
        for c in ["speciesKey"]:
            d[c] = d[c].astype(str); tx[c] = tx[c].astype(str)
        d = d.merge(tx, on="speciesKey", how="inner")
        d["set"] = "0"
    else:
        d = pd.read_parquet(parquet, columns=["speciesKey", *tax_cols, "filename", "set"])
    d = d[d["set"].astype(str) == "0"]
    d = d.dropna(subset=["genus", "specificEpithet"])
    n = d.groupby("speciesKey").size()
    d = d[d["speciesKey"].isin(n[n >= per_species].index)]

    rng = np.random.default_rng(seed)
    if all_species:
        # For "how accurate is it", the class COUNT dominates: zero-shot over 25 classes and over
        # 400 is not the same measurement. So take every species that clears the image floor and
        # let n_species cap it, rather than filtering to genera with congeners.
        chosen = list(d["speciesKey"].unique())
        rng.shuffle(chosen)
        chosen = chosen[:n_species]
        d = d[d["speciesKey"].isin(chosen)]
        d = d.sample(frac=1.0, random_state=seed).groupby("speciesKey", sort=False).head(per_species)
        if "image_path" in d.columns:
            d["path"] = d["image_path"].astype(str)
        else:
            d["path"] = d["speciesKey"].astype(str) + "/" + d["filename"]
        d = d[[Path(img_dir, p_).exists() for p_ in d["path"]]]
        print(f"sampled {d['speciesKey'].nunique()} species (all-species mode), {len(d)} images")
        return d
    per_genus = d.drop_duplicates("speciesKey").groupby("genus")["speciesKey"].apply(list)
    multi = per_genus[per_genus.map(len) >= 2]
    genera = list(multi.index)
    rng.shuffle(genera)
    chosen: list = []
    for g in genera:
        chosen.extend(multi[g][:3])
        if len(chosen) >= n_species:
            break
    chosen = chosen[:n_species]
    d = d[d["speciesKey"].isin(chosen)]
    d = d.sample(frac=1.0, random_state=seed).groupby("speciesKey", sort=False).head(per_species)
    if "image_path" in d.columns:
        d["path"] = d["image_path"].astype(str)
    else:
        d["path"] = d["speciesKey"].astype(str) + "/" + d["filename"]
    d = d[[Path(img_dir, p).exists() for p in d["path"]]]
    print(f"sampled {d['speciesKey'].nunique()} species over {d['genus'].nunique()} genera, "
          f"{len(d)} images ({d.groupby('genus')['speciesKey'].nunique().mean():.2f} species/genus)")
    return d


@torch.no_grad()
def embed(model, preprocess, paths, img_dir: Path, device, bs=64):
    from PIL import Image
    outs = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([preprocess(Image.open(Path(img_dir, p)).convert("RGB"))
                             for p in paths[i:i + bs]]).to(device)
        pooled = model.visual(batch)          # proj stripped by the caller -> 1024
        outs.append(pooled.float().cpu())
        if i % (bs * 20) == 0:
            print(f"  embedded {i + len(batch)}/{len(paths)}", flush=True)
    return torch.cat(outs)


def centroid_top1(emb: torch.Tensor, labels: np.ndarray, seed: int):
    """Split each species' images into enrol/query, then nearest-centroid top-1 on the queries."""
    emb = F.normalize(emb, dim=1)
    rng = np.random.default_rng(seed)
    enrol_idx, query_idx = [], []
    for s in np.unique(labels):
        idx = np.where(labels == s)[0]
        rng.shuffle(idx)
        k = max(1, int(0.7 * len(idx)))
        enrol_idx.extend(idx[:k]); query_idx.extend(idx[k:])
    enrol_idx, query_idx = np.array(enrol_idx), np.array(query_idx)
    classes = np.unique(labels)
    cents = torch.stack([F.normalize(emb[enrol_idx][labels[enrol_idx] == s].mean(0), dim=0)
                         for s in classes])
    sims = emb[query_idx] @ cents.T
    pred = classes[sims.argmax(1).numpy()]
    return float((pred == labels[query_idx]).mean()), cents, classes


def sibling_stats(cents: torch.Tensor, classes: np.ndarray, meta: pd.DataFrame):
    """Mean centroid cosine for same-genus pairs vs different-family pairs, and the gap."""
    g = meta.set_index("speciesKey").loc[classes]
    # .to_numpy(str) is load-bearing: the parquet columns are Arrow-backed and do not support
    # numpy broadcasting indexing.
    gen = g["genus"].astype(str).to_numpy()
    fam = g["family"].astype(str).to_numpy()
    sim = (cents @ cents.T).numpy()
    iu = np.triu_indices(len(classes), 1)
    same_genus = (gen[:, None] == gen[None, :])[iu]
    diff_fam = (fam[:, None] != fam[None, :])[iu]
    s = sim[iu]
    return {"same_genus_cos": float(s[same_genus].mean()),
            "diff_family_cos": float(s[diff_fam].mean()),
            "gap": float(s[diff_fam].mean() - s[same_genus].mean()),
            "n_same_genus_pairs": int(same_genus.sum())}


@torch.no_grad()
def zero_shot(model, tokenizer, emb768, labels, classes, meta, device, template: str):
    # dict records, not itertuples: `class` is a Python keyword and itertuples silently renames it.
    g = meta.set_index("speciesKey").loc[classes].astype(str).to_dict("records")
    if template == "binomial":
        texts = [f"a photo of {r['genus']} {r['specificEpithet']}." for r in g]
    else:  # BioCLIP-style full taxonomic hierarchy
        texts = [f"a photo of {' '.join(r[h] for h in HIER)} {r['specificEpithet']}." for r in g]
    tf = model.encode_text(tokenizer(texts).to(device)).float().cpu()
    tf = F.normalize(tf, dim=1)
    sims = F.normalize(emb768, dim=1) @ tf.T
    pred = classes[sims.argmax(1).numpy()]
    return float((pred == labels).mean())


def main(a):
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = sample(a.parquet, Path(a.img_dir), a.n_species, a.per_species, a.seed, a.taxonomy_from, a.all_species)
    meta = d.drop_duplicates("speciesKey")[["speciesKey", *HIER, "specificEpithet"]]
    paths, labels = list(d["path"]), d["speciesKey"].to_numpy()

    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    model = model.to(device).eval()
    proj = model.visual.proj                      # keep it: needed for the 768-d space and text

    print("\nembedding (1024-d pooled, proj stripped) ...")
    model.visual.proj = None
    e1024 = embed(model, preprocess, paths, Path(a.img_dir), device)
    e768 = e1024 @ proj.detach().float().cpu()    # exactly what visual() would have returned
    model.visual.proj = proj

    res = {"n_species": int(len(np.unique(labels))), "n_images": len(paths)}
    for name, emb in (("1024_pooled", e1024), ("768_projected", e768)):
        acc, cents, classes = centroid_top1(emb, labels, a.seed)
        res[f"centroid_top1_{name}"] = round(acc, 4)
        res[f"geometry_{name}"] = {k: round(v, 4) for k, v in
                                   sibling_stats(cents, classes, meta).items()}

    classes = np.unique(labels)
    for tmpl in ("binomial", "hierarchy"):
        res[f"zeroshot_top1_{tmpl}"] = round(
            zero_shot(model, tokenizer, e768, labels, classes, meta, device, tmpl), 4)

    print("\n" + json.dumps(res, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="data/global/0032836-250426092105405_processing_metadata_"
                                        "postprocessed_quality_filtered.parquet")
    p.add_argument("--img-dir", default="data/global/images")
    p.add_argument("--n-species", type=int, default=200)
    p.add_argument("--per-species", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-species", action="store_true",
                   help="skip the congener constraint; use every species clearing the floor")
    p.add_argument("--taxonomy-from", default=None,
                   help="parquet to join taxon NAMES from, by speciesKey (for the trap set, whose "
                        "parquet carries only GBIF keys)")
    p.add_argument("--out", default="data/tol_overlap/feature_probe.json")
    main(p.parse_args())
