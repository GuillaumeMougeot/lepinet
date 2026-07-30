"""Visualise and *measure* the embedding space — plain cosine vs ArcFace. (#8, dev-only)

ArcFace's claim is geometric: the angular margin should give each class a tighter, better-separated
region, and — the property that matters for open-set — leave *meaningful* distance for classes never
seen in training. A picture alone can mislead (UMAP/t-SNE will happily invent clusters), so this
reports **numbers first, picture second**:

Quantitative (the honest part — these are what to quote):
  * ``intra``  mean cosine(embedding, its own class prototype)      -- higher = tighter classes
  * ``inter``  mean max cosine to a *wrong* prototype               -- lower  = better separated
  * ``margin`` intra − inter                                        -- the separation ArcFace targets
  * ``silhouette`` on cosine distance                               -- standard cluster quality

Qualitative: a 2-D UMAP (falls back to t-SNE) coloured by species, one panel per model, so the two
can be eyeballed side by side.

Deliberately small: a handful of species × a few hundred images. The metrics are unstable and the
plot unreadable at 12 k classes, and the point is a *comparison*, not a census.

    python dev/053_embedding_viz.py \\
        --models plain=/path/independent.pt arcface=/path/arcface_zscore.pt \\
        --parquet meta.parquet --img-dir images/ --n-species 12 --per-species 40 --out emb/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def embeddings_for(model, dls, df, device, num_workers=8):
    """L2-normalised penultimate embeddings — the space ArcFace is supposed to shape.

    Taken from the head's ``preclassification`` (bottleneck + LeakyReLU + L2-norm), i.e. exactly the
    vectors the class prototypes are compared against — not the backbone features, and not the logits.
    """
    from lepinet.test import dl_num_workers

    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    dl = dls.test_dl(df, num_workers=nw)
    body, head = model[0], model[1].head
    model.to(device).eval()
    out = []
    for batch in dl:
        feats = body(batch[0].to(device))
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1) if feats.ndim == 4 else feats
        out.append(head.preclassification(pooled.float()).cpu())
    return torch.cat(out).numpy()


def geometry(emb: np.ndarray, labels: np.ndarray, prototypes: np.ndarray | None = None) -> dict:
    """Intra/inter class cosine statistics + silhouette. Prototypes default to class means."""
    classes = sorted(set(labels.tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    if prototypes is None:
        prototypes = np.stack([emb[labels == c].mean(0) for c in classes])
    prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
    sims = emb @ prototypes.T                                   # [N, C] cosine to every prototype
    own = np.array([idx[c] for c in labels])
    intra = sims[np.arange(len(emb)), own]
    masked = sims.copy()
    masked[np.arange(len(emb)), own] = -np.inf
    inter = masked.max(1)
    res = {"intra": float(intra.mean()), "inter": float(inter.mean()),
           "margin": float((intra - inter).mean()), "n": int(len(emb)), "classes": len(classes)}
    try:
        from sklearn.metrics import silhouette_score
        res["silhouette"] = float(silhouette_score(emb, labels, metric="cosine"))
    except Exception:
        res["silhouette"] = None
    return res


def project(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(emb)
    except Exception:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, metric="cosine", init="random", random_state=seed).fit_transform(emb)


def subset(parquet, levels, n_species, per_species, seed=0):
    df = pd.read_parquet(parquet)
    df = df[df["set"].astype(str) == "0"]
    rng = np.random.default_rng(seed)
    counts = df[levels[0]].value_counts()
    keep = counts[counts >= per_species].index.to_numpy()
    chosen = rng.choice(keep, size=min(n_species, len(keep)), replace=False)
    parts = [df[df[levels[0]] == s].sample(per_species, random_state=seed) for s in chosen]
    out = pd.concat(parts).reset_index(drop=True)
    for lv in levels:
        out[lv] = out[lv].astype(str)
    if "image_path" not in out.columns:
        out["image_path"] = out[levels[0]].astype(str) + "/" + out["filename"]
    out["is_valid"] = np.arange(len(out)) % 5 == 0
    return out


def main(a):
    from lepinet.data import DEFAULT_LEVELS, make_dls
    from lepinet.test import load_model, resolve_checkpoint_path

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    levels = DEFAULT_LEVELS
    df = subset(a.parquet, levels, a.n_species, a.per_species, a.seed)
    print(f"{len(df)} images | {df[levels[0]].nunique()} species")

    report, panels = {}, {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        ckpt = torch.load(resolve_checkpoint_path(path), map_location="cpu", weights_only=False)
        model, _ = load_model(ckpt, img_size=a.img_size)
        vocabs = ckpt["vocabs"]
        sub = df[df[levels[0]].isin({str(v) for v in vocabs[levels[0]]})]
        dls = make_dls(sub[["image_path", "is_valid", *levels]].reset_index(drop=True), vocabs,
                       a.img_dir, int(a.img_size * 460 / 256), a.img_size, 64, a.num_workers,
                       lowmem=False, levels=levels)
        emb = embeddings_for(model, dls, sub, device, a.num_workers)
        labels = sub[levels[0]].to_numpy()
        # Use the model's OWN trained prototypes where the classes are in vocab: that is the
        # geometry the loss actually shaped, rather than an after-the-fact class mean.
        w = model[1].head.layers[0].weight.detach().cpu().numpy()
        vidx = {str(v): i for i, v in enumerate(vocabs[levels[0]])}
        protos = np.stack([w[vidx[c]] for c in sorted(set(labels))])
        report[name] = geometry(emb, labels, protos)
        panels[name] = (project(emb, a.seed), labels)
        print(f"  {name:10s} intra {report[name]['intra']:.3f}  inter {report[name]['inter']:.3f}  "
              f"margin {report[name]['margin']:.3f}  silhouette {report[name]['silhouette']}")

    (out_dir / "geometry.json").write_text(json.dumps(report, indent=2))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), squeeze=False)
        for ax, (name, (xy, labels)) in zip(axes[0], panels.items()):
            codes = pd.factorize(labels)[0]
            ax.scatter(xy[:, 0], xy[:, 1], c=codes, cmap="tab20", s=9, alpha=.85, linewidths=0)
            m = report[name]
            ax.set_title(f"{name}\nmargin {m['margin']:.3f} · silhouette {m['silhouette']:.3f}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Embedding space — {report[list(report)[0]]['classes']} species, "
                     f"{report[list(report)[0]]['n']} images")
        fig.tight_layout()
        fig.savefig(out_dir / "embeddings.png", dpi=150)
        print("wrote", out_dir / "embeddings.png")
    except Exception as e:  # a headless box without matplotlib still gets the numbers
        print("plot skipped:", e)
    print("wrote", out_dir / "geometry.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=/path/to/ckpt.pt (repeatable)")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", default="emb")
    ap.add_argument("--n-species", type=int, default=12)
    ap.add_argument("--per-species", type=int, default=40)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
