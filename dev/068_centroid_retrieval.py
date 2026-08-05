"""Can we throw the head away at inference? Class centroids + nearest-neighbour vs the linear head.

The owner's question, and the cheapest real test of the head-scaling direction
(`journal/2026-08-05-scaling-the-head.md`). At 1 M species a 1280 x 1M prototype matrix is 5 GB; an
ANN index over 1 M centroids is a few GB on disk and never materialises 1 M logits. But that only
works if centroids are as good as trained prototypes, which is an empirical question about the
embedding's geometry.

**Why this project should win it.** ArcFace exists to make classes tight and well separated *in
angle*, which is exactly what nearest-neighbour retrieval needs. The plain cosine head's clusters are
diffuse (mean max-cos 0.144) while ArcFace x z-score's are tight (0.671) — so the prediction is that
centroid retrieval nearly matches the linear head for the ArcFace model and fails for the plain one.
Running both is what makes the result interpretable rather than a single number.

Three retrieval variants, because "centroid" hides a modelling assumption:

* ``mean``    — normalised mean embedding per class. Optimal if a class is an isotropic blob.
* ``medoid``  — the training image closest to that mean. Robust to outliers and mislabels; also the
                only variant that stores a *real* example, which matters for interpretability.
* ``kmeans``  — k centroids per class (``--k``). Handles multimodal classes (sexual dimorphism,
                worn specimens, dorsal vs lateral) that a single mean averages into nothing.

Also reports the **rank-truncated** prototype accuracy, which answers option 0 of the same journal
entry in the same pass: if the trained prototypes are effectively low-rank, factorising the head is
nearly free and none of the exotic options are needed.

    python dev/068_centroid_retrieval.py --model '...*.pt' --parquet ... --img-dir ... --out r.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def embed(model, dls, df, device, num_workers=32):
    """Per-image normalised embeddings — the vector the head would dot with its prototypes."""
    from lepinet.test import dl_num_workers

    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    dl = dls.test_dl(df, num_workers=nw)
    model.to(device).eval()
    body, head = model[0], model[1].head
    out = []
    t0 = time.time()
    for batch in dl:
        feats = body(batch[0].to(device))
        pooled = (F.adaptive_avg_pool2d(feats, 1).flatten(1) if feats.ndim == 4 else feats).float()
        out.append(F.normalize(head.preclassification(pooled), dim=1).cpu())
    e = torch.cat(out)
    print(f"  embedded {len(e)} images in {time.time()-t0:.0f}s")
    return e


def build_centroids(emb, y, n_classes, kind="mean", k=3, seed=0):
    """Class summaries in embedding space. Returns (centroids [M,d], their class ids [M])."""
    cents, owners = [], []
    g = torch.Generator().manual_seed(seed)
    for c in range(n_classes):
        m = y == c
        if not m.any():
            continue
        e = emb[m]
        if kind == "mean":
            cents.append(F.normalize(e.mean(0, keepdim=True), dim=1))
            owners.append(c)
        elif kind == "medoid":
            mu = F.normalize(e.mean(0, keepdim=True), dim=1)
            cents.append(e[(e @ mu.T).squeeze(1).argmax()].unsqueeze(0))
            owners.append(c)
        elif kind == "kmeans":
            kk = min(k, len(e))
            idx = torch.randperm(len(e), generator=g)[:kk]
            cen = e[idx].clone()
            for _ in range(10):                       # Lloyd on the sphere
                a = (e @ cen.T).argmax(1)
                for j in range(kk):
                    if (a == j).any():
                        cen[j] = F.normalize(e[a == j].mean(0), dim=0)
            cents.append(cen)
            owners.extend([c] * kk)
    return torch.cat(cents), torch.tensor(owners)


def macro_f1(pred, true, n_classes):
    """Same definition as the package's metric: mean per-class F1 over classes present in `true`."""
    f1s = []
    for c in torch.unique(true):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def main(a):
    import pandas as pd

    from lepinet.data import DEFAULT_LEVELS, filter_df, make_dls
    from lepinet.test import load_model, resolve_checkpoint_path

    ckpt = torch.load(resolve_checkpoint_path(a.model), map_location="cpu", weights_only=False)
    levels = ckpt.get("levels", DEFAULT_LEVELS)
    vocabs = ckpt["vocabs"]
    lvl = levels[0]
    idx = {str(v): i for i, v in enumerate(vocabs[lvl])}
    n_classes = len(idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(ckpt, img_size=a.img_size)

    def prep(set_id, cap):
        df = filter_df(pd.read_parquet(a.parquet), keep_in=[set_id], levels=levels)
        df = df[df[lvl].astype(str).isin(idx)]
        if "image_path" not in df.columns:
            df["image_path"] = df[lvl].astype(str) + "/" + df["filename"]
        for lv in levels:
            df[lv] = df[lv].astype(str)
        if cap and len(df) > cap:
            # Cap per class, not globally: a global sample would starve rare species of the very
            # examples a centroid needs, and the metric is macro.
            #
            # Done by index rather than `groupby.apply`, because that drops the grouping column from
            # the result (pandas keeps it as the index) and the next line then fails with
            # `KeyError: ['speciesKey'] not in index` -- a message about a column the caller can see
            # in the dataframe it passed in.
            per = max(1, cap // n_classes)
            keep = df.groupby(lvl, sort=False).head(per).index if per == 1 else (
                df.sample(frac=1.0, random_state=0).groupby(lvl, sort=False).head(per).index)
            df = df.loc[keep]
        df = df.reset_index(drop=True)
        df["is_valid"] = np.arange(len(df)) % 5 == 0
        return df

    train_df, test_df = prep(a.train_set, a.train_cap), prep(a.test_set, a.test_cap)
    print(f"train (for centroids): {len(train_df)} | test: {len(test_df)}")
    dls = make_dls(train_df[["image_path", "is_valid", *levels]], vocabs, a.img_dir,
                   int(a.img_size * 460 / 256), a.img_size, 128, a.num_workers,
                   lowmem=False, levels=levels)

    e_tr = embed(model, dls, train_df, device, a.num_workers)
    e_te = embed(model, dls, test_df, device, a.num_workers)
    y_tr = torch.tensor([idx[s] for s in train_df[lvl]])
    y_te = torch.tensor([idx[s] for s in test_df[lvl]])

    res = {"head": ckpt.get("head"), "n_classes": n_classes,
           "n_train": len(e_tr), "n_test": len(e_te), "macro_f1": {}}

    # --- the incumbent: the trained prototype matrix
    W = F.normalize(model[1].head.layers[0].weight.detach().cpu().float(), dim=1)
    res["macro_f1"]["linear_head"] = macro_f1((e_te @ W.T).argmax(1), y_te, n_classes)

    # --- retrieval variants
    for kind in a.variants.split(","):
        cen, owners = build_centroids(e_tr, y_tr, n_classes, kind=kind, k=a.k)
        pred = owners[(e_te @ cen.T).argmax(1)]
        res["macro_f1"][f"centroid_{kind}"] = macro_f1(pred, y_te, n_classes)
        res[f"n_centroids_{kind}"] = int(len(cen))

    # --- option 0: how low-rank is the trained prototype matrix?
    s = torch.linalg.svdvals(W)
    energy = (s ** 2).cumsum(0) / (s ** 2).sum()
    res["prototype_spectrum"] = {
        "rank_for_90pct_energy": int((energy < 0.90).sum()) + 1,
        "rank_for_99pct_energy": int((energy < 0.99).sum()) + 1,
        "participation_ratio": float((s.sum() ** 2) / (s ** 2).sum()),
        "d": int(W.shape[1]),
    }
    res["macro_f1_truncated"] = {}
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    for r in [int(x) for x in a.ranks.split(",") if int(x) < min(W.shape)]:
        Wr = F.normalize((U[:, :r] * S[:r]) @ Vh[:r], dim=1)
        res["macro_f1_truncated"][str(r)] = macro_f1((e_te @ Wr.T).argmax(1), y_te, n_classes)

    print("\nmacro-F1:")
    for k, v in res["macro_f1"].items():
        d = v - res["macro_f1"]["linear_head"]
        print(f"  {k:22s} {v:.4f}  ({d:+.4f} vs linear head)")
    print(f"\nprototype matrix: {res['prototype_spectrum']}")
    print("rank-truncated linear head:")
    for r, v in res["macro_f1_truncated"].items():
        print(f"  rank {r:>5s}  {v:.4f}  ({v - res['macro_f1']['linear_head']:+.4f})")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", default="centroid_retrieval.json")
    ap.add_argument("--train-set", default="1", help="Fold to build centroids from (NOT the test fold).")
    ap.add_argument("--test-set", default="0")
    ap.add_argument("--train-cap", type=int, default=250000)
    ap.add_argument("--test-cap", type=int, default=150000)
    ap.add_argument("--variants", default="mean,medoid,kmeans")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ranks", default="64,128,256,512")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=32)
    main(ap.parse_args())
