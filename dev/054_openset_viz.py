"""The *right* plot for an angular open-set effect: 1-D score distributions, not a 2-D projection.

`dev/053` showed why UMAP/t-SNE are the wrong tool here: silhouette barely moved (0.617 -> 0.641)
because **cluster separability was never the problem** — closed-set accuracy is ~0.911 for both
heads. What ArcFace changes is the *absolute angular position* of embeddings relative to the class
prototypes, and a projection deliberately discards absolute geometry (it is invariant to exactly the
thing we care about). So visualise the quantity the decision actually uses:

  **Panel A — the money plot.** Histogram of the novelty score (`max cos θ` over all prototypes) for
  KNOWN vs NOVEL species, one row per model. Overlap area *is* the error; the AUROC is the
  probability a novel image scores below a known one. This shows the 0.601 -> 0.91 result directly
  instead of implying it.

  **Panel B — the mechanism.** Histogram of cos to the *own* prototype vs the *nearest wrong* one.
  The gap between the two curves is the margin, and their absolute position is what a fixed
  threshold can exploit.

Novel species come from global_lepi classes the model never trained on (excluded by
`min_img_per_spc`), so this isolates novelty from domain shift.

    python dev/054_openset_viz.py --models plain=a.pt arcface-zscore=b.pt \\
        --parquet meta.parquet --img-dir images/ --out openset/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sample_known_novel(parquet, species_vocab, levels, n_known, n_novel, seed=0):
    """Held-out images of KNOWN species, and images of species the model has never seen."""
    df = pd.read_parquet(parquet)
    df = df[df["set"].astype(str) == "0"]
    df[levels[0]] = df[levels[0]].astype(str)
    known_set = {str(v) for v in species_vocab}
    is_known = df[levels[0]].isin(known_set)
    rng = np.random.default_rng(seed)
    known = df[is_known].sample(min(n_known, int(is_known.sum())), random_state=seed)
    novel_pool = df[~is_known]
    novel = novel_pool.sample(min(n_novel, len(novel_pool)), random_state=seed) if len(novel_pool) else novel_pool
    out = pd.concat([known.assign(_novel=False), novel.assign(_novel=True)]).reset_index(drop=True)
    if "image_path" not in out.columns:
        out["image_path"] = out[levels[0]].astype(str) + "/" + out["filename"]
    for lv in levels:
        out[lv] = out[lv].astype(str)
    out["is_valid"] = np.arange(len(out)) % 5 == 0
    print(f"  known {int((~out._novel).sum())} | novel {int(out._novel.sum())} "
          f"({out.loc[out._novel, levels[0]].nunique()} unseen species)")
    return out


@torch.no_grad()
def cosines(model, dls, df, device, num_workers=8):
    """Per image: cos to every prototype -> (max cos, cos to own prototype if known)."""
    from lepinet.test import dl_num_workers

    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    dl = dls.test_dl(df, num_workers=nw)
    body, head = model[0], model[1].head
    model.to(device).eval()
    w = torch.nn.functional.normalize(head.layers[0].weight.detach(), dim=1).to(device)
    out = []
    for batch in dl:
        feats = body(batch[0].to(device))
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1) if feats.ndim == 4 else feats
        emb = head.preclassification(pooled.float())
        out.append((emb @ w.T).cpu())
    return torch.cat(out).numpy()


def auroc(known_scores, novel_scores):
    """P(known scores above novel) — Mann-Whitney, no sklearn needed."""
    all_s = np.concatenate([known_scores, novel_scores])
    ranks = all_s.argsort().argsort()
    r_known = ranks[: len(known_scores)].sum()
    n_k, n_n = len(known_scores), len(novel_scores)
    return float((r_known - n_k * (n_k - 1) / 2) / (n_k * n_n))


def main(a):
    from lepinet.data import DEFAULT_LEVELS, make_dls
    from lepinet.test import load_model, resolve_checkpoint_path

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    levels = DEFAULT_LEVELS
    report, panels = {}, {}

    for spec in a.models:
        name, path = spec.split("=", 1)
        print(f"[{name}]")
        ckpt = torch.load(resolve_checkpoint_path(path), map_location="cpu", weights_only=False)
        vocabs = ckpt["vocabs"]
        df = sample_known_novel(a.parquet, vocabs[levels[0]], levels, a.n_known, a.n_novel, a.seed)
        model, _ = load_model(ckpt, img_size=a.img_size)
        dls = make_dls(df[["image_path", "is_valid", *levels]].reset_index(drop=True), vocabs,
                       a.img_dir, int(a.img_size * 460 / 256), a.img_size, 64, a.num_workers,
                       lowmem=False, levels=levels)
        sims = cosines(model, dls, df, device, a.num_workers)

        novel = df["_novel"].to_numpy()
        max_cos = sims.max(1)
        vidx = {str(v): i for i, v in enumerate(vocabs[levels[0]])}
        own_idx = np.array([vidx.get(s, -1) for s in df[levels[0]]])
        k = ~novel
        own_cos = sims[np.arange(len(sims))[k], own_idx[k]]
        masked = sims[k].copy()
        masked[np.arange(k.sum()), own_idx[k]] = -np.inf
        wrong_cos = masked.max(1)

        report[name] = {
            "auroc": auroc(max_cos[k], max_cos[novel]),
            "known_maxcos": float(max_cos[k].mean()), "novel_maxcos": float(max_cos[novel].mean()),
            "own": float(own_cos.mean()), "nearest_wrong": float(wrong_cos.mean()),
            "margin": float((own_cos - wrong_cos).mean()),
            "n_known": int(k.sum()), "n_novel": int(novel.sum()),
        }
        panels[name] = (max_cos[k], max_cos[novel], own_cos, wrong_cos)
        r = report[name]
        print(f"  AUROC {r['auroc']:.4f} | known maxcos {r['known_maxcos']:.3f} vs novel "
              f"{r['novel_maxcos']:.3f} | own {r['own']:.3f} nearest-wrong {r['nearest_wrong']:.3f}")

    (out_dir / "openset.json").write_text(json.dumps(report, indent=2))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(panels)
        fig, axes = plt.subplots(2, n, figsize=(6 * n, 8), squeeze=False)
        for j, (name, (mk, mn, own, wrong)) in enumerate(panels.items()):
            r = report[name]
            ax = axes[0][j]
            bins = np.linspace(min(mn.min(), mk.min()), max(mn.max(), mk.max()), 60)
            ax.hist(mk, bins=bins, alpha=.65, label=f"known (n={r['n_known']})", color="#1b6b4a", density=True)
            ax.hist(mn, bins=bins, alpha=.65, label=f"novel (n={r['n_novel']})", color="#c2492f", density=True)
            ax.set_title(f"{name} — novelty score\nAUROC {r['auroc']:.3f}")
            ax.set_xlabel("max cos θ over prototypes")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
            ax = axes[1][j]
            bins = np.linspace(min(wrong.min(), own.min()), max(wrong.max(), own.max()), 60)
            ax.hist(own, bins=bins, alpha=.65, label="own prototype", color="#1b6b4a", density=True)
            ax.hist(wrong, bins=bins, alpha=.65, label="nearest wrong", color="#8a8f88", density=True)
            ax.set_title(f"class geometry — margin {r['margin']:.3f}")
            ax.set_xlabel("cos θ")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
        fig.suptitle("Open-set behaviour: the effect is angular, so plot the angles", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "openset.png", dpi=150)
        print("wrote", out_dir / "openset.png")
    except Exception as e:
        print("plot skipped:", e)
    print("wrote", out_dir / "openset.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", default="openset")
    ap.add_argument("--n-known", type=int, default=1500)
    ap.add_argument("--n-novel", type=int, default=1500)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
