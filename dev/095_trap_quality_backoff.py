"""Does the back-off policy answer at a coarser rank when a trap crop is poor, and does it stay right?

For every probe crop: sharpness (variance of the Laplacian of the greyscale crop resized to 128 px on
its short side, so it is not a proxy for size) and pixel area. The release cascade (`dev/083`:
per-rank thresholds at 95 % precision conditional on reaching the rank, fitted on the seed-0 half of
the trap nights, as in `dev/086`) is applied to the other half, then broken down by quintile.

    python dev/095_trap_quality_backoff.py            # caches data/trap_quality/probe_quality.parquet

journal/2026-10-23-does-the-model-know-when-a-trap-image-is-too-poor.md.
"""
from __future__ import annotations

import importlib.util
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "flemming" / "images"
CACHE = ROOT / "data" / "trap_quality" / "probe_quality.parquet"
MODELS = {"B8 (T = 1.91)": "b8", "P5": "p5v2", "B3rep5x (T = 1.70)": "b3"}
_s = importlib.util.spec_from_file_location("rel", ROOT / "dev" / "083_hf_release.py")
rel = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rel)


def quality(path):
    im = Image.open(IMG / path)
    w, h = im.size
    g = im.convert("L")
    s = 128 / min(w, h)
    g = np.asarray(g.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR), dtype=np.float32)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var()), w * h


def probe_quality():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    p = pd.read_parquet(ROOT / "data/flemming/splits/flemming_probe.parquet", columns=["image_path"])
    with ProcessPoolExecutor(4) as ex:
        q = list(ex.map(quality, p.image_path, chunksize=64))
    p["sharpness"], p["area"] = [x[0] for x in q], [x[1] for x in q]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(CACHE)
    return p


def cascade_by(res, th, key, q=5):
    rows = []
    res = res.assign(bin=pd.qcut(res[key], q, labels=[f"Q{i + 1}" for i in range(q)]))
    for b, d in res.groupby("bin", observed=True):
        c = rel.cascade(d, th)
        rows.append({"bin": b, "n": len(d), f"{key} median": d[key].median(),
                     **{f"{lv} %": 100 * c[lv]["coverage"] for lv in rel.LEVELS},
                     **{f"{lv} prec": c[lv]["precision"] for lv in rel.LEVELS},
                     "abstain %": 100 * c["abstain"], "correct answer %": 100 * c["useful_rate"]})
    return pd.DataFrame(rows)


def main():
    q = probe_quality()
    group = q.image_path.map(lambda s: "_".join(re.search(r"TRAPNAME_([^_]+)_IMAGENAME_(\d{8})", s).groups()))
    for name, folder in MODELS.items():
        res = pd.read_parquet(ROOT / f"data/hf_release/{folder}/eval_probe.parquet")
        assert len(res) == len(q) and (res.group.astype(str).values == group.values).all(), "row order differs"
        res = pd.concat([res, q[["sharpness", "area"]]], axis=1)
        groups = np.array(sorted(res.group.unique(), key=str))
        fit_g = set(np.random.default_rng(0).permutation(groups)[: len(groups) // 2])
        th = rel.fit_thresholds(res[res.group.isin(fit_g)], 0.95)
        held = res[~res.group.isin(fit_g)]
        print(f"\n### {name}: thresholds {', '.join(f'{k} {v:.4f}' for k, v in th.items())}; "
              f"{len(held)} held-out crops")
        for key in ("sharpness", "area"):
            print(f"\nby {key} quintile (Q1 = lowest):")
            print(cascade_by(held, th, key).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
