"""Does the trap adapt/probe split leak individual moths across nights?

dev/064 splits the trap crops by (trap, night), so the timelapse frames of one night never straddle
adapt and probe. What that grouping cannot see is a moth that stays on the sheet through the day and
is photographed again the next night: same individual, same pose, two groups, possibly two splits.

This finds such twins at the pixel level. For every probe crop, the most similar adapt crop of the
same species and trap (normalised cross-correlation of 32x32 greyscale thumbnails, crop areas within
15 %), and the same statistic for reference populations: other frames of the same night (known to
hold the same individuals) and nights >= 30 apart (cannot). With --preds, the probe macro-F1 of each
model is recomputed with and without the twins.

    python dev/088_trap_split_twins.py --groups data/flemming/splits/flemming_groups.parquet \
        --img-dir data/flemming/images [--preds name=predictions.csv ...]

journal/2026-10-12-do-trap-moths-leak-across-nights.md has the result (2026-10-12).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

IMG_DIR = ""


def thumb(path):
    im = Image.open(f"{IMG_DIR}/{path}").convert("L")
    v = np.asarray(im.resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
    return im.size[0] * im.size[1], (v - v.mean()) / (v.std() + 1e-6) / np.sqrt(v.size)


def _init(d):
    global IMG_DIR
    IMG_DIR = d


def nearest(g, V, area, night, rows, cols_mask):
    """For each row in `rows`, best NCC over same-(trap, species) crops where cols_mask(row) holds."""
    best = np.full(len(g), np.nan)
    gap = np.zeros(len(g), int)
    for _, d in g.groupby(["trap", "species_folder"]):
        idx = d.index.values
        S = V[idx] @ V[idx].T
        a = area[idx]
        S[np.minimum.outer(a, a) / np.maximum.outer(a, a) <= 0.85] = -1
        for k, i in enumerate(idx):
            if not rows[i]:
                continue
            m = cols_mask(i, idx)
            if m.any():
                j = np.argmax(np.where(m, S[k], -9))
                best[i], gap[i] = S[k, j], abs(night[i] - night[idx[j]])
    return best, gap


def main(a):
    g = pd.read_parquet(a.groups).reset_index(drop=True)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.img_dir,)) as ex:
        out = list(ex.map(thumb, g.image_path, chunksize=256))
    area = np.array([o[0] for o in out], np.float64)
    V = np.stack([o[1] for o in out]).astype(np.float32)
    night = pd.to_datetime(g.night).values.astype("datetime64[D]").astype(np.int64)
    ts = g.timestamp.values
    split = g.split.values
    probe = split == "probe"

    refs = {
        "same night, other frame": lambda i, idx: (night[idx] == night[i]) & (ts[idx] != ts[i]),
        "next night (any split)": lambda i, idx: np.abs(night[idx] - night[i]) == 1,
        ">= 30 nights apart": lambda i, idx: np.abs(night[idx] - night[i]) >= 30,
    }
    print("share of crops whose nearest same-species, same-trap crop exceeds the NCC threshold")
    for name, mask in refs.items():
        b, _ = nearest(g, V, area, night, np.ones(len(g), bool), mask)
        b = b[~np.isnan(b)]
        print(f"  {name:26s} n={len(b):6d}  " +
              "  ".join(f">{t}: {(b > t).mean() * 100:6.3f} %" for t in a.thresholds))

    twin, gap = nearest(g, V, area, night, probe, lambda i, idx: split[idx] == "adapt")
    print(f"\nprobe crops with a twin in adapt ({probe.sum()} probe crops):")
    for t in a.thresholds:
        m = probe & (twin > t)
        print(f"  NCC > {t}: {m.sum():4d} ({m.sum() / probe.sum() * 100:.2f} %), "
              f"{g[m].species_folder.nunique()} species; nights to the twin: "
              f"{pd.Series(gap[m]).value_counts().sort_index().to_dict()}")

    if a.preds:
        from sklearn.metrics import f1_score

        def mf1(d):
            d = d[d.known_label]
            return f1_score(d.label.astype(str), d.prediction.astype(str), average="macro",
                            labels=sorted(d.label.astype(str).unique()), zero_division=0)
        print("\nprobe species macro-F1 recomputed (known-label rows), without the twins:")
        for spec in a.preds:
            name, path = spec.split("=", 1)
            d = pd.read_csv(path)
            d = d[d.level == 0]
            cells = [f"all {mf1(d):.4f}"]
            for t in a.thresholds:
                drop = set(g[probe & (twin > t)].image_path)
                cells.append(f"-twins>{t} {mf1(d[~d.filename.isin(drop)]):.4f}")
            print(f"  {name:12s} " + "  ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="data/flemming/splits/flemming_groups.parquet")
    ap.add_argument("--img-dir", default="data/flemming/images")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.95, 0.97])
    ap.add_argument("--preds", nargs="*", default=[], help="name=path/to/predictions.csv")
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args())
