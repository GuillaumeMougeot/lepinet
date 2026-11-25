"""Z24: the eval-only MixRes check. Does resolution alone make the frozen ToL head's small-crop deficit on the trap probe?

    CUDA_VISIBLE_DEVICES="" taskset -c 0-3 nice -n 10 .venv/bin/python -I dev/121_mixres_downscale_check.py [--limit N]

Z22 found the frozen ToL head (tol_full's own head restricted to the moth species it shares with lepinet) at 83.4 %
accuracy on the probe's largest crops (absolute quintile qa5) and 65.2 % on the smallest (qa1), list rule, one view.
This shrinks the qa5 crops to the qa3 and qa1 median sizes (sqrt(area) 178 and 124 px; PIL bilinear, which
antialiases when reducing; aspect kept) and scores them the same way; the release's reader then upsamples them to 256
as it does real small crops. qa5 and qa1 as they are must reproduce Z22's numbers first (within 0.5 points).

Prediction: species logits from the release (`trunk.onnx`, `heads/tol.safetensors`, `predict.py`'s reader and score,
the read dev/109's evaluation used), restricted to ToL rows that map to a lepinet species (`data/tol_eval/
map_species.csv`, the species with most training images per ToL row, as dev/109 `_maps`) on the north-European list,
argmax; truth from the crop's folder name (`data/flemming/name2id.csv`). Per-crop results and a bootstrap over species
(500 draws, paired) go to `data/population_probes/z24/`. Journal: 2026-11-13 beyond scaling, Z24.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/release_checks/release_tol"
PROBE = ROOT / "data/population_probes/z22/probe_crops.parquet"
IMAGES = ROOT / "data/flemming/images"
NE_LIST = Path.home() / "codes/mini_trainer/dev/releases/mambo_v3/presets/north_europe.classes"
OUT = ROOT / "data/population_probes/z24"
Z22 = {"qa5": 0.834, "qa1": 0.652}
BATCH = 16


def release():
    spec = importlib.util.spec_from_file_location("release_predict", RELEASE / "predict.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["release_predict"] = m
    spec.loader.exec_module(m)
    return m


def read(path, size, target=None):
    """The release reader (largest centred square, PIL bilinear to `size`), after an optional shrink to sqrt(area)
    = target."""
    with Image.open(path) as im:
        im = im.convert("RGB")
    if target is not None:
        f = target / np.sqrt(im.width * im.height)
        if f < 1:
            im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.Resampling.BILINEAR)
    w, h = im.size
    side = int((w / size if w / size < h / size else h / size) * size)
    left, top = int(0.5 * (w - side)), int(0.5 * (h - side))
    im = im.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(im).transpose(2, 0, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=0, help="a quick run on N crops per condition")
    p.add_argument("--reps", type=int, default=500)
    a = p.parse_args()
    import onnxruntime as ort
    P = release()
    cfg = json.loads((RELEASE / "config.json").read_text())
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 4, 1
    sess = ort.InferenceSession(str(RELEASE / cfg["trunk"]["onnx"]), so, providers=["CPUExecutionProvider"])
    head = P.load_head(RELEASE / "heads/tol.safetensors")
    labels = json.loads((RELEASE / "heads/tol.json").read_text())["labels"][0]
    size = cfg["preprocessing"]["read_size"]

    sp = pd.read_csv(ROOT / "data/tol_eval/map_species.csv", dtype={"speciesKey": str})
    sp = sp[sp.tol_class.notna()].sort_values("n_train", ascending=False).drop_duplicates("tol_class")
    assert all(labels[int(i)] == c for i, c in zip(sp.tol_index, sp.tol_class)), "release rows != dev/109's tol_index"
    ne = set(open(NE_LIST).read().split())
    cand = sp[sp.speciesKey.isin(ne)]
    rows, keys = cand.tol_index.astype(int).to_numpy(), cand.speciesKey.to_numpy()
    print(f"{len(sp):,} ToL rows mapped to lepinet species; {len(rows):,} on the list", flush=True)
    n2i = pd.read_csv(ROOT / "data/flemming/name2id.csv", dtype={"speciesKey": str}).set_index("verbatimScientificName")

    q = pd.read_parquet(PROBE)
    q["truth"] = n2i.loc[q.species, "speciesKey"].to_numpy()
    med = q.groupby("abs_q", observed=True)["size"].median()
    conds = [("qa5", "qa5", None), ("qa1", "qa1", None),
             ("qa5_to_qa3", "qa5", float(med["qa3"])), ("qa5_to_qa1", "qa5", float(med["qa1"]))]
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for name, qq, target in conds:
        d = q[q.abs_q == qq].reset_index(drop=True)
        if a.limit:
            d = d.sample(min(a.limit, len(d)), random_state=0).reset_index(drop=True)
        pred, t0 = [], time.time()
        for s in range(0, len(d), BATCH):
            px = np.stack([read(IMAGES / f, size, target) for f in d.image_path[s:s + BATCH]])
            feats, _ = sess.run(["features", "embedding"], {"images": px})
            _, ranks = P.score(feats, head)
            pred.append(keys[ranks[0][:, rows].argmax(1)])
        d["pred"] = np.concatenate(pred)
        d["ok"] = d.pred == d.truth
        d[["image_path", "species", "size", "truth", "pred", "ok"]].to_parquet(OUT / f"{name}.parquet")
        acc = d.ok.mean()
        print(f"{name:12s} {len(d):,} crops, target {target and round(target)} px: accuracy {acc:.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        res.append({"condition": name, "crops": len(d), "target_px": target, "accuracy": acc})
    for name in ("qa5", "qa1"):
        r = next(x for x in res if x["condition"] == name)
        print(f"check {name}: {r['accuracy']:.4f} against Z22's {Z22[name]}")
    base = pd.read_parquet(OUT / "qa5.parquet").set_index("image_path")
    rng = np.random.default_rng(0)
    for name in ("qa5_to_qa3", "qa5_to_qa1"):
        d = pd.read_parquet(OUT / f"{name}.parquet").set_index("image_path")
        j = base[["species", "ok"]].join(d["ok"], rsuffix="_s")
        sp_ = j.species.unique()
        g = j.groupby("species")[["ok", "ok_s"]].agg(["sum", "count"])
        boot = []
        for _ in range(a.reps):
            pick = rng.choice(sp_, len(sp_))
            gg = g.loc[pick]
            boot.append(gg[("ok_s", "sum")].sum() / gg[("ok_s", "count")].sum() - gg[("ok", "sum")].sum() / gg[("ok", "count")].sum())
        diff = j.ok_s.mean() - j.ok.mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{name} - qa5: {100 * diff:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}] (bootstrap over species)")
        next(x for x in res if x["condition"] == name).update(diff=diff, ci=[lo, hi])
    (OUT / "z24.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT / "z24.json")


if __name__ == "__main__":
    main()
