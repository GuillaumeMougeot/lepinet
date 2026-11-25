"""BT6, the reverse of C3's order half: Kim Bjerge's AMI broad-taxon classifier and the frozen ToL model, both scored on
ToL's own test fold in Kim's 15 arthropod categories. C3 (dev/115) scored ToL on Kim's trap crops; this scores Kim's
model on ToL's photos. CPU only, at most 4 threads.

    .venv/bin/python -I dev/118_broad_taxa_reverse.py sample   # stratified sample of ToL's test fold, before any score
    .venv/bin/python -I dev/118_broad_taxa_reverse.py embed    # ToL release (ONNX) and Kim's ResNet50 on every image, resumable
    .venv/bin/python -I dev/118_broad_taxa_reverse.py score    # three classifiers, macro-F1 over the 15 categories

Sample: ToL test rows with no `exclude` flag, mapped to a category by dev/115's V0 mapping (family names, written before
C3's first score); Vegetation (kingdom Plantae) is left out, since Kim's Vegetation is trap background, not plant
photos. Up to `--per-cat` images per category, drawn at random (seed 0) from rows sorted by path, so a category keeps
its own mix of species and image types (citizen science, museum specimen).

Classifiers:
- kim: his published weights (`dhc_best_128.pth`, loaded as tensors only) and his inference preprocessing (squash to
  128 x 128, bilinear without antialiasing, RGB, [0, 1], no normalisation), as dev/117 reproduces it.
- tol_zero: zero training, ToL's species posterior summed into the 16 categories through its families (dev/115 V0).
- tol_head_a: C3's head (a), logistic regression on ToL's embedding fitted on Kim's training list (his trap crops,
  C as chosen in BT1), applied to ToL's photos: the trap-trained head on the general trunk.
Each is scored as trained (16-way, so a Vegetation answer is wrong) and restricted to the 15 arthropod categories.
Kim's model classifies detector crops; ToL's photos are whole photos, often close-ups. The comparison measures how far
each model travels from its own domain, not which is the better classifier.
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bt115", ROOT / "dev/115_broad_taxa.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
CATS, THREADS = B.CATS, B.THREADS
VEG = CATS.index("Vegetation")
META = Path("/data/datasets/tol256/metadata.parquet")
IMAGES = Path("/data/datasets/tol256/images")
KIM_WEIGHTS = B.KIM / "dhc_best_128.pth"
OUT = B.OUT / "reverse"
SHARD, BATCH = 1024, 16


def sample(a):
    import pyarrow.parquet as pq
    cols = ["path", "species", "kingdom", "phylum", "class", "order", "family", "img_type", "split", "exclude"]
    t = pq.read_table(META, columns=cols, filters=[("split", "=", "test")]).to_pandas()
    t = t[t.exclude.isna()].sort_values("path").reset_index(drop=True)
    key = t[["kingdom", "phylum", "class", "order", "family"]].fillna("").agg("|".join, axis=1)
    uniq = key.unique()
    cat, _ = B.family_category(list(uniq), "V0")
    t["cat"] = key.map(dict(zip(uniq, cat))).astype(int)
    t = t[(t.cat >= 0) & (t.cat != VEG)]
    rng = np.random.default_rng(0)
    s = pd.concat([g.iloc[rng.permutation(len(g))[:a.per_cat]] for _, g in t.groupby("cat")])
    s = s.sort_values(["cat", "path"]).reset_index(drop=True)
    s["category"] = [CATS[c] for c in s.cat]
    s["img_type"] = s.img_type.fillna("unknown")
    OUT.mkdir(parents=True, exist_ok=True)
    s[["path", "species", "order", "family", "img_type", "cat", "category"]].to_csv(OUT / "index.csv", index=False)
    print(f"{len(s):,} images of {s.species.nunique():,} species (pool {len(t):,})")
    print(pd.crosstab(s.category, s.img_type.str.split(":").str[0], margins=True).to_string())


def _squash128(path):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    with Image.open(path) as im:
        x = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float()[None]
    x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False, antialias=False)
    return x[0].round().clamp(0, 255) / 255


def _kim_model():
    import torch
    import torch.nn as nn
    import torchvision

    class KimResNet50(nn.Module):   # dev/117's class, without the ImageNet download
        def __init__(self):
            super().__init__()
            self.model_ft = torchvision.models.resnet50(weights=None)
            self.model_ft.fc = nn.Identity()
            self.drop = nn.Dropout(p=0.5)
            self.linear_lvl1 = nn.Linear(2048, 512)
            self.relu_lv1 = nn.ReLU(inplace=False)
            self.softmax_reg1 = nn.Linear(512, len(CATS))

        def forward(self, x):
            return self.softmax_reg1(self.relu_lv1(self.linear_lvl1(self.drop(self.model_ft(x)))))

    m = KimResNet50()
    m.load_state_dict(torch.load(KIM_WEIGHTS, map_location="cpu", weights_only=True))
    return m.eval()


def embed(a):
    import onnxruntime as ort
    import torch
    from scipy import sparse
    torch.set_num_threads(THREADS)
    P = B.predict_module()
    idx = pd.read_csv(OUT / "index.csv")
    n = a.limit or len(idx)
    cfg = json.loads((B.RELEASE / "config.json").read_text())
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = THREADS, 1
    sess = ort.InferenceSession(str(B.RELEASE / cfg["trunk"]["onnx"]), so, providers=["CPUExecutionProvider"])
    head = P.load_head(B.RELEASE / "heads/tol.safetensors")
    fam = head.parents[1][head.parents[0]]
    S = sparse.csr_matrix((np.ones(len(fam), np.float32), (np.arange(len(fam)), fam)), shape=(len(fam), head.sizes[2]))
    M = B.category_matrix("V0")
    read = cfg["preprocessing"]["read_size"]
    kim = _kim_model()
    shards = OUT / ("emb_smoke" if a.limit else "emb")
    shards.mkdir(exist_ok=True)
    n_sh = (n + SHARD - 1) // SHARD
    t0, done = time.time(), 0
    for s in range(n_sh):
        dst = shards / f"shard_{s:04d}.npz"
        if dst.exists():
            continue
        rows = np.arange(s * SHARD, min((s + 1) * SHARD, n))
        out = {k: [] for k in ("embedding", "cat_mass", "max_p", "kim_logits")}
        for b in range(0, len(rows), BATCH):
            paths = [IMAGES / p for p in idx.path.to_numpy()[rows[b:b + BATCH]]]
            px = np.stack([P.read_square(p, read) for p in paths])
            feats, _ = sess.run(["features", "embedding"], {"images": px})
            emb, ranks = P.score(feats, head)
            p = B._softmax(ranks[0])
            out["embedding"].append(emb.astype(np.float16))
            out["cat_mass"].append((S.T @ p.T).T.astype(np.float32) @ M)
            out["max_p"].append(p.max(1))
            with torch.no_grad():
                out["kim_logits"].append(kim(torch.stack([_squash128(q) for q in paths])).numpy())
        tmp = shards / f".tmp_{s:04d}.npz"
        np.savez(tmp, rows=rows, **{k: np.concatenate(v) for k, v in out.items()})
        os.replace(tmp, dst)
        done += len(rows)
        rate = done / (time.time() - t0)
        print(f"shard {s + 1}/{n_sh}  {rate:.2f} img/s  eta {(n - (s + 1) * SHARD) / rate / 3600:.2f} h", flush=True)


def _macro15(y, p):
    r = B.macro_prf(y, p)
    present = [c for c in CATS if r["per_class"][c]["support"] > 0]
    return {"f1": float(np.mean([r["per_class"][c]["f1"] for c in present])), "accuracy": r["accuracy"], "n": r["n"],
            "vegetation_share": float((p == VEG).mean()),
            "per_class_f1": {c: round(r["per_class"][c]["f1"], 4) for c in present}}


def score(a):
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    files = sorted((OUT / "emb").glob("shard_*.npz"))
    z = {k: [] for k in ("rows", "embedding", "cat_mass", "kim_logits")}
    for f in files:
        with np.load(f) as d:
            for k in z:
                z[k].append(d[k])
    z = {k: np.concatenate(v) for k, v in z.items()}
    assert len(z["rows"]) == len(idx) and (z["rows"] == np.arange(len(idx))).all(), "embed is not finished"
    y = idx.cat.to_numpy()
    # C3's head (a): Kim's training list, C as BT1 chose it on his fold
    kidx, kz = B._load_all(("embedding",))
    tr = (kidx.split == "train").to_numpy()
    C = json.loads((B.OUT / "head.json").read_text())["C_chosen"]["a"][0]
    lr = B.LR(C).fit(B._features(kz, "embedding")[tr], kidx.label.to_numpy()[tr])
    X = z["embedding"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    scores = {"kim": z["kim_logits"], "tol_zero": np.log(z["cat_mass"] + 1e-6), "tol_head_a": lr.logits(X)}
    group = idx.img_type.str.split(":").str[0].replace({"Museum Specimen": "museum", "Citizen Science": "citizen"})
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(y), len(y)) for _ in range(a.boot)]
    res = {"C_head_a": C, "n": int(len(y)), "groups": group.value_counts().to_dict()}
    for name, s in scores.items():
        for mode in ("as_trained", "15_way"):
            s2 = s.copy()
            if mode == "15_way":
                s2[:, VEG] = -np.inf
            p = s2.argmax(1)
            r = _macro15(y, p)
            boot = [_macro15(y[d], p[d])["f1"] for d in draws]
            r["f1_95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            r["by_group"] = {g: {k: v for k, v in _macro15(y[m], p[m]).items() if k != "per_class_f1"}
                             for g in group.unique() for m in [(group == g).to_numpy()] if m.sum() >= 500}
            res[f"{name}|{mode}"] = r
            print(f"{name:11s} {mode:10s} macro-F1 {r['f1']:.3f} [{r['f1_95'][0]:.3f}, {r['f1_95'][1]:.3f}]  "
                  f"acc {r['accuracy']:.3f}  Vegetation {r['vegetation_share']:.3f}  "
                  + "  ".join(f"{g} {v['f1']:.3f}" for g, v in r["by_group"].items()), flush=True)
    (OUT / "scores.json").write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["sample", "embed", "score"])
    p.add_argument("--per-cat", type=int, default=2000, help="sample: images per category at most")
    p.add_argument("--limit", type=int, default=0, help="embed: only the first N images (a smoke test)")
    p.add_argument("--boot", type=int, default=200, help="score: bootstrap draws over images")
    a = p.parse_args()
    {"sample": sample, "embed": embed, "score": score}[a.step](a)


if __name__ == "__main__":
    main()
