"""BioCLIP-2's evaluation suite on frozen encoders (paper B, claim C2): stock BioCLIP-2 reproduced first, then tol_full.

CPU only. BioCLIP-2's protocols are ported from its own code, github.com/Imageomics/bioclip-2 at 49e1cf5
(`src/evaluation/{classification,newt,awa2,plantdoc,simpleshot}.py`, `src/imageomics/naming_eval.py`; a read-only
clone sits in SUITE/_code and is never imported). The ToL model runs through its release folder
(`data/release_checks/release_tol`: trunk.onnx and the numpy head, verified against mini_trainer).

    python dev/116_bioclip2_suite.py index                          # image lists -> data/bioclip2_suite/index/<set>.parquet
    python dev/116_bioclip2_suite.py embed --model bioclip2|tol [--sets a,b] [--threads 4] [--limit N]
    python dev/116_bioclip2_suite.py fewshot                        # species tasks 1/5-shot, PlantDoc 1-shot
    python dev/116_bioclip2_suite.py zeroshot                       # BioCLIP-2 text; ToL head marginals; coverage
    python dev/116_bioclip2_suite.py newt | awa2                    # the non-species probes
    python dev/116_bioclip2_suite.py overlap                        # benchmark images among ToL's rows
    python dev/116_bioclip2_suite.py report                         # tables -> data/bioclip2_suite/results.md

Run under `taskset -c 0-3` (the workstation's agent slot); `--threads` sets torch, onnxruntime and sklearn alike.
Journal: journal/2026-11-20-can-one-frozen-trunk-with-specialisation-heads-stand-in-for-specialist-models.md (C2).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUITE = Path("/data/au761367/benchmarks/bioclip2_suite")
RARE = Path("/data/au761367/benchmarks/rare_species")
OUT = ROOT / "data/bioclip2_suite"
RELEASE = ROOT / "data/release_checks/release_tol"
TOL_E1 = ROOT / "data/population_probes/tol_e1"     # TL1's Rare Species embeddings (both models), reused
TOL_META = Path("/data/datasets/tol256/metadata.parquet")
SEED = 0                                            # BioCLIP-2's params.py default
META = {"plankton": "PLK", "insects": "INS", "insects2": "INS_2", "plantnet": "PLT_NET", "fungi": "FNG",
        "plantvillage": "PLT_VIL", "medicinal_leaf": "MED_LF"}
CAMERA = ["desert-lion", "ENA24", "island", "orinoquia", "ohio-small-animals"]   # slurm/eval.sh's order
SETS = [*META, "camera_trap", "plantdoc", "newt", "awa2"]                         # embedding order: species tasks first
SPECIES_TASKS = ["plankton", "insects", "insects2", "camera_trap", "plantnet", "fungi", "plantvillage",
                 "medicinal_leaf", "rare_species"]
# BioCLIP-2, arXiv 2505.23883 Table 1 (zero-shot, 1-shot, 5-shot) and Table 2
PAPER = {"nabirds": (74.9, 82.4, 92.4), "plankton": (3.9, 32.0, 50.5), "insects": (55.3, 74.6, 89.3),
         "insects2": (27.7, 28.4, 44.3), "camera_trap": (53.9, 48.1, 67.7), "plantnet": (96.8, 85.8, 94.4),
         "fungi": (83.8, 70.3, 85.0), "plantvillage": (25.1, 67.6, 83.9), "medicinal_leaf": (57.8, 92.0, 98.4),
         "rare_species": (76.8, 59.5, 77.2)}
PAPER_OTHER = {"fishnet": 39.8, "newt": 89.1, "awa2": 69.5, "herbarium19": 48.6, "plantdoc": 40.4}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ------------------------------------------------------------------------------------------------- index
def _taxon_com(row: dict) -> str:
    """naming_eval.Taxon(...).taxon_common_name, as BioCLIP-2's `taxon_com` text type builds a class."""
    f = {k: str(row.get(k, "") or "") for k in ("kingdom", "phylum", "cls", "order", "family", "genus",
                                                 "species", "subspecies", "common_name")}
    parts = [f["kingdom"].capitalize(), f["phylum"].capitalize(), f["cls"].capitalize(), f["order"].capitalize(),
             f["family"].capitalize(), f["genus"].capitalize(), f["species"].lower()]
    if f["subspecies"]:
        parts.append(f["subspecies"].lower())
    if f["kingdom"] == "":
        parts = [f["genus"].capitalize(), f["species"].lower()] + ([f["subspecies"].lower()] if f["subspecies"] else [])
    tax = " ".join(parts)
    return tax if f["common_name"] == "" else tax + " with common name " + f["common_name"]


def _with_idx(d: pd.DataFrame) -> pd.DataFrame:
    """DatasetFromFile's class_to_idx: classes in order of first appearance."""
    order = list(dict.fromkeys(d["cls"]))
    return d.assign(cls_idx=d["cls"].map({c: i for i, c in enumerate(order)}).astype(int))


def index(a) -> None:
    out = OUT / "index"
    out.mkdir(parents=True, exist_ok=True)
    for task, code in META.items():
        base = SUITE / "meta_album_mini" / f"{code}_Mini"
        d = pd.read_csv(base / "bioclip2_metadata.csv", index_col=0).fillna("")
        d = d.rename(columns={"class": "cls"})
        d["path"] = [str(base / f"{code}_Mini" / "images" / Path(p).name) for p in d["filepath"]]
        lab = pd.read_csv(base / f"{code}_Mini" / "labels.csv")
        lab_of = dict(zip(lab["FILE_NAME"], lab["CATEGORY"].map(nfc)))
        assert all(lab_of[Path(p).name] == nfc(c) for p, c in zip(d["path"], d["cls"])), task  # same image, same label
        assert all(Path(p).exists() for p in d["path"]), task
        _with_idx(d[["path", "cls"]]).to_parquet(out / f"{task}.parquet")
        print(f"{task}: {len(d):,} images, {d['cls'].nunique()} classes")
    rows = []
    for sub in CAMERA:
        d = pd.read_csv(SUITE / "camera_trap" / f"{sub}-balanced.csv", index_col=0).fillna("")
        text = [_taxon_com(r) for r in d.to_dict("records")]
        d = d.rename(columns={"cls": "tax_class"}).assign(cls=text)
        d["path"] = [str(SUITE / "camera_trap/data/test" / p) for p in d["filepath"]]
        assert all(Path(p).exists() for p in d["path"]), sub
        rows.append(_with_idx(d[["path", "cls", "kingdom", "phylum", "tax_class", "order", "family", "genus",
                                 "species", "common_name"]]).assign(subset=sub))
    d = pd.concat(rows, ignore_index=True)
    d.to_parquet(out / "camera_trap.parquet")
    print(f"camera_trap: {len(d):,} images, {d.groupby('subset')['cls'].nunique().to_dict()} classes")
    # PlantDoc: torchvision ImageFolder per split, classes in code-point order. Train has one class more
    # ("Tomato two spotted spider mites leaf"), which does NOT sort last: the two lowercase "grape" classes
    # follow it, so in BioCLIP-2's plantdoc.py their test indices are one lower than their train indices.
    # cls_idx: ImageFolder's index within the split (as BioCLIP-2 ran); cls_idx_by_name: train's index.
    rows = []
    for split in ("train", "test"):
        classes = sorted(e.name for e in os.scandir(SUITE / "plantdoc" / split) if e.is_dir())
        for i, c in enumerate(classes):
            for f in sorted(os.scandir(SUITE / "plantdoc" / split / c), key=lambda e: e.name):
                if f.is_file() and Path(f.name).suffix.lower() in {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm",
                                                                     ".tif", ".tiff", ".webp"}:
                    rows.append({"path": f.path, "cls": c, "cls_idx": i, "split": split})
    d = pd.DataFrame(rows)
    tr = d[d.split == "train"].groupby("cls")["cls_idx"].first()
    d["cls_idx_by_name"] = d["cls"].map(tr).astype(int)
    off = d[(d.split == "test") & (d.cls_idx != d.cls_idx_by_name)]
    d.to_parquet(out / "plantdoc.parquet")
    print(f"plantdoc: {d.split.value_counts().to_dict()} images, {tr.size} / {d[d.split == 'test'].cls.nunique()} "
          f"classes; test images whose ImageFolder index differs from train's: {len(off)} "
          f"({off.groupby('cls').size().to_dict()})")
    d = pd.read_csv(SUITE / "newt" / "newt2021_labels.csv")
    d["path"] = [str(SUITE / "newt/newt2021_images" / f"{i}.jpg") for i in d["id"]]
    d.to_parquet(out / "newt.parquet")
    print(f"newt: {len(d):,} images, {d.task.nunique()} tasks")
    base = SUITE / "awa2" / "Animals_with_Attributes2"
    classes = [line.split()[1] for line in (base / "classes.txt").read_text().splitlines() if line.strip()]
    pm = np.loadtxt(base / "predicate-matrix-binary.txt", dtype=int)
    split = {**{c.strip(): "train" for c in (base / "trainclasses.txt").read_text().split()},
             **{c.strip(): "test" for c in (base / "testclasses.txt").read_text().split()}}
    rows = []
    for c in classes:
        for name in sorted(os.listdir(base / "JPEGImages" / c)):
            rows.append({"path": str(base / "JPEGImages" / c / name), "cls": c, "cls_idx": classes.index(c), "split": split[c]})
    d = pd.DataFrame(rows)
    d.to_parquet(out / "awa2.parquet")
    np.save(out / "awa2_predicates.npy", pm)
    print(f"awa2: {len(d):,} images, split {d.split.value_counts().to_dict()}")


# ------------------------------------------------------------------------------------------------- embed
def _release_module():
    import sys
    if "tol_release_predict" in sys.modules:
        return sys.modules["tol_release_predict"]
    spec = importlib.util.spec_from_file_location("tol_release_predict", RELEASE / "predict.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m   # its dataclass looks its module up while the module executes
    spec.loader.exec_module(m)
    return m


def _encoder(model: str, threads: int):
    """A function paths -> {name: array}. CPU only, `threads` intra-op threads."""
    if model == "bioclip2":
        import open_clip
        import torch
        from PIL import Image
        torch.set_num_threads(threads)
        net, _, tf = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        net = net.eval()

        def load(paths):
            ims = []
            for p in paths:   # classification.py's img_loader; the transform converts any other mode to RGB
                im = Image.open(p)
                ims.append(tf(im.convert("RGB") if im.mode in ("RGBA", "P") else im))
            return torch.stack(ims)

        def run(x):
            with torch.inference_mode():
                return {"emb": net.encode_image(x).float().numpy()}
        return load, run
    import onnxruntime as ort
    rel = _release_module()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    sess = ort.InferenceSession(str(RELEASE / "trunk.onnx"), opts, providers=["CPUExecutionProvider"])

    def load(paths):
        return np.stack([rel.read_square(p, 256) for p in paths])

    def run(x):
        f, e = sess.run(["features", "embedding"], {"images": x})
        return {"feat": f.astype(np.float32), "emb": e.astype(np.float32)}
    return load, run


def embed(a) -> None:
    from concurrent.futures import ThreadPoolExecutor
    load, run = _encoder(a.model, a.threads)
    for name in (a.sets.split(",") if a.sets else SETS):
        dst = OUT / ("emb_smoke" if a.limit else "emb") / a.model / name   # a smoke test never counts as done
        if (dst / "emb.npy").exists() and not a.limit:
            print(f"{name}: done", flush=True)
            continue
        paths = pd.read_parquet(OUT / "index" / f"{name}.parquet")["path"].tolist()[: a.limit or None]
        parts = dst / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        t0, n0 = time.time(), 0
        shards = [paths[i:i + a.shard] for i in range(0, len(paths), a.shard)]
        with ThreadPoolExecutor(1) as pool:   # read the next batch while the model runs
            for s, shard in enumerate(shards):
                f = parts / f"{s:05d}.npz"
                if f.exists():
                    continue
                batches = [shard[i:i + a.batch] for i in range(0, len(shard), a.batch)]
                nxt, out = pool.submit(load, batches[0]), []
                for b in range(len(batches)):
                    x = nxt.result()
                    if b + 1 < len(batches):
                        nxt = pool.submit(load, batches[b + 1])
                    out.append(run(x))
                np.savez(f, **{k: np.concatenate([o[k] for o in out]) for k in out[0]})
                n0 += len(shard)
                print(f"{name}: {min((s + 1) * a.shard, len(paths)):,}/{len(paths):,} "
                      f"({n0 / (time.time() - t0):.2f} img/s)", flush=True)
        got = [np.load(parts / f"{s:05d}.npz") for s in range(len(shards))]
        for k in got[0].files:
            np.save(dst / f"{k}.npy", np.concatenate([g[k] for g in got]))
        (dst / "paths.json").write_text(json.dumps(paths))
        (dst / "meta.json").write_text(json.dumps({"model": a.model, "n": len(paths), "threads": a.threads,
                                                   "limit": a.limit}, indent=1))
        if not a.limit:
            for g in parts.iterdir():
                g.unlink()
            parts.rmdir()
        print(f"{name}: {len(paths):,} images -> {dst}", flush=True)


def _emb(model: str, name: str, layer: str = "emb") -> np.ndarray:
    d = OUT / "emb" / model / name
    paths = json.loads((d / "paths.json").read_text())
    ix = pd.read_parquet(OUT / "index" / f"{name}.parquet")
    assert paths == ix["path"].tolist(), f"{model} {name}: embedding rows are not the index's"
    return np.load(d / f"{layer}.npy").astype(np.float32)


# ----------------------------------------------------------------------------------------------- few-shot
def simpleshot_bioclip(feats: np.ndarray, target: np.ndarray, kshot: int, fold: int, keep=None) -> tuple[float, float]:
    """classification.py: split_few_shot_examples + get_few_shot_accuracy (features already normalised).
    random.seed(fold); classes in order of first appearance; shuffle each class's list; first k support, the
    rest query; no class skipped. Distances by the expanded square (same argmin as LA.norm, chunked).
    Shuffles row indices, which draws the same permutation as shuffling the feature list (same length, same state).
    Returns (accuracy, accuracy over the queries with keep True; nan without keep)."""
    select: dict = {}
    for i, t in enumerate(target):
        select.setdefault(int(t), []).append(i)
    random.seed(fold)
    train, test, label, tgt = [], [], [], []
    for c, ids in select.items():
        if len(ids) < kshot:
            raise ValueError(f"class {c} has {len(ids)} images, fewer than {kshot}")
        ids = ids[:]
        random.shuffle(ids)
        train.append(ids[:kshot]); test.extend(ids[kshot:])
        label.extend([c] * kshot); tgt.extend([c] * (len(ids) - kshot))
    tr, te, label, tgt = feats[np.concatenate(train)], feats[np.array(test)], np.array(label), np.array(tgt)
    m = tr.mean(0)
    tr = (tr - m) / np.linalg.norm(tr - m, 2, 1)[:, None]
    te = (te - m) / np.linalg.norm(te - m, 2, 1)[:, None]
    centre = tr.reshape(len(train), kshot, -1).mean(1)
    labels = label[::kshot]
    cn = (centre ** 2).sum(1)
    pred = np.concatenate([labels[(cn[None] - 2 * te[i:i + 2048] @ centre.T).argmin(1)] for i in range(0, len(te), 2048)])
    ok = pred == tgt
    return float(ok.mean()), (float(ok[keep[np.array(test)]].mean()) if keep is not None else float("nan"))


def _fewshot(feats, target, ks=(1, 5), folds=5, keep=None) -> dict:
    """{k: (mean, sd)}; with keep also {("clean", k): (mean, sd)} over the kept queries of the same draws."""
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    res = {}
    for k in ks:
        accs = np.array([simpleshot_bioclip(f, target, k, fold, keep) for fold in range(folds)])
        res[k] = (100 * float(accs[:, 0].mean()), 100 * float(accs[:, 0].std()))
        if keep is not None:
            res[("clean", k)] = (100 * float(accs[:, 1].mean()), 100 * float(accs[:, 1].std()))
    return res


def plantdoc_1shot(feats: np.ndarray, d: pd.DataFrame, repeats: int = 5, by_name: bool = False) -> tuple[float, float]:
    """plantdoc.py: normalised features; per repeat np.random.seed(SEED + r), one train image per class
    (choose_k_per_class), simpleshot.py's centring and sklearn NearestCentroid, accuracy on all of test.
    by_name=False keeps BioCLIP-2's per-split ImageFolder labels (two test classes off by one); True joins by name."""
    from sklearn.neighbors import NearestCentroid
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    trm = (d["split"] == "train").to_numpy()
    yte_col = "cls_idx_by_name" if by_name else "cls_idx"
    xtr, ytr, xte, yte = f[trm], d.loc[trm, "cls_idx"].to_numpy(), f[~trm], d.loc[~trm, yte_col].to_numpy()
    scores = []
    for r in range(repeats):
        random.seed(SEED + r); np.random.seed(SEED + r)
        idx = np.array([], dtype=int)
        for c in np.unique(ytr):
            ci = np.where(ytr == c)[0]
            np.random.shuffle(ci)
            idx = np.concatenate((idx, ci[:1]))
        np.random.shuffle(idx)
        x, y = xtr[idx], ytr[idx]
        mu = x.mean(0, keepdims=True)
        x = (x - mu) / np.linalg.norm(x - mu, axis=1, keepdims=True)
        q = (xte - mu) / np.linalg.norm(xte - mu, axis=1, keepdims=True)
        clf = NearestCentroid().fit(x, y)
        dist = np.linalg.norm(q[:, None] - clf.centroids_[None], axis=2)
        scores.append(float((dist.argmin(1) == yte).mean()))   # index == label: classes 0..C-1, as there
    return 100 * float(np.mean(scores)), 100 * float(np.std(scores))


def _rare(model: str, layer: str = "emb"):
    """TL1's Rare Species embeddings (dev/109 rare's mapping), BioCLIP-2's metadata order."""
    sub = {"tol": "emb_rare", "bioclip2": "emb_rare_bioclip2"}[model]
    d = TOL_E1 / sub
    if layer != "emb":
        return None, None
    e = np.load(d / "embeddings.npy").astype(np.float32)
    paths = json.loads((d / "paths.json").read_text())
    meta = pd.read_csv(RARE / "metadata.csv")
    side = pd.read_parquet(ROOT / "data/tol_eval/index_rare.parquet").set_index("rarespecies_id")
    row = {p: i for i, p in enumerate(paths)}
    feats = e[[row[p] for p in side.loc[meta["rarespecies_id"], "path"]]]
    cls = list(dict.fromkeys(meta["sciName"]))
    return feats, meta["sciName"].map({c: i for i, c in enumerate(cls)}).to_numpy()


def _flagged() -> set:
    """Benchmark images found among ToL's training rows (overlap): an exact id match on a trained row, or a
    difference hash within BENCH_HAMMING bits of a trained row in the same genus/family that `verify` confirms
    as a copy (thumbnail correlation >= VERIFY_CORR; the hash alone before `verify` ran). Empty before `overlap`."""
    d = OUT / "overlap"
    out = set()
    if (d / "verified.parquet").exists():
        v = pd.read_parquet(d / "verified.parquet")
        out |= set(v.loc[v["corr"] >= VERIFY_CORR, "path"])
    elif (d / "nearest.parquet").exists():
        n = pd.read_parquet(d / "nearest.parquet")
        out |= set(n.loc[n["dist_trained"] <= BENCH_HAMMING, "path"])
    for f in d.glob("exact_*.parquet"):
        e = pd.read_parquet(f)
        out |= set(e.loc[e["trained"], "path"])
    return out


def fewshot(a) -> None:
    rows = []
    flagged = _flagged()
    for model, layers in (("bioclip2", ["emb"]), ("tol", ["emb", "feat"])):
        for layer in layers:
            for task in SPECIES_TASKS:
                if task == "rare_species":
                    feats, target = _rare(model, layer)
                    if feats is None:
                        continue
                    res = _fewshot(feats, target)
                elif task == "camera_trap":
                    ix = pd.read_parquet(OUT / "index/camera_trap.parquet")
                    if not (OUT / "emb" / model / "camera_trap" / f"{layer}.npy").exists():
                        continue
                    e = _emb(model, "camera_trap", layer)
                    keep = ~ix["path"].isin(flagged).to_numpy()
                    per = {s: _fewshot(e[(ix.subset == s).to_numpy()], ix.loc[ix.subset == s, "cls_idx"].to_numpy(),
                                       keep=keep[(ix.subset == s).to_numpy()]) for s in CAMERA}
                    res = {k: (float(np.mean([per[s][k][0] for s in CAMERA])), float(np.mean([per[s][k][1] for s in CAMERA])))
                           for k in (1, 5, ("clean", 1), ("clean", 5))}
                    for s in CAMERA:
                        rows.append({"model": model, "layer": layer, "task": f"camera_trap/{s}",
                                     "1-shot": per[s][1][0], "5-shot": per[s][5][0], "1-shot sd": per[s][1][1], "5-shot sd": per[s][5][1]})
                else:
                    if not (OUT / "emb" / model / task / f"{layer}.npy").exists():
                        continue
                    ix = pd.read_parquet(OUT / "index" / f"{task}.parquet")
                    e = _emb(model, task, layer)
                    keep = ~ix["path"].isin(flagged).to_numpy()
                    res = _fewshot(e, ix["cls_idx"].to_numpy(), keep=keep)
                if task != "rare_species" and ("clean", 1) in res:
                    # clean queries: the same draws, scored only on queries not found among ToL's training rows (a
                    # flagged image can still be a support image; dropping it there would empty some 5-shot classes)
                    n_flag = int((~keep).sum())
                    rows.append({"model": model, "layer": layer, "task": f"{task}, clean queries", "images": int(keep.sum()),
                                 "flagged": n_flag, "1-shot": res[("clean", 1)][0], "5-shot": res[("clean", 5)][0],
                                 "1-shot sd": res[("clean", 1)][1], "5-shot sd": res[("clean", 5)][1]})
                rows.append({"model": model, "layer": layer, "task": task, "1-shot": res[1][0], "5-shot": res[5][0],
                             "1-shot sd": res[1][1], "5-shot sd": res[5][1],
                             "paper 1-shot": PAPER[task][1], "paper 5-shot": PAPER[task][2]})
                for r in rows[-2:] if rows[-2:-1] and rows[-2]["task"].endswith("clean queries") else rows[-1:]:
                    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            if (OUT / "emb" / model / "plantdoc" / f"{layer}.npy").exists():
                for by_name in (False, True):
                    m, s = plantdoc_1shot(_emb(model, "plantdoc", layer), pd.read_parquet(OUT / "index/plantdoc.parquet"),
                                          by_name=by_name)
                    rows.append({"model": model, "layer": layer,
                                 "task": "plantdoc, labels by name" if by_name else "plantdoc (as BioCLIP-2's code)",
                                 "1-shot": m, "1-shot sd": s, "paper 1-shot": PAPER_OTHER["plantdoc"]})
                    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "fewshot.csv", index=False)
    print("written", OUT / "fewshot.csv")


def report(a) -> None:
    """The BS tables in one markdown file, from the CSVs the other steps wrote."""
    lines = ["# BioCLIP-2's suite: stock BioCLIP-2 against frozen tol_full (dev/116)", ""]
    for name in ("fewshot", "zeroshot", "newt", "awa2", "overlap"):
        f = OUT / f"{name}.csv"
        if f.exists():
            d = pd.read_csv(f).round(2).fillna("")
            table = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
            table += ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
            lines += [f"## {name}", "", *table, ""]
    (OUT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------- zero-shot
def zeroshot_bioclip2(a) -> list[dict]:
    """classification.py's zero_shot_classifier: 80 OpenAI templates per class name (as built by the task's
    text type), mean of normalised text embeddings, renormalised; top-1 of image features @ classifier."""
    import open_clip
    import torch
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
    torch.set_num_threads(a.threads)
    net, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    net = net.eval()
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    assert len(OPENAI_IMAGENET_TEMPLATES) == 80

    def classifier(names):
        w = []
        with torch.inference_mode():
            for n in names:
                t = net.encode_text(tok([tpl(n) for tpl in OPENAI_IMAGENET_TEMPLATES]))
                t = torch.nn.functional.normalize(t, dim=-1).mean(0)
                w.append((t / t.norm()).numpy())
        return np.stack(w, 1)

    rows = []
    for task in [t for t in SPECIES_TASKS if t != "rare_species"]:
        ix = pd.read_parquet(OUT / "index" / f"{task}.parquet")
        if not (OUT / "emb/bioclip2" / task / "emb.npy").exists():
            continue
        e = _emb("bioclip2", task)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        accs = []
        for s in (CAMERA if task == "camera_trap" else [None]):
            m = (ix.subset == s).to_numpy() if s else np.ones(len(ix), bool)
            names = list(dict.fromkeys(ix.loc[m, "cls"]))
            acc = float(((e[m] @ classifier(names)).argmax(1) == ix.loc[m, "cls_idx"].to_numpy()).mean())
            accs.append(acc)
        rows.append({"model": "bioclip2 text", "task": task, "zero-shot": 100 * float(np.mean(accs)),
                     "paper": PAPER[task][0]})
        print(rows[-1], flush=True)
    return rows


# Common crop names of PlantVillage (and PlantDoc) to taxa: a written table, taxonomy only.
CROPS = {"Apple": "Malus domestica", "Blueberry": "Vaccinium corymbosum", "Cherry_(including_sour)": "Prunus",
         "Corn_(maize)": "Zea mays", "Grape": "Vitis vinifera", "Orange": "Citrus sinensis", "Peach": "Prunus persica",
         "Pepper,_bell": "Capsicum annuum", "Potato": "Solanum tuberosum", "Raspberry": "Rubus idaeus",
         "Soybean": "Glycine max", "Squash": "Cucurbita", "Strawberry": "Fragaria ananassa", "Tomato": "Solanum lycopersicum"}
KINGDOMS = {"plankton": None, "insects": "Animalia", "insects2": "Animalia", "plantnet": "Plantae", "fungi": "Fungi",
            "plantvillage": "Plantae", "medicinal_leaf": "Plantae", "camera_trap": "Animalia", "rare_species": None}


def _tol_names():
    """ToL head vocabulary: binomial / genus / family / order -> list of (rank, index, kingdom)."""
    lab = json.loads((RELEASE / "heads/tol.json").read_text())["labels"]
    by = {}
    for r, names in enumerate(lab[:4]):
        for i, path in enumerate(names):
            parts = path.split("|")
            key = parts[-1].replace("_", " ").lower()
            by.setdefault(key, []).append((r, i, parts[0]))
    return lab, by


def _class_names(task: str, ix: pd.DataFrame) -> dict:
    """Benchmark class -> candidate scientific names, finest first (as given; no synonym resolution)."""
    out = {}
    for c, g in ix.groupby("cls", sort=False):
        c0 = nfc(str(c))
        if task == "camera_trap" or task == "rare_species":
            r = g.iloc[0]
            gen, sp = str(r.get("genus", "") or ""), str(r.get("species", "") or "")
            out[c] = [f"{gen} {sp}".strip()] if gen and sp else []
            out[c] += [x for x in (gen, r.get("family", ""), r.get("order", "")) if x]
        elif task == "plantvillage":
            crop = c0.split("___")[0]
            name = CROPS.get(crop, "")
            out[c] = [name, name.split()[0]] if " " in name else [name]
        elif task == "insects":
            m = re.findall(r"\(([^()]*)\)", c0)
            out[c] = [m[-1].strip()] if m else [c0.strip()]
            if m and " " in m[-1].strip():
                out[c].append(m[-1].strip().split()[0])
        elif task == "medicinal_leaf":
            name = re.sub(r"\(.*?\)", "", c0).strip()
            out[c] = [name, name.split()[0]] if " " in name else [name]
        elif task == "fungi":
            w = c0.split()
            out[c] = [" ".join(w[:2]), w[0]]
        else:   # plankton, plantnet, insects2: underscores as spaces
            name = c0.replace("_", " ").strip()
            w = name.split()
            out[c] = [" ".join(w[:2]), w[0]] if len(w) >= 2 else [name]
    return out


def map_classes(task: str, ix: pd.DataFrame, by: dict) -> pd.DataFrame:
    """Each class's ToL node (rank, index) at the finest rank whose name matches; kingdom-checked."""
    king = KINGDOMS.get(task)
    rows = []
    for c, names in _class_names(task, ix).items():
        hit = None
        for n in names:
            cands = [x for x in by.get(str(n).lower().strip(), []) if king is None or x[2] == king]
            want = 0 if len(str(n).split()) == 2 else None
            cands = [x for x in cands if want is None or x[0] == 0] if want == 0 else [x for x in cands if x[0] > 0]
            if len(cands) >= 1:
                hit = sorted(cands)[0]   # homonyms within a kingdom: the first (rare; counted in `ambiguous`)
                rows.append({"cls": c, "name": n, "rank": hit[0], "node": hit[1], "ambiguous": len(cands) > 1})
                break
        if hit is None:
            rows.append({"cls": c, "name": names[0] if names else "", "rank": -1, "node": -1, "ambiguous": False})
    return pd.DataFrame(rows)


ZS_VARIANTS = ["marginal", "marginal, no class bias", "prototype angle"]


def _species_ancestors(head) -> list[np.ndarray]:
    """anc[r][s]: the rank-r node above species row s (anc[0] = identity)."""
    anc = [np.arange(len(head.direction))]
    for parent in head.parents:
        anc.append(parent[anc[-1]].astype(np.int64))
    return anc


def _node_scores(emb: np.ndarray, head, nodes: list[tuple[int, int]], anc, rel, batch: int = 256) -> dict:
    """Per ZS_VARIANTS, images x nodes. emb: the head's embedding (unit length; the release's `embedding`).
    'marginal': P(node) under the full 190,199-species softmax, logits as trained (magnitude g, bias b), coarse
    nodes the sum over their species (the head's logsumexp); 'marginal, no class bias': the same with b = 0;
    'prototype angle': cosine to the node's prototype (its species row, or the renormalised mean of its rows),
    the analogue of BioCLIP-2's text embedding."""
    members = [np.flatnonzero(anc[r] == j) for r, j in nodes]
    proto = np.stack([head.direction[m].mean(0) for m in members])
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    dim = head.direction.shape[1]
    out = {v: np.empty((len(emb), len(nodes))) for v in ZS_VARIANTS}
    for i in range(0, len(emb), batch):
        e = emb[i:i + batch].astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        cos = e @ head.direction.T
        za = (rel.zscore(cos, dim) * head.magnitude) if head.magnitude_after_angle else rel.zscore(cos * head.magnitude, dim)
        for v, z in (("marginal", za + head.bias), ("marginal, no class bias", za)):
            z = z.astype(np.float64)
            mx = z.max(1, keepdims=True)
            ez = np.exp(z - mx)
            tot = ez.sum(1)
            out[v][i:i + batch] = np.stack([ez[:, m].sum(1) for m in members], 1) / tot[:, None]
        out["prototype angle"][i:i + batch] = e @ proto.T
    return out


def zeroshot_tol(a, rel) -> tuple[list[dict], pd.DataFrame]:
    """The head's rows as class prototypes. Each benchmark class is mapped by name to a ToL node (species, else
    genus, family or order); scored by that node's marginal probability minus the mass of benchmark classes
    nested inside it (variants in ZS_VARIANTS); unmapped classes are never predicted; classes on one node tie
    (argmax keeps the first: in expectation over classes, 1/k each, the ceiling reported)."""
    head = rel.load_head(RELEASE / "heads/tol.safetensors")
    anc = _species_ancestors(head)
    lab, by = _tol_names()
    rows, maps = [], []
    for task in SPECIES_TASKS:
        if task == "rare_species":
            meta = pd.read_csv(RARE / "metadata.csv")
            ix = _with_idx(pd.DataFrame({"cls": meta["sciName"], "genus": meta["genus"], "species": meta["species"],
                                         "family": meta["family"], "order": meta["order"]}))
            emb, _ = _rare("tol")   # TL1's stored head embedding (mini_trainer's; the release's equals it)
        else:
            if not (OUT / "emb/tol" / task / "emb.npy").exists():
                continue
            ix = pd.read_parquet(OUT / "index" / f"{task}.parquet")
            emb = _emb("tol", task, "emb")
        acc = {v: [] for v in ZS_VARIANTS}
        by_rank = {v: {} for v in ZS_VARIANTS}
        ceil = []
        for s in (CAMERA if task == "camera_trap" else [None]):
            m = (ix.subset == s).to_numpy() if s else np.ones(len(ix), bool)
            sub = ix[m]
            mp = map_classes(task, sub, by).set_index("cls").loc[list(dict.fromkeys(sub["cls"]))]
            mp["task"], mp["subset"] = task, s
            maps.append(mp.reset_index())
            ok = mp["rank"].to_numpy() >= 0
            uniq = list(dict.fromkeys((int(r), int(j)) for r, j in mp.loc[ok, ["rank", "node"]].to_numpy()))
            col = {n: u for u, n in enumerate(uniq)}
            sc = _node_scores(emb[m], head, uniq, anc, rel)
            # nested benchmark taxa (marginals only): a coarser node keeps the mass not claimed by a finer class
            rep = [int(np.flatnonzero(anc[r] == j)[0]) for r, j in uniq]   # one species row below each node
            for u, (r, j) in enumerate(uniq):
                inner = [w for w, (r2, _) in enumerate(uniq) if r2 < r and anc[r][rep[w]] == j]
                for v in ZS_VARIANTS[:2]:
                    if inner:
                        sc[v][:, u] = np.maximum(sc[v][:, u] - sc[v][:, inner].sum(1), 0)
            y = sub["cls_idx"].to_numpy()
            y = pd.Series(y).map({c: i for i, c in enumerate(dict.fromkeys(y))}).to_numpy()
            rank_of_query = mp["rank"].to_numpy()[y]
            for v in ZS_VARIANTS:
                score = np.full((m.sum(), len(mp)), -np.inf)
                for k, (rk, nd) in enumerate(mp[["rank", "node"]].to_numpy()):
                    if rk >= 0:
                        score[:, k] = sc[v][:, col[(int(rk), int(nd))]]
                hit = score.argmax(1) == y
                acc[v].append(float(hit.mean()))
                for label, mask in (("species", rank_of_query == 0), ("coarser", rank_of_query > 0), ("unmapped", rank_of_query < 0)):
                    by_rank[v].setdefault(label, []).append((int(hit[mask].sum()), int(mask.sum())))
            node_of = [tuple(x) if x[0] >= 0 else None for x in mp[["rank", "node"]].to_numpy().tolist()]
            share = Counter(n for n in node_of if n is not None)
            reach = np.array([1.0 / share[n] if n is not None else 0.0 for n in node_of])
            ceil.append(float(reach[y].mean()))
        mpt = pd.concat([x for x in maps if x["task"].iloc[0] == task])
        row = {"model": "tol head", "task": task, "ceiling (mapped, untied)": 100 * float(np.mean(ceil)),
               "classes": len(mpt), "species": int((mpt["rank"] == 0).sum()), "genus": int((mpt["rank"] == 1).sum()),
               "family or order": int((mpt["rank"] >= 2).sum()), "unmapped": int((mpt["rank"] < 0).sum()),
               "paper (BioCLIP-2 text)": PAPER[task][0]}
        for v in ZS_VARIANTS:
            row[f"zero-shot, {v}"] = 100 * float(np.mean(acc[v]))   # Camera Trap: mean of the five subsets
            for label, lst in by_rank[v].items():
                n = sum(t for _, t in lst)
                row[f"{v}: {label} queries"] = 100 * sum(h for h, _ in lst) / n if n else float("nan")
        rows.append(row)
        print({k: (round(x, 1) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
    return rows, pd.concat(maps, ignore_index=True)


def zeroshot(a) -> None:
    rows = zeroshot_bioclip2(a)
    rows.append({"model": "bioclip2 text", "task": "rare_species", "zero-shot": 76.7, "paper": 76.8,
                 "note": "2026-11-07 entry's reproduction (Hugging Face column names), not rerun"})
    rel = _release_module()
    t, maps = zeroshot_tol(a, rel)
    pd.DataFrame(rows + t).to_csv(OUT / "zeroshot.csv", index=False)
    maps.to_csv(OUT / "zeroshot_class_map.csv", index=False)
    print("written", OUT / "zeroshot.csv")


# ---------------------------------------------------------------------------------------------- NeWT, AwA2
def newt(a) -> None:
    """newt.py: per task, centre on the train mean and L2-normalise; RandomizedSearchCV over a scaled SVC
    (100 draws, 5-fold); accuracy pooled over every task's test images; runs with seeds SEED + run."""
    import scipy.stats
    import sklearn.model_selection
    import sklearn.pipeline
    import sklearn.preprocessing
    import sklearn.svm
    ix = pd.read_parquet(OUT / "index/newt.parquet")
    flagged = _flagged()
    rows = []
    for model, layer in (("bioclip2", "emb"), ("tol", "emb"), ("tol", "feat")):
        if not (OUT / "emb" / model / "newt" / f"{layer}.npy").exists() or (a.models and model not in a.models.split(",")):
            continue
        e = _emb(model, "newt", layer)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        scores, by_cluster, extra = [], {}, {}
        for run in range(a.runs):
            seed = SEED + run
            random.seed(seed); np.random.seed(seed)
            yp, yt, cl, fl, un = [], [], [], [], []
            for task, g in ix.groupby("task", sort=False):
                tr = (g["split"] == "train").to_numpy()
                x = e[g.index.to_numpy()]
                mu = x[tr].mean(0, keepdims=True)
                x = (x - mu) / np.linalg.norm(x - mu, axis=1, keepdims=True)
                svc = sklearn.model_selection.RandomizedSearchCV(
                    sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC(C=1.0, kernel="rbf")),
                    {"svc__C": scipy.stats.loguniform(a=1e-3, b=1e1), "svc__kernel": ["rbf", "linear", "sigmoid", "poly"],
                     "svc__gamma": scipy.stats.loguniform(a=1e-4, b=1e-3)},
                    n_iter=100, n_jobs=a.threads, random_state=seed)
                y = g["label"].to_numpy()
                svc.fit(x[tr], y[tr])
                yp.append(svc.predict(x[~tr])); yt.append(y[~tr]); cl += [g["task_cluster"].iloc[0]] * int((~tr).sum())
                fl.append(g.loc[~tr, "path"].isin(flagged).to_numpy())
                un.append(np.full(int((~tr).sum()), task.startswith("inat_unobserved")))
            yp, yt, fl, un = np.concatenate(yp), np.concatenate(yt), np.concatenate(fl), np.concatenate(un)
            ok = yp == yt
            te_paths = np.concatenate([g.loc[g["split"] != "train", "path"].to_numpy() for _, g in ix.groupby("task", sort=False)])
            (OUT / "newt_pred").mkdir(exist_ok=True)   # per test image, so another clean split needs no refit
            pd.DataFrame({"path": te_paths, "correct": ok}).to_parquet(OUT / "newt_pred" / f"{model}_{layer}_run{run}.parquet")
            scores.append(100 * float(ok.mean()))
            for c, okc in pd.Series(ok).groupby(np.array(cl)):
                by_cluster.setdefault(c, []).append(100 * float(okc.mean()))
            # the overlap audit covers the inat_unobserved tasks only (their labels name a genus)
            for name, m in (("flagged test images out", ~fl), ("inat_unobserved tasks", un),
                            ("inat_unobserved, clean test images", un & ~fl), ("inat_unobserved, flagged test images", un & fl)):
                extra.setdefault(name, []).append(100 * float(ok[m].mean()))
                extra[f"{name}: n"] = [int(m.sum())]
            print(f"newt {model} {layer} run {run}: {scores[-1]:.2f}", flush=True)
        rows.append({"model": model, "layer": layer, "newt": float(np.mean(scores)), "sd": float(np.std(scores)),
                     **{f"cluster {c}": float(np.mean(v)) for c, v in by_cluster.items()},
                     **{k: float(np.mean(v)) for k, v in extra.items()}, "paper": PAPER_OTHER["newt"]})
    res = pd.DataFrame(rows)
    if (OUT / "newt.csv").exists():   # keep other models' rows from an earlier run
        old = pd.read_csv(OUT / "newt.csv")
        res = pd.concat([old[~old.set_index(["model", "layer"]).index.isin(res.set_index(["model", "layer"]).index)], res])
    res.to_csv(OUT / "newt.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string())


def awa2(a) -> None:
    """awa2.py: unnormalised features; Linear(d,512), Dropout(0.5), Linear(512,85); BCEWithLogits, Adam lr 1e-5,
    50 epochs, batch 256, StepLR(20, 0.5); train on trainclasses, score testclasses; predict logit > 0.5;
    macro-F1 over the attributes present in test; runs with seeds SEED + run."""
    import sklearn.metrics
    import torch
    torch.set_num_threads(a.threads)
    ix = pd.read_parquet(OUT / "index/awa2.parquet")
    pm = np.load(OUT / "index/awa2_predicates.npy")
    y_all = pm[ix["cls_idx"].to_numpy()].astype(np.float32)
    tr = (ix["split"] == "train").to_numpy()
    # The recipe (lr 1e-5, threshold on the logit at 0.5) is tuned to BioCLIP-2's unnormalised features, mean norm
    # about 48; ToL's embedding has norm 1 and its trunk features about 1.9, so with the code as is the probe barely
    # leaves its initialisation. Second variant: each model's features times one constant, to BioCLIP-2's mean norm.
    ref = OUT / "emb" / "bioclip2" / "awa2" / "emb.npy"
    ref_norm = float(np.linalg.norm(np.load(ref).astype(np.float32), axis=1).mean()) if ref.exists() else None
    rows = []
    variants = [(m, l, s) for m, l in (("bioclip2", "emb"), ("tol", "emb"), ("tol", "feat"))
                for s in ((False,) if m == "bioclip2" else (False, True))]
    for model, layer, scaled in variants:
        if not (OUT / "emb" / model / "awa2" / f"{layer}.npy").exists() or (a.models and model not in a.models.split(",")):
            continue
        if scaled and ref_norm is None:
            continue
        xn = _emb(model, "awa2", layer)
        factor = ref_norm / float(np.linalg.norm(xn, axis=1).mean()) if scaled else 1.0
        x = torch.from_numpy(xn * factor)
        y = torch.from_numpy(y_all)
        scores = []
        for run in range(a.runs):
            seed = SEED + run
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            clf = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 512), torch.nn.Dropout(0.5), torch.nn.Linear(512, 85))
            opt = torch.optim.Adam(clf.parameters(), lr=1e-5)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
            ds = torch.utils.data.TensorDataset(x[tr], y[tr])
            for _ in range(50):
                clf.train()
                for xb, yb in torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True):
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
            clf.eval()
            with torch.no_grad():
                pred = (clf(x[~tr]) > 0.5).numpy()
            yt = y_all[~tr]
            scores.append(100 * float(sklearn.metrics.f1_score(yt, pred, average="macro", labels=np.unique(yt))))
            print(f"awa2 {model} {layer} x{factor:.1f} run {run}: {scores[-1]:.2f}", flush=True)
        rows.append({"model": model, "layer": layer, "features": f"x {factor:.1f} (mean norm {ref_norm:.1f})" if scaled else "as is",
                     "awa2": float(np.mean(scores)), "sd": float(np.std(scores)), "paper": PAPER_OTHER["awa2"]})
    res = pd.DataFrame(rows)
    if (OUT / "awa2.csv").exists():   # keep other models' rows from an earlier run
        old = pd.read_csv(OUT / "awa2.csv")
        if "features" not in old:
            old["features"] = "as is"
        key = ["model", "layer", "features"]
        res = pd.concat([old[~old.set_index(key).index.isin(res.set_index(key).index)], res])
    res.to_csv(OUT / "awa2.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string())


# ------------------------------------------------------------------------------------------------ overlap
# Which benchmark images are among ToL's rows (tol256-v1 trained on split == "train" with no `exclude`)?
# Exact ids where the benchmark keeps them (Danish Fungi file names start with the GBIF occurrence id; Pl@ntNet-300K
# names are Pl@ntNet image hashes, which appear in Pl@ntNet's GBIF media URLs; SPIPOLL ids), then a difference hash
# within genus (dev/103's rule for Rare Species: same genus, at most 6 bits of 64). Written genus tables (taxonomy
# only) for common-name classes; a class without one is not audited (counted).
BENCH_HAMMING = 6
AWA2_GENERA = {
    "antelope": ["Antilope", "Tragelaphus", "Aepyceros", "Antidorcas", "Gazella", "Eudorcas", "Oryx", "Kobus",
                 "Hippotragus", "Connochaetes", "Redunca", "Alcelaphus", "Damaliscus", "Antilocapra"],
    "grizzly+bear": ["Ursus"], "killer+whale": ["Orcinus"], "beaver": ["Castor"], "dalmatian": ["Canis"],
    "persian+cat": ["Felis"], "horse": ["Equus"], "german+shepherd": ["Canis"], "blue+whale": ["Balaenoptera"],
    "siamese+cat": ["Felis"], "skunk": ["Mephitis", "Spilogale", "Conepatus"], "mole": ["Talpa", "Scalopus", "Condylura", "Scapanus"],
    "tiger": ["Panthera"], "hippopotamus": ["Hippopotamus"], "leopard": ["Panthera"], "moose": ["Alces"],
    "spider+monkey": ["Ateles"], "humpback+whale": ["Megaptera"], "elephant": ["Loxodonta", "Elephas"], "gorilla": ["Gorilla"],
    "ox": ["Bos"], "fox": ["Vulpes", "Urocyon"], "sheep": ["Ovis"],
    "seal": ["Phoca", "Halichoerus", "Mirounga", "Pusa", "Leptonychotes", "Arctocephalus", "Zalophus"],
    "chimpanzee": ["Pan"], "hamster": ["Mesocricetus", "Phodopus", "Cricetus"], "squirrel": ["Sciurus", "Tamiasciurus"],
    "rhinoceros": ["Ceratotherium", "Diceros", "Rhinoceros"], "rabbit": ["Oryctolagus", "Sylvilagus"], "bat": [],
    "giraffe": ["Giraffa"], "wolf": ["Canis"], "chihuahua": ["Canis"], "rat": ["Rattus"], "weasel": ["Mustela"],
    "otter": ["Lutra", "Lontra", "Enhydra"], "buffalo": ["Bubalus", "Syncerus", "Bison"], "zebra": ["Equus"],
    "giant+panda": ["Ailuropoda"], "deer": ["Odocoileus", "Cervus", "Capreolus", "Dama"], "bobcat": ["Lynx"],
    "pig": ["Sus"], "lion": ["Panthera"], "mouse": ["Mus", "Peromyscus", "Apodemus"], "polar+bear": ["Ursus"],
    "collie": ["Canis"], "walrus": ["Odobenus"], "raccoon": ["Procyon"], "cow": ["Bos"],
    "dolphin": ["Tursiops", "Delphinus", "Stenella", "Lagenorhynchus"]}
PLANTDOC_GENERA = {"Apple": "Malus", "Bell_pepper": "Capsicum", "Blueberry": "Vaccinium", "Cherry": "Prunus", "Corn": "Zea",
                   "Peach": "Prunus", "Potato": "Solanum", "Raspberry": "Rubus", "Soyabean": "Glycine", "Squash": "Cucurbita",
                   "Strawberry": "Fragaria", "Tomato": "Solanum", "grape": "Vitis"}


def dhash(path) -> int:
    """dev/103's difference hash: grey, LANCZOS to 9x8, horizontal gradient signs, 64 bits."""
    from PIL import Image
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return int(np.packbits((g[:, 1:] > g[:, :-1]).ravel()).view(">u8")[0])


def _bench_records() -> pd.DataFrame:
    """One row per benchmark image and taxon group it is audited in: set, path, kingdom (or ''), level
    ('genus' | 'family'), name; plus exact ids where the benchmark has them."""
    _, by = _tol_names()
    is_genus = {k for k, v in by.items() if any(r == 1 for r, _, _ in v)}
    is_family = {k for k, v in by.items() if any(r == 2 for r, _, _ in v)}
    rows = []

    def add(name, path, kingdom, level, taxon, **ids):
        rows.append({"set": name, "path": path, "kingdom": kingdom or "", "level": level, "taxon": taxon.lower(), **ids})

    for task in [*META, "camera_trap"]:
        ix = pd.read_parquet(OUT / "index" / f"{task}.parquet")
        names = _class_names(task, ix)
        for c, g in ix.groupby("cls", sort=False):
            cand = [str(n).strip() for n in names[c] if n]
            gen = next((n.split()[0] for n in cand if n.split() and n.split()[0].lower() in is_genus), None)
            fam = None if gen else next((n for n in cand if " " not in n and n.lower() in is_family), None)
            for p in g["path"]:
                ids = {}
                if task == "fungi":
                    ids["gbif_id"] = Path(p).name.split("-")[0]
                if task == "plantnet":
                    ids["plantnet_hash"] = Path(p).stem.lower()
                if task == "insects":
                    ids["spipoll_id"] = Path(p).stem
                if gen:
                    add(task, p, KINGDOMS.get(task), "genus", gen, **ids)
                elif fam:
                    add(task, p, KINGDOMS.get(task), "family", fam, **ids)
                else:
                    add(task, p, KINGDOMS.get(task), "none", "", **ids)
    d = pd.read_parquet(OUT / "index/plantdoc.parquet")
    for p, c in zip(d["path"], d["cls"]):
        add("plantdoc", p, "Plantae", "genus", PLANTDOC_GENERA[c.split()[0]])
    d = pd.read_parquet(OUT / "index/awa2.parquet")
    for p, c in zip(d["path"], d["cls"]):
        for gname in AWA2_GENERA[c] or [None]:
            add("awa2", p, "Animalia", "genus" if gname else "none", gname or "")
    d = pd.read_parquet(OUT / "index/newt.parquet")
    d = d[d["task"].str.startswith("inat_unobserved")]
    for p, t in zip(d["path"], d["text_label"]):
        add("newt (inat_unobserved tasks)", p, "", "genus", t.split()[0])
    return pd.DataFrame(rows)


def overlap(a) -> None:
    """Exact ids during one streaming pass over ToL's rows; difference hashes of the benchmark images and of the
    trained ToL rows in the audited taxa (streamed to disk in chunks, resumable); nearest distance within taxon."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from multiprocessing import get_context
    out = OUT / "overlap"
    out.mkdir(parents=True, exist_ok=True)
    bench = _bench_records()
    bench.to_parquet(out / "bench_records.parquet")
    print(bench.groupby(["set", "level"]).size().unstack(fill_value=0).to_string(), flush=True)
    cand_file, exact_file = out / "tol_candidates.parquet", out / "tol_exact.parquet"
    if not cand_file.exists():   # one streaming pass over ToL's 65.7 M rows; candidates written as they come
        g_any = set(bench.loc[(bench.level == "genus") & (bench.kingdom == ""), "taxon"])
        g_k = {f"{k}|{g}" for k, g in zip(bench.loc[(bench.level == "genus") & (bench.kingdom != ""), "kingdom"],
                                          bench.loc[(bench.level == "genus") & (bench.kingdom != ""), "taxon"])}
        f_k = {f"{k}|{f}" for k, f in zip(bench.loc[bench.level == "family", "kingdom"], bench.loc[bench.level == "family", "taxon"])}
        gbif = set(bench["gbif_id"].dropna())
        pnet = set(bench["plantnet_hash"].dropna())
        spip = set(bench["spipoll_id"].dropna())
        pf = pq.ParquetFile(TOL_META)
        writer, exact, n, t0 = None, [], 0, time.time()
        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i, columns=["uuid", "path", "kingdom", "family", "genus", "split", "exclude", "gbif_id", "url"])
            trained = pc.and_(pc.equal(pc.fill_null(t["split"], ""), "train"), pc.is_null(t["exclude"]))
            gl = pc.utf8_lower(pc.fill_null(t["genus"], ""))
            fl = pc.utf8_lower(pc.fill_null(t["family"], ""))
            kg = pc.fill_null(t["kingdom"], "")
            m = pc.is_in(gl, value_set=pa_set(g_any))
            m = pc.or_(m, pc.is_in(pc.binary_join_element_wise(kg, gl, "|"), value_set=pa_set(g_k)))
            m = pc.or_(m, pc.is_in(pc.binary_join_element_wise(kg, fl, "|"), value_set=pa_set(f_k)))
            url = pc.fill_null(t["url"], "")
            hexes = pc.struct_field(pc.extract_regex(url, r"(?P<h>[0-9a-f]{40})"), [0])
            digits = pc.struct_field(pc.extract_regex(url, r"(?P<d>[0-9]{13,})"), [0])
            ids = pc.or_(pc.is_in(pc.fill_null(t["gbif_id"], ""), value_set=pa_set(gbif)),
                         pc.or_(pc.fill_null(pc.is_in(hexes, value_set=pa_set(pnet)), False),
                                pc.fill_null(pc.is_in(digits, value_set=pa_set(spip)), False)))
            if pc.any(ids).as_py():
                e = t.append_column("trained", trained).append_column("url_hex", hexes).append_column("url_digits", digits)
                exact.append(e.filter(ids).drop_columns(["url"]))
            keep = t.append_column("trained", trained).filter(pc.and_(m, trained)).select(
                ["uuid", "path", "kingdom", "family", "genus"])
            if writer is None:
                writer = pq.ParquetWriter(cand_file.with_suffix(".tmp"), keep.schema)
            writer.write_table(keep)
            n += keep.num_rows
            if i % 100 == 0:
                print(f"  row group {i}/{pf.metadata.num_row_groups}: {n:,} trained candidates, {time.time() - t0:.0f} s", flush=True)
        writer.close()
        (pa.concat_tables(exact).to_pandas() if exact else pd.DataFrame(
            columns=["uuid", "trained", "gbif_id", "url_hex", "url_digits"])).to_parquet(exact_file)
        cand_file.with_suffix(".tmp").rename(cand_file)
    ex = pd.read_parquet(exact_file)
    rows = []
    for col, tcol in (("gbif_id", "gbif_id"), ("plantnet_hash", "url_hex"), ("spipoll_id", "url_digits")):
        b = bench.dropna(subset=[col]).drop_duplicates("path")
        hit = b[["path", col]].merge(ex[[tcol, "trained", "uuid"]].dropna(subset=[tcol]), left_on=col, right_on=tcol)
        rows.append({"id": col, "benchmark images with that id": int(b["path"].nunique()),
                     "matched in ToL (any split)": int(hit["path"].nunique()),
                     "matched in ToL training rows": int(hit.loc[hit.trained, "path"].nunique())})
        hit[["path", "uuid", "trained"]].to_parquet(out / f"exact_{col}.parquet")
    pd.DataFrame(rows).to_csv(OUT / "overlap_exact.csv", index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)
    if a.scan_only:
        return
    bh_file = out / "bench_hash.parquet"
    with get_context("spawn").Pool(a.threads) as pool:
        if not bh_file.exists():
            paths = bench["path"].drop_duplicates().tolist()
            pd.DataFrame({"path": paths, "h": pool.map(_safe_hash, paths, chunksize=64)}).to_parquet(bh_file)
            print(f"benchmark hashed: {len(paths):,}", flush=True)
        hd = out / "tol_hash"
        hd.mkdir(exist_ok=True)
        pf = pq.ParquetFile(cand_file)
        t0, done = time.time(), 0
        for i in range(pf.metadata.num_row_groups):   # one hash file per candidate row group: resumable
            f = hd / f"{i:05d}.parquet"
            if f.exists():
                continue
            t = pf.read_row_group(i, columns=["kingdom", "family", "genus", "path"]).to_pandas()
            files = [str(TOL_META.parent / "images" / p) for p in t["path"]]
            t["h"] = pool.map(_safe_hash, files, chunksize=256)
            t.drop(columns="path").to_parquet(f)
            done += len(t)
            print(f"  ToL hashed: group {i + 1}/{pf.metadata.num_row_groups}, {done / (time.time() - t0):.0f}/s", flush=True)
    bh = bench.merge(pd.read_parquet(bh_file), on="path").dropna(subset=["h"])
    th = pd.concat([pd.read_parquet(f) for f in sorted((out / "tol_hash").glob("*.parquet"))], ignore_index=True)
    th = th.dropna(subset=["h"])
    th["genus_l"] = th["genus"].fillna("").str.lower().astype("category")
    th["family_l"] = th["family"].fillna("").str.lower().astype("category")
    th["kingdom"] = th["kingdom"].fillna("").astype("category")
    th["hu"] = _u64(th["h"])
    th = th[["kingdom", "genus_l", "family_l", "hu"]]
    best = []
    for (lvl, king, tax), g in bh[bh.level != "none"].groupby(["level", "kingdom", "taxon"]):
        col = "genus_l" if lvl == "genus" else "family_l"
        x = th[(th[col] == tax) & ((th["kingdom"] == king) if king else True)]
        gh = _u64(g["h"])
        d = np.full(len(gh), 64)
        if len(x):
            xh = x["hu"].to_numpy()
            for i in range(0, len(gh), 64):   # bounded memory: 64 benchmark images x the taxon's trained rows
                d[i:i + 64] = np.bitwise_count(gh[i:i + 64, None] ^ xh[None, :]).min(1)
        best.append(pd.DataFrame({"path": g["path"].to_numpy(), "dist_trained": d, "tol_rows": len(x)}))
    best = pd.concat(best).groupby("path").agg(dist_trained=("dist_trained", "min"), tol_rows=("tol_rows", "sum")).reset_index()
    best.to_parquet(out / "nearest.parquet")
    res = []
    for s_, g in bench.drop_duplicates(["set", "path"]).groupby("set", sort=False):
        audited = g[g.level != "none"]["path"].drop_duplicates()
        nb = best[best.path.isin(audited)]
        res.append({"set": s_, "images": int(g["path"].nunique()), "audited (taxon known)": int(audited.size),
                    "trained ToL rows compared": int(nb["tol_rows"].sum()),
                    f"within {BENCH_HAMMING} bits of a trained row": int((nb["dist_trained"] <= BENCH_HAMMING).sum()),
                    "within 3 bits": int((nb["dist_trained"] <= 3).sum()),
                    "equal hash": int((nb["dist_trained"] == 0).sum())})
    res = pd.DataFrame(res)
    res.to_csv(OUT / "overlap.csv", index=False)
    print(res.to_string(), flush=True)


VERIFY_CORR = 0.95   # by eye on the borderline pairs (C2 journal, 2026-11-21): copies >= 0.96, other frames <= 0.94


def _thumb(path, side: int = 32):
    """Whole image, grey, LANCZOS to side x side, zero mean and unit norm (Pearson correlation by dot product)."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((side, side), Image.Resampling.LANCZOS), dtype=np.float32).ravel()
    except Exception:  # noqa: BLE001 - unreadable file
        return None
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 0 else None


def _pair_corr(pair):
    a, b = _thumb(pair[0]), _thumb(pair[1])
    return float(a @ b) if a is not None and b is not None else float("nan")


def verify(a) -> None:
    """The hash rule (dev/103's, set on Rare Species photographs) passes different camera-trap night frames and
    plain-background AwA2 images at 6 bits. For each flagged benchmark image: every trained ToL row within
    BENCH_HAMMING bits in its audited taxon (nearest 20), and the best Pearson correlation of 32 x 32 grey
    thumbnails. A copy, resized or recompressed, correlates near 1; a different frame does not."""
    import pyarrow.parquet as pq
    from multiprocessing import get_context
    out = OUT / "overlap"
    bench = pd.read_parquet(out / "bench_records.parquet").merge(pd.read_parquet(out / "bench_hash.parquet"), on="path")
    near = pd.read_parquet(out / "nearest.parquet")
    flagged = set(near.loc[near["dist_trained"] <= BENCH_HAMMING, "path"])
    fb = bench[bench["path"].isin(flagged) & (bench["level"] != "none")].dropna(subset=["h"]).copy()
    fb["hu"] = _u64(fb["h"])
    groups = {k: g for k, g in fb.groupby(["level", "kingdom", "taxon"])}
    pf = pq.ParquetFile(out / "tol_candidates.parquet")
    pairs = []
    for i in range(pf.metadata.num_row_groups):
        th = pd.read_parquet(out / "tol_hash" / f"{i:05d}.parquet")
        th["path"] = pf.read_row_group(i, columns=["path"]).column(0).to_numpy(zero_copy_only=False)
        th = th.dropna(subset=["h"])
        gl, fl, kg = th["genus"].fillna("").str.lower(), th["family"].fillna("").str.lower(), th["kingdom"].fillna("")
        for (lvl, king, tax), g in groups.items():
            m = ((gl if lvl == "genus" else fl) == tax) & ((kg == king) if king else True)
            if not m.any():
                continue
            xh = _u64(th.loc[m, "h"])
            xp = th.loc[m, "path"].to_numpy()
            d = np.bitwise_count(g["hu"].to_numpy()[:, None] ^ xh[None, :])
            for bi, ti in zip(*np.nonzero(d <= BENCH_HAMMING)):
                pairs.append((g["path"].iloc[bi], xp[ti], int(d[bi, ti])))
        if i % 100 == 0:
            print(f"  hash group {i}/{pf.metadata.num_row_groups}: {len(pairs):,} pairs within {BENCH_HAMMING} bits", flush=True)
    p = pd.DataFrame(pairs, columns=["path", "tol_path", "dist"]).drop_duplicates(["path", "tol_path"])
    p = p.sort_values(["path", "dist"]).groupby("path").head(20).reset_index(drop=True)
    img = TOL_META.parent / "images"
    with get_context("spawn").Pool(a.threads) as pool:
        p["corr"] = pool.map(_pair_corr, [(b, str(img / t)) for b, t in zip(p["path"], p["tol_path"])], chunksize=64)
    p.to_parquet(out / "verify_pairs.parquet")
    best = p.sort_values("corr", ascending=False).drop_duplicates("path")
    best = best.merge(bench[["set", "path"]].drop_duplicates("path"), on="path")
    best.to_parquet(out / "verified.parquet")
    edges = [-1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.01]
    tab = best.groupby(["set", pd.cut(best["corr"], edges)], observed=False).size().unstack(fill_value=0)
    print(tab.to_string(), flush=True)
    res = best.groupby("set").agg(flagged=("path", "size"), verified=("corr", lambda c: int((c >= VERIFY_CORR).sum())))
    res.to_csv(OUT / "overlap_verified.csv")
    print(res.to_string())


def clean(a) -> None:
    """After `verify`: the verified counts per set at VERIFY_CORR, and NeWT's clean split recomputed from the saved
    per-image predictions (no refit)."""
    v = pd.read_parquet(OUT / "overlap" / "verified.parquet")
    res = v.groupby("set").agg(hash_flagged=("path", "size"), copies=("corr", lambda c: int((c >= VERIFY_CORR).sum())))
    res.to_csv(OUT / "overlap_verified.csv")
    print(res.to_string(), flush=True)
    flagged = _flagged()
    ix = pd.read_parquet(OUT / "index/newt.parquet")
    unobs = set(ix.loc[ix["task"].str.startswith("inat_unobserved"), "path"])
    nt = pd.read_csv(OUT / "newt.csv")
    for i, r in nt.iterrows():
        runs = sorted((OUT / "newt_pred").glob(f"{r['model']}_{r['layer']}_run*.parquet"))
        if not runs:
            continue
        extra = {}
        for f in runs:
            p = pd.read_parquet(f)
            fl, un = p["path"].isin(flagged).to_numpy(), p["path"].isin(unobs).to_numpy()
            for name, m in (("flagged test images out", ~fl), ("inat_unobserved tasks", un),
                            ("inat_unobserved, clean test images", un & ~fl), ("inat_unobserved, flagged test images", un & fl)):
                extra.setdefault(name, []).append(100 * float(p["correct"].to_numpy()[m].mean()))
                extra[f"{name}: n"] = [int(m.sum())]
        for k, vals in extra.items():
            nt.loc[i, k] = float(np.mean(vals))
    nt.to_csv(OUT / "newt.csv", index=False)
    print(nt.round(2).to_string())


def pa_set(values):
    import pyarrow as pa
    return pa.array(sorted(values), type=pa.string())


def _safe_hash(p):
    """The hash as 16 hex digits (64-bit values do not fit pandas' int64), None if unreadable."""
    try:
        return f"{dhash(p):016x}"
    except Exception:  # noqa: BLE001 - unreadable file: no hash
        return None


def _u64(hexes) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("index", index), ("embed", embed), ("fewshot", fewshot), ("zeroshot", zeroshot),
                     ("newt", newt), ("awa2", awa2), ("overlap", overlap), ("verify", verify), ("clean", clean),
                     ("report", report)):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        sp.add_argument("--threads", type=int, default=4)
        if name == "embed":
            sp.add_argument("--model", choices=["bioclip2", "tol"], required=True)
            sp.add_argument("--sets", default="")
            sp.add_argument("--batch", type=int, default=16)
            sp.add_argument("--shard", type=int, default=1024)
            sp.add_argument("--limit", type=int, default=0, help="first N images only (smoke test; kept apart)")
        if name in ("newt", "awa2"):
            sp.add_argument("--runs", type=int, default=3)
            sp.add_argument("--models", default="", help="bioclip2,tol (default: every model embedded)")
        if name == "overlap":
            sp.add_argument("--scan-only", action="store_true", help="the ToL metadata pass and exact ids; no hashing")
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
