"""Evaluation of a ToL model (tol256-v1, seven ranks keyed by full path) from a mini_trainer checkpoint.

CPU steps run with lepinet's venv; `embed` needs mini_trainer on the path (branch feature/lepinet-integration)
and a GPU:

    python dev/109_tol_eval.py map                      # ToL species -> lepinet's legacy GBIF keys: data/tol_eval/map_*.csv
    python dev/109_tol_eval.py index --set S            # S: testfold | trap | heldout | rare -> data/tol_eval/index_S.json
    python dev/109_tol_eval.py classlist [--keys F]     # lepinet keys (e.g. north_europe.classes) -> ToL classes, for --class_list
    python dev/109_tol_eval.py relabel --pred CSV       # a ToL model's mini_metric.csv -> lepinet's three ranks and keys
    PYTHONPATH=$MT python dev/109_tol_eval.py embed --weights last.pt --set S [--prototypes]   # GPU; loader workers min(4, CPUs) unless --num_workers
    python dev/109_tol_eval.py placement                # held-out taxa: right genus / family / order, 1- and 5-shot
    python dev/109_tol_eval.py rare                     # Rare Species 1- and 5-shot, BioCLIP-2's protocol
    python dev/109_tol_eval.py congeners                # Z7: is the nearest species prototype a congener?
    python dev/109_tol_eval.py selftest                 # the scoring code on synthetic data

Test fold and trap crops go through mini_trainer's own prediction (`mt_hpredict`) with `--data_index` from `index`
and `--class_list` from `classlist` (the model restricted to ToL species that have a lepinet key), then
`relabel` turns its output into the three-rank, GBIF-keyed CSV that `dev/lepinet/score.py` and
`dev/lepinet/flemming.py score` read. Lepidoptera in ToL carry Catalogue of Life XR names (dev/102 colxr), lepinet
legacy GBIF Backbone keys; the bridge is dev/101's frozen `dev/colxr/name_map.csv`. Plan row W3.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/tol_eval"
TOL = Path("/data/datasets/tol256")
SPEC = TOL / "train/class_spec.json"
LEPI = ROOT / "data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"
LEPI_IMAGES = ROOT / "data/global/images"
TRAP = Path("/data/au761367/lepinet/ucloud_mirror/data/flemming/flemming_names_labels.parquet")
TRAP_IMAGES = ROOT / "data/flemming/images"
NE_LIST = Path.home() / "codes/mini_trainer/dev/releases/mambo_v3/presets/north_europe.classes"
RARE = Path("/data/au761367/benchmarks/rare_species")
FLOOR = 50                       # lepinet's training floor: 12,041 species, 629,742 test images
HELDOUT_CAP = 50                 # images per held-out species embedded
DEPTH_RANK = {"near": 1, "mid": 2, "far": 3}   # held out a species / a genus / a family: placed at genus / family / order
RANKS = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]
BASE = OUT / "baselines"         # frozen baselines' embeddings (independent of any ToL checkpoint)
SCORES = ["max softmax", "max logit", "energy (logsumexp)"]   # species head, higher = known


def _emb_dir(a, name: str) -> Path:
    """Where an embed of set `name` lives: --emb if given, else <root>/emb_<set> for the ToL model and
    data/tol_eval/baselines/emb_<set>_<model> for a frozen baseline."""
    if getattr(a, "emb", ""):
        return Path(a.emb)
    model = getattr(a, "model", "tol")
    return Path(a.root) / f"emb_{name}" if model == "tol" else BASE / f"emb_{name}_{model}"


def _f103():
    spec = importlib.util.spec_from_file_location("f103", ROOT / "dev/103_tol_freeze.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["f103"] = m
    spec.loader.exec_module(m)
    return m


def canon(name) -> str:
    """'Ctenoplusia (Ctenoplusia) accentifera (Lefèbvre, 1827)' -> 'Ctenoplusia accentifera'."""
    s = re.sub(r"\s*\([A-Z][a-z]+\)", "", str(name))
    w = s.split()
    return " ".join(w[:2]) if len(w) >= 2 and w[1][:1].islower() else ""


# ------------------------------------------------------------------------------------------------- map
def lepi_species() -> pd.DataFrame:
    t = pq.read_table(LEPI, columns=["speciesKey", "genusKey", "familyKey", "set"]).to_pandas()
    t["test"] = t["set"] == "0"
    g = t.groupby("speciesKey").agg(genusKey=("genusKey", "first"), familyKey=("familyKey", "first"),
                                     n_test=("test", "sum"), n=("test", "size"))
    g["n_train"] = g["n"] - g["n_test"]
    g["vocab"] = g["n_train"] >= FLOOR
    return g.drop(columns="n")


def map_(a):
    spec = json.load(open(SPEC))
    idx = spec["cls2idx"]["0"]
    tol: dict[str, list[str]] = {}
    for k in idx:
        p = k.split("|")
        if p[3] == "Lepidoptera":
            tol.setdefault(p[6].replace("_", " "), []).append(k)
    nm = pd.read_csv(ROOT / "dev/colxr/name_map.csv", dtype=str).drop_duplicates("legacy_species_key")
    tax = pd.read_csv(ROOT / "dev/colxr/taxonomy.csv", dtype=str).set_index("species_key")
    sp = lepi_species()
    nm = nm[nm["legacy_species_key"].isin(sp.index)]
    rows = []
    for r in nm.itertuples():
        got, how = None, ""
        for how_, name in (("colxr", canon(r.new_species_name)), ("legacy", canon(r.training_name))):
            cands = tol.get(name, [])
            if len(cands) > 1:   # a homonym inside Lepidoptera: keep the one in the COL XR family
                fam = tax["family_name"].get(r.new_species_key, "")
                cands = [c for c in cands if c.split("|")[4] == fam]
            if len(cands) == 1:
                got, how = cands[0], how_
                break
        rows.append({"speciesKey": r.legacy_species_key, "tol_class": got, "tol_index": idx[got] if got else -1, "how": how or "unmapped",
                     "colxr_name": canon(r.new_species_name), "legacy_name": canon(r.training_name)})
    m = pd.DataFrame(rows).set_index("speciesKey").join(sp)
    # Several legacy keys can land on one ToL class (synonyms merged in COL XR): the ToL model names the class
    # once, so `relabel` answers with the key that has the most training images; the others cannot be told apart.
    m["n_keys_on_class"] = m.groupby("tol_class")["tol_class"].transform("size").where(m["tol_class"].notna(), 0)
    OUT.mkdir(parents=True, exist_ok=True)
    m.reset_index().to_csv(OUT / "map_species.csv", index=False)
    ok = m[m["tol_class"].notna()].copy()
    for rank, cut, key in (("genus", 6, "genusKey"), ("family", 5, "familyKey")):
        ok["tol"] = ok["tol_class"].str.split("|").str[:cut].str.join("|")
        v = ok.groupby(["tol", key])["n_train"].sum().reset_index().sort_values("n_train", ascending=False)
        best = v.drop_duplicates("tol")
        best.rename(columns={"tol": "tol_class", key: f"{rank}Key"}).to_csv(OUT / f"map_{rank}.csv", index=False)
        print(f"{rank}: {len(best):,} ToL {rank} paths -> {best[f'{rank}Key'].nunique():,} lepinet keys; "
              f"{int((v.groupby('tol').size() > 1).sum())} paths split over several keys (majority by training images)")
    trap = pq.read_table(TRAP, columns=["speciesKey"]).to_pandas()["speciesKey"].astype("int64").astype(str)
    ne = [x.strip() for x in open(NE_LIST) if x.strip()] if NE_LIST.exists() else []
    mapped = set(ok.index)
    v = m[m["vocab"]]
    cov = {
        "lepinet classes (12,632)": [len(m), int(m["tol_class"].notna().sum())],
        "lepinet vocabulary, >= 50 training images (species)": [len(v), int(v["tol_class"].notna().sum())],
        "test fold images of the vocabulary": [int(v["n_test"].sum()), int(v.loc[v["tol_class"].notna(), "n_test"].sum())],
        "trap species": [trap.nunique(), len(set(trap) & mapped)],
        "trap crops": [len(trap), int(trap.isin(mapped).sum())],
        "north_europe list": [len(ne), len(set(ne) & mapped)],
        "by colxr name / legacy name": [int((m["how"] == "colxr").sum()), int((m["how"] == "legacy").sum())],
        "mapped keys sharing a ToL class with another key": int((ok["n_keys_on_class"] > 1).sum()),
        "ToL Lepidoptera species classes": sum(len(x) for x in tol.values()),
    }
    (OUT / "map_coverage.json").write_text(json.dumps(cov, indent=1))
    for k, x in cov.items():
        print(f"{k}: " + (f"{x[1]:,} of {x[0]:,} ({x[1] / x[0]:.1%})" if isinstance(x, list) and k[:2] != "by" else f"{x}"))
    un = m[m["tol_class"].isna() & m["vocab"]].sort_values("n_test", ascending=False)
    un.reset_index()[["speciesKey", "colxr_name", "legacy_name", "n_train", "n_test"]].to_csv(OUT / "map_unmapped.csv", index=False)


def _maps():
    sp = pd.read_csv(OUT / "map_species.csv", dtype={"speciesKey": str, "genusKey": str, "familyKey": str})
    sp = sp[sp["tol_class"].notna()]
    inv = sp.sort_values("n_train", ascending=False).drop_duplicates("tol_class").set_index("tol_class")["speciesKey"]
    g = pd.read_csv(OUT / "map_genus.csv", dtype=str).set_index("tol_class")["genusKey"]
    f = pd.read_csv(OUT / "map_family.csv", dtype=str).set_index("tol_class")["familyKey"]
    return sp, [inv.to_dict(), g.to_dict(), f.to_dict()]


# ------------------------------------------------------------------------------------------------ index
def index(a):
    OUT.mkdir(parents=True, exist_ok=True)
    side = None
    if a.set == "testfold":
        t = pq.read_table(LEPI, columns=["speciesKey", "genusKey", "familyKey", "set", "filename"]).to_pandas()
        sp = lepi_species()
        t = t[(t["set"] == "0") & t["speciesKey"].isin(sp.index[sp["vocab"]])]   # as score.py: the test fold, never filtered further
        paths = [str(LEPI_IMAGES / s / f) for s, f in zip(t["speciesKey"], t["filename"])]
        labels = [[s, g, f, "-1", "-1", "-1", "-1"] for s, g, f in zip(t["speciesKey"], t["genusKey"], t["familyKey"])]
    elif a.set == "trap":
        t = pq.read_table(TRAP, columns=["image_path", "speciesKey", "genusKey", "familyKey"]).to_pandas()
        k = t[["speciesKey", "genusKey", "familyKey"]].astype("int64").astype(str)
        paths = [str(TRAP_IMAGES / p) for p in t["image_path"]]
        labels = [[s, g, f, "-1", "-1", "-1", "-1"] for s, g, f in k.itertuples(index=False)]
    elif a.set == "heldout":
        f103 = _f103()
        ho = pd.read_csv(TOL / "heldout_taxa.csv", dtype=str, keep_default_na=False)
        cols = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
        ho["key"] = ho[cols].agg("|".join, axis=1)
        depth = ho.set_index("key")["depth"]
        parts = []
        for t in _row_groups(TOL / "metadata.parquet", ["uuid", "path", "exclude", *cols]):
            x = t.to_pandas()
            x = x[x["exclude"] == "heldout_taxa"]
            if len(x):
                parts.append(x)
        x = pd.concat(parts, ignore_index=True)
        x["key"] = x[cols].fillna("").agg("|".join, axis=1)
        x = x[x["key"].isin(depth.index)]
        x["h"] = pd.util.hash_array(x["uuid"].to_numpy(dtype=object), hash_key="heldout000000001", categorize=False)
        x = x.sort_values(["key", "h"]).groupby("key").head(HELDOUT_CAP)
        x["depth"] = x["key"].map(depth)
        keys = [f103.keys_of(tuple(r)) for r in x[cols].fillna("").itertuples(index=False)]
        paths = [str(TOL / "images" / p) for p in x["path"]]
        labels = [list(k) for k in keys]
        side = pd.DataFrame({"path": paths, "species": [k[0] for k in keys], "depth": x["depth"].to_numpy(),
                             **{r: [k[i] for k in keys] for i, r in enumerate(RANKS)}})
        print(f"{len(x):,} images of {x['key'].nunique():,} held-out species (of {len(ho):,}); by depth "
              f"{x.groupby('depth')['key'].nunique().to_dict()}")
    elif a.set == "known":
        # E2's in-distribution side: test-split images of trained species, a uniform sample by uuid hash, as many
        # as the held-out index has (held out is capped at 50 per species, this is not: head-heavy, as deployment)
        n = len(json.load(open(OUT / "index_heldout.json"))["path"])
        parts = []
        for t in _row_groups(TOL / "train/train.parquet", ["image_path", "uuid", "set", "speciesKey"]):
            x = t.to_pandas()
            x = x[x["set"] == "0"]
            x["h"] = pd.util.hash_array(x["uuid"].to_numpy(dtype=object), hash_key="known00000000001", categorize=False)
            parts.append(x[x["h"] < np.uint64(int(2**64 * 0.06))])   # about 180 k of 3.05 M, cut to n below
        x = pd.concat(parts, ignore_index=True).sort_values("h").head(n)
        assert len(x) == n, (len(x), n)
        paths = [str(TOL / "images" / p) for p in x["image_path"]]
        labels = [["-1"] * 7 for _ in paths]
        side = pd.DataFrame({"path": paths, "speciesKey": x["speciesKey"].to_numpy()})
        print(f"{n:,} test-split images of {x['speciesKey'].nunique():,} trained species")
    elif a.set == "rare":
        idx = json.load(open(RARE / "images_index.json"))
        paths = [str(RARE / "images" / r["path"]) for r in idx]
        labels = [["-1"] * 7 for _ in idx]
        side = pd.DataFrame({"path": paths, "rarespecies_id": [r["rarespecies_id"] for r in idx]})
    else:
        raise SystemExit(a.set)
    data = {"path": paths, "label": labels, "class": [0] * len(paths), "split": ["test"] * len(paths)}
    (OUT / f"index_{a.set}.json").write_text(json.dumps(data))
    if side is not None:
        side.to_parquet(OUT / f"index_{a.set}.parquet")
    print(f"{len(paths):,} images -> {OUT / f'index_{a.set}.json'}")


def _row_groups(path, cols):
    f = pq.ParquetFile(path)
    for i in range(f.metadata.num_row_groups):
        yield f.read_row_group(i, columns=cols)


def classlist(a):
    sp, _ = _maps()
    if a.keys:
        want = {x.strip() for x in open(a.keys) if x.strip()}
        sp = sp[sp["speciesKey"].isin(want)]
        print(f"{len(want):,} keys in {a.keys}; {sp['tol_class'].nunique():,} ToL classes")
    out = Path(a.out) if a.out else OUT / (f"{Path(a.keys).stem}.tol.classes" if a.keys else "lepinet.tol.classes")
    out.write_text("\n".join(sorted(sp["tol_class"].unique())) + "\n")
    print(f"{sp['tol_class'].nunique():,} classes -> {out}")


# ---------------------------------------------------------------------------------------------- relabel
def relabel(a):
    """Seven ToL ranks -> lepinet's species, genus, family; predictions to legacy keys. known_label is recomputed
    against lepinet's vocabulary (12,041 species and their genera and families), as for every lepinet model:
    score.py drops known_label 0 rows, so a label the ToL model cannot name must stay in and count wrong
    (CLAUDE.md: never filter the test fold). The share it can name is printed. mt's stale `correct` and
    `prediction_made` columns are dropped."""
    sp, maps = _maps()
    voc = pd.read_csv(OUT / "map_species.csv", dtype={"speciesKey": str, "genusKey": str, "familyKey": str})
    voc = voc[voc["vocab"]]
    known = [set(voc["speciesKey"]), set(voc["genusKey"]), set(voc["familyKey"])]
    d = pd.read_csv(a.pred, dtype={"label": str, "prediction": str})
    d = d[d["level"] <= 2].drop(columns=["correct", "prediction_made"], errors="ignore").copy()
    for lvl in range(3):
        m = d["level"] == lvl
        pred = d.loc[m, "prediction"].map(maps[lvl])
        bad = pred.isna()
        if bad.any():
            print(f"level {lvl}: {int(bad.sum()):,} predictions outside the map (scored wrong), e.g. "
                  f"{d.loc[m, 'prediction'][bad].iloc[0][:100]}")
        d.loc[m, "prediction"] = pred.fillna("unmapped").to_numpy()
        d.loc[m, "known_label"] = d.loc[m, "label"].isin(known[lvl]).astype(int).to_numpy()
    out = Path(a.out) if a.out else Path(a.pred).with_name("mini_metric.lepinet.csv")
    d.to_csv(out, index=False)
    s = d[d["level"] == 0]
    named = s["label"].isin(set(maps[0].values()))
    right = s["label"] == s["prediction"]
    print(f"{s['instance_id'].nunique():,} images; in lepinet's vocabulary {s['known_label'].mean():.1%}; "
          f"labels the ToL model can name {named.mean():.1%}; species accuracy {right[s['known_label'] == 1].mean():.4f} "
          f"(on the nameable {right[named].mean():.4f}) -> {out}")


# ------------------------------------------------------------------------------------------------ embed
def effective_cpus() -> int:
    """CPUs this process may use: the cgroup quota if set, else the affinity mask (as dev/082; inside a UCloud
    job os.cpu_count() reports the host's 256 cores)."""
    import os
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, round(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def embed(a):
    """GPU. The head's input embedding and every rank's argmax for the images of index_<set>.json, through
    mini_trainer's own predict (preprocessing from the checkpoint). --prototypes also saves the species weights."""
    import torch
    from mini_trainer.hierarchical.predict import cli
    from mini_trainer.logging import RawResultCollector
    from mini_trainer.modeling import classification_module
    from mini_trainer.modeling.context import EmbeddingContext
    from mini_trainer.predict import main as predict

    if a.model != "tol":
        return embed_frozen(a)
    out = Path(a.out) if a.out else _emb_dir(a, a.set)
    keep_protos = a.prototypes

    class Collector(RawResultCollector):
        def __init__(self, model=None, **kwargs):
            super().__init__()
            meta = classification_module(model).metadata
            self.cls2idx = meta.get("cls2idx")
            self.emb, self.arg, self.conf, self.paths, self.sp = [], [], [], [], []
            if keep_protos:
                n = len(self.cls2idx["0"])
                cands = {k: v for k, v in model.state_dict().items() if torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == n and v.shape[1] > 1}
                key = sorted(cands)[-1]
                out.mkdir(parents=True, exist_ok=True)
                np.save(out / "prototypes.npy", cands[key].float().cpu().numpy().astype(np.float16))
                (out / "prototypes.json").write_text(json.dumps({"tensor": key, "shape": list(cands[key].shape)}))

        def collect(self, paths=None, predictions=None, labels=None, **kwargs):
            outs = list(predictions) if isinstance(predictions, (list, tuple)) else [predictions]
            e = EmbeddingContext.get()
            assert e is not None, "the model published no embedding"
            self.emb.append(e.float().cpu().numpy().astype(np.float16))
            top = [o.float().softmax(-1).max(-1) for o in outs]
            self.arg.append(np.stack([t.indices.cpu().numpy() for t in top], 1).astype(np.int32))
            self.conf.append(np.stack([t.values.cpu().numpy() for t in top], 1).astype(np.float16))
            s = outs[0].float()   # species logits (class_spec order: species first), for E2's open-set scores
            self.sp.append(torch.stack([s.softmax(-1).max(-1).values, s.max(-1).values, torch.logsumexp(s, -1)], 1).cpu().numpy())
            self.paths.extend(paths)

        def save(self, dst, *args, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "embeddings.npy", np.concatenate(self.emb))
            np.save(out / "argmax.npy", np.concatenate(self.arg))
            np.save(out / "confidence.npy", np.concatenate(self.conf))
            np.save(out / "species_scores.npy", np.concatenate(self.sp).astype(np.float32))   # columns: SCORES
            (out / "paths.json").write_text(json.dumps(list(self.paths)))
            (out / "classes.json").write_text(json.dumps(self.cls2idx))
            print(f"{len(self.paths):,} images, embedding dim {self.emb[0].shape[1]} -> {out}")

    ix = Path(a.index) if a.index else OUT / f"index_{a.set}.json"
    if a.set == "known":   # no labels: mt needs no class for them
        pass
    sys.argv = [sys.argv[0], "--weights", a.weights, "--input", str(a.input or Path(json.load(open(ix))["path"][0]).parent),
                "--data_index", str(ix), "--output", str(out.parent), "--name", f"{out.name}_mt",
                "--class_spec", str(a.class_spec), *a.extra]   # extra: mt_hpredict flags, e.g. --batch_size 256
    # mt's loader defaults to one worker per host core, each holding a copy: 10 GB at 16 images on the
    # workstation, where systemd-oomd then kills the owner's tmux sessions. At most 4 unless given; on UCloud
    # pass the job's cgroup count with --num_workers.
    if not any(x.split("=")[0] == "--num_workers" for x in a.extra):
        sys.argv += ["--num_workers", str(min(4, effective_cpus()))]
    kwargs = cli(description="ToL embeddings")
    kwargs["collector_cls"] = Collector
    with torch.inference_mode():
        predict(**kwargs)


def embed_frozen(a):
    """E1's frozen baselines, no head: BioCLIP-2's projected image feature (768-d, `encode_image`, as the Rare
    Species port that reproduced its paper) and raw DINOv3 ConvNeXt-L's pooled feature (timm, num_classes=0,
    1,536-d; the ToL model's starting trunk). Each model's own eval preprocessing, fp32 as the port."""
    import torch
    from PIL import Image
    ix = Path(a.index) if a.index else OUT / f"index_{a.set}.json"
    paths = json.load(open(ix))["path"]
    out = Path(a.out) if a.out else _emb_dir(a, a.set)
    if a.model == "bioclip2":
        import open_clip
        model, _, tf = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        fwd, name = model.encode_image, "hf-hub:imageomics/bioclip-2, encode_image"
    else:
        import timm
        model = timm.create_model("convnext_large.dinov3_lvd1689m", pretrained=True, num_classes=0)
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
        fwd, name = model, "timm convnext_large.dinov3_lvd1689m, num_classes=0"
    model = model.eval().to("cuda")
    nw = next((int(x.split("=")[1]) if "=" in x else int(a.extra[i + 1]) for i, x in enumerate(a.extra)
               if x.split("=")[0] == "--num_workers"), min(4, effective_cpus()))

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return tf(Image.open(paths[i]).convert("RGB"))

    embs = []
    with torch.inference_mode():
        for x in torch.utils.data.DataLoader(Images(), batch_size=128, num_workers=nw):
            embs.append(fwd(x.to("cuda", non_blocking=True)).float().cpu().numpy().astype(np.float16))
    out.mkdir(parents=True, exist_ok=True)
    e = np.concatenate(embs)
    np.save(out / "embeddings.npy", e)
    (out / "paths.json").write_text(json.dumps(paths))
    (out / "meta.json").write_text(json.dumps({"model": name, "dim": int(e.shape[1]), "transform": str(tf)}, indent=1))
    print(f"{len(paths):,} images, embedding dim {e.shape[1]} ({a.model}) -> {out}")


def _load_emb(a, name):
    """An embed folder: embeddings, argmax per rank (None for a frozen baseline), paths, classes (None likewise)."""
    d = _emb_dir(a, name)
    e = np.load(d / "embeddings.npy").astype(np.float32)
    arg = np.load(d / "argmax.npy") if (d / "argmax.npy").exists() else None
    classes = json.loads((d / "classes.json").read_text()) if (d / "classes.json").exists() else None
    return d, e, arg, json.loads((d / "paths.json").read_text()), classes


# ------------------------------------------------------------------------------------- few-shot (BioCLIP-2)
def few_shot(feats: np.ndarray, target: np.ndarray, kshot: int, fold: int) -> tuple[float, np.ndarray, np.ndarray]:
    """BioCLIP-2's SimpleShot (src/evaluation/classification.py, as ported for Rare Species on 2026-11-08):
    features L2-normalised; random.seed(fold); per class in order of first appearance, shuffle, first k support;
    subtract the support mean, L2-normalise, class centroids, nearest centroid; top-1. Returns accuracy and the
    query targets and predictions (for group breakdowns). Classes with fewer than k+1 images are skipped."""
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    select: dict = {}
    for x, t in zip(f, target):
        select.setdefault(t, []).append(x)
    random.seed(fold)
    train, test, label, tgt = [], [], [], []
    for c, fs in select.items():
        if len(fs) <= kshot:
            continue
        fs = fs[:]
        random.shuffle(fs)
        train.append(fs[:kshot]); test.extend(fs[kshot:])
        label.extend([c] * kshot); tgt.extend([c] * (len(fs) - kshot))
    tr, te, label, tgt = np.vstack(train), np.vstack(test), np.array(label), np.array(tgt)
    m = tr.mean(0)
    cl2n = lambda x: (x - m) / np.linalg.norm(x - m, 2, 1)[:, None]  # noqa: E731
    tr, te = cl2n(tr), cl2n(te)
    centre = tr.reshape(len(train), kshot, -1).mean(1)
    labels = label[::kshot]
    cn = (centre ** 2).sum(1)
    pred = np.concatenate([labels[(cn[None] - 2 * te[i:i + 1024] @ centre.T).argmin(1)] for i in range(0, len(te), 1024)])
    return float((pred == tgt).mean()), tgt, pred


def _fewshot_table(feats, target, groups=None):
    res = {}
    for k in (1, 5):
        accs, by = [], {}
        for fold in range(5):
            acc, tgt, pred = few_shot(feats, target, k, fold)
            accs.append(acc)
            if groups is not None:
                for g, ok in pd.Series(pred == tgt).groupby(pd.Series(tgt).map(groups).to_numpy()):
                    by.setdefault(g, []).append(float(ok.mean()))
        res[f"{k}shot"] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs)), "folds": accs,
                           **({"by_group": {g: float(np.mean(v)) for g, v in by.items()}} if by else {})}
    return res


# ----------------------------------------------------------------------------------------------- placement
def placement_scores(arg: np.ndarray, truth: pd.DataFrame, idx2cls: list[dict]) -> pd.DataFrame:
    """Per held-out species: share of its images whose argmax at the placement rank is the true taxon there
    (genus for a held-out species, family for a held-out genus, order for a held-out family), plus every rank."""
    rows = []
    for depth, r in DEPTH_RANK.items():
        m = (truth["depth"] == depth).to_numpy()
        if not m.any():
            continue
        for rr in range(1, 7):
            pred = np.array([idx2cls[rr][i] for i in arg[m, rr]])
            ok = pred == truth.loc[m, RANKS[rr]].to_numpy()
            per = pd.Series(ok).groupby(truth.loc[m, "species"].to_numpy()).mean()
            known = set(idx2cls[rr].values())
            no_class = truth.loc[m].groupby("species")[RANKS[rr]].first().map(lambda x: x not in known)
            rows.append({"held out": {"near": "species (genus trained)", "mid": "whole genus", "far": "whole family"}[depth],
                         "rank": RANKS[rr], "placement rank": rr == r, "species": len(per), "images": int(m.sum()),
                         "species whose taxon here is not a model class": int(no_class.sum()),
                         "accuracy, macro over species": float(per.mean()), "accuracy, images": float(ok.mean())})
    return pd.DataFrame(rows)


def placement(a):
    d, e, arg, paths, classes = _load_emb(a, "heldout")
    side = pd.read_parquet(OUT / "index_heldout.parquet").set_index("path").loc[paths].reset_index()
    if arg is not None:   # a frozen baseline has no head: few-shot only
        idx2cls = [{v: k for k, v in classes[str(r)].items()} for r in range(7)]
        tab = placement_scores(arg, side, idx2cls)
        pd.set_option("display.width", 200)
        print(tab[tab["placement rank"]].drop(columns="placement rank").round(4).to_string(index=False))
        tab.to_csv(d / "placement.csv", index=False)
    target = pd.factorize(side["species"])[0]
    fs = _fewshot_table(e, target, groups=dict(enumerate(side.groupby(target)["depth"].first())))
    for k, v in fs.items():
        print(f"held-out species, {k}: {v['mean']:.4f} +- {v['sd']:.4f}; by depth "
              f"{ {g: round(x, 4) for g, x in v.get('by_group', {}).items()} }")
    (d / "fewshot.json").write_text(json.dumps(fs, indent=1))


# ---------------------------------------------------------------------------------------------- rare
def rare(a):
    """BioCLIP-2's Rare Species few-shot on the ToL model's embeddings. tol256-v1 excluded every training image
    within 6 bits (dHash, same genus) of a Rare Species image, so the benchmark is decontaminated; the table
    also splits queries by whether the species is a ToL training class (seen) or not."""
    d, e, arg, paths, _ = _load_emb(a, "rare")
    meta = pd.read_csv(RARE / "metadata.csv")
    side = pd.read_parquet(OUT / "index_rare.parquet").set_index("rarespecies_id")
    row = {p: i for i, p in enumerate(paths)}
    feats = e[[row[p] for p in side.loc[meta["rarespecies_id"], "path"]]]
    cls = list(dict.fromkeys(meta["sciName"]))
    target = meta["sciName"].map({c: i for i, c in enumerate(cls)}).to_numpy()
    trained = {k.split("|")[6].replace("_", " ") for k in json.load(open(SPEC))["cls2idx"]["0"]}
    seen = {i: ("seen species" if c in trained else "unseen species") for i, c in enumerate(cls)}
    res = _fewshot_table(feats, target, groups=seen)
    res["species"] = {"total": len(cls), "seen (a ToL training class of that name)": sum(v == "seen species" for v in seen.values())}
    for k in ("1shot", "5shot"):
        print(f"Rare Species {k}: {res[k]['mean']:.4f} +- {res[k]['sd']:.4f}; {res[k].get('by_group')}")
    print(res["species"], "(BioCLIP-2 reference, our port: 0.604 / 0.770)")
    (d / "rare_species.json").write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------------- congeners
def congener_stats(w: np.ndarray, s2g: np.ndarray, s2f: np.ndarray, chunk: int = 512) -> dict:
    """Z7 (dev/107 prototype_stats), at 190 k classes: for each species prototype, is the nearest other one a
    congener (species with a congener only), or in its family; torch on GPU when there is one."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.nn.functional.normalize(torch.from_numpy(w).float().to(dev), dim=1)
    g, f = torch.from_numpy(s2g).to(dev), torch.from_numpy(s2f).to(dev)
    nn = torch.empty(len(W), dtype=torch.long, device=dev)
    for i in range(0, len(W), chunk):
        cos = W[i:i + chunk] @ W.T
        cos[torch.arange(len(cos)), torch.arange(i, i + len(cos))] = -2.0
        nn[i:i + chunk] = cos.argmax(1)
    gs = torch.bincount(g)[g]
    has = gs > 1
    cong = (g[nn] == g)[has].float().mean().item()
    fam = (f[nn] == f).float().mean().item()
    chance = ((gs[has] - 1).float() / (len(W) - 1)).mean().item()
    return {"nearest is congener": cong, "chance": chance, "nearest in family": fam,
            "species with a congener": int(has.sum()), "species": len(W), "dim": int(w.shape[1])}


def congeners(a):
    d = _emb_dir(a, "heldout")
    w = np.load(d / "prototypes.npy").astype(np.float32)
    spec = json.load(open(SPEC))
    order = sorted(spec["cls2idx"]["0"], key=spec["cls2idx"]["0"].get)
    assert len(order) == len(w), (len(order), w.shape)
    s2g = pd.factorize(np.array([spec["labels"][k][1] for k in order]))[0]
    s2f = pd.factorize(np.array([spec["labels"][k][2] for k in order]))[0]
    st = congener_stats(w, s2g, s2f)
    wn = w / np.linalg.norm(w, axis=1, keepdims=True)
    ce = congener_stats(wn - wn.mean(0), s2g, s2f)
    st["nearest is congener, centred"] = ce["nearest is congener"]
    st["nearest in family, centred"] = ce["nearest in family"]
    print(json.dumps(st, indent=1))
    (d / "congeners.json").write_text(json.dumps(st, indent=1))


# ------------------------------------------------------------------------------------------------ openset
def openset_scores(known: np.ndarray, held: np.ndarray, depth: np.ndarray) -> pd.DataFrame:
    """AUROC of known (positive) against held-out images, per score column (higher = known) and depth."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for i, name in enumerate(SCORES):
        for dn in ["near", "mid", "far", "all"]:
            h = held[:, i] if dn == "all" else held[depth == dn, i]
            y = np.r_[np.ones(len(known)), np.zeros(len(h))]
            rows.append({"score": name, "held out": dn, "n held out": len(h), "n known": len(known),
                         "AUROC": float(roc_auc_score(y, np.r_[known[:, i], h]))})
    return pd.DataFrame(rows)


def openset(a):
    """E2 (replaces C3b for ToL models): held-out images (near = genus kept, mid = genus withheld, far = family
    withheld) against as many test-split images of trained species, by the species head's max softmax, max logit
    and energy."""
    dh, dk = _emb_dir(a, "heldout"), _emb_dir(a, "known")
    held, known = np.load(dh / "species_scores.npy"), np.load(dk / "species_scores.npy")
    paths = json.loads((dh / "paths.json").read_text())
    depth = pd.read_parquet(OUT / "index_heldout.parquet").set_index("path").loc[paths, "depth"].to_numpy()
    tab = openset_scores(known, held, depth)
    print(tab.pivot(index="score", columns="held out", values="AUROC")[["near", "mid", "far", "all"]].round(4).to_string())
    tab.to_csv(Path(a.root) / "openset.csv", index=False)


# ------------------------------------------------------------------------------------------------ report
def report(a):
    """One markdown table of whatever E1-E2 outputs exist: <root> for the ToL model, data/tol_eval/baselines."""
    root, lines = Path(a.root), []
    dirs = {"ToL": (root / "emb_heldout", root / "emb_rare"),
            "BioCLIP-2": (BASE / "emb_heldout_bioclip2", BASE / "emb_rare_bioclip2"),
            "DINOv3 ConvNeXt-L, raw": (BASE / "emb_heldout_dinov3", BASE / "emb_rare_dinov3")}
    pct = lambda x: f"{100 * x:.1f}"   # noqa: E731
    lines += ["| model | held out 1-shot (near / mid / far) | held out 5-shot (near / mid / far) | Rare Species 1 / 5-shot |",
              "|---|---|---|---|"]
    for m, (h, r) in dirs.items():
        cell = []
        for k in ("1shot", "5shot"):
            f = h / "fewshot.json"
            if f.exists():
                v = json.loads(f.read_text())[k]
                g = v.get("by_group", {})
                cell.append(f"{pct(v['mean'])} ({' / '.join(pct(g[x]) for x in ('near', 'mid', 'far') if x in g)})")
            else:
                cell.append("not run")
        f = r / "rare_species.json"
        rs = json.loads(f.read_text()) if f.exists() else None
        cell.append(f"{pct(rs['1shot']['mean'])} / {pct(rs['5shot']['mean'])}" if rs else "not run")
        lines.append(f"| {m} | " + " | ".join(cell) + " |")
    f = root / "emb_heldout" / "placement.csv"
    if f.exists():
        t = pd.read_csv(f)
        t = t[t["placement rank"]]
        lines += ["", "ToL placement, macro over species: " + "; ".join(
            f"{h} at {r} {pct(x)}" for h, r, x in zip(t["held out"], t["rank"], t["accuracy, macro over species"]))]
    f = root / "emb_heldout" / "congeners.json"
    if f.exists():
        c = json.loads(f.read_text())
        lines.append(f"ToL Z7: nearest prototype a congener {pct(c['nearest is congener'])} % "
                     f"(centred {pct(c['nearest is congener, centred'])}, chance {100 * c['chance']:.2f})")
    f = root / "openset.csv"
    if f.exists():
        t = pd.read_csv(f).pivot(index="score", columns="held out", values="AUROC")[["near", "mid", "far", "all"]]
        lines += ["", "| ToL open-set AUROC | near | mid | far | all |", "|---|---|---|---|---|"]
        lines += [f"| {s} | " + " | ".join(f"{x:.4f}" for x in t.loc[s]) + " |" for s in SCORES if s in t.index]
    text = "\n".join(lines) + "\n"
    (root / "results.md").write_text(text)
    print(text)


# ------------------------------------------------------------------------------------------------ selftest
def selftest(a):
    """The scoring code on synthetic data with known answers."""
    rng = np.random.default_rng(0)
    # 40 genera x 5 species, embeddings = genus centre + species offset + noise
    G, S, N, D = 40, 5, 12, 32
    gc = rng.normal(size=(G, D)) * 4
    sc = gc.repeat(S, 0) + rng.normal(size=(G * S, D))
    target = np.repeat(np.arange(G * S), N)
    feats = sc[target] + 0.05 * rng.normal(size=(len(target), D))
    acc, _, _ = few_shot(feats, target, 5, 0)
    assert acc > 0.99, acc
    noise = rng.normal(size=feats.shape)
    acc0, _, _ = few_shot(noise, target, 1, 0)
    assert acc0 < 0.03, acc0
    # congeners: prototypes = species centres -> nearest is a congener
    s2g = np.repeat(np.arange(G), S)
    st = congener_stats(sc, s2g, s2g // 4)
    assert st["nearest is congener"] > 0.95 and st["chance"] < 0.05, st
    st = congener_stats(rng.normal(size=sc.shape), s2g, s2g // 4)
    assert st["nearest is congener"] < 0.2, st
    # placement: argmax at genus rank right for "near" rows, family for "mid"
    truth = pd.DataFrame({"species": ["a", "a", "b", "c"], "depth": ["near", "near", "mid", "far"],
                          **{r: [f"{r}{i}" for i in (0, 0, 1, 2)] for r in RANKS}})
    idx2cls = [{0: f"{r}0", 1: f"{r}1", 2: f"{r}2", 3: f"{r}9"} for r in RANKS]
    arg = np.array([[0] * 7, [3] * 7, [1] * 7, [2] * 7])
    t = placement_scores(arg, truth, idx2cls)
    t = t[t["placement rank"]].set_index("held out")["accuracy, macro over species"]
    assert t["species (genus trained)"] == 0.5 and t["whole genus"] == 1.0 and t["whole family"] == 1.0, t
    # canonical names
    assert canon("Ctenoplusia (Ctenoplusia) accentifera (Lefèbvre, 1827)") == "Ctenoplusia accentifera"
    assert canon("Xyleutes strix Linnaeus, 1758") == "Xyleutes strix"
    # open set: known scores higher than near, near higher than far -> AUROC 1 for "far", 0.5-ish for overlap
    known = np.tile(np.array([[1.0, 1.0, 1.0]]), (50, 1)) + rng.normal(size=(50, 3)) * 0.01
    held = np.r_[np.full((50, 3), 0.0), np.full((50, 3), 1.0) + rng.normal(size=(50, 3)) * 0.01]
    depth = np.array(["far"] * 50 + ["near"] * 50)
    t = openset_scores(known, held, depth).set_index(["score", "held out"])["AUROC"]
    assert t[("max logit", "far")] == 1.0 and 0.3 < t[("max logit", "near")] < 0.7, t
    print("selftest: few-shot, congeners, placement, names, open set OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["map", "index", "classlist", "relabel", "embed", "placement", "rare", "congeners",
                                    "openset", "report", "selftest"])
    p.add_argument("--set", choices=["testfold", "trap", "heldout", "rare", "known"])
    p.add_argument("--model", default="tol", choices=["tol", "bioclip2", "dinov3"],
                   help="embed, placement, rare: the ToL checkpoint (--weights) or a frozen baseline (no head)")
    p.add_argument("--root", default=str(OUT), help="folder of the ToL model's emb_<set> outputs (default data/tol_eval)")
    p.add_argument("--keys", default="", help="classlist: lepinet keys, one per line (default: every mapped species)")
    p.add_argument("--pred", help="relabel: a ToL model's mini_metric.csv")
    p.add_argument("--out", default="")
    p.add_argument("--weights", help="embed: mini_trainer checkpoint")
    p.add_argument("--class_spec", default=str(SPEC))
    p.add_argument("--input", default="", help="embed: image root passed to predict (default: the index's first folder)")
    p.add_argument("--prototypes", action="store_true", help="embed: also save the species head's weights")
    p.add_argument("--emb", default="", help="placement, rare, congeners: an embed output folder (default: see --root, --model)")
    p.add_argument("--index", default="", help="embed: a data index other than data/tol_eval/index_<set>.json")
    a, a.extra = p.parse_known_args()
    if a.extra and a.step != "embed":
        p.error(f"unknown arguments {a.extra}")
    {"map": map_, "index": index, "classlist": classlist, "relabel": relabel, "embed": embed, "placement": placement,
     "rare": rare, "congeners": congeners, "openset": openset, "report": report, "selftest": selftest}[a.step](a)
