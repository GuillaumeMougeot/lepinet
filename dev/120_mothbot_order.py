"""Mothbot: the owner's AMI IAS crops (Kim Bjerge's order classifier run on box tree moth snapshots, mothbot format)
re-annotated with our best order classifier, plus ToL's species read of every crop. CPU only, ONNX, 4 threads.

    CUDA_VISIBLE_DEVICES="" taskset -c 0-3 nice -n 10 .venv/bin/python -I dev/120_mothbot_order.py fit
    ... dev/120_mothbot_order.py check                     # CPU read == dev/117's GPU padsq embedding on Kim's crops
    ... dev/120_mothbot_order.py run --zip IN.zip --out OUT.zip
    ... dev/120_mothbot_order.py kimbox                    # box tree moths in Kim's own dataset, by ToL's species head

The order classifier is BT1-BT3's head (a) on the pad-to-square read: multinomial logistic regression (C = 0.1, the
value chosen on his training list) on tol_full's frozen, normalised embedding of the crop padded to a square with the
border-ring mean, fit on Kim's training list (100,101 crops). It scored macro F1 0.961 on his validation list and 0.900
pooled over held-out sites (`data/broad_taxa/views.json`, journal 2026-11-20), level with BT4 (0.962 / 0.897) and
CPU-only. `fit` refits it from dev/117's GPU embeddings and asserts the 0.961; `check` asserts that this script's CPU
read (release ONNX trunk, dev/117's pad and read) reproduces the GPU embeddings it was fit on.

The zip is downloaded data: it is read in memory (no extraction), images are decoded by PIL only, run with `python -I`.
Output: the same entries in the same order. The parquet's label, labelid, score, abovethreshold, algorithm and orderlabel
are ours (score = our softmax probability; abovethreshold keeps the input's rule, score >= 0.045, which every 16-way
argmax probability meets); score_order and abovethreshold_order are not Kim's order score (they anti-correlate with it),
so they are left as they were. Images are renamed with the input's pattern
`{label}_{score, 3 decimals, trailing zeros cut}_{projectid}_{partnerid}_{filename}_crop_{detectionid}.jpg`.
A side table (`--out` with `.csv`) holds Kim's and our answer per crop, ToL's species top 1 and its probability for the
box tree moth (ToL row 28774, *Cryptographis perspectalis*: the Catalogue of Life build's name for *Cydalima
perspectalis*), and the nearest crop of Kim's dataset (cosine), since the zip holds 2022 crops from his sites.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/broad_taxa"
GPU = OUT / "gpu"
HEAD = OUT / "order_head_a_padsq.npz"
BOX_TREE_MOTH = 28774          # Cryptographis_perspectalis in tol.json labels[0]
C_KIM = 0.1                    # head.json C_chosen["a"][kim]
F1_KIM = 0.9607603947075196    # views.json a_padsq kim f1
THRESH = 0.045                 # the input parquet's abovethreshold rule
BATCH = 16


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "dev" / file)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


B115 = _load("b115", "115_broad_taxa.py")
B117 = _load("b117", "117_broad_taxa_gpu.py")


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fit(_):
    import torch
    torch.set_num_threads(B115.THREADS)
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    E = np.load(GPU / "emb_padsq.npy", mmap_mode="r")
    y = idx.label.to_numpy()
    tr, te = (idx.split == "train").to_numpy(), (idx.split == "val").to_numpy()
    m = B115.LR(C_KIM).fit(_norm(E[np.flatnonzero(tr)]), y[tr])
    f1 = B115.macro_prf(y[te], m.predict(_norm(E[np.flatnonzero(te)])))["f1"]
    print(f"head (a) padsq refit: his validation list macro F1 {f1:.4f} (recorded {F1_KIM:.4f})", flush=True)
    assert abs(f1 - F1_KIM) < 2e-3, "the refit does not reproduce the recorded score"
    np.savez(HEAD, mu=m.mu, sd=m.sd, W=m.W.detach().numpy(), b=m.b.detach().numpy(), f1_kim=f1, n_train=int(tr.sum()))
    print("wrote", HEAD)


def order_logits(emb, h):
    return ((emb - h["mu"]) / h["sd"]).astype(np.float32) @ h["W"] + h["b"]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Reader:
    """tol_full's release trunk on the padded read: image -> (embedding, species logits)."""

    def __init__(self):
        import onnxruntime as ort
        self.P = B115.predict_module()
        cfg = json.loads((B115.RELEASE / "config.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = B115.THREADS, 1
        self.sess = ort.InferenceSession(str(B115.RELEASE / cfg["trunk"]["onnx"]), so, providers=["CPUExecutionProvider"])
        self.head = self.P.load_head(B115.RELEASE / "heads/tol.safetensors")
        self.read = cfg["preprocessing"]["read_size"]
        assert self.read == B117.READ

    def pixels(self, image):
        from PIL import Image
        img = np.asarray(image.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
        sq = np.ascontiguousarray(B117.pad_square(img)).transpose(1, 2, 0)
        return B117.read_square_image(Image.fromarray(sq), self.read)

    def __call__(self, px):
        feats, _ = self.sess.run(["features", "embedding"], {"images": np.stack(px)})
        emb, ranks = self.P.score(feats, self.head)
        return emb.astype(np.float32), ranks[0]


def check(a):
    from PIL import Image
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    rows = np.random.default_rng(0).choice(len(idx), a.n, replace=False)
    E = np.load(GPU / "emb_padsq.npy", mmap_mode="r")
    h = np.load(HEAD)
    R = Reader()
    embs = []
    for s in range(0, len(rows), BATCH):
        px = []
        for i in rows[s:s + BATCH]:
            with Image.open(B115.IMAGES / idx.folder[i] / idx.file[i]) as im:
                px.append(R.pixels(im))
        embs.append(R(px)[0])
    cpu, gpu = _norm(np.concatenate(embs)), _norm(E[rows])
    cos = (cpu * gpu).sum(1)
    same = (order_logits(cpu, h).argmax(1) == order_logits(gpu, h).argmax(1)).mean()
    print(f"CPU read against the GPU padsq embedding, {len(rows)} of Kim's crops: cosine min {cos.min():.5f}, "
          f"median {np.median(cos):.5f}; same order answer {same:.3f}")
    assert cos.min() > 0.999 and same == 1.0


NAME = re.compile(r"^(?P<dir>.*/)(?P<label>[^/]*)_(?P<s>[0-9.]+)_(?P<proj>[a-z]+)_(?P<partner>\d+)_(?P<fname>[^/]*)"
                  r"_crop_(?P<det>[^/]*)\.jpg$")


def fmt(score):
    return ("%.3f" % score).rstrip("0").rstrip(".")


def load_input(path):
    """The mothbot zip: its parquet, Kim's 16 label names in his id order, and each crop's image entry, the filename
    pattern checked on every crop."""
    zin = zipfile.ZipFile(path)
    names = zin.namelist()
    pq = [n for n in names if n.endswith(".parquet")]
    assert len(pq) == 1, pq
    meta = pd.read_parquet(io.BytesIO(zin.read(pq[0])))
    id2label = meta.drop_duplicates("labelid").set_index("labelid").label.astype(str)
    labels = [id2label[str(i)] for i in range(16)]   # Kim's order = CATS order in dev/115
    assert [l.replace(" ", "").replace("Trichocera", "Trichoceridae").replace("Macros", "Macro").replace("Micros", "Micro")
            for l in labels] == B115.CATS, labels
    jpgs = {}
    for n in names:
        m = NAME.match(n)
        if m:
            jpgs[m["det"]] = (n, m)
    assert len(jpgs) == len(meta) == meta.detectionid.nunique(), (len(jpgs), len(meta))
    meta = meta.set_index("detectionid", drop=False)
    for det, (n, m) in jpgs.items():
        r = meta.loc[det]
        assert (m["label"], m["s"], m["proj"], m["partner"], m["fname"]) == (
            r.label, fmt(r.score), r.projectid, r.partnerid, r.filename), n
    return zin, pq[0], meta, labels, jpgs


def infer(zin, jpgs, cache):
    """tol_full on the padded read of every crop, cached: embedding, family posterior (ToL's own head, species summed
    per family), species top 1 and its probability, the box tree moth's probability."""
    from PIL import Image
    dets = list(jpgs)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        assert list(z["dets"]) == dets, "the cache is for another zip"
        return dets, {k: z[k] for k in z.files if k != "dets"}
    R = Reader()
    out = {k: [] for k in ("emb", "family_p", "sp_top", "sp_p", "p_btm")}
    t0 = time.time()
    for s in range(0, len(dets), BATCH):
        px = []
        for d in dets[s:s + BATCH]:
            with Image.open(io.BytesIO(zin.read(jpgs[d][0]))) as im:
                px.append(R.pixels(im))
        feats, _ = R.sess.run(["features", "embedding"], {"images": np.stack(px)})
        e, ranks = R.P.score(feats, R.head)
        p = softmax(ranks[0])
        out["emb"].append(e.astype(np.float32))
        out["family_p"].append(softmax(ranks[2]).astype(np.float32))
        out["sp_top"].append(ranks[0].argmax(1))
        out["sp_p"].append(p.max(1))
        out["p_btm"].append(p[:, BOX_TREE_MOTH])
        if (s // BATCH) % 20 == 0:
            done = s + len(px)
            print(f"{done:,}/{len(dets):,} crops, {done / (time.time() - t0):.2f} crops/s", flush=True)
    out = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(cache, dets=np.array(dets), **out)
    return dets, out


def write_zip(zin, pq_name, meta, jpgs, dets, label, labelid, score, algorithm, out):
    """The input zip with our label, labelid, score, abovethreshold, algorithm and orderlabel, images renamed with the
    input's pattern; everything else byte for byte, the parquet in the input's Arrow schema. Returns the new names."""
    new = meta.loc[dets].copy()
    new["label"] = new["orderlabel"] = label
    new["labelid"] = [str(i) for i in labelid]
    new["score"] = np.asarray(score, np.float64)
    new["abovethreshold"] = new.score >= THRESH
    new["algorithm"] = algorithm
    renamed = {}
    for d in dets:
        n, m = jpgs[d]
        r = new.loc[d]
        renamed[n] = f"{m['dir']}{r.label}_{fmt(r.score)}_{r.projectid}_{r.partnerid}_{r.filename}_crop_{d}.jpg"
    assert len(set(renamed.values())) == len(renamed)
    out_meta = new.loc[meta.detectionid].reset_index(drop=True).astype(meta.dtypes.to_dict())   # input row order
    tmp = Path(str(out) + ".part")
    with zipfile.ZipFile(tmp, "w") as zout:
        for info in zin.infolist():
            n = info.filename
            if n == pq_name:
                zout.writestr(zipfile.ZipInfo(n, info.date_time), parquet_like(out_meta, zin.read(n)),
                              compress_type=zipfile.ZIP_DEFLATED)
            elif n in renamed:
                zout.writestr(zipfile.ZipInfo(renamed[n], info.date_time), zin.read(n), compress_type=zipfile.ZIP_STORED)
            else:
                zout.writestr(info, zin.read(n), compress_type=info.compress_type)
    tmp.replace(out)
    return [renamed[jpgs[d][0]].rsplit("/", 1)[1] for d in dets]


def run(a):
    """Head (a) on the padded read."""
    h = np.load(HEAD)
    zin, pq_name, meta, labels, jpgs = load_input(a.zip)
    dets, z = infer(zin, jpgs, a.cache)
    emb = z["emb"]
    p_ord = softmax(order_logits(emb, h))
    # nearest crop of Kim's dataset (padsq embedding), for the 2022 crops from his sites
    E = np.load(GPU / "emb_padsq.npy", mmap_mode="r")
    kim = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    near = np.full(len(dets), -1.0, np.float32)
    near_i = np.zeros(len(dets), np.int64)
    for s in range(0, E.shape[0], 16384):
        sims = emb @ _norm(E[s:s + 16384]).T
        j = sims.argmax(1)
        v = sims[np.arange(len(dets)), j]
        upd = v > near
        near[upd], near_i[upd] = v[upd], s + j[upd]
    sp_names = json.loads((B115.RELEASE / "heads/tol.json").read_text())["labels"][0]
    lab = p_ord.argmax(1)
    m = meta.loc[dets]
    side = pd.DataFrame({
        "detectionid": dets, "country": m.country.to_numpy(), "code": m.code.to_numpy(), "year": m.year.to_numpy(),
        "kim_label": m.label.to_numpy(), "kim_score": m.score.to_numpy(),
        "tol_label": [labels[i] for i in lab], "tol_score": p_ord.max(1),
        "tol_species": [sp_names[i].split("|")[-1] for i in z["sp_top"]], "tol_species_p": z["sp_p"],
        "p_box_tree_moth": z["p_btm"], "kim_dataset_nearest_cos": near,
        "kim_dataset_nearest": [f"{kim.folder[i]}/{kim.file[i]}" for i in near_i]})
    side["new_file"] = write_zip(zin, pq_name, meta, jpgs, dets, side.tol_label.to_numpy(), lab, side.tol_score.to_numpy(),
                                 "lepinet_tol_order_a_padsq", a.out)
    side.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print(f"wrote {a.out} and {Path(a.out).with_suffix('.csv')}")
    summary(side)


def raw_groups(labels):
    """ToL families -> groups: Kim's 16 categories by dev/115's V0 mapping (written before any score; Plantae =
    Vegetation), every other family by its ToL order, so taxa outside his scheme (snails, other insect orders, fungi)
    keep their own name. Returns (group index per family, group names); Kim's 16 come first, in his id order."""
    fc, missing = B115.family_category(labels, "V0")
    assert not missing
    names = []
    for path in labels:
        p = path.split("|")
        order = p[3] or p[2] or p[1] or p[0]
        # a family left unmapped under one of Kim's orders is a mis-pathed node (Asterolepididae, a fossil fish under
        # Lepidoptera): named by family, so no second "Lepidoptera" group appears
        names.append(f"{p[4]} ({p[1]})" if order in ("Lepidoptera", "Diptera", "Hymenoptera") else order)
    extra = sorted({names[i] for i in np.flatnonzero(fc < 0)})
    gid = {n: 16 + k for k, n in enumerate(extra)}
    g = np.array([c if c >= 0 else gid[names[i]] for i, c in enumerate(fc)])
    return g, extra


def raw(a):
    """Zero training: ToL's own head, its family posterior summed into groups (Kim's 16 + ToL orders outside them)."""
    zin, pq_name, meta, labels, jpgs = load_input(a.zip)
    dets, z = infer(zin, jpgs, a.cache)
    tol = json.loads((B115.RELEASE / "heads/tol.json").read_text())["labels"]
    g, extra = raw_groups(tol[2])
    names = labels + extra
    G = np.zeros((len(g), len(names)), np.float32)
    G[np.arange(len(g)), g] = 1
    mass = z["family_p"] @ G
    lab = mass.argmax(1)
    fam = z["family_p"].argmax(1)
    m = meta.loc[dets]
    side = pd.DataFrame({
        "detectionid": dets, "country": m.country.to_numpy(), "code": m.code.to_numpy(), "year": m.year.to_numpy(),
        "kim_label": m.label.to_numpy(), "kim_score": m.score.to_numpy(),
        "raw_label": [names[i] for i in lab], "raw_labelid": lab, "raw_score": mass.max(1),
        "raw_family": [tol[2][i].split("|", 3)[-1] for i in fam], "raw_family_p": z["family_p"].max(1),
        "tol_species": [tol[0][i].split("|")[-1] for i in z["sp_top"]], "tol_species_p": z["sp_p"],
        "p_box_tree_moth": z["p_btm"]})
    side["new_file"] = write_zip(zin, pq_name, meta, jpgs, dets, side.raw_label.to_numpy(), lab, side.raw_score.to_numpy(),
                                 "lepinet_tol_raw_v0_padsq", a.out)
    side.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print(f"wrote {a.out} and {Path(a.out).with_suffix('.csv')}; labelid 16+ = ToL orders outside Kim's scheme")
    print(side.raw_label.value_counts().head(30).to_string())


def combo(a):
    """Head (a) times ToL's own category posterior (product of experts over Kim's 16, ToL's mass renormalised over
    them): 0.949 on his list and 0.899 site held out against (a)'s 0.961 / 0.900 (supervisor's check, 2026-11-25, on
    dev/117's padsq embeddings and views_logmass.npz). Where ToL puts more than half its mass outside his 16
    categories the crop is labelled Other (labelid 16), scored by that mass: dirt and taxa outside his scheme (snails).
    ToL's name for the group is kept in the side table only; on blurry crops it is unreliable (a streak read as a gecko
    at a Danish site, a blob as a songbird at 0.86)."""
    h = np.load(HEAD)
    zin, pq_name, meta, labels, jpgs = load_input(a.zip)
    dets, z = infer(zin, jpgs, a.cache)
    tol = json.loads((B115.RELEASE / "heads/tol.json").read_text())["labels"]
    g, extra = raw_groups(tol[2])
    G = np.zeros((len(g), 16 + len(extra)), np.float32)
    G[np.arange(len(g)), g] = 1
    mass = z["family_p"] @ G
    inside = mass[:, :16].sum(1)
    poe = np.log(softmax(order_logits(z["emb"], h))) + np.log(mass[:, :16] / inside[:, None] + 1e-6)
    p = softmax(poe)
    out = inside < 0.5
    names = labels + ["Other"]
    lab = np.where(out, 16, p.argmax(1))
    score = np.where(out, 1 - inside, p.max(1))
    m = meta.loc[dets]
    side = pd.DataFrame({
        "detectionid": dets, "country": m.country.to_numpy(), "code": m.code.to_numpy(), "year": m.year.to_numpy(),
        "kim_label": m.label.to_numpy(), "kim_score": m.score.to_numpy(),
        "combo_label": [names[i] for i in lab], "combo_labelid": lab, "combo_score": score, "tol_mass_outside_16": 1 - inside,
        "tol_other_group": [extra[i] for i in mass[:, 16:].argmax(1)], "tol_other_group_mass": mass[:, 16:].max(1),
        "tol_species": [tol[0][i].split("|")[-1] for i in z["sp_top"]], "tol_species_p": z["sp_p"]})
    side["new_file"] = write_zip(zin, pq_name, meta, jpgs, dets, side.combo_label.to_numpy(), lab, score,
                                 "lepinet_tol_order_a_x_raw_padsq", a.out)
    side.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print(f"wrote {a.out}; {int(out.sum())} crops labelled Other")
    print(side.combo_label.value_counts().head(20).to_string())


def parquet_like(df, original: bytes) -> bytes:
    """`df` as parquet with the original file's Arrow schema (string, not large_string; area_px int32 even when all
    null), so a reader of the input reads the output the same way."""
    import pyarrow as pa
    import pyarrow.parquet as pqt
    schema = pqt.read_schema(io.BytesIO(original))
    table = pa.Table.from_pandas(df, preserve_index=False).select(schema.names).cast(schema)
    buf = io.BytesIO()
    pqt.write_table(table, buf)
    assert pqt.read_schema(io.BytesIO(buf.getvalue())).remove_metadata().equals(schema.remove_metadata())
    return buf.getvalue()


def repack(a):
    """Rewrite the parquet inside an existing output with the input's schema (no re-inference)."""
    zin, zold = zipfile.ZipFile(a.zip), zipfile.ZipFile(a.out)
    name = next(n for n in zin.namelist() if n.endswith(".parquet"))
    data = parquet_like(pd.read_parquet(io.BytesIO(zold.read(name))), zin.read(name))
    tmp = Path(str(a.out) + ".part")
    with zipfile.ZipFile(tmp, "w") as zout:
        for info in zold.infolist():
            if info.filename == name:
                zout.writestr(zipfile.ZipInfo(name, info.date_time), data, compress_type=zipfile.ZIP_DEFLATED)
            else:
                zout.writestr(info, zold.read(info.filename), compress_type=info.compress_type)
    tmp.replace(a.out)
    print("repacked", a.out)


def summary(side):
    btm = side.tol_species == "Cryptographis_perspectalis"
    print(f"\nToL species top 1 = box tree moth: {btm.sum():,} of {len(side):,} crops "
          f"({(side.p_box_tree_moth > 0.5).sum():,} with probability above 0.5)")
    print("\nKim's label against ours, all crops:")
    print(pd.crosstab(side.kim_label, side.tol_label, margins=True).to_string())
    print("\nOn the crops ToL calls box tree moth: Kim's label / ours")
    print(side[btm].kim_label.value_counts().to_string())
    print(side[btm].tol_label.value_counts().to_string())
    print(f"\nnearest Kim crop at cosine >= 0.95: {(side.kim_dataset_nearest_cos >= 0.95).sum()}, >= 0.98: "
          f"{(side.kim_dataset_nearest_cos >= 0.98).sum()}")


def kimbox(_):
    """ToL's species top 1 on every crop of Kim's dataset (dev/115's shards, the centre read)."""
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    z = B115.load_shards(("top5", "top5_p"))
    top1, p1 = z["top5"][:, 0], z["top5_p"][:, 0]
    in5 = (z["top5"] == BOX_TREE_MOTH).any(1)
    btm = top1 == BOX_TREE_MOTH
    cat = np.array(B115.CATS)[idx.label.to_numpy()]
    print(f"Kim's dataset: {len(idx):,} crops; ToL top 1 = box tree moth on {btm.sum():,} "
          f"({(btm & (p1 > 0.5)).sum():,} above 0.5); in the top 5 on {in5.sum():,}")
    print(pd.Series(cat[btm]).value_counts().to_string())
    print("by trap site:", pd.Series(idx.trap.str[:2].to_numpy()[btm]).value_counts().to_dict())
    lep = np.isin(cat, ["LepidopteraMacro", "LepidopteraMicro"])
    print(f"Lepidoptera crops: {lep.sum():,}; box tree moth share {btm[lep].mean():.4%}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["fit", "check", "run", "raw", "combo", "repack", "kimbox"])
    p.add_argument("--zip", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--cache", type=Path, default=OUT / "mothbot_cache.npz", help="the trunk pass, reused by run and raw")
    a = p.parse_args()
    {"fit": fit, "check": check, "run": run, "raw": raw, "combo": combo, "repack": repack, "kimbox": kimbox}[a.step](a)


if __name__ == "__main__":
    main()
