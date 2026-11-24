"""C3, the order half: the frozen ToL trunk on the AMI broad-taxon dataset (MCC24; Bjerge, Karstoft, Hoye 2025)
against Kim Bjerge's specialist (ResNet50v2 at 128 px). CPU only, ONNX Runtime, at most 4 threads.

    .venv/bin/python -I dev/115_broad_taxa.py split      # Kim's 80/20 split recovered from his published logits; his metrics
    .venv/bin/python -I dev/115_broad_taxa.py embed      # ToL trunk + head on every crop (his validation split first), resumable
    .venv/bin/python -I dev/115_broad_taxa.py mapping    # ToL families -> the 16 categories, the pre-registered variants
    .venv/bin/python -I dev/115_broad_taxa.py zeroshot   # no training: category masses from the hierarchical marginals
    .venv/bin/python -I dev/115_broad_taxa.py head       # BT1: every labelled rung, on Kim's list and site held out
    .venv/bin/python -I dev/115_broad_taxa.py nolabels   # BT5: EM prior (L0), temperature, pseudo-label head
    .venv/bin/python -I dev/115_broad_taxa.py leak       # near-duplicates across each split, 1-NN, byte-identical pairs
    .venv/bin/python -I dev/115_broad_taxa.py night      # diagnostic: 80/20 by trap-night within the sites, 1-NN and (a)
    .venv/bin/python -I dev/115_broad_taxa.py views      # BT3: tta3bm and pad-to-square, from dev/117's embeddings
    .venv/bin/python -I dev/115_broad_taxa.py gpu        # BT2 (Kim's recipe retrained), BT4 (frozen-trunk head)

Folds: `kim` (his training list -> his validation list) and LV / OH / SS (the other two sites -> that site, pooled
as "site held out"). The GPU job's outputs (dev/117, /12347837/repos/broad_taxa/out) go in data/broad_taxa/gpu.

Data (CC-BY-4.0, Zenodo 15105293, `broadTaxonClassifier.zip`, md5 checked) and Kim's published files
(github.com/kimbjerge/MCC24-trap, `model_order_100524/`) sit in /data/au761367/benchmarks/mcc24; they are downloaded
data, so run this with `python -I` and read Kim's pickles with an unpickler that refuses every global.

Kim did not publish his split lists, but he published his model's logits on the full set (`predictLabelsTestTrain.pkl`,
125,135 rows) and on the validation split (`predictLabelsTest.pkl`, 25,034 rows, the "samples" of his Table 4). The
full set is in torchvision ImageFolder order (folders sorted, files sorted by name: checked by running his own
weights on two whole folders, 100 % nearest-row agreement), and every validation row equals one full-set row exactly,
so `split` recovers his validation list file by file. His "validation" is the only held-out set he reports on, and his
checkpoint is `dhc_best`, so it also chose the checkpoint.

The trunk is mini_trainer's release of tol_full (`mt_release`, journal 2026-11-19): `trunk.onnx` with
`heads/tol.safetensors` scored by the release's own `predict.py`, verified against mini_trainer's evaluation on
300 images (100 % top-1 at every rank, identical pixels). Results: journal 2026-11-20 (one frozen trunk), RESULTS BT1.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/au761367/benchmarks/mcc24")
IMAGES = DATA / "broadTaxonClassifier"
KIM = DATA / "kim_repo"
OUT = ROOT / "data/broad_taxa"
RELEASE = ROOT / "data/release_checks/release_tol"
CATS = ["Araneae", "Coleoptera", "DipteraBrachycera", "DipteraNematocera", "DipteraTipulidae", "DipteraTrichoceridae",
        "Ephemeroptera", "Hemiptera", "HymenopteraOther", "HymenopteraVespidae", "LepidopteraMacro", "LepidopteraMicro",
        "Neuroptera", "Opiliones", "Trichoptera", "Vegetation"]   # Kim's class order (his folders, sorted)
IMG_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}   # torchvision ImageFolder's
SHARD = 1024
BATCH = 16
THREADS = 4


# ------------------------------------------------------------------------------------------------ helpers
class _NoGlobals(pickle.Unpickler):
    """Kim's pickles hold lists of floats, ints and strings only; anything else is refused."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def kim_pickle(name):
    with open(KIM / name, "rb") as f:
        logits, labels, names = _NoGlobals(f).load()
    return np.asarray(logits, np.float32), np.asarray(labels, np.int64), names


def macro_prf(y, p, n=len(CATS)):
    """Per-class precision, recall, F1 and their unweighted means (sklearn's average='macro', zero_division=0)."""
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(n)), zero_division=0)
    return {"precision": float(pr.mean()), "recall": float(rc.mean()), "f1": float(f1.mean()),
            "accuracy": float((y == p).mean()), "n": int(len(y)),
            "per_class": {CATS[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                                    "support": int(sup[i])} for i in range(n)}}


def predict_module():
    """The release's own predict.py (mini_trainer/release/standalone.py, copied into the release)."""
    spec = importlib.util.spec_from_file_location("release_predict", RELEASE / "predict.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["release_predict"] = m   # its dataclasses look themselves up there
    spec.loader.exec_module(m)
    return m


CROP = re.compile(r"^crop_TRAPNAME_([A-Z]{2}\d)_IMAGENAME_(\d{14})-(\d+)-snapshot_CROPNUMBER_(\d+)_UUID_")
MOTH = re.compile(r"^(.*?)_([A-Z]{2}\d)_(\d+)_(\d+)_(\d+)_(\d+)\.(?:JPG|jpg)$")


def parse_name(f):
    """Trap and time from a file name. Crops: source image time stamp and crop number. Lepidoptera files:
    'Genus species_TRAP_dayofyear_hour_minute_seq' (no year; the species name is not a category label)."""
    m = CROP.match(f)
    if m:
        return {"trap": m.group(1), "stamp": m.group(2), "source": f"{m.group(1)}_{m.group(2)}", "species_name": ""}
    m = MOTH.match(f)
    if m:
        return {"trap": m.group(2), "stamp": f"doy{int(m.group(3)):03d}_{int(m.group(4)):02d}{int(m.group(5)):02d}",
                "source": "", "species_name": m.group(1).strip()}
    return {"trap": "", "stamp": "", "source": "", "species_name": ""}


# ------------------------------------------------------------------------------------------------- split
def split(_):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c, d in enumerate(CATS):
        files = sorted(f for f in os.listdir(IMAGES / d) if os.path.splitext(f)[1].lower() in IMG_EXT)
        rows += [{"folder": d, "file": f, "label": c} for f in files]
    idx = pd.DataFrame(rows)
    LA, yA, names = kim_pickle("predictLabelsTestTrain.pkl")
    LV, yV, _ = kim_pickle("predictLabelsTest.pkl")
    assert len(idx) == len(LA) == 125_135 and (idx.label.to_numpy() == yA).all(), "folder counts differ from Kim's"
    print("Kim's classes:", names)
    # every validation row is one full-set row, bit for bit; validation order is a subsequence of full order
    key = [LA[i].tobytes() for i in range(len(LA))]
    where = {}
    for i, k in enumerate(key):
        where.setdefault(k, []).append(i)
    val_rows, twins, last = [], [], {}
    for j in range(len(LV)):
        cand = [i for i in where.get(LV[j].tobytes(), []) if yA[i] == yV[j]]
        assert cand, f"validation row {j} has no exact match"
        nxt = [i for i in cand if i > last.get(yV[j], -1)]
        val_rows.append(nxt[0])
        last[yV[j]] = nxt[0]
        twins.append(len(cand))
    idx["split"] = "train"
    idx.loc[val_rows, "split"] = "val"
    idx["logit_twins"] = [len(where[k]) for k in key]   # >1: another crop gives bit-identical logits (a duplicate file?)
    meta = pd.DataFrame([parse_name(f) for f in idx.file])
    idx = pd.concat([idx, meta], axis=1)
    idx.insert(0, "kim_idx", np.arange(len(idx)))
    # duplicate files, by content
    idx["md5"] = [hashlib.md5((IMAGES / d / f).read_bytes()).hexdigest() for d, f in zip(idx.folder, idx.file)]
    idx.to_csv(OUT / "index.csv", index=False)
    dup = idx[idx.duplicated("md5", keep=False)]
    cross = dup.groupby("md5").split.nunique().gt(1).sum()
    print(f"validation {int((idx.split == 'val').sum()):,} (ambiguous ties {sum(t > 1 for t in twins)}); "
          f"files with a byte-identical twin {len(dup):,} in {dup.md5.nunique():,} groups, {cross} groups on both sides")
    # Kim's numbers, from his own logits
    pv = LV.argmax(1)
    res = {"validation": macro_prf(yV, pv)}
    th = pd.read_csv(KIM / "thresholdsTestTrain.csv")["Threshold"].to_numpy()
    keep = LV[np.arange(len(LV)), pv] >= th[pv]
    res["validation_without_uncertain"] = macro_prf(yV[keep], pv[keep])
    res["full_set_incl_train"] = macro_prf(yA, LA.argmax(1))
    (OUT / "kim_metrics.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"Kim {k}: n {v['n']:,}  macro P {v['precision']:.4f} R {v['recall']:.4f} F1 {v['f1']:.4f}  acc {v['accuracy']:.4f}")


# ------------------------------------------------------------------------------------------------- embed
def embed(a):
    import onnxruntime as ort
    from scipy import sparse

    P = predict_module()
    idx = pd.read_csv(OUT / "index.csv")
    order = np.r_[idx.index[idx.split == "val"], idx.index[idx.split == "train"]]   # Kim's validation first
    if a.limit:
        order = order[:a.limit]
    cfg = json.loads((RELEASE / "config.json").read_text())
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = THREADS, 1
    sess = ort.InferenceSession(str(RELEASE / cfg["trunk"]["onnx"]), so, providers=["CPUExecutionProvider"])
    head = P.load_head(RELEASE / "heads/tol.safetensors")
    fam = head.parents[1][head.parents[0]]            # species -> family
    S = sparse.csr_matrix((np.ones(len(fam), np.float32), (np.arange(len(fam)), fam)), shape=(len(fam), head.sizes[2]))
    read = cfg["preprocessing"]["read_size"]
    shards = OUT / ("emb_smoke" if a.limit else "emb")
    shards.mkdir(exist_ok=True)
    n_sh = (len(order) + SHARD - 1) // SHARD
    t0, done = time.time(), 0
    for s in range(n_sh):
        dst = shards / f"shard_{s:04d}.npz"
        if dst.exists():
            continue
        rows = order[s * SHARD:(s + 1) * SHARD]
        out = {k: [] for k in ("features", "embedding", "family_p", "top5", "top5_p", "max_logit", "lse", "max_p")}
        for b in range(0, len(rows), BATCH):
            r = rows[b:b + BATCH]
            px = np.stack([P.read_square(IMAGES / idx.folder[i] / idx.file[i], read) for i in r])
            feats, _ = sess.run(["features", "embedding"], {"images": px})
            emb, ranks = P.score(feats, head)
            z = ranks[0]
            m = z.max(1, keepdims=True)
            e = np.exp(z - m)
            tot = e.sum(1, keepdims=True)
            p = e / tot
            t5 = np.argpartition(-p, 5, axis=1)[:, :5]
            t5 = np.take_along_axis(t5, np.argsort(-np.take_along_axis(p, t5, 1), 1), 1)
            out["features"].append(feats.astype(np.float16))
            out["embedding"].append(emb.astype(np.float16))
            out["family_p"].append((S.T @ p.T).T.astype(np.float32))
            out["top5"].append(t5.astype(np.int32))
            out["top5_p"].append(np.take_along_axis(p, t5, 1).astype(np.float32))
            out["max_logit"].append(m[:, 0])
            out["lse"].append((np.log(tot) + m)[:, 0])
            out["max_p"].append(p.max(1))
        arrays = {k: np.concatenate(v) for k, v in out.items()}
        arrays["family_p"] = arrays["family_p"].astype(np.float16)
        tmp = shards / f".tmp_{s:04d}.npz"
        np.savez(tmp, kim_idx=rows, **arrays)
        os.replace(tmp, dst)
        done += len(rows)
        rate = done / (time.time() - t0)
        print(f"shard {s + 1}/{n_sh}  {rate:.2f} img/s  eta {(len(order) - (s + 1) * SHARD) / rate / 3600:.2f} h", flush=True)


# ----------------------------------------------------------------------------------------------- mapping
# Written into the journal before any score (2026-11-20). Taxonomy only; nothing here was chosen by looking at a result.
# Diptera suborders: Pape, Blagoderov & Mostovski 2011, "Order Diptera", Zootaxa 3148: 222-229 (lower Diptera
# infraorders = "Nematocera", a paraphyletic grade; everything else Brachycera). Every Diptera family in ToL is
# listed on one side; `mapping` fails if one is missing.
NEMATOCERA = """Anisopodidae Axymyiidae Bibionidae Blephariceridae Bolitophilidae Canthyloscelidae Cecidomyiidae
Ceratopogonidae Chaoboridae Chironomidae Corethrellidae Culicidae Cylindrotomidae Deuterophlebiidae Diadocidiidae
Ditomyiidae Dixidae Hesperinidae Keroplatidae Limoniidae Lygistorrhinidae Mycetophilidae Nymphomyiidae Pachyneuridae
Pediciidae Perissommatidae Psychodidae Ptychopteridae Rangomaramidae Scatopsidae Sciaridae Simuliidae Tanyderidae
Thaumaleidae Tipulidae Trichoceridae""".split()
BRACHYCERA = """Acartophthalmidae Acroceridae Agromyzidae Anthomyiidae Anthomyzidae Asilidae Asteiidae Athericidae
Aulacigastridae Bombyliidae Brachystomatidae Calliphoridae Canacidae Carnidae Chloropidae Chyromyidae Clusiidae
Coelopidae Conopidae Cryptochetidae Diastatidae Diopsidae Dolichopodidae Drosophilidae Dryomyzidae Empididae
Ephydridae Fanniidae Helcomyzidae Heleomyzidae Heterocheilidae Hippoboscidae Hybotidae Inbiomyiidae Iteaphilidae
Lauxaniidae Lonchaeidae Lonchopteridae Megamerinidae Micropezidae Milichiidae Muscidae Mydidae Nemestrinidae Neriidae
Odiniidae Oestridae Opetiidae Opomyzidae Oreogetonidae Pallopteridae Phaeomyiidae Phoridae Piophilidae Pipunculidae
Platypezidae Platystomatidae Polleniidae Psilidae Pyrgotidae Rhagionidae Richardiidae Sarcophagidae Scathophagidae
Scenopinidae Sciomyzidae Sepsidae Somatiidae Sphaeroceridae Stratiomyidae Strongylophthalmyiidae Syrphidae Tabanidae
Tachinidae Tanypezidae Tephritidae Therevidae Ulidiidae Vermileonidae Xylomyidae Xylophagidae""".split()
CRANEFLIES = ["Tipulidae", "Limoniidae", "Pediciidae", "Cylindrotomidae"]   # Tipuloidea (variant V1 only)
# Vespidae in current classification includes the potter and pollen wasps, which ToL keeps as families of their own.
VESPIDAE = ["Vespidae", "Eumenidae", "Masaridae"]
# Macro-Lepidoptera: the superfamilies of the British and Danish macro-moth convention (Hepialoidea, Cossoidea incl.
# Sesiidae, Zygaenoidea: the families in Waring & Townsend's macro-moth guide and Danish "storsommerfugle"), the
# Macroheterocera (Mimallonoidea, Drepanoidea, Lasiocampoidea, Bombycoidea, Geometroidea, Noctuoidea) and
# Papilionoidea; family placement from van Nieukerken et al. 2011, "Order Lepidoptera", Zootaxa 3148: 212-221.
# Every other Lepidoptera family is micro. Kim: "grouping species of Lepidoptera at the family level" (no list).
MACRO = """Hepialidae Brachodidae Castniidae Cossidae Metarbelidae Sesiidae Aididae Dalceridae Epipyropidae
Heterogynidae Lacturidae Limacodidae Megalopygidae Phaudidae Somabrachyidae Zygaenidae Mimallonidae Cimeliidae Doidae
Drepanidae Thyatiridae Epicopeiidae Lasiocampidae Anthelidae Apatelodidae Bombycidae Brahmaeidae Carthaeidae Endromidae
Eupterotidae Phiditiidae Saturniidae Sphingidae Geometridae Sematuridae Uraniidae Erebidae Euteliidae Noctuidae Nolidae
Notodontidae Oenosandridae Scranciidae Hedylidae Hesperiidae Lycaenidae Nymphalidae Papilionidae Pieridae
Riodinidae""".split()
ORDERS = {"Araneae": "Araneae", "Coleoptera": "Coleoptera", "Ephemeroptera": "Ephemeroptera", "Hemiptera": "Hemiptera",
          "Neuroptera": "Neuroptera", "Opiliones": "Opiliones", "Trichoptera": "Trichoptera"}
VARIANTS = {
    "V0": "primary: names as Kim's labels say; Tipulidae alone; Vegetation = kingdom Plantae",
    "V1": "broad readings: crane flies = Tipuloidea (Tipulidae, Limoniidae, Pediciidae, Cylindrotomidae); "
          "Neuroptera = Neuropterida (+ Megaloptera, Raphidioptera); Vegetation = Plantae",
    "V2": "V0, but Vegetation = open-set reading: every family not in the 15 arthropod categories "
          "(other orders, other animals, plants, fungi)",
}


def family_category(labels, variant="V0"):
    """ToL family index -> category index (-1: not mapped). Path: Kingdom|Phylum|Class|Order|Family; the order and
    family are matched by name, so ToL's few mis-pathed nodes (e.g. Riodinidae under Cnidaria) still map."""
    cat = {c: i for i, c in enumerate(CATS)}
    out = np.full(len(labels), -1)
    missing = []
    for i, path in enumerate(labels):
        p = path.split("|")
        kingdom, order, family = p[0], p[3], p[4]
        if order in ORDERS:
            out[i] = cat[ORDERS[order]]
        elif variant == "V1" and order in ("Megaloptera", "Raphidioptera"):
            out[i] = cat["Neuroptera"]
        elif order == "Diptera":
            if family == "Trichoceridae":
                out[i] = cat["DipteraTrichoceridae"]
            elif family == "Tipulidae" or (variant == "V1" and family in CRANEFLIES):
                out[i] = cat["DipteraTipulidae"]
            elif family in NEMATOCERA:
                out[i] = cat["DipteraNematocera"]
            elif family in BRACHYCERA:
                out[i] = cat["DipteraBrachycera"]
            else:
                missing.append(path)
        elif order == "Hymenoptera":
            out[i] = cat["HymenopteraVespidae"] if family in VESPIDAE else cat["HymenopteraOther"]
        elif order == "Lepidoptera":
            if p[2] != "Insecta" or p[1] not in ("Arthropoda", "Cnidaria"):   # Asterolepididae: a fish under 'Lepidoptera'
                continue
            out[i] = cat["LepidopteraMacro"] if family in MACRO else cat["LepidopteraMicro"]
        elif variant in ("V0", "V1") and kingdom == "Plantae":
            out[i] = cat["Vegetation"]
    if variant == "V2":
        out[out < 0] = cat["Vegetation"]
    return out, missing


def mapping(_):
    labels = json.loads((RELEASE / "heads/tol.json").read_text())["labels"][2]
    doc = {"variants": VARIANTS, "nematocera": NEMATOCERA, "brachycera": BRACHYCERA, "craneflies_V1": CRANEFLIES,
           "vespidae": VESPIDAE, "macro_lepidoptera": MACRO, "orders": ORDERS}
    for v in VARIANTS:
        out, missing = family_category(labels, v)
        assert not missing, f"Diptera families on neither list: {missing}"
        doc[v] = {CATS[c]: sorted(labels[i].split("|", 3)[-1] for i in np.flatnonzero(out == c)) for c in range(len(CATS))}
        print(v, {CATS[c][:12]: int((out == c).sum()) for c in range(len(CATS))}, "unmapped", int((out < 0).sum()))
    leps = sorted({l.split("|")[4] for l in labels if l.split("|")[3] == "Lepidoptera"})
    print("Lepidoptera families macro:", [f for f in leps if f in MACRO])
    print("Lepidoptera families micro:", [f for f in leps if f not in MACRO])
    (OUT / "mapping.json").write_text(json.dumps(doc, indent=1))


def load_shards(cols=("embedding",)):
    """Every finished shard, rows in Kim's full-set order (kim_idx); missing rows absent. Each column is allocated
    once and filled shard by shard, so the peak is the result plus one shard (the family masses are 1.3 GB in fp16)."""
    files = sorted((OUT / "emb").glob("shard_*.npz"))
    ki = []
    for f in files:
        with np.load(f) as z:
            ki.append(z["kim_idx"])
    rows = np.sort(np.concatenate(ki))
    pos = np.full(int(rows.max()) + 1, -1)
    pos[rows] = np.arange(len(rows))
    out = {"kim_idx": rows}
    for f, k in zip(files, ki):
        with np.load(f) as z:
            for c in cols:
                a = z[c]
                if c not in out:
                    out[c] = np.empty((len(rows),) + a.shape[1:], a.dtype)
                out[c][pos[k]] = a
    return out


def family_mass_reduce(fp, M, chunk=8192):
    """fp (n x families, fp16) @ M (families x k), in float32 chunks."""
    return np.concatenate([fp[s:s + chunk].astype(np.float32) @ M for s in range(0, len(fp), chunk)])


# ----------------------------------------------------------------------------------------------- scoring
# Diagnostics added after the predictions (not pre-registered): the 16 categories collapsed to their 11 orders
# (Diptera, Hymenoptera and Lepidoptera merged), and the routing decision C3's pipeline needs (Lepidoptera or not).
ORDER_OF = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 9, 10])
LEP = {CATS.index("LepidopteraMacro"), CATS.index("LepidopteraMicro")}


def diagnostics(y, p):
    from sklearn.metrics import precision_recall_fscore_support
    o = macro_prf(ORDER_OF[y], ORDER_OF[p], n=11)
    yl, pl = np.isin(y, list(LEP)), np.isin(p, list(LEP))
    pr, rc, f1, _ = precision_recall_fscore_support(yl, pl, average="binary")
    return {"order11_macro_f1": o["f1"], "order11_accuracy": o["accuracy"],
            "lepidoptera_routing": {"precision": float(pr), "recall": float(rc), "f1": float(f1)}}


def kim_val():
    LV, yV, _ = kim_pickle("predictLabelsTest.pkl")
    return yV, LV.argmax(1)


def _line(name, r):
    d = r.get("diag", {})
    return (f"{name:34s} n {r['n']:7,}  macroF1 {r['f1']:.4f}  P {r['precision']:.4f}  R {r['recall']:.4f}  "
            f"acc {r['accuracy']:.4f}" + (f"  | 11 orders F1 {d['order11_macro_f1']:.4f}  Lep routing F1 "
                                          f"{d['lepidoptera_routing']['f1']:.4f}" if d else ""))


def zeroshot(_):
    from sklearn.metrics import confusion_matrix
    labels = json.loads((RELEASE / "heads/tol.json").read_text())["labels"][2]
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    z = load_shards(("family_p",))
    rows = z["kim_idx"]
    y = idx.label.to_numpy()[rows]
    val = idx.split.to_numpy()[rows] == "val"
    fp = z["family_p"]   # fp16; reduced in float32 chunks
    res = {"n_embedded": int(len(rows)), "n_val_embedded": int(val.sum())}
    yk, pk = kim_val()
    res["kim_val"] = {**macro_prf(yk, pk), "diag": diagnostics(yk, pk)}
    print(_line("Kim's ResNet50, his validation", res["kim_val"]))
    for v in VARIANTS:
        fc, _ = family_category(labels, v)
        M = np.zeros((len(labels), len(CATS)), np.float32)
        M[np.flatnonzero(fc >= 0), fc[fc >= 0]] = 1
        mass = family_mass_reduce(fp, M)
        off = np.clip(1 - mass.sum(1), 0, 1)
        pred = mass.argmax(1)
        for name, sel in (("full", np.ones(len(rows), bool)), ("val", val)):
            r = macro_prf(y[sel], pred[sel])
            r["diag"] = diagnostics(y[sel], pred[sel])
            r["set_aside_mass_mean"] = float(off[sel].mean())
            r["set_aside_largest"] = float((off[sel] > mass[sel].max(1)).mean())   # off-list mass beats every category
            r["confusion"] = confusion_matrix(y[sel], pred[sel], labels=list(range(len(CATS)))).tolist()
            res[f"{v}_{name}"] = r
            print(_line(f"ToL zero training {v} {name}", r), f" set-aside mass {r['set_aside_mass_mean']:.3f}")
        if v == "V0":   # where the set-aside mass goes
            colmean = sum(fp[s:s + 8192].astype(np.float32).sum(0) for s in range(0, len(fp), 8192)) / len(fp)
            off_cols = np.flatnonzero(fc < 0)
            fams = off_cols[np.argsort(-colmean[off_cols])[:12]]
            res["V0_set_aside_families"] = {labels[i]: float(colmean[i]) for i in fams}
    (OUT / "zeroshot.json").write_text(json.dumps(res, indent=1))
    r = res["V0_val"]["per_class"]
    print("per class, V0 on Kim's validation (F1; Kim's F1):")
    kp = res["kim_val"]["per_class"]
    for c in CATS:
        print(f"  {c:22s} {r[c]['f1']:.3f}  {kp[c]['f1']:.3f}  (P {r[c]['precision']:.3f} R {r[c]['recall']:.3f}, n {r[c]['support']})")


def _features(z, key):
    x = z[key].astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True) if key == "embedding" else x


def fold_masks(idx):
    """kim: his training list -> his validation list; LV, OH, SS: the other two sites -> that site."""
    site = idx.trap.str[:2].to_numpy()
    out = {"kim": ((idx.split == "train").to_numpy(), (idx.split == "val").to_numpy())}
    for s in ("LV", "OH", "SS"):
        out[s] = ((site != s) & (site != ""), site == s)
    return out


def score_folds(fn, y, masks, name):
    """fn(tr, te) -> predictions on te. Kim's list scored alone; the three sites pooled (each crop predicted by the
    model that did not see its site)."""
    res, pooled_y, pooled_p = {}, [], []
    for f, (tr, te) in masks.items():
        p = fn(tr, te)
        if f == "kim":
            res["kim"] = {**macro_prf(y[te], p), "diag": diagnostics(y[te], p)}
        else:
            res[f"site_{f}"] = {k: v for k, v in macro_prf(y[te], p).items() if k != "per_class"}
            pooled_y.append(y[te])
            pooled_p.append(p)
    yy, pp = np.concatenate(pooled_y), np.concatenate(pooled_p)
    res["site"] = {**macro_prf(yy, pp), "diag": diagnostics(yy, pp)}
    print(_line(f"{name} | Kim's list", res["kim"]))
    print(_line(f"{name} | site held out", res["site"]), flush=True)
    return res


class LR:
    """Multinomial logistic regression, sklearn's objective (C * sum of cross-entropies + |W|^2 / 2) on inputs
    standardised on the training part, solved with full-batch L-BFGS in torch (fp32): sklearn's lbfgs took minutes
    per fit at 100 k x 1,536."""

    def __init__(self, C):
        self.C = C

    def fit(self, X, y):
        import torch
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
        yt = torch.from_numpy(y.astype(np.int64))
        self.W = torch.zeros(X.shape[1], len(CATS), requires_grad=True)
        self.b = torch.zeros(len(CATS), requires_grad=True)
        opt = torch.optim.LBFGS([self.W, self.b], max_iter=500, tolerance_grad=1e-4, line_search_fn="strong_wolfe")  # sklearn tol
        lam = 1.0 / (self.C * len(y))

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xt @ self.W + self.b, yt) + 0.5 * lam * (self.W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return self

    def logits(self, X):
        import torch
        with torch.no_grad():
            return (torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32)) @ self.W + self.b).numpy()

    def predict(self, X):
        return self.logits(X).argmax(1)


def _choose_c(X, y, seed=0, verbose=False):
    hold = np.random.default_rng(seed).random(len(y)) < 0.1
    scores = {C: macro_prf(y[hold], LR(C).fit(X[~hold], y[~hold]).predict(X[hold]))["f1"] for C in (0.1, 1, 10, 100)}
    if verbose:
        print("   C on 10 % of the training part:", {k: round(v, 4) for k, v in scores.items()}, flush=True)
    return max(scores, key=scores.get)


def _lr_rung(X, y, chosen, tag):
    def fn(tr, te):
        C = _choose_c(X[tr], y[tr])
        chosen[tag].append(C)
        return LR(C).fit(X[tr], y[tr]).predict(X[te])
    return fn


def category_matrix(variant="V0"):
    labels = json.loads((RELEASE / "heads/tol.json").read_text())["labels"][2]
    fc, _ = family_category(labels, variant)
    M = np.zeros((len(labels), len(CATS)), np.float32)
    M[np.flatnonzero(fc >= 0), fc[fc >= 0]] = 1
    return M


def category_logmass(fp, variant="V0"):
    return np.log(family_mass_reduce(fp, category_matrix(variant)) + 1e-6)


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_bias(lm, y):
    import torch
    t, yt = torch.from_numpy(lm.astype(np.float32)), torch.from_numpy(y)
    b = torch.zeros(len(CATS), requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(t + b, yt)
        loss.backward()
        return loss
    opt.step(closure)
    return b.detach().numpy()


def fit_temperature(lm, y):
    import torch
    t, yt = torch.from_numpy(lm.astype(np.float32)), torch.from_numpy(y)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(t / logT.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())


def nll_ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = sum(abs((pred[m] == y[m]).mean() - conf[m].mean()) * m.mean()
              for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf > lo) & (conf <= hi)).any())
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()), float(ece)


def em_prior(p, iters=50):
    """dev/107's em_prior with a uniform source prior: the maximum-likelihood class prior of an unlabelled set, the
    model's posteriors held fixed (Saerens et al. 2002)."""
    c = p.shape[1]
    pi = np.full(c, 1.0 / c)
    for _ in range(iters):
        r = (pi * c).astype(np.float32)
        d = np.maximum(p @ r, 1e-30)
        s = r.astype(np.float64) * (p.T @ (1.0 / d).astype(np.float32))
        pi = s / s.sum()
    return pi


def _load_all(cols):
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    z = load_shards(cols)
    assert len(z["kim_idx"]) == len(idx), f"embed is not finished: {len(z['kim_idx'])} of {len(idx)} rows"
    return idx, z


def head(_):
    """BT1: every labelled rung (L1-L3, (a)-(c)), on Kim's list and site held out."""
    idx, z = _load_all(("embedding", "features", "family_p"))
    y = idx.label.to_numpy()
    masks = fold_masks(idx)
    lm = category_logmass(z["family_p"])
    # L3's candidates: a family with mean mass above 1e-4 on a fold's training part (at least half the crops) has
    # mean mass above 5e-5 over all crops, so the columns above 1e-5 hold every candidate of every fold
    fp16 = z["family_p"]
    colmean = sum(fp16[s:s + 8192].astype(np.float32).sum(0) for s in range(0, len(fp16), 8192)) / len(fp16)
    cand = np.flatnonzero(colmean > 1e-5)
    fp = z.pop("family_p")[:, cand].astype(np.float32)
    print(f"L3 candidates: {len(cand)} families of {len(colmean)}", flush=True)
    chosen = {k: [] for k in ("L2", "L3", "a", "c")}
    res = {}

    def l1(tr, te):
        return (lm[te] + fit_bias(lm[tr], y[tr])).argmax(1)
    res["L1_bias_only"] = score_folds(l1, y, masks, "L1 zero training + 16 biases")
    res["L2_linear_on_16_masses"] = score_folds(_lr_rung(lm, y, chosen, "L2"), y, masks, "L2 linear on 16 log masses")

    def l3(tr, te):
        keep = fp[tr].mean(0) > 1e-4
        lf = np.log(fp[:, keep] + 1e-6)
        return _lr_rung(lf, y, chosen, "L3")(tr, te)
    res["L3_lr_on_family_posterior"] = score_folds(l3, y, masks, "L3 LR on log family masses")
    X = _features(z, "embedding")
    res["a_lr_embedding"] = score_folds(_lr_rung(X, y, chosen, "a"), y, masks, "(a) LR on the embedding")

    def ncm(tr, te):
        mu = np.stack([X[tr & (y == c)].mean(0) for c in range(len(CATS))])
        mu /= np.linalg.norm(mu, axis=1, keepdims=True)
        return (X[te] @ mu.T).argmax(1)
    res["b_nearest_class_mean"] = score_folds(ncm, y, masks, "(b) nearest class mean")
    res["c_lr_features"] = score_folds(_lr_rung(_features(z, "features"), y, chosen, "c"), y, masks, "(c) LR on features")
    res["C_chosen"] = {k: v for k, v in chosen.items()}
    (OUT / "head.json").write_text(json.dumps(res, indent=1))


def nolabels(_):
    """BT5: L0 (EM prior), temperature, pseudo-label head. Labels are used only to score (and T is fit on Kim's
    training list)."""
    idx, z = _load_all(("embedding", "family_p"))
    y = idx.label.to_numpy()
    masks = fold_masks(idx)
    val = masks["kim"][1]
    site = idx.trap.str[:2].to_numpy()
    lm = category_logmass(z.pop("family_p"))
    p = _softmax(lm)
    res = {}

    def report(tag, pred):
        r = {"kim": {**macro_prf(y[val], pred[val]), "diag": diagnostics(y[val], pred[val])},
             "all": {k: v for k, v in macro_prf(y, pred).items() if k != "per_class"}}
        print(_line(f"{tag} | Kim's list", r["kim"]))
        print(_line(f"{tag} | all crops", r["all"]), flush=True)
        return r
    res["zero"] = report("zero training", p.argmax(1))
    for T_name, T in (("T1", 1.0), ("Ttrain", fit_temperature(lm[masks["kim"][0]], y[masks["kim"][0]]))):
        pt = _softmax(lm / T)
        nll, ece = nll_ece(pt[val], y[val])
        res[f"calibration_{T_name}"] = {"T": T, "nll_val": nll, "ece_val": ece}
        print(f"temperature {T:.3f}: NLL {nll:.4f}, ECE {ece:.4f} on Kim's list")
        pi = em_prior(pt)
        res[f"L0_pooled_{T_name}"] = {**report(f"L0 EM prior, all crops, T {T:.2f}", (pt * pi).argmax(1)),
                                      "prior": pi.tolist()}
        ps = pt.copy()
        for s in ("LV", "OH", "SS"):
            m = site == s
            ps[m] = pt[m] * em_prior(pt[m])
        ps[site == ""] = pt[site == ""] * pi
        res[f"L0_per_site_{T_name}"] = report(f"L0 EM prior, per site, T {T:.2f}", ps.argmax(1))
    # pseudo-labels: (a) fit on the 70 % most confident zero-training answers of the fold's training part
    X = _features(z, "embedding")
    conf, pl = p.max(1), p.argmax(1)

    def pseudo(tr, te):
        rows = np.flatnonzero(tr)
        keep = rows[conf[rows] >= np.quantile(conf[rows], 0.30)]
        C = _choose_c(X[keep], pl[keep])
        return LR(C).fit(X[keep], pl[keep]).predict(X[te])
    res["pseudo_label_head"] = score_folds(pseudo, y, masks, "pseudo-label head (70 % gate)")
    gate = conf >= np.quantile(conf[masks["kim"][0]], 0.30)
    res["pseudo_label_correct_in_gate"] = float((pl[masks["kim"][0] & gate] == y[masks["kim"][0] & gate]).mean())
    print(f"pseudo-labels correct in the gate (Kim's training list): {res['pseudo_label_correct_in_gate']:.3f}")
    (OUT / "nolabels.json").write_text(json.dumps(res, indent=1))


def leak(_):
    """Near-duplicates across each split (nearest training embedding), 1-NN, byte-identical pairs."""
    import torch
    torch.set_num_threads(THREADS)
    idx, z = _load_all(("embedding",))
    X = torch.from_numpy(_features(z, "embedding"))
    y = idx.label.to_numpy()
    masks = fold_masks(idx)

    def nn1(q, ref):
        best, arg = np.full(q.sum(), -2.0, np.float32), np.zeros(q.sum(), int)
        R, ri, qi = X[ref], np.flatnonzero(ref), np.flatnonzero(q)
        for s in range(0, len(qi), 4096):
            v, i = (X[qi[s:s + 4096]] @ R.T).max(1)
            best[s:s + 4096], arg[s:s + 4096] = v.numpy(), ri[i.numpy()]
        return best, arg

    res, cos_site, pred_site, y_site = {}, [], [], []
    for f, (tr, te) in masks.items():
        cos, arg = nn1(te, tr)
        same = y[arg] == y[te]
        r = {"n": int(te.sum()), "cos_ge_0.98": float((cos >= 0.98).mean()), "cos_ge_0.95": float((cos >= 0.95).mean()),
             "cos_ge_0.98_same_class": float(((cos >= 0.98) & same).mean()), "cos_median": float(np.median(cos)),
             "nn1": {k: v for k, v in macro_prf(y[te], y[arg]).items() if k != "per_class"}}
        md5_tr = set(idx.md5[tr])
        r["byte_identical_in_training_part"] = int(idx.md5[te].isin(md5_tr).sum())
        res[f] = r
        if f != "kim":
            cos_site.append(cos), pred_site.append(y[arg]), y_site.append(y[te])
        print(f"{f}: n {r['n']:,}  nearest training crop cos >= 0.98 {r['cos_ge_0.98']:.3f} (same class "
              f"{r['cos_ge_0.98_same_class']:.3f}), >= 0.95 {r['cos_ge_0.95']:.3f}, median {r['cos_median']:.3f}; "
              f"1-NN macro F1 {r['nn1']['f1']:.4f} acc {r['nn1']['accuracy']:.4f}; byte-identical {r['byte_identical_in_training_part']}",
              flush=True)
    c, p_, yy = np.concatenate(cos_site), np.concatenate(pred_site), np.concatenate(y_site)
    res["site_pooled"] = {"cos_ge_0.98": float((c >= 0.98).mean()), "cos_ge_0.95": float((c >= 0.95).mean()),
                          "cos_median": float(np.median(c)),
                          "nn1": {k: v for k, v in macro_prf(yy, p_).items() if k != "per_class"}}
    print("site pooled:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["site_pooled"].items() if k != "nn1"},
          f"1-NN macro F1 {res['site_pooled']['nn1']['f1']:.4f}")
    (OUT / "leak.json").write_text(json.dumps(res, indent=1))


def night_groups(idx):
    """(trap, night) per crop: the evening's date (hours before noon belong to the night before). Crops carry a full
    time stamp; Lepidoptera files only a day of year, so their nights merge across 2022 and 2023 (larger groups)."""
    out = []
    for trap, st in zip(idx.trap, idx.stamp.astype(str)):
        if not trap:
            out.append("")
        elif st.startswith("doy"):
            doy, hour = int(st[3:6]), int(st[7:9])
            out.append(f"{trap}|d{doy - int(hour < 12)}")
        else:
            t = pd.Timestamp(f"{st[:4]}-{st[4:6]}-{st[6:8]}") - pd.Timedelta(days=int(int(st[8:10]) < 12))
            out.append(f"{trap}|{t.date()}")
    return np.array(out)


def night(_):
    """Post-hoc diagnostic (2026-11-20, prediction written first): Kim's 80/20, but by trap-night instead of by crop,
    within the same sites. Separates what his random split shares within a night (the same individuals across
    frames, the same conditions) from what a new site changes."""
    import torch
    torch.set_num_threads(THREADS)
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False, dtype={"stamp": str})
    z = load_shards(("embedding", "family_p"))
    y = idx.label.to_numpy()
    g = night_groups(idx)
    ug = np.unique(g[g != ""])
    test = set(np.random.default_rng(0).choice(ug, size=round(0.2 * len(ug)), replace=False))
    te = np.isin(g, list(test))
    tr = (g != "") & ~te
    X = _features(z, "embedding")
    res = {"groups": int(len(ug)), "n_train": int(tr.sum()), "n_test": int(te.sum())}
    Xt = torch.from_numpy(X)
    R, ri, qi = Xt[tr], np.flatnonzero(tr), np.flatnonzero(te)
    cos, arg = np.empty(len(qi), np.float32), np.empty(len(qi), int)
    for s0 in range(0, len(qi), 4096):
        v, i = (Xt[qi[s0:s0 + 4096]] @ R.T).max(1)
        cos[s0:s0 + 4096], arg[s0:s0 + 4096] = v.numpy(), ri[i.numpy()]
    res["near_duplicates"] = {"cos_ge_0.98": float((cos >= 0.98).mean()), "cos_ge_0.95": float((cos >= 0.95).mean()),
                              "cos_median": float(np.median(cos))}
    res["nn1"] = {k: v for k, v in macro_prf(y[te], y[arg]).items() if k != "per_class"}
    C = json.loads((OUT / "head.json").read_text())["C_chosen"]["a"][0]   # (a)'s C on Kim's fold
    p = LR(C).fit(X[tr], y[tr]).predict(X[te])
    res["a_lr_embedding"] = {**macro_prf(y[te], p), "diag": diagnostics(y[te], p), "C": C}
    lm = category_logmass(z["family_p"])
    res["zero_training_same_crops"] = {k: v for k, v in macro_prf(y[te], lm[te].argmax(1)).items() if k != "per_class"}
    print(f"night split: {len(ug)} trap-nights, train {tr.sum():,}, test {te.sum():,}; nearest training crop cos >= 0.98 "
          f"{res['near_duplicates']['cos_ge_0.98']:.3f}, >= 0.95 {res['near_duplicates']['cos_ge_0.95']:.3f}, median "
          f"{res['near_duplicates']['cos_median']:.3f}")
    print(_line("1-NN, night held out", res["nn1"]))
    print(_line(f"(a) LR on the embedding, night held out, C {C}", res["a_lr_embedding"]))
    print(_line("zero training, same crops", res["zero_training_same_crops"]))
    (OUT / "night.json").write_text(json.dumps(res, indent=1))


# --------------------------------------------------------------------------------------- the GPU job's outputs
GPU = OUT / "gpu"   # /12347837/repos/broad_taxa/out, downloaded


def views(_):
    """BT3: zero training and head (a) under tta3bm and pad-to-square, from dev/117's embeddings (one row per crop,
    index order). The ToL head is scored in chunks straight to the 16 V0 category log masses (species -> category),
    so no 190 k-wide array is kept. Head (a) reuses each fold's C from `head`; one fit on r0 serves r0 and tta3bm
    (the three views' logits averaged), one on padsq serves padsq."""
    import torch
    torch.set_num_threads(THREADS)
    meta = json.loads((GPU / "emb_meta.json").read_text())
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    assert meta["kim_idx"] == list(range(len(idx))), "the job's rows are not in index order"
    y = idx.label.to_numpy()
    masks = fold_masks(idx)
    E = {v: np.load(GPU / f"emb_{v}.npy", mmap_mode="r") for v in meta["views"]}
    tta = ("r0", "rot-30bm_pad25", "rot30bm_pad25")
    cache = OUT / "views_logmass.npz"
    if not cache.exists():
        P = predict_module()
        h = P.load_head(RELEASE / "heads/tol.safetensors")
        D, g, b = (torch.from_numpy(t) for t in (h.direction, h.magnitude, h.bias))
        SM = torch.from_numpy(category_matrix("V0")[h.parents[1][h.parents[0]]])   # species -> category
        sc = math.sqrt(D.shape[1] - 2)
        out = {k: np.empty((len(idx), len(CATS)), np.float32) for k in ("r0", "tta3bm", "padsq")}
        t0 = time.time()
        for st in range(0, len(idx), 256):
            z = {}
            for v in meta["views"]:
                e = torch.from_numpy(np.asarray(E[v][st:st + 256], dtype=np.float32))
                e = e / e.norm(dim=1, keepdim=True)
                z[v] = sc * torch.asin((g * (e @ D.T)).clamp(-1 + 1e-7, 1 - 1e-7)) + b   # predict.score's formula
            pol = {"r0": z["r0"], "padsq": z["padsq"], "tta3bm": sum(z[v] for v in tta) / 3}
            for k, zz in pol.items():
                out[k][st:st + 256] = torch.log(torch.softmax(zz, 1) @ SM + 1e-6).numpy()
            if st % 16384 == 0:
                print(f"views: {st:,}/{len(idx):,} ({time.time() - t0:.0f} s)", flush=True)
        np.savez(cache, **out)
    lms = np.load(cache)
    res = {}
    # the job's r0 against the CPU (ONNX) run: embeddings and the zero-training answer, every crop
    zc = load_shards(("embedding", "family_p"))
    ref = category_logmass(zc["family_p"]).argmax(1)
    cos = np.concatenate([(lambda a, b: (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))(
        np.asarray(E["r0"][s:s + 8192], dtype=np.float32), zc["embedding"][s:s + 8192].astype(np.float32))
        for s in range(0, len(idx), 8192)])
    res["r0_against_cpu"] = {"embedding_cos_min": float(cos.min()), "embedding_cos_median": float(np.median(cos)),
                             "same_category": float((lms["r0"].argmax(1) == ref).mean())}
    print("GPU r0 against the CPU ONNX run:", res["r0_against_cpu"], flush=True)
    del zc
    for k in ("r0", "tta3bm", "padsq"):
        pred = lms[k].argmax(1)
        res[f"zero_{k}"] = score_folds(lambda tr, te, p=pred: p[te], y, masks, f"zero training, {k}")
    Cs = dict(zip(("kim", "LV", "OH", "SS"), json.loads((OUT / "head.json").read_text())["C_chosen"]["a"]))

    def emb(v, rows):
        x = np.asarray(E[v][rows], dtype=np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    fits = {}

    def fold_of(te):
        return next(f for f, (_, t) in masks.items() if t is te)

    def a_pred(tr, te, train_view, test_views):
        f = fold_of(te)
        if (f, train_view) not in fits:
            fits[(f, train_view)] = LR(Cs[f]).fit(emb(train_view, np.flatnonzero(tr)), y[tr])
        m, rows = fits[(f, train_view)], np.flatnonzero(te)
        return np.mean([m.logits(emb(v, rows)) for v in test_views], 0).argmax(1)
    res["a_r0"] = score_folds(lambda tr, te: a_pred(tr, te, "r0", ("r0",)), y, masks, "(a) on the job's r0")
    res["a_tta3bm"] = score_folds(lambda tr, te: a_pred(tr, te, "r0", tta), y, masks, "(a) fit on r0, tested tta3bm")
    res["a_padsq"] = score_folds(lambda tr, te: a_pred(tr, te, "padsq", ("padsq",)), y, masks, "(a) padsq")
    (OUT / "views.json").write_text(json.dumps(res, indent=1))


def gpu(_):
    """BT2 (Kim's recipe) and BT4 (frozen-trunk head) from the job's logits."""
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    y = idx.label.to_numpy()
    res = {}
    for tag, key in (("BT2_kim_recipe", "kim"), ("BT4_frozen_head", "frozen")):
        files = {f: GPU / f"{key}_{f}.npz" for f in ("kim", "LV", "OH", "SS")}
        if not all(p.exists() for p in files.values()):
            print(tag, "missing", [f for f, p in files.items() if not p.exists()])
            continue
        r, yy, pp = {}, [], []
        for f, p in files.items():
            z = np.load(p)
            assert (y[z["kim_idx"]] == z["label"]).all()
            lg = z["logits_last"] if "logits_last" in z.files else z["logits"]
            pred = lg.argmax(1)
            if f == "kim":
                r["kim_last"] = {**macro_prf(z["label"], pred), "diag": diagnostics(z["label"], pred)}
                if "logits_best" in z.files:
                    pb = z["logits_best"].argmax(1)
                    r["kim_best_epoch"] = {**macro_prf(z["label"], pb), "epoch": int(z["best_epoch"]),
                                           "history": json.loads(str(z["history"]))}
                print(_line(f"{tag} | Kim's list, last epoch", r["kim_last"]))
                if "kim_best_epoch" in r:
                    print(_line(f"{tag} | Kim's list, best epoch {r['kim_best_epoch']['epoch']} (his protocol)",
                                r["kim_best_epoch"]))
            else:
                r[f"site_{f}"] = {k: v for k, v in macro_prf(z["label"], pred).items() if k != "per_class"}
                if "history" in z.files:   # the held-out site's score per epoch (reported, never used to choose)
                    h = json.loads(str(z["history"]))
                    r[f"site_{f}"]["f1_by_epoch"] = [round(e["macro_f1"], 4) for e in h]
                    print(f"{tag} | site {f} macro F1 by epoch: min {min(e['macro_f1'] for e in h):.3f}, "
                          f"max {max(e['macro_f1'] for e in h):.3f}, last 5 mean "
                          f"{np.mean([e['macro_f1'] for e in h[-5:]]):.3f}")
                yy.append(z["label"]), pp.append(pred)
        yy, pp = np.concatenate(yy), np.concatenate(pp)
        r["site"] = {**macro_prf(yy, pp), "diag": diagnostics(yy, pp)}
        print(_line(f"{tag} | site held out (last epoch)", r["site"]), flush=True)
        res[tag] = r
    (OUT / "gpu.json").write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["split", "embed", "mapping", "zeroshot", "head", "nolabels", "leak", "night", "views",
                                    "gpu"])
    p.add_argument("--limit", type=int, default=0, help="embed: only the first N crops (a smoke test)")
    a = p.parse_args()
    {"split": split, "embed": embed, "mapping": mapping, "zeroshot": zeroshot, "head": head, "nolabels": nolabels,
     "leak": leak, "night": night, "views": views, "gpu": gpu}[a.step](a)


if __name__ == "__main__":
    main()
