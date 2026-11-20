"""C3, the order half: the frozen ToL trunk on the AMI broad-taxon dataset (MCC24; Bjerge, Karstoft, Hoye 2025)
against Kim Bjerge's specialist (ResNet50v2 at 128 px). CPU only, ONNX Runtime, at most 4 threads.

    .venv/bin/python -I dev/115_broad_taxa.py split      # Kim's 80/20 split recovered from his published logits; his metrics
    .venv/bin/python -I dev/115_broad_taxa.py embed      # ToL trunk + head on every crop (his validation split first), resumable
    .venv/bin/python -I dev/115_broad_taxa.py mapping    # ToL families -> the 16 categories, the pre-registered variants
    .venv/bin/python -I dev/115_broad_taxa.py zeroshot   # no training: category masses from the hierarchical marginals
    .venv/bin/python -I dev/115_broad_taxa.py head       # heads on the frozen embedding, fit on Kim's train split only
    .venv/bin/python -I dev/115_broad_taxa.py leak       # how close each validation crop is to a training crop

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
    """Every finished shard, rows in Kim's full-set order (kim_idx); missing rows absent."""
    parts = {c: [] for c in ("kim_idx",) + tuple(cols)}
    for f in sorted((OUT / "emb").glob("shard_*.npz")):
        with np.load(f) as z:
            for c in parts:
                parts[c].append(z[c])
    arr = {c: np.concatenate(v) for c, v in parts.items()}
    o = np.argsort(arr["kim_idx"])
    return {c: v[o] for c, v in arr.items()}


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
    fp = z["family_p"].astype(np.float32)
    res = {"n_embedded": int(len(rows)), "n_val_embedded": int(val.sum())}
    yk, pk = kim_val()
    res["kim_val"] = {**macro_prf(yk, pk), "diag": diagnostics(yk, pk)}
    print(_line("Kim's ResNet50, his validation", res["kim_val"]))
    for v in VARIANTS:
        fc, _ = family_category(labels, v)
        M = np.zeros((len(labels), len(CATS)), np.float32)
        M[np.flatnonzero(fc >= 0), fc[fc >= 0]] = 1
        mass = fp @ M
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
            top = np.argsort(-fp[:, fc < 0].sum(0))[:12]
            fams = np.flatnonzero(fc < 0)[top]
            res["V0_set_aside_families"] = {labels[i]: float(fp[:, i].mean()) for i in fams}
    (OUT / "zeroshot.json").write_text(json.dumps(res, indent=1))
    r = res["V0_val"]["per_class"]
    print("per class, V0 on Kim's validation (F1; Kim's F1):")
    kp = res["kim_val"]["per_class"]
    for c in CATS:
        print(f"  {c:22s} {r[c]['f1']:.3f}  {kp[c]['f1']:.3f}  (P {r[c]['precision']:.3f} R {r[c]['recall']:.3f}, n {r[c]['support']})")


def _features(z, key):
    x = z[key].astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True) if key == "embedding" else x


def _fit_lr(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, tol=1e-4)).fit(X, y)


def _choose_c(X, y, seed=0):
    hold = np.random.default_rng(seed).random(len(y)) < 0.1
    scores = {}
    for C in (0.1, 1, 10, 100):
        m = _fit_lr(X[~hold], y[~hold], C)
        scores[C] = macro_prf(y[hold], m.predict(X[hold]))["f1"]
        print(f"  C {C:6}: macro F1 on 10 % of the training list {scores[C]:.4f}", flush=True)
    return max(scores, key=scores.get), scores


def head(_):
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    z = load_shards(("embedding", "features"))
    assert len(z["kim_idx"]) == len(idx), f"embed is not finished: {len(z['kim_idx'])} of {len(idx)} rows"
    y = idx.label.to_numpy()
    tr, va = (idx.split == "train").to_numpy(), (idx.split == "val").to_numpy()
    res = {}
    for name, key in (("a_lr_embedding", "embedding"), ("c_lr_features", "features")):
        X = _features(z, key)
        t0 = time.time()
        C, grid = _choose_c(X[tr], y[tr])
        m = _fit_lr(X[tr], y[tr], C)
        p = m.predict(X[va])
        res[name] = {**macro_prf(y[va], p), "diag": diagnostics(y[va], p), "C": C, "grid": grid,
                     "fit_seconds": time.time() - t0}
        print(_line(f"({name[0]}) LR on {key}, C {C}", res[name]), flush=True)
        if key == "embedding":
            np.save(OUT / "head_a_val_pred.npy", p)
    X = _features(z, "embedding")
    mu = np.stack([X[tr & (y == c)].mean(0) for c in range(len(CATS))])
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    p = (X[va] @ mu.T).argmax(1)
    res["b_nearest_class_mean"] = {**macro_prf(y[va], p), "diag": diagnostics(y[va], p)}
    print(_line("(b) nearest class mean", res["b_nearest_class_mean"]))
    res.update(ladder(idx, y, tr, va))
    (OUT / "head.json").write_text(json.dumps(res, indent=1))


def ladder(idx, y, tr, va):
    """C5's decomposition (added after the zero-training score, predictions written first): from the frozen ToL
    head's own output to a head on the embedding, each rung fit on Kim's training list only.
    L1: 16 biases on the zero-training log masses (class prior only); L2: a linear map of the 16 log masses
    (the mapping relearned); L3: logistic regression on the log family posterior (5,039 families, those with mean
    mass above 1e-4 on the training list)."""
    import torch
    labels = json.loads((RELEASE / "heads/tol.json").read_text())["labels"][2]
    z = load_shards(("family_p",))
    fp = z["family_p"].astype(np.float32)
    fc, _ = family_category(labels, "V0")
    M = np.zeros((len(labels), len(CATS)), np.float32)
    M[np.flatnonzero(fc >= 0), fc[fc >= 0]] = 1
    lm = np.log(fp @ M + 1e-6)
    out = {}
    t = torch.from_numpy(lm[tr])
    b = torch.zeros(len(CATS), requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=200)
    yt = torch.from_numpy(y[tr])

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(t + b, yt)
        loss.backward()
        return loss
    opt.step(closure)
    p = (lm[va] + b.detach().numpy()).argmax(1)
    out["L1_bias_only"] = {**macro_prf(y[va], p), "diag": diagnostics(y[va], p), "bias": b.detach().tolist()}
    print(_line("L1 zero training + 16 biases", out["L1_bias_only"]), flush=True)
    C, grid = _choose_c(lm[tr], y[tr])
    p = _fit_lr(lm[tr], y[tr], C).predict(lm[va])
    out["L2_linear_on_16_masses"] = {**macro_prf(y[va], p), "diag": diagnostics(y[va], p), "C": C}
    print(_line(f"L2 linear on 16 log masses, C {C}", out["L2_linear_on_16_masses"]), flush=True)
    keep = fp[tr].mean(0) > 1e-4
    lf = np.log(fp[:, keep] + 1e-6)
    C, grid = _choose_c(lf[tr], y[tr])
    p = _fit_lr(lf[tr], y[tr], C).predict(lf[va])
    out["L3_lr_on_family_posterior"] = {**macro_prf(y[va], p), "diag": diagnostics(y[va], p), "C": C,
                                        "families": int(keep.sum())}
    print(_line(f"L3 LR on {keep.sum()} log family masses, C {C}", out["L3_lr_on_family_posterior"]), flush=True)
    return out


def leak(_):
    """1-NN from Kim's training list to his validation list; the head trained on two sites, scored on the third."""
    import torch
    torch.set_num_threads(THREADS)
    idx = pd.read_csv(OUT / "index.csv", keep_default_na=False)
    z = load_shards(("embedding",))
    assert len(z["kim_idx"]) == len(idx)
    X = torch.from_numpy(_features(z, "embedding"))
    y = idx.label.to_numpy()
    site = idx.trap.str[:2].to_numpy()
    tr, va = (idx.split == "train").to_numpy(), (idx.split == "val").to_numpy()
    res = {}

    def nn1(q, ref):
        best, arg = np.full(q.sum(), -2.0, np.float32), np.zeros(q.sum(), int)
        R = X[ref]
        ri = np.flatnonzero(ref)
        for s in range(0, q.sum(), 4096):
            sim = X[np.flatnonzero(q)[s:s + 4096]] @ R.T
            v, i = sim.max(1)
            best[s:s + 4096], arg[s:s + 4096] = v.numpy(), ri[i.numpy()]
        return best, arg

    cos, arg = nn1(va, tr)
    p = y[arg]
    res["nn1_random_split"] = {**macro_prf(y[va], p), "cos_ge_0.95": float((cos >= 0.95).mean()),
                               "cos_ge_0.90": float((cos >= 0.90).mean()), "cos_median": float(np.median(cos))}
    print(_line("1-NN, Kim's train -> val", res["nn1_random_split"]), f" nearest cos median {np.median(cos):.3f}, "
          f">=0.95 {(cos >= 0.95).mean():.3f}")
    head_json = json.loads((OUT / "head.json").read_text())
    C = head_json["a_lr_embedding"]["C"]
    pred = np.full(len(y), -1)
    cos_site = np.full(len(y), np.nan, np.float32)
    Xn = X.numpy()
    for s in ("LV", "OH", "SS"):
        te, fit = site == s, (site != s) & (site != "")
        m = _fit_lr(Xn[fit], y[fit], C)
        pred[te] = m.predict(Xn[te])
        cos_site[te] = nn1(te, fit)[0]
        r = macro_prf(y[te], pred[te])
        res[f"site_{s}"] = r
        print(_line(f"(a) trained without {s}, scored on {s}", r), flush=True)
    ok = pred >= 0
    res["site_pooled"] = {**macro_prf(y[ok], pred[ok]), "diag": diagnostics(y[ok], pred[ok]),
                          "nn_cos_median": float(np.nanmedian(cos_site)),
                          "nn_cos_ge_0.95": float((cos_site[ok] >= 0.95).mean())}
    print(_line("(a) site held out, pooled", res["site_pooled"]))
    (OUT / "leak.json").write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["split", "embed", "mapping", "zeroshot", "head", "leak"])
    p.add_argument("--limit", type=int, default=0, help="embed: only the first N crops (a smoke test)")
    a = p.parse_args()
    {"split": split, "embed": embed, "mapping": mapping, "zeroshot": zeroshot, "head": head, "leak": leak}[a.step](a)


if __name__ == "__main__":
    main()
