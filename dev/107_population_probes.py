"""Three CPU checks on files already on the workstation (no GPU, a few GB at most).

Journal: journal/2026-11-13-beyond-scaling-what-would-make-a-better-biodiversity-backbone.md
(predictions written there before these ran). RESULTS section 2, group Z.

  observer   Z1: the test fold split by observer (`recordedBy`). S = the observer has training
             images of the same species, O = the observer has training images of other species
             only, N = the observer is absent from training. Species recall per group, paired
             over the species present in both groups (recall decomposes over images, macro-F1
             does not), for L0 (uniform sampling) and singlehead (sqrt-oversampling).
  ghosts     Z2: C3b's saved scores on the 47,905 trap crops, read per group of crops with the
             same predicted species instead of per crop.
  geometry   Z3: how taxonomic the species prototypes are (nearest prototype a congener; mean
             cosine by tree distance), for the released bundles.

  prior      Z1d: on trap crops, do tail-weighted models name rare species that occur at no trap?
  siteprior  Z4: the traps' class prior estimated by EM on unlabelled adapt-night crops (saved
             single-view logits of Nemo and P5), applied on the probe nights.
  presence   Z2c: a label-free presence filter (species with a confident crop on the adapt nights),
             scored on the probe nights in mini_metrics' zero-threshold convention.

Run from the repo root:  .venv/bin/python dev/107_population_probes.py {observer,prior,ghosts,presence,siteprior,geometry}
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"
PREDS = ROOT / "data/ucloud_preds"
OUT = ROOT / "data/population_probes"
RNG = np.random.default_rng(0)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted half."""
    x = np.concatenate([pos, neg])
    ranks = pd.Series(x).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(values: np.ndarray, n: int = 2000) -> tuple[float, float]:
    idx = RNG.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------------------------------------------------------- Z1
def species_preds(run: str) -> pd.DataFrame:
    path = glob.glob(str(PREDS / run / "*/eval/predictions.csv"))[0]
    d = pd.read_csv(path, usecols=["filename", "level", "label", "prediction", "known_label"])
    d = d[(d.level == 0) & d.known_label]
    return pd.DataFrame({"key": d.filename.to_numpy(), "correct": (d.label == d.prediction).to_numpy()})


def observer(args) -> None:
    cols = ["filename", "speciesKey", "set", "recordedBy", "eventDate"]
    m = pd.read_parquet(META, columns=cols)
    m["obs"] = m.recordedBy.astype("string").str.strip().str.lower()
    m.loc[m.obs.isin(["", "nan", "none", "unknown", "anonymous"]), "obs"] = pd.NA
    m["day"] = m.eventDate.astype("string").str.slice(0, 10)
    m["key"] = m.speciesKey.astype(str) + "/" + m.filename.astype(str)
    is_test = m.set.astype(str) == "0"
    train, test = m[~is_test], m[is_test]
    print(f"rows {len(m):,}  train {len(train):,}  test {len(test):,}  sets {sorted(m.set.astype(str).unique())[:12]}")
    print(f"recordedBy missing: train {train.obs.isna().mean():.3f}  test {test.obs.isna().mean():.3f}; "
          f"distinct observers in train {train.obs.nunique():,}")

    n_train = train.groupby("speciesKey").size()
    tr = train.dropna(subset=["obs"])
    obs_all = set(tr.obs.unique())
    pair = set(zip(tr.obs, tr.speciesKey))
    triple = set(zip(tr.obs, tr.speciesKey, tr.day))

    t = test[["key", "speciesKey", "obs", "day"]].copy()
    has = t.obs.notna().to_numpy()
    o, s, d = t.obs.to_numpy(), t.speciesKey.to_numpy(), t.day.to_numpy()
    in_pair = np.fromiter(((a, b) in pair for a, b in zip(o, s)), bool, len(t))
    in_trip = np.fromiter(((a, b, c) in triple for a, b, c in zip(o, s, d)), bool, len(t))
    in_obs = np.fromiter((a in obs_all for a in o), bool, len(t))
    t["group"] = np.select([~has, in_pair, in_obs], ["missing", "S", "O"], default="N")
    t["sameday"] = in_trip & has
    t["n_train"] = t.speciesKey.map(n_train).fillna(0).astype(int)
    t["rare"] = t.n_train < 200

    runs = {"L0": "L0-eval", "singlehead": "singlehead"}
    for name, run in runs.items():
        p = species_preds(run)
        t = t.merge(p.rename(columns={"correct": name}), on="key", how="left")
    print(f"test images with predictions: {t.L0.notna().sum():,} (L0), {t.singlehead.notna().sum():,} (singlehead)")
    t = t.dropna(subset=list(runs))
    for name in runs:
        t[name] = t[name].astype(float)

    print("\nshare of test images per group, and image-weighted accuracy")
    g = t.groupby("group").agg(images=("key", "size"), species=("speciesKey", "nunique"),
                               L0=("L0", "mean"), singlehead=("singlehead", "mean"))
    g["share"] = g.images / g.images.sum()
    print(g.round(4).to_string())
    sd = t[t.group == "S"].groupby("sameday").agg(images=("key", "size"), L0=("L0", "mean"),
                                                   singlehead=("singlehead", "mean"))
    print("\nwithin S, by whether the observer has a training image of the species on the same day")
    print(sd.round(4).to_string())

    def paired(a: str, b: str, sub: pd.DataFrame, label: str) -> dict:
        ra = sub[sub.group == a].groupby("speciesKey")[list(runs)].mean()
        rb = sub[sub.group == b].groupby("speciesKey")[list(runs)].mean()
        both = ra.index.intersection(rb.index)
        ra, rb = ra.loc[both], rb.loc[both]
        out = {"contrast": f"{a} vs {b}", "subset": label, "species": len(both)}
        for name in runs:
            diff = (ra[name] - rb[name]).to_numpy()
            lo, hi = boot_ci(diff)
            out[f"{name}_{a}"] = ra[name].mean()
            out[f"{name}_{b}"] = rb[name].mean()
            out[f"{name}_diff"] = diff.mean()
            out[f"{name}_ci"] = f"[{lo:+.4f}, {hi:+.4f}]"
        gain_a = (ra.singlehead - ra.L0).to_numpy()
        gain_b = (rb.singlehead - rb.L0).to_numpy()
        lo, hi = boot_ci(gain_a - gain_b)
        out.update(gain_a=gain_a.mean(), gain_b=gain_b.mean(), gain_diff=(gain_a - gain_b).mean(),
                   gain_diff_ci=f"[{lo:+.4f}, {hi:+.4f}]")
        return out

    rows = []
    for label, sub in [("all species", t), ("rare (<200 train images)", t[t.rare]),
                       ("common (>=200)", t[~t.rare])]:
        for a, b in [("S", "N"), ("S", "O"), ("O", "N")]:
            rows.append(paired(a, b, sub, label))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\npaired macro recall over the species present in both groups (bootstrap over species)")
    print(res.round(4).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "z1_observer.csv", index=False)
    g.to_csv(OUT / "z1_groups.csv")


# ----------------------------------------------------------------------------- Z2
def ghosts(args) -> None:
    sc = pd.read_parquet(PREDS / "trap-openset/trap-openset-C3b.parquet")
    sc["msp"] = -sc.msp  # stored as a novelty score (higher = more novel)
    taxa = json.load(open(ROOT / "data/trap_openset/c3b_taxa.json"))
    near, mid = set(taxa["near"]), set(taxa["mid"])
    path = glob.glob(str(PREDS / "C3b-shift/*/eval/predictions.csv"))[0]
    p = pd.read_csv(path, usecols=["filename", "level", "prediction", "confidence"])
    p = p[p.level == 0].rename(columns={"filename": "image_path"})
    d = sc.merge(p, on="image_path", how="inner")
    assert len(d) == len(sc), (len(d), len(sc))
    assert np.allclose(d.msp, d.confidence, atol=2e-3), "scores and predictions disagree"
    d["label"] = d.label.astype(str)
    d["pred"] = d.prediction.astype(str)
    d["stratum"] = np.where(~d.novel, "known", np.where(d.label.isin(near), "near",
                            np.where(d.label.isin(mid), "mid", "oov")))
    d["species_folder"] = d.image_path.str.split("/").str[0]
    d["trap"] = d.image_path.str.extract(r"TRAPNAME_([A-Z]+\d)_")[0]
    print(d.stratum.value_counts().to_string())
    kn, nv = d[~d.novel], d[d.novel]
    print(f"\nper crop AUROC (msp): all novel {auroc(-nv.msp.to_numpy(), -kn.msp.to_numpy()):.4f}  "
          f"near {auroc(-d[d.stratum == 'near'].msp.to_numpy(), -kn.msp.to_numpy()):.4f}")

    def groups(keys: list[str], min_n: int) -> pd.DataFrame:
        rows = []
        for k, sub in d.groupby(keys):
            if len(sub) < min_n:
                continue
            pred = k[0] if isinstance(k, tuple) else k
            rows.append({"key": k, "n": len(sub), "share_true": (sub.label == pred).mean(),
                         "share_novel": sub.novel.mean(), "q90": sub.msp.quantile(0.9),
                         "mean": sub.msp.mean(), "frac90": (sub.msp > 0.9).mean(),
                         "top_folder": sub.species_folder.mode().iat[0]})
        g = pd.DataFrame(rows)
        g["kind"] = np.where(g.share_true > 0.5, "present", np.where(g.share_novel > 0.5, "ghost", "confusion"))
        return g

    for keys, title in [(["pred"], "groups by predicted species, all traps pooled"),
                        (["pred", "trap"], "groups by (predicted species, trap)")]:
        g = groups(keys, 10)
        print(f"\n== {title} (n >= 10): {g.kind.value_counts().to_dict()}")
        pres, gh, conf = g[g.kind == "present"], g[g.kind == "ghost"], g[g.kind == "confusion"]
        for stat in ["q90", "mean", "frac90"]:
            line = f"  {stat:7s}"
            if len(gh):
                a = auroc(-gh[stat].to_numpy(), -pres[stat].to_numpy())
                boots = []
                for _ in range(1000):
                    bg = gh[stat].to_numpy()[RNG.integers(0, len(gh), len(gh))]
                    bp = pres[stat].to_numpy()[RNG.integers(0, len(pres), len(pres))]
                    boots.append(auroc(-bg, -bp))
                line += f" ghost vs present {a:.3f} [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]"
            if len(conf):
                line += f" | confusion vs present {auroc(-conf[stat].to_numpy(), -pres[stat].to_numpy()):.3f}"
                both = pd.concat([gh, conf])
                line += f" | not-present vs present {auroc(-both[stat].to_numpy(), -pres[stat].to_numpy()):.3f}"
            print(line)
        print("  medians  present q90 %.3f mean %.3f | ghost q90 %.3f mean %.3f | confusion q90 %.3f mean %.3f" % (
            pres.q90.median(), pres["mean"].median(), gh.q90.median() if len(gh) else np.nan,
            gh["mean"].median() if len(gh) else np.nan, conf.q90.median() if len(conf) else np.nan,
            conf["mean"].median() if len(conf) else np.nan))
        if keys == ["pred"]:
            print(g[g.kind == "ghost"].sort_values("n", ascending=False).round(3).to_string(index=False))
            OUT.mkdir(parents=True, exist_ok=True)
            g.to_csv(OUT / "z2_groups.csv", index=False)

    print("\nwhere the novel crops go (predicted species per novel species folder, top 3)")
    for folder, sub in nv.groupby("species_folder"):
        top = sub.pred.value_counts().head(3)
        print(f"  {folder:32s} n={len(sub):4d} {sub.stratum.iat[0]:5s} " +
              ", ".join(f"{k}:{v}" for k, v in top.items()))

    print("\nZ2b bags: mean msp of n crops, near-novel bags against known bags (5,000 each)")
    from statistics import NormalDist
    nd = NormalDist()
    near_s, known_s = d[d.stratum == "near"].msp.to_numpy(), kn.msp.to_numpy()
    a1 = auroc(-near_s, -known_s)
    for n in [1, 3, 10, 30, 100]:
        bn = near_s[RNG.integers(0, len(near_s), (5000, n))].mean(axis=1)
        bk = known_s[RNG.integers(0, len(known_s), (5000, n))].mean(axis=1)
        print(f"  n={n:3d}  AUROC {auroc(-bn, -bk):.3f}   independence curve {nd.cdf(np.sqrt(n) * nd.inv_cdf(a1)):.3f}")
    # the deployable version: a bag is drawn inside one predicted species
    print("  bags inside one predicted species (classes with >= 30 near-novel and >= 30 known crops)")
    for pred, sub in d.groupby("pred"):
        a, b = sub[sub.stratum == "near"].msp.to_numpy(), sub[~sub.novel].msp.to_numpy()
        if len(a) < 30 or len(b) < 30:
            continue
        out = [f"n=1 {auroc(-a, -b):.3f}"]
        for n in [10, 30]:
            ba = a[RNG.integers(0, len(a), (3000, n))].mean(axis=1)
            bb = b[RNG.integers(0, len(b), (3000, n))].mean(axis=1)
            out.append(f"n={n} {auroc(-ba, -bb):.3f}")
        print(f"    pred {pred}: near {len(a)}, known {len(b)} (of which truly this species "
              f"{(sub[~sub.novel].label == pred).mean():.2f}); " + ", ".join(out))


# ----------------------------------------------------------------------------- Z1d, Z2c
def trap_preds(run: str) -> pd.DataFrame:
    path = glob.glob(str(PREDS / run / "*/eval/predictions.csv"))[0]
    d = pd.read_csv(path, usecols=["filename", "level", "label", "prediction", "confidence"])
    d = d[d.level == 0].rename(columns={"filename": "image_path"})
    g = pd.read_parquet(ROOT / "data/flemming/splits/flemming_groups.parquet",
                        columns=["image_path", "trap", "split"])
    d = d.merge(g, on="image_path", how="left")
    d["label"], d["pred"] = d.label.astype(str), d.prediction.astype(str)
    return d


def prior(args) -> None:
    """Z1d: do tail-weighted models name rare, absent species on trap crops?"""
    m = pd.read_parquet(META, columns=["speciesKey", "set"])
    n_train = m[m.set.astype(str) != "0"].groupby("speciesKey").size()
    n_train.index = n_train.index.astype(str)
    rows = []
    for name, run in [("L0 (uniform)", "L0-shift"), ("singlehead (sqrt-oversampling)", "base-shift"),
                      ("L1 (balanced softmax)", "L1-shift"), ("L2 (both)", "L2-shift")]:
        d = trap_preds(run)
        at_traps = set(d.label.unique())
        absent = ~d.pred.isin(at_traps)
        rare = d.pred.map(n_train).fillna(0) < 200
        wrong = d.pred != d.label
        rows.append({"model": name, "accuracy": 1 - wrong.mean(), "wrong": wrong.mean(),
                     "pred absent species": absent.mean(), "absent and rare": (absent & rare).mean(),
                     "absent and common": (absent & ~rare).mean(),
                     "wrong, species at traps": (wrong & ~absent).mean(),
                     "distinct species predicted": d.pred.nunique(),
                     "distinct absent species predicted": d.pred[absent].nunique()})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"trap crops {len(d):,}, species at the traps {len(at_traps)}")
    print(res.round(4).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "z1d_prior.csv", index=False)


def f1_all(label: np.ndarray, pred: np.ndarray, keep: np.ndarray) -> tuple[float, float, int]:
    """Macro-F1 over labelled and predicted species (dropped predictions name nothing), macro
    recall over the labelled species, and the number of species predicted."""
    df = pd.DataFrame({"label": label, "pred": np.where(keep, pred, "__none__")})
    tp = df[df.label == df.pred].groupby("label").size()
    n_lab = df.groupby("label").size()
    n_pred = df[df.pred != "__none__"].groupby("pred").size()
    idx = n_lab.index.union(n_pred.index)
    tp, nl, npd = tp.reindex(idx, fill_value=0), n_lab.reindex(idx, fill_value=0), n_pred.reindex(idx, fill_value=0)
    f1 = (2 * tp / (nl + npd)).fillna(0)
    return float(f1.mean()), float((tp / nl)[nl > 0].mean()), int((npd > 0).sum())


def presence(args) -> None:
    """Z2c: a label-free presence filter from the adapt nights, scored on the probe nights."""
    rows = []
    for name, run in [("C3b (plain head)", "C3b-shift"), ("B1 (margin, trap aug)", "B1-shift"),
                      ("L0 (plain, uniform)", "L0-shift"), ("singlehead", "base-shift")]:
        d = trap_preds(run)
        ad, pr = d[d.split == "adapt"], d[d.split == "probe"]
        lab, pred, conf = pr.label.to_numpy(), pr.pred.to_numpy(), pr.confidence.to_numpy()
        core = ad.groupby("pred").confidence.max()
        present = set(core[core >= 0.9].index)
        keep_presence = np.isin(pred, list(present))
        # the labelled alternative: one confidence threshold chosen on the adapt nights
        best_t, best = 0.0, -1.0
        for t in np.linspace(0, 0.99, 34):
            f, _, _ = f1_all(ad.label.to_numpy(), ad.pred.to_numpy(), ad.confidence.to_numpy() >= t)
            if f > best:
                best_t, best = t, f
        for rule, keep in [("none", np.ones(len(pr), bool)),
                           ("presence filter (no labels)", keep_presence),
                           (f"threshold {best_t:.2f} (fitted with adapt labels)", conf >= best_t),
                           ("both", keep_presence & (conf >= best_t))]:
            f, r, n = f1_all(lab, pred, keep)
            rows.append({"model": name, "rule": rule, "macro-F1 (labelled + predicted)": f,
                         "macro recall (labelled)": r, "species predicted": n,
                         "crops answered": keep.mean(), "accuracy on answered": (lab == pred)[keep].mean()})
        rows[-4]["species labelled"] = len(set(lab))
        rows[-3]["present set"] = len(present)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 280)
    print(res.round(4).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "z2c_presence.csv", index=False)


# ----------------------------------------------------------------------------- Z4
LOGITS = Path("/data/au761367/otto_eval/logits")  # single-view logits saved by the TTA study


def softmax_rows(z, idx: np.ndarray) -> np.ndarray:
    out = np.empty((len(idx), z.shape[1]), np.float32)
    for i in range(0, len(idx), 4096):
        a = np.asarray(z[idx[i:i + 4096]], dtype=np.float32)
        a -= a.max(axis=1, keepdims=True)
        np.exp(a, out=a)
        a /= a.sum(axis=1, keepdims=True)
        out[i:i + 4096] = a
    return out


def em_prior(p: np.ndarray, iters: int = 50, eps: float = 0.0) -> np.ndarray:
    """Maximum-likelihood class prior of an unlabelled set, the model's posteriors held fixed."""
    n, c = p.shape
    pi = np.full(c, 1.0 / c)
    for _ in range(iters):
        s = np.zeros(c)
        for i in range(0, n, 4096):
            q = p[i:i + 4096] * pi
            q /= q.sum(axis=1, keepdims=True)
            s += q.sum(axis=0)
        pi = (s + eps) / (n + c * eps)
    return pi


def siteprior(args) -> None:
    lab = trap_preds("C3b-shift")[["image_path", "label", "trap", "split"]]
    g = pd.read_parquet(ROOT / "data/flemming/splits/flemming_groups.parquet",
                        columns=["image_path", "species_heldout"])
    lab = lab.merge(g, on="image_path")
    rows = []
    for name, folder in [("Nemo", "nemo17"), ("P5", "p517")]:
        keys = np.array(json.load(open(LOGITS / folder / "keys.json")))
        meta = pd.DataFrame({"image_path": json.load(open(LOGITS / folder / "crops.json"))})
        meta = meta.merge(lab, on="image_path", how="left")
        assert meta.label.notna().all()
        z = np.load(LOGITS / folder / "r0.npy", mmap_mode="r")
        site = meta.trap.str[:2].to_numpy()
        is_probe = (meta.split == "probe").to_numpy()
        pool = ~is_probe                      # every crop of the adapt nights, unlabelled
        pool_adapt = (meta.split == "adapt").to_numpy()  # without the held-out species' crops
        y = meta.label.to_numpy()
        ho = meta.species_heldout.to_numpy()
        i_probe = np.where(is_probe)[0]
        pp = softmax_rows(z, i_probe)
        yp, hop, sitep = y[i_probe], ho[i_probe], site[i_probe]

        def score(rule: str, prior: np.ndarray | None, mask: np.ndarray | None = None) -> None:
            q = pp if prior is None else pp * prior.astype(np.float32)
            pred = keys[q.argmax(axis=1)]
            m = np.ones(len(yp), bool) if mask is None else mask
            f, r, n = f1_all(yp[m], pred[m], np.ones(m.sum(), bool))
            hit = pred == yp
            rows.append({"model": name, "rule": rule, "crops": int(m.sum()), "accuracy": hit[m].mean(),
                         "macro-F1 (labelled + predicted)": f, "macro recall (labelled)": r,
                         "species predicted": n, "accuracy, held-out species": hit[m & hop].mean(),
                         "accuracy, other species": hit[m & ~hop].mean()})

        score("no prior", None)
        p_pool = softmax_rows(z, np.where(pool)[0])
        pi = em_prior(p_pool)
        score("EM prior, all adapt-night crops", pi)
        print(f"{name}: EM prior has {(pi > 1e-6).sum()} species above 1e-6, "
              f"{(pi > 1.0 / len(p_pool)).sum()} above one crop's worth; labelled species on those nights "
              f"{len(set(y[pool]))}")
        score("EM prior, smoothed (0.01 pseudo-crop per class)", em_prior(p_pool, eps=0.01))
        counts = pd.Series(y[pool]).value_counts()
        oracle = pd.Series(counts, index=keys).fillna(0).to_numpy() + 1e-9
        score("oracle prior (true labels of the adapt nights)", oracle / oracle.sum())
        del p_pool
        score("EM prior, adapt split only (held-out species' crops removed)",
              em_prior(softmax_rows(z, np.where(pool_adapt)[0])))
        for s in ["LV", "OH", "SS"]:
            m = sitep == s
            score(f"site {s}: no prior", None, m)
            score(f"site {s}: EM prior from the two other sites",
                  em_prior(softmax_rows(z, np.where(pool & (site != s))[0])), m)
            score(f"site {s}: EM prior from its own adapt nights",
                  em_prior(softmax_rows(z, np.where(pool & (site == s))[0])), m)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print(res.round(4).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "z4_siteprior.csv", index=False)


# ----------------------------------------------------------------------------- Z3
def geometry(args) -> None:
    from safetensors import safe_open
    rows = []
    for name, folder in [("B3rep5x (V2-S, margin, no marginal loss)", "b3"),
                         ("B8 (DINOv3 ConvNeXt-L, margin, marginal loss)", "b8"),
                         ("P5 (BioCLIP-2, three heads)", "p5v2")]:
        base = ROOT / "data/hf_release" / folder
        tax = json.load(open(base / "taxonomy.json"))
        s2g = np.array(tax["parents"]["species_to_genus"])
        g2f = np.array(tax["parents"]["genus_to_family"])
        s2f = g2f[s2g]
        with safe_open(str(base / "model.safetensors"), "np") as f:
            w = f.get_tensor("1.head.layers.0.parametrizations.weight.original1").astype(np.float32)
            g = f.get_tensor("1.head.layers.0.parametrizations.weight.original0").astype(np.float32).ravel()
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        c = len(w)
        genus_size = np.bincount(s2g)[s2g]
        has_cong = genus_size > 1
        nn_cong = np.zeros(c, bool)
        nn_fam = np.zeros(c, bool)
        sums = np.zeros(3)
        counts = np.zeros(3)
        nn_cos = np.zeros(c, np.float32)
        for i in range(0, c, 1024):
            cos = w[i:i + 1024] @ w.T
            idx = np.arange(i, min(i + 1024, c))
            cos[np.arange(len(idx)), idx] = -2.0
            j = cos.argmax(axis=1)
            nn_cos[idx] = cos.max(axis=1)
            nn_cong[idx] = s2g[j] == s2g[idx]
            nn_fam[idx] = s2f[j] == s2f[idx]
            same_g = s2g[idx][:, None] == s2g[None, :]
            same_f = s2f[idx][:, None] == s2f[None, :]
            valid = cos > -1.5
            for k, mask in enumerate([same_g & valid, same_f & ~same_g, ~same_f]):
                sums[k] += cos[mask].sum()
                counts[k] += mask.sum()
        chance = ((genus_size[has_cong] - 1) / (c - 1)).mean()
        rows.append({"model": name, "dim": w.shape[1], "row scale mean": g.mean(),
                     "nearest is congener": nn_cong[has_cong].mean(), "chance": chance,
                     "nearest in family": nn_fam.mean(), "nearest cos": nn_cos.mean(),
                     "cos congener": sums[0] / counts[0], "cos same family": sums[1] / counts[1],
                     "cos other": sums[2] / counts[2], "species with a congener": int(has_cong.sum())})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(4).T.to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "z3_geometry.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for fn in (observer, prior, ghosts, presence, siteprior, geometry):
        sub.add_parser(fn.__name__).set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)
