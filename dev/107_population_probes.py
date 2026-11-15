"""CPU checks on files already on the workstation (no GPU, a few GB at most).

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

  siteprior2 Z5: the same prior on Nemo, P5 and OT3, one view and tta3bm, against the north_europe
             list, with the training prior divided out, with a temperature, and per site and season.
  siteprior3 Z5g, Z8: list, EM, both, and the EM inside the list, overall and per site, for any
             folder of saved logits (--folders name=path,...).
  observers  Z6: the observer split on every saved test-fold prediction file, next to each run's
             probe accuracy.
  congeners  Z7: Z3's statistic on the campaign's heads (checkpoints in data/campaign_weights).
  severity   Z7b: among wrong species answers, the share naming a congener, per head.

Run from the repo root:  .venv/bin/python dev/107_population_probes.py {observer,prior,ghosts,presence,siteprior,geometry,siteprior2,siteprior3,observers,congeners,severity}
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


def observer_groups() -> pd.DataFrame:
    """One row per test image: its key, species, observer group (S, O, N or missing)."""
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
    return t


def observer(args) -> None:
    t = observer_groups()
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


def em_prior(p: np.ndarray, iters: int = 50, eps: float = 0.0, source: np.ndarray | None = None,
             base: np.ndarray | None = None, m: float = 0.0) -> np.ndarray:
    """Maximum-likelihood class prior of an unlabelled set, the model's posteriors held fixed
    (Saerens et al. 2002). `source` is the model's training prior; None means uniform, which is
    what Z4 assumed. `eps` adds that many pseudo-crops to every class; `base` and `m` shrink toward
    a base prior with the weight of m crops. Each iteration is two matrix-vector products."""
    n, c = p.shape
    src = np.full(c, 1.0 / c) if source is None else source
    extra = np.full(c, float(eps)) + (m * base if base is not None else 0.0)
    pi = np.full(c, 1.0 / c) if base is None else base.copy()
    for _ in range(iters):
        r = (pi / src).astype(np.float32)
        d = np.maximum(p @ r, 1e-30)
        s = r.astype(np.float64) * (p.T @ (1.0 / d).astype(np.float32))
        pi = (s + extra) / (s.sum() + extra.sum())
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


# ----------------------------------------------------------------------------- Z5
NE_LIST = Path.home() / "codes/mini_trainer/dev/releases/mambo_v3/presets/north_europe.classes"
TTA3BM = ["r0", "rot-30bm_pad25", "rot30bm_pad25"]
GROUPS = ROOT / "data/flemming/splits/flemming_groups.parquet"


def posteriors(folder: str, views: list[str], rows: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """softmax(mean logits over the views / temp) for the given crops, float32."""
    zs = [np.load(LOGITS / folder / f"{v}.npy", mmap_mode="r") for v in views]
    out = np.empty((len(rows), zs[0].shape[1]), np.float32)
    for i in range(0, len(rows), 2048):
        idx = rows[i:i + 2048]
        a = np.asarray(zs[0][idx], dtype=np.float32)
        for z in zs[1:]:
            a += z[idx]
        a *= 1.0 / (len(zs) * temp)
        a -= a.max(axis=1, keepdims=True)
        np.exp(a, out=a)
        a /= a.sum(axis=1, keepdims=True)
        out[i:i + 2048] = a
    return out


def top1(p: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:
        return p.argmax(axis=1)
    w = (w / w.max()).astype(np.float32)
    out = np.empty(len(p), np.int64)
    for i in range(0, len(p), 4096):
        out[i:i + 4096] = (p[i:i + 4096] * w).argmax(axis=1)
    return out


def siteprior2(args) -> None:
    lab = trap_preds("C3b-shift")[["image_path", "label", "trap", "split"]]
    lab = lab.merge(pd.read_parquet(GROUPS, columns=["image_path", "species_heldout", "night"]), on="image_path")
    m = pd.read_parquet(META, columns=["speciesKey", "set"])
    n_train = m[m.set.astype(str) != "0"].groupby("speciesKey").size()
    n_train.index = n_train.index.astype(str)
    ne = set(open(NE_LIST).read().split())
    rows = []
    models = [("Nemo", "nemo17"), ("P5", "p517"), ("OT3", "ot3")]
    if args.models:
        models = [x for x in models if x[0] in args.models.split(",")]
    for name, folder in models:
        keys = np.array(json.load(open(LOGITS / folder / "keys.json")))
        meta = pd.DataFrame({"image_path": json.load(open(LOGITS / folder / "crops.json"))})
        meta = meta.merge(lab, on="image_path", how="left")
        assert meta.label.notna().all()
        is_probe = (meta.split == "probe").to_numpy()
        order = np.concatenate([np.where(~is_probe)[0], np.where(is_probe)[0]])  # the pool first
        n_pool = int((~is_probe).sum())
        y = meta.label.to_numpy()[order]
        site = meta.trap.str[:2].to_numpy()[order]
        day = pd.to_datetime(meta.night).map(pd.Timestamp.toordinal).to_numpy()[order]
        ho = meta.species_heldout.to_numpy()[order]
        yp, hop, sitep, dayp = y[n_pool:], ho[n_pool:], site[n_pool:], day[n_pool:]
        sitepool, daypool = site[:n_pool], day[:n_pool]
        src = np.maximum(n_train.reindex(keys).fillna(0).to_numpy(float), 1.0)
        in_list = np.isin(keys, list(ne)).astype(float)
        counts = pd.Series(y[:n_pool]).value_counts().reindex(keys).fillna(0).to_numpy(float) + 1e-9
        oracle = counts / counts.sum()
        key_index = {k: i for i, k in enumerate(keys)}
        y_idx = np.array([key_index.get(k, -1) for k in y])

        def score(view: str, rule: str, pred_idx: np.ndarray, **extra) -> None:
            pred = keys[pred_idx]
            f, r, n = f1_all(yp, pred, np.ones(len(yp), bool))
            hit = pred == yp
            rows.append({"model": name, "views": view, "rule": rule, "accuracy": hit.mean(),
                         "macro-F1 (labelled + predicted)": f, "macro recall (labelled)": r,
                         "species predicted": n, "accuracy, held-out species": hit[hop].mean(),
                         "accuracy, other species": hit[~hop].mean(), **extra})
            print(f"  {name:5s} {view:9s} {rule:62s} acc {hit.mean():.4f}  F1 {f:.4f}  recall {r:.4f}  species {n}",
                  flush=True)

        for view, views in [("one view", ["r0"]), ("tta3bm", TTA3BM)]:
            p = posteriors(folder, views, order)
            pool, pp = p[:n_pool], p[n_pool:]
            score(view, "no prior", top1(pp))
            score(view, "north_europe list", top1(pp, in_list))
            score(view, "EM, not smoothed (Z4)", top1(pp, em_prior(pool)))
            pi = em_prior(pool, eps=0.01)
            score(view, "EM", top1(pp, pi), species_in_prior=int((pi > 1.0 / n_pool).sum()))
            score(view, "north_europe list and EM", top1(pp, pi * in_list))
            for a in (0.5, 1.0):
                s_a = src ** a / (src ** a).sum()
                pia = em_prior(pool, eps=0.01, source=s_a)
                score(view, f"EM, training prior^{a} divided out", top1(pp, pia / s_a),
                      species_in_prior=int((pia > 1.0 / n_pool).sum()))
                score(view, f"true adapt-night frequencies / training prior^{a}", top1(pp, oracle / s_a))
            score(view, "true adapt-night frequencies (Z4's oracle)", top1(pp, oracle))
            score(view, "EM on the probe crops themselves", top1(pp, em_prior(pp, eps=0.01)))
            score(view, "EM on all crops, adapt and probe nights", top1(pp, em_prior(p, eps=0.01)))
            if view == "one view":
                if name == "Nemo":
                    score(view, "EM, 200 iterations", top1(pp, em_prior(pool, iters=200, eps=0.01)))
                # per site and season: one prior per (site, night) of the probe, shrunk to the pooled one
                windows = {
                    "EM per site, all nights, shrunk to pooled (100 crops)": lambda s_, d_: sitepool == s_,
                    "EM all sites, nights within 15, shrunk": lambda s_, d_: np.abs(daypool - d_) <= 15,
                    "EM per site, nights within 15, shrunk": lambda s_, d_: (sitepool == s_) & (np.abs(daypool - d_) <= 15),
                    "EM per site, the 30 nights before, shrunk": lambda s_, d_: (sitepool == s_) & (daypool < d_) & (daypool >= d_ - 30),
                }
                for rule, select in windows.items():
                    pred = np.empty(len(yp), np.int64)
                    sizes, cache = [], {}
                    for s_, d_ in sorted(set(zip(sitep, dayp))):
                        sel = np.where(select(s_, d_))[0]
                        key = sel.tobytes()
                        if key not in cache:
                            cache[key] = em_prior(pool[sel], iters=30, base=pi, m=100.0) if len(sel) else pi
                        g = (sitep == s_) & (dayp == d_)
                        pred[g] = top1(pp[g], cache[key])
                        sizes.append(len(sel))
                    score(view, rule, pred, median_pool=float(np.median(sizes)))
            del p, pool, pp
            if view == "one view" and name in ("Nemo", "OT3"):
                known = y_idx[:n_pool] >= 0
                for temp in (0.5, 0.7, 1.4, 2.0):
                    p = posteriors(folder, views, order, temp)
                    pool, pp = p[:n_pool], p[n_pool:]
                    nll = -np.log(np.maximum(pool[np.where(known)[0], y_idx[:n_pool][known]], 1e-30)).mean()
                    score(view, f"T = {temp}: no prior", top1(pp), adapt_nll=float(nll))
                    score(view, f"T = {temp}: EM", top1(pp, em_prior(pool, eps=0.01)))
                    del p, pool, pp
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / ("z5_siteprior.csv" if not args.models else f"z5_siteprior_{args.models.replace(',', '_')}.csv")
    res.to_csv(out, index=False)
    print("written", out)


def em_in_list(pool: np.ndarray, in_list: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """The EM estimate when only the list's species can receive mass."""
    idx = np.where(in_list > 0)[0]
    w = np.zeros(pool.shape[1])
    w[idx] = em_prior(np.ascontiguousarray(pool[:, idx]), eps=eps)
    return w


def siteprior3(args) -> None:
    """Z5g and Z8: list, EM, both, and the EM inside the list; overall and per site, with the prior
    taken from the site's own unlabelled crops. --folders name=path,... (path under LOGITS or absolute)."""
    lab = trap_preds("C3b-shift")[["image_path", "label", "trap", "split"]]
    ne = set(open(NE_LIST).read().split())
    rows = []
    for item in args.folders.split(","):
        name, folder = item.split("=")
        keys = np.array(json.load(open(LOGITS / folder / "keys.json")))
        meta = pd.DataFrame({"image_path": json.load(open(LOGITS / folder / "crops.json"))})
        meta = meta.merge(lab, on="image_path", how="left")
        assert meta.label.notna().all()
        is_probe = (meta.split == "probe").to_numpy()
        order = np.concatenate([np.where(~is_probe)[0], np.where(is_probe)[0]])
        n_pool = int((~is_probe).sum())
        y = meta.label.to_numpy()[order]
        site = meta.trap.str[:2].to_numpy()[order]
        yp, sitep, sitepool = y[n_pool:], site[n_pool:], site[:n_pool]
        in_list = np.isin(keys, list(ne)).astype(float)
        p = posteriors(folder, ["r0"], order)
        pool, pp = p[:n_pool], p[n_pool:]

        def score(scope: str, rule: str, w: np.ndarray | None, mask: np.ndarray) -> None:
            pred = keys[top1(pp[mask], w)]
            f, r, n = f1_all(yp[mask], pred, np.ones(int(mask.sum()), bool))
            acc = (pred == yp[mask]).mean()
            rows.append({"model": name, "scope": scope, "rule": rule, "crops": int(mask.sum()), "accuracy": acc,
                         "macro-F1 (labelled + predicted)": f, "macro recall (labelled)": r, "species predicted": n})
            print(f"  {name:22s} {scope:4s} {rule:44s} acc {acc:.4f}  F1 {f:.4f}  recall {r:.4f}  species {n}", flush=True)

        everything = np.ones(len(yp), bool)
        pi = em_prior(pool, eps=0.01)
        w_in = em_in_list(pool, in_list)
        for scope, mask in [("all", everything)] + [(s_, sitep == s_) for s_ in ("LV", "OH", "SS")]:
            score(scope, "no prior", None, mask)
            score(scope, "north_europe list", in_list, mask)
            score(scope, "EM, all sites", pi, mask)
            score(scope, "list and EM, all sites", pi * in_list, mask)
            score(scope, "EM inside the list, all sites", w_in, mask)
            if scope != "all":
                own = pool[sitepool == scope]
                score(scope, "list and EM, own site", em_prior(own, eps=0.01) * in_list, mask)
                score(scope, "EM inside the list, own site", em_in_list(own, in_list), mask)
                other = pool[sitepool != scope]
                score(scope, "EM inside the list, the two other sites", em_in_list(other, in_list), mask)
        del p, pool, pp
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"{args.out}.csv", index=False)
    print("written", OUT / f"{args.out}.csv")


# ----------------------------------------------------------------------------- Z6
def _species_level(path: str) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    t = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        include_columns=["filename", "level", "label", "prediction"]))
    t = t.filter(pc.equal(t["level"], 0))
    d = t.to_pandas()
    return pd.DataFrame({"filename": d.filename, "correct": (d.label == d.prediction).to_numpy()})


def _probe_accuracy(path: str, probe_paths: set) -> tuple[float, int]:
    d = _species_level(path)
    name = d.filename.str.split("/").str[-2:].str.join("/")
    keep = name.isin(probe_paths).to_numpy()
    return float(d.correct[keep].mean()), int(keep.sum())


def observers(args) -> None:
    t = observer_groups()[["key", "speciesKey", "group"]]
    g = pd.read_parquet(GROUPS, columns=["image_path", "split"])
    probe_paths = set(g.image_path[g.split == "probe"])
    files = sorted(glob.glob(str(PREDS / "*/*/eval/predictions.csv")) + glob.glob(str(PREDS / "c-*/*/predictions.csv"))
                   + glob.glob(str(PREDS / "*/whole-set0/predictions.csv")))
    rows = []
    for path in files:
        if Path(path).stat().st_size < 100e6:      # trap files are small; the test fold is 170 MB or more
            continue
        rel = Path(path).relative_to(PREDS).parts
        run = rel[1] if rel[0].startswith("c-") else rel[0]
        d = _species_level(path)
        d["key"] = d.filename.str.split("/").str[-2:].str.join("/")
        d = d.merge(t, on="key", how="inner")
        d["correct"] = d.correct.astype(float)
        rec = {grp: sub.groupby("speciesKey").correct.mean() for grp, sub in d.groupby("group")}
        row = {"run": run, "images": len(d), "macro recall": d.groupby("speciesKey").correct.mean().mean()}
        for grp in "SON":
            row[f"accuracy {grp}"] = d.correct[d.group == grp].mean()
        for a, b in [("S", "N"), ("S", "O")]:
            both = rec[a].index.intersection(rec[b].index)
            row[f"recall {a} (paired with {b})"] = rec[a].loc[both].mean()
            row[f"recall {b} (paired with {a})"] = rec[b].loc[both].mean()
            row[f"gap {a}-{b}"] = (rec[a].loc[both] - rec[b].loc[both]).mean()
        # the run's probe accuracy, from the nearest trap prediction file
        base = rel[0]
        for suffix in ("-eval", "-cleanfold", "-clean"):
            base = base.removesuffix(suffix)
        base = {"singlehead": "base", "marginal": "marg"}.get(base, base)
        cands = ([str(Path(path).parent / "trap/probe/predictions.csv")] if rel[0].startswith("c-") else
                 glob.glob(str(PREDS / f"{base}-probe/*/eval/predictions.csv")) +
                 glob.glob(str(PREDS / f"{base}-shift/*/eval/predictions.csv")))
        cands = [c for c in cands if Path(c).exists()]
        if cands:
            row["probe accuracy"], row["probe crops"] = _probe_accuracy(cands[0], probe_paths)
        rows.append(row)
        print(f"{run[:70]:70s} n {len(d):7d}  S-N {row['gap S-N']:+.4f}  S-O {row['gap S-O']:+.4f}  "
              f"probe {row.get('probe accuracy', float('nan')):.4f}", flush=True)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "z6_observers.csv", index=False)
    print("written", OUT / "z6_observers.csv")


# ----------------------------------------------------------------------------- Z3
def prototype_stats(w: np.ndarray, s2g: np.ndarray, s2f: np.ndarray) -> dict:
    """How taxonomic a set of species prototypes is: who is the nearest other prototype, and the
    mean cosine to congeners, to the rest of the family, and to other families."""
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
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
    return {"dim": w.shape[1], "nearest is congener": nn_cong[has_cong].mean(), "chance": chance,
            "nearest in family": nn_fam.mean(), "nearest cos": nn_cos.mean(),
            "cos congener": sums[0] / counts[0], "cos same family": sums[1] / counts[1],
            "cos other": sums[2] / counts[2], "species with a congener": int(has_cong.sum())}


def congeners(args) -> None:
    """Z7: the same statistic on the campaign's heads (F, BU, I, TD on three backbones, three seeds)."""
    import torch
    base = ROOT / "data/campaign_weights"
    spec = json.load(open(base / "class_spec.json"))
    order = sorted(spec["cls2idx"]["0"], key=spec["cls2idx"]["0"].get)
    s2g = pd.factorize(np.array([spec["labels"][k][1] for k in order]))[0]
    s2f = pd.factorize(np.array([spec["labels"][k][2] for k in order]))[0]
    rows = []
    for path in sorted(base.glob("*/last.pt")):
        run = path.parent.name
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        cands = {k: v for k, v in sd.items() if hasattr(v, "ndim") and v.ndim == 2 and v.shape[0] == len(order)
                 and v.shape[1] > 1}
        if not cands:
            continue
        key = sorted(cands)[-1]   # I and TD register the leaf layer twice (linear is layers.0): same tensor
        bb, head, seed = run.split("_")[:3]
        w = cands[key].float().numpy()
        st = prototype_stats(w, s2g, s2f)
        wn = w / np.linalg.norm(w, axis=1, keepdims=True)
        ce = prototype_stats(wn - wn.mean(axis=0), s2g, s2f)   # the common direction removed
        st["nearest is congener, centred"] = ce["nearest is congener"]
        st["nearest in family, centred"] = ce["nearest in family"]
        rows.append({"run": run, "backbone": bb, "head": head, "seed": seed, "tensor": key, **st})
        print(f"{run:16s} dim {st['dim']:5d}  nearest is congener {st['nearest is congener']:.4f}  in family "
              f"{st['nearest in family']:.4f}  cos {st['cos congener']:.3f} / {st['cos same family']:.3f} / "
              f"{st['cos other']:.3f}   {key}", flush=True)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "z7_congeners.csv", index=False)
    if len(res):
        print(res.groupby(["backbone", "head"])[["nearest is congener", "nearest is congener, centred",
                                                  "nearest in family", "nearest in family, centred"]]
              .agg(["mean", "min", "max"]).round(4).to_string())


def severity(args) -> None:
    """Z7b: among wrong species answers, how often is the named species a congener, or in the family?
    The campaign's seed-42 heads with per-image predictions, on the test fold and the probe crops."""
    spec = json.load(open(ROOT / "data/campaign_weights/class_spec.json"))
    genus = {k: v[1] for k, v in spec["labels"].items()}
    family = {k: v[2] for k, v in spec["labels"].items()}
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    rows = []
    for folder in sorted(glob.glob(str(PREDS / "c-*/*_s42"))):
        run = Path(folder).name
        for scope, path in [("test fold", f"{folder}/predictions.csv"), ("probe", f"{folder}/trap/probe/predictions.csv")]:
            if not Path(path).exists():
                continue
            t = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
                include_columns=["level", "label", "prediction"], column_types={"label": "string", "prediction": "string"}))
            d = t.filter(pc.equal(t["level"], 0)).to_pandas()
            d = d[d.label.isin(genus.keys())]
            wrong = d[d.label != d.prediction]
            same_g = wrong.label.map(genus).to_numpy() == wrong.prediction.map(genus).to_numpy()
            same_f = wrong.label.map(family).to_numpy() == wrong.prediction.map(family).to_numpy()
            # per true species, then averaged: the tail counts as much as the head
            per = pd.DataFrame({"label": wrong.label.to_numpy(), "g": same_g, "f": same_f}).groupby("label").mean()
            rows.append({"run": run, "scope": scope, "images": len(d), "accuracy": 1 - len(wrong) / len(d),
                         "errors": len(wrong), "errors in the true genus": same_g.mean(),
                         "errors in the true family": same_f.mean(), "same, macro over species: genus": per.g.mean(),
                         "family": per.f.mean()})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "z7b_severity.csv", index=False)


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
        rows.append({"model": name, "row scale mean": g.mean(), **prototype_stats(w, s2g, s2f)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(4).T.to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "z3_geometry.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for fn in (observer, prior, ghosts, presence, siteprior, geometry, siteprior2, siteprior3, observers, congeners, severity):
        sp = sub.add_parser(fn.__name__)
        sp.set_defaults(fn=fn)
        if fn is siteprior2:
            sp.add_argument("--models", default="", help="comma-separated subset of Nemo,P5,OT3")
        if fn is siteprior3:
            sp.add_argument("--folders", required=True, help="name=folder,... under the logits root, or absolute")
            sp.add_argument("--out", default="z8_siteprior")
    a = ap.parse_args()
    a.fn(a)
