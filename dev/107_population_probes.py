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

  calibrated     Z9: Z8's rules scored with the campaign's scorer (species macro-F1 calibrated by
                 (trap, night)), the prior added to the species logits.
  discrimination Z10b: AUROC of the logit difference between two species of the probe, crops of one
                 against crops of the other; no prior and no per-crop offset can move it.
  invariance     Z11: recall per species on the test fold against the probe crops, one model.
  overlap        Z13: crop by crop, whether the errors self-training repairs are the ones a prior repairs.
  newsite        Z15: each site as a new site, on the calibrated macro-F1 of its own probe crops.
  tolgap         Z17: the test-fold gap between the ToL model's Lepidoptera head and VD, by group of species.
  frozen_index   Z20: support and query images for comparing frozen encoders on Lepidoptera.
  frozen         Z20: nearest class mean on the embeddings of dev/109 `embed`, test fold and probe.

Run from the repo root:  .venv/bin/python dev/107_population_probes.py {observer,prior,ghosts,presence,siteprior,geometry,siteprior2,siteprior3,observers,congeners,severity,calibrated,discrimination,invariance,overlap,newsite,tolgap,frozen_index,frozen}
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


# ----------------------------------------------------------------------------- Z9-Z11
MT = Path.home() / "codes/mini_trainer/.worktrees/lepinet-integration"
MM = ROOT / "data/tools/mini_metrics-61db373"


def _group_logsumexp(x: np.ndarray, inv: np.ndarray, n: int) -> np.ndarray:
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(n))
    xo = x[:, order]
    m = np.maximum.reduceat(xo, starts, axis=1)
    return np.log(np.add.reduceat(np.exp(xo - m[:, inv[order]]), starts, axis=1)) + m


def _campaign_score(folder: str, views: list[str], w: np.ndarray, dest: Path, probe_rows: np.ndarray,
                    crops: list, keys: np.ndarray, truth: np.ndarray, genus_of: np.ndarray,
                    family_of: np.ndarray, extra_sets: dict | None = None) -> dict:
    """The campaign's score.py for one rule, on the probe crops: w multiplies the species posteriors
    (species with w = 0 are dropped, as a list does), genus and family are logsumexp over the kept
    species, confidence is the top-1 softmax. Then flemming.py score on probe and probe-HO (species
    macro-F1 at zero threshold and calibrated by (trap, night))."""
    import subprocess
    sel = np.flatnonzero(w > 0)
    logw = np.log(w[sel] / w[sel].max()).astype(np.float32)
    g_codes, g_inv = np.unique(genus_of[sel], return_inverse=True)
    fam_of_genus = pd.Series(family_of[sel], index=genus_of[sel]).groupby(level=0).first().loc[g_codes].to_numpy()
    f_codes, f_inv = np.unique(fam_of_genus, return_inverse=True)
    vocab = [set(keys), set(genus_of), set(family_of)]
    zs = [np.load(LOGITS / folder / f"{v}.npy", mmap_mode="r") for v in views]
    parts = []
    for start in range(0, len(probe_rows), 2048):
        ids = probe_rows[start:start + 2048]
        leaf = np.zeros((len(ids), len(keys)), np.float32)
        for z in zs:
            leaf += np.asarray(z[ids], dtype=np.float32) / np.float32(len(zs))
        sp = leaf[:, sel] + logw
        ge = _group_logsumexp(sp, g_inv, len(g_codes))
        fa = _group_logsumexp(ge, f_inv, len(f_codes))
        for lvl, (x, labels) in enumerate(((sp, keys[sel]), (ge, g_codes), (fa, f_codes))):
            i = x.argmax(1)
            conf = 1.0 / np.exp(x - x[np.arange(len(i)), i][:, None]).sum(1)
            lab = truth[ids, lvl]
            parts.append(pd.DataFrame({"instance_id": ids, "filename": [crops[k] for k in ids], "level": lvl,
                                       "label": lab, "prediction": labels[i], "confidence": conf, "threshold": 0.0,
                                       "known_label": [int(t in vocab[lvl]) for t in lab]}))
    df = pd.concat(parts).sort_values(["instance_id", "level"], kind="stable")
    (dest / "trap").mkdir(parents=True, exist_ok=True)
    csv = dest / "trap" / "mini_metric.csv"
    df.to_csv(csv, index=False)
    import os
    subprocess.run([str(ROOT / ".venv/bin/python"), str(MT / "dev/lepinet/flemming.py"), "score", "--predictions", str(csv),
                    "--lepinet", str(ROOT), "--seeds", "20", "--sets", "probe", "probe-ho",
                    *[x for k, v in (extra_sets or {}).items() for x in ("--set-file", f"{k}={v}")]],
                   env={**os.environ, "PYTHONPATH": str(MM)}, capture_output=True, text=True, check=True)
    out = {}
    for st in ("probe", "probe-ho", *(extra_sets or {})):
        sc = pd.read_csv(dest / "trap" / st / "scores.csv")
        f1 = sc[(sc.metric == "f1") & (sc.level == 0)].iloc[0]
        out[f"{st}, zero"], out[f"{st}, calibrated"] = float(f1.zero), float(f1.calg_mean)
    sp0 = df[df.level == 0]
    out["_pred"] = sp0.set_index("instance_id").prediction
    return out


def _trap_setup(folder: str):
    """Crops, labels and taxonomy of one folder of saved logits; the pool (not probe) is ordered first."""
    lab = trap_preds("C3b-shift")[["image_path", "label", "trap", "split"]]
    lab = lab.merge(pd.read_parquet(GROUPS, columns=["image_path", "species_heldout"]), on="image_path")
    keys = np.array(json.load(open(LOGITS / folder / "keys.json")))
    crops = json.load(open(LOGITS / folder / "crops.json"))
    meta = pd.DataFrame({"image_path": crops}).merge(lab, on="image_path", how="left")
    assert meta.label.notna().all()
    is_probe = (meta.split == "probe").to_numpy()
    order = np.concatenate([np.where(~is_probe)[0], np.where(is_probe)[0]])
    tax = pd.read_parquet(META, columns=["speciesKey", "genusKey", "familyKey"]).astype(str)
    tax = tax.drop_duplicates("speciesKey").set_index("speciesKey")
    return keys, crops, meta, order, int((~is_probe).sum()), tax


def calibrated(args) -> None:
    """Z9: Z8's rules on the papers' metric. --folders name=folder=views,... (views: r0 or tta3bm)."""
    ne = set(open(NE_LIST).read().split())
    n2i = pd.read_csv(ROOT / "data/flemming/name2id.csv").astype(
        {k: str for k in ("speciesKey", "genusKey", "familyKey")}).set_index("verbatimScientificName")
    rows = []
    for item in args.folders.split(","):
        name, folder, vname = item.split("=")
        views = TTA3BM if vname == "tta3bm" else ["r0"]
        keys, crops, meta, order, n_pool, tax = _trap_setup(folder)
        genus_of, family_of = tax.loc[keys, "genusKey"].to_numpy(), tax.loc[keys, "familyKey"].to_numpy()
        truth = n2i.loc[[c.split("/")[0] for c in crops], ["speciesKey", "genusKey", "familyKey"]].to_numpy()
        p = posteriors(folder, views, order)
        pool, pp = p[:n_pool], p[n_pool:]
        probe_rows = order[n_pool:]
        yp = meta.label.to_numpy()[probe_rows]
        in_list = np.isin(keys, list(ne)).astype(float)
        rules = {"nothing": np.ones(len(keys)), "list": in_list,
                 "list and EM": em_prior(pool, eps=0.01) * in_list,
                 "list and EM, probe nights included": em_prior(p, eps=0.01) * in_list,
                 "the probe's own frequencies (diagnostic)":
                     pd.Series(yp).value_counts().reindex(keys).fillna(0).to_numpy(float)}
        if args.rules:
            rules = {k: v for k, v in rules.items() if k in args.rules.split(";")}
        for rule, w in rules.items():
            pred = keys[top1(pp, w)]
            f, r, _ = f1_all(yp, pred, np.ones(len(yp), bool))
            sc = _campaign_score(folder, views, w, OUT / "z9" / f"{name}_{vname}" / rule.split(" (")[0].replace(" ", "_").replace(",", "").replace("'", ""),
                                 np.sort(probe_rows), crops, keys, truth, genus_of, family_of)
            agree = (sc.pop("_pred").loc[probe_rows].to_numpy() == pred).mean()
            assert agree > 0.999, (name, rule, agree)  # the scorer's table and this script name the same species
            rows.append({"model": name, "views": vname, "rule": rule, "accuracy": (pred == yp).mean(),
                         "macro recall (labelled)": r, "macro-F1 (labelled + predicted)": f, **sc})
            print(f"  {name:24s} {vname:7s} {rule:42s} acc {rows[-1]['accuracy']:.4f} recall {r:.4f}  probe cal "
                  f"{sc['probe, calibrated']:.4f} zero {sc['probe, zero']:.4f}  probe-HO cal {sc['probe-ho, calibrated']:.4f} "
                  f"zero {sc['probe-ho, zero']:.4f}", flush=True)
        del p, pool, pp
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"{args.out}.csv", index=False)
    print("written", OUT / f"{args.out}.csv")


def discrimination(args) -> None:
    """Z10b: pairwise discrimination among the probe's species, free of any prior. Whatever constants a
    prior adds to the logits of species k and j, the choice between them on a crop rests on
    z_k(x) - z_j(x). Its AUROC, crops of k against crops of j, is unchanged by a constant per species
    (a prior, a list) and by a constant per crop (the softmax ignores it, and fp16s dumps subtract one).
    Mean of 1 - AUROC over pairs of species with at least 5 probe crops; per species, the mean over
    its pairs."""
    rows, per = [], []
    common = None   # --common: only species every folder's model can name (models with different vocabularies)
    if getattr(args, "common", False):
        common = set.intersection(*[set(json.load(open(LOGITS / item.split("=")[1] / "keys.json"))) for item in args.folders.split(",")])
    for item in args.folders.split(","):
        name, folder, vname = item.split("=")
        views = TTA3BM if vname == "tta3bm" else ["r0"]
        keys, crops, meta, order, n_pool, tax = _trap_setup(folder)
        probe_rows = np.sort(order[n_pool:])
        y = meta.label.to_numpy()[probe_rows]
        held = set(meta.label[meta.species_heldout.to_numpy()])
        counts = pd.Series(y).value_counts()
        col = {k: i for i, k in enumerate(keys)}
        species = sorted(k for k in counts.index[counts >= 5] if k in col and (common is None or k in common))
        cols = np.array([col[k] for k in species])
        z = np.zeros((len(probe_rows), len(cols)), np.float32)
        for v in views:
            zz = np.load(LOGITS / folder / f"{v}.npy", mmap_mode="r")
            for i in range(0, len(probe_rows), 2048):
                z[i:i + 2048] += np.asarray(zz[probe_rows[i:i + 2048]], dtype=np.float32)[:, cols] / len(views)
        genus = tax.genusKey.reindex(species).to_numpy()
        family = tax.familyKey.reindex(species).to_numpy()
        # margin of each species over every other one, on its own crops
        d = [z[y == k][:, [a]] - z[y == k] for a, k in enumerate(species)]
        n = len(species)
        err = np.full((n, n), np.nan)
        for a in range(n):
            for b in range(a + 1, n):
                err[a, b] = err[b, a] = 1 - auroc(d[a][:, b], -d[b][:, a])
        iu = np.triu_indices(n, 1)
        same_g = (genus[:, None] == genus[None, :])[iu]
        same_f = (family[:, None] == family[None, :])[iu] & ~same_g
        e = err[iu]
        per_species = np.nanmean(err, axis=1)
        cong = np.array([np.nanmean(err[a][(genus == genus[a]) & (np.arange(n) != a)]) if ((genus == genus[a]).sum() > 1) else np.nan
                         for a in range(n)])
        for a, k in enumerate(species):
            per.append({"model": name, "species": k, "crops": int(counts[k]), "held out": k in held,
                        "pairwise error": per_species[a], "pairwise error, congeners": cong[a]})
        is_held = np.array([k in held for k in species])
        rows.append({"model": name, "views": vname, "species": n, "pairs": len(e), "pairwise error": e.mean(),
                     "pairs with error > 0.01": float((e > 0.01).mean()),
                     "congener pairs": int(same_g.sum()), "pairwise error, congeners": e[same_g].mean(),
                     "same family, other genus": e[same_f].mean(), "other family": e[~same_g & ~same_f].mean(),
                     "held-out species (mean over their pairs)": per_species[is_held].mean()})
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        del z, d
    pd.DataFrame(per).to_csv(OUT / f"{args.out}_per_species.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"{args.out}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(5).to_string(index=False))


def invariance(args) -> None:
    """Z11: under label shift alone a species keeps its recall, whatever the mix of species around
    it. Recall per species on the test fold and on the probe crops, same model, full vocabulary."""
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    spec = json.load(open(ROOT / "data/campaign_weights/class_spec.json"))
    genus = {k: v[1] for k, v in spec["labels"].items()}

    def read(path: str) -> pd.DataFrame:
        t = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
            include_columns=["level", "label", "prediction"], column_types={"label": "string", "prediction": "string"}))
        return t.filter(pc.equal(t["level"], 0)).to_pandas()

    rows = []
    for folder in sorted(glob.glob(str(PREDS / "c-*/*_s42"))):
        if not Path(f"{folder}/trap/probe/predictions.csv").exists() or not Path(f"{folder}/predictions.csv").exists():
            continue
        test, probe = read(f"{folder}/predictions.csv"), read(f"{folder}/trap/probe/predictions.csv")
        n = probe.groupby("label").size()
        sp = n.index[n >= 5].intersection(pd.Index(test.label.unique()))
        r_test = (test.label == test.prediction).groupby(test.label).mean().reindex(sp)
        r_probe = (probe.label == probe.prediction).groupby(probe.label).mean().reindex(sp)

        def congener_share(d: pd.DataFrame) -> float:
            w = d[d.label.isin(sp) & (d.label != d.prediction) & d.prediction.isin(genus.keys())]
            return float((w.label.map(genus).to_numpy() == w.prediction.map(genus).to_numpy()).mean())

        lo, hi = boot_ci((r_test - r_probe).to_numpy())
        rows.append({"run": Path(folder).name, "species": len(sp), "recall, test fold": r_test.mean(),
                     "recall, probe": r_probe.mean(), "gap": (r_test - r_probe).mean(), "gap lo": lo, "gap hi": hi,
                     "species with lower recall on the probe": float((r_probe < r_test).mean()),
                     "errors naming a congener, test fold": congener_share(test),
                     "errors naming a congener, probe": congener_share(probe)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))
    res.to_csv(OUT / "z11_invariance.csv", index=False)


def overlap(args) -> None:
    """Z13: crop by crop on the probe, are the errors that self-training repairs the ones a prior
    repairs with no training? --pairs name=unadapted folder=self-trained folder;..."""
    ne = set(open(NE_LIST).read().split())
    rows = []
    for pair in args.pairs.split(";"):
        name, fa, fb = pair.split("=")
        keys, crops, meta, order, n_pool, tax = _trap_setup(fa)
        assert json.load(open(LOGITS / fb / "keys.json")) == list(keys) and json.load(open(LOGITS / fb / "crops.json")) == crops
        yp = meta.label.to_numpy()[order[n_pool:]]
        fauna = set(meta.label)  # every species labelled at the traps, adapt and probe nights
        genus = tax.genusKey
        in_list = np.isin(keys, list(ne)).astype(float)
        p = posteriors(fa, ["r0"], order)
        pool, pp = p[:n_pool], p[n_pool:]
        a0, a_list = keys[top1(pp)], keys[top1(pp, in_list)]
        a_em = keys[top1(pp, em_prior(pool, eps=0.01) * in_list)]
        del p, pool, pp
        pb = posteriors(fb, ["r0"], order[n_pool:])
        b0, b_list = keys[top1(pb)], keys[top1(pb, in_list)]
        del pb

        def kind(pred: np.ndarray, m: np.ndarray) -> dict:
            absent = ~np.isin(pred[m], list(fauna))
            cong = genus.reindex(pred[m]).to_numpy() == genus.reindex(yp[m]).to_numpy()
            return {"n": int(m.sum()), "name an absent species": float(absent.mean()),
                    "name a species outside the list": float((~np.isin(pred[m], list(ne))).mean()),
                    "name a congener": float(cong.mean())}

        wrong = a0 != yp
        fixed_st = wrong & (b0 == yp)
        left_st = b0 != yp
        r = {"pair": name, "probe crops": len(yp), "wrong, unadapted": int(wrong.sum()),
             "repaired by self-training": int(fixed_st.sum()),
             "repaired by the list": int((wrong & (a_list == yp)).sum()),
             "repaired by list and EM": int((wrong & (a_em == yp)).sum()),
             "of those self-training repairs: also the list": float((a_list == yp)[fixed_st].mean()),
             "of those self-training repairs: also list and EM": float((a_em == yp)[fixed_st].mean()),
             "of those list and EM repair: also self-training": float((b0 == yp)[wrong & (a_em == yp)].mean()),
             "broken by self-training (right before)": int((~wrong & (b0 != yp)).sum()),
             "broken by list and EM (right before)": int((~wrong & (a_em != yp)).sum()),
             "wrong for both best rules": int(((a_em != yp) & (b_list != yp)).sum()),
             "wrong, unadapted with list and EM": int((a_em != yp).sum()),
             "wrong, self-trained with list": int((b_list != yp).sum())}
        for label, pred, m in [("errors of the unadapted model", a0, wrong), ("errors self-training repairs", a0, fixed_st),
                               ("errors self-training leaves", b0, left_st),
                               ("errors left, unadapted with list and EM", a_em, a_em != yp),
                               ("errors left, self-trained with list", b_list, b_list != yp)]:
            for k, v in kind(pred, m).items():
                r[f"{label}: {k}"] = v
        rows.append(r)
    res = pd.DataFrame(rows).set_index("pair").T
    pd.set_option("display.width", 200)
    print(res.round(4).to_string())
    res.to_csv(OUT / "z13_overlap.csv")


def newsite(args) -> None:
    """Z15: each site as a new site, on the calibrated macro-F1 of its own probe crops. --unadapted,
    --selftrained and --fold are folders of saved logits ({site} in --fold is replaced by lv, oh, ss)."""
    ne = set(open(NE_LIST).read().split())
    n2i = pd.read_csv(ROOT / "data/flemming/name2id.csv").astype(
        {k: str for k in ("speciesKey", "genusKey", "familyKey")}).set_index("verbatimScientificName")
    keys, crops, meta, order, n_pool, tax = _trap_setup(args.unadapted)
    genus_of, family_of = tax.loc[keys, "genusKey"].to_numpy(), tax.loc[keys, "familyKey"].to_numpy()
    truth = n2i.loc[[c.split("/")[0] for c in crops], ["speciesKey", "genusKey", "familyKey"]].to_numpy()
    in_list = np.isin(keys, list(ne)).astype(float)
    site = meta.trap.str[:2].to_numpy()
    is_probe = (meta.split == "probe").to_numpy()
    probe_rows = np.sort(order[n_pool:])
    sets = {}
    (OUT / "z15").mkdir(parents=True, exist_ok=True)
    for s_ in ("LV", "OH", "SS"):
        sets[f"site{s_.lower()}"] = OUT / "z15" / f"probe_{s_}.parquet"
        pd.DataFrame({"image_path": np.array(crops)[is_probe & (site == s_)]}).to_parquet(sets[f"site{s_.lower()}"])
    p = posteriors(args.unadapted, ["r0"], np.arange(len(crops)))
    rows = []

    def run(model: str, folder: str, rule: str, w: np.ndarray, only: str | None = None) -> None:
        sc = _campaign_score(folder, ["r0"], w, OUT / "z15" / model.replace(" ", "_") / rule.replace(" ", "_").replace(",", ""),
                             probe_rows, crops, keys, truth, genus_of, family_of, extra_sets=sets)
        sc.pop("_pred")
        for s_ in ("LV", "OH", "SS"):
            if only in (None, s_):
                rows.append({"site": s_, "model": model, "rule": rule, "calibrated": sc[f"site{s_.lower()}, calibrated"],
                             "zero": sc[f"site{s_.lower()}, zero"]})
                print(rows[-1], flush=True)

    run("never saw a trap", args.unadapted, "list", in_list)
    for s_ in ("LV", "OH", "SS"):
        own = p[~is_probe & (site == s_)]
        run("never saw a trap", args.unadapted, f"list and EM from {s_}'s other nights", em_prior(own, eps=0.01) * in_list, s_)
        run("never saw a trap", args.unadapted, f"list and EM from all of {s_}'s crops", em_prior(p[site == s_], eps=0.01) * in_list, s_)
        run(f"self-trained without {s_}", args.fold.replace("{site}", s_.lower()), "list", in_list, s_)
    run("self-trained on all sites", args.selftrained, "list", in_list)
    res = pd.DataFrame(rows)
    res["rule"] = res.rule.str.replace(r"(LV|OH|SS)'s", "the site's", regex=True)
    res["model"] = res.model.str.replace(r"without (LV|OH|SS)", "without the site", regex=True)
    pd.set_option("display.width", 200)
    print(res.pivot_table(index=["model", "rule"], columns="site", values="calibrated", sort=False).round(4).to_string())
    print(res.pivot_table(index=["model", "rule"], columns="site", values="zero", sort=False).round(4).to_string())
    res.to_csv(OUT / "z15_newsite.csv", index=False)


E3 = Path("/data/au761367/otto_eval/e3")   # the lepinet session's E3 files (W3), read only


def _class_f1(path: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-class F1 at zero threshold over the union of labelled and predicted species (mini_metrics'
    convention), with each class's labelled and predicted counts, from a scorer's predictions.csv."""
    d = pd.read_csv(path, usecols=["level", "label", "prediction"])
    d = d[d["level"] == 0]
    nl, npr = d["label"].value_counts(), d["prediction"].value_counts()
    classes = nl.index.union(npr.index)
    tp = (d["label"] == d["prediction"]).groupby(d["label"]).sum().reindex(classes, fill_value=0)
    nl, npr = nl.reindex(classes, fill_value=0), npr.reindex(classes, fill_value=0)
    return 2 * tp / (nl + npr), nl, npr


def tolgap(args) -> None:
    """Z17: the test-fold gap between the ToL model's Lepidoptera head (E3, line 1) and VD, split over groups of
    species. Macro-F1 is a mean over classes, so each group's share of the gap is exact."""
    m = pd.read_csv(ROOT / "data/tol_eval/map_species.csv")
    cnt = pd.read_csv(OUT / "tol_train_counts.csv").set_index("tol_class")["tol_n_train"]
    m["tol_n"] = m["tol_class"].map(cnt)
    m = m.set_index("speciesKey")
    f_t, nl, np_t = _class_f1(E3 / "tol/e3_testfold/line1/predictions.csv")
    f_v, nl_v, np_v = _class_f1(E3 / "dinov3_bu_s42_test/e3/line1/predictions.csv")
    assert (nl[nl > 0].sort_index() == nl_v[nl_v > 0].sort_index()).all()
    print(f"macro-F1 recomputed: ToL {f_t.mean():.4f} over {len(f_t):,} classes, VD {f_v.mean():.4f} over {len(f_v):,} "
          f"(sheet: 0.8751, 0.9120); predicted-only classes: ToL {(nl.reindex(f_t.index) == 0).sum()}, "
          f"VD {(nl_v.reindex(f_v.index) == 0).sum()}")
    lab = nl[nl > 0].index
    d = pd.DataFrame({"tol": f_t.reindex(lab), "vd": f_v.reindex(lab), "n_test": nl.reindex(lab)}).join(
        m[["how", "n_keys_on_class", "n_train", "tol_n", "tol_class"]])
    d["pred_by_tol"] = np_t.reindex(lab, fill_value=0)
    shared = d["n_keys_on_class"] >= 2
    d["group"] = np.where(d["how"] == "unmapped", "1 no ToL class (never trained)",
                 np.where(shared & (d["pred_by_tol"] == 0), "2 shares a ToL class, the key ToL never answers",
                 np.where(shared, "3 shares a ToL class, the key ToL answers", "one to one")))
    ratio = d["tol_n"] / d["n_train"]
    one = d["group"] == "one to one"
    bins = pd.cut(ratio[one], [0, 0.5, 0.9, 1.1, 2, 1e9], labels=["4 one to one, ToL images < 0.5x lepinet's", "5 one to one, 0.5-0.9x",
                                                                   "6 one to one, 0.9-1.1x", "7 one to one, 1.1-2x", "8 one to one, > 2x"])
    d.loc[one, "group"] = bins.astype(str)
    rows = []
    for g, x in list(d.groupby("group")) + [("all labelled species", d), ("one to one, all", d[one])]:
        rows.append({"group": g, "species": len(x), "test images": int(x["n_test"].sum()),
                     "lepinet train images, median": x["n_train"].median(), "ToL train images, median": x["tol_n"].median(),
                     "F1 ToL": x["tol"].mean(), "F1 VD": x["vd"].mean(),
                     "gap, points on these species": 100 * (x["tol"].mean() - x["vd"].mean()),
                     "share of the total gap, points": 100 * (x["tol"].sum() / len(f_t) - x["vd"].sum() / len(f_v))})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print(f"total gap {100 * (f_t.mean() - f_v.mean()):.2f} points; of it, the denominators (predicted-only classes) "
          f"{100 * (d['vd'].sum() / len(f_v) - d['vd'].sum() / len(f_t)):.2f}")
    # one-to-one species by lepinet training count, and by how much ToL's count differs
    x = d[one].copy()
    x["train bin"] = pd.cut(x["n_train"], [0, 100, 200, 500, 1000, 1e9], labels=["50-100", "100-200", "200-500", "500-1000", "> 1000"])
    print(x.groupby("train bin", observed=True).agg(species=("tol", "size"), tol=("tol", "mean"), vd=("vd", "mean"),
                                                    ratio=("tol_n", lambda s: (s / x.loc[s.index, "n_train"]).median())).round(3).to_string())
    res.to_csv(OUT / "z17_tolgap.csv", index=False)
    d.to_csv(OUT / "z17_tolgap_per_species.csv")


Z20 = OUT / "z20"


def frozen_index(args) -> None:
    """Z20, step 1: the images three frozen encoders are compared on. Support: 10 training images of each of
    lepinet's 12,041 vocabulary species. Queries: up to 8 test-fold images per species whose GBIF occurrence is
    outside ToL-200M (unseen by BioCLIP-2; the ToL model excluded the whole fold). The trap crops come from
    data/tol_eval/index_trap.json. Embed with dev/109 `embed --index data/population_probes/z20/index_global.json`."""
    vocab = pd.read_csv(ROOT / "data/tol_eval/map_species.csv")
    vocab = set(vocab.loc[vocab["vocab"], "speciesKey"].astype(str))
    meta = pd.read_parquet(META, columns=["gbifID", "speciesKey", "filename", "set"])
    meta["speciesKey"] = meta["speciesKey"].astype(str)
    meta = meta[meta["speciesKey"].isin(vocab)]
    clean = set(pd.read_parquet(ROOT / "data/tol_overlap/test_ids_clean_of_tol.parquet")["gbifID"])
    sup = meta[meta["set"] != "0"].groupby("speciesKey", group_keys=False).sample(n=10, random_state=0).assign(role="support")
    q = meta[(meta["set"] == "0") & meta["gbifID"].isin(clean)].sample(frac=1.0, random_state=0)
    q = q[q.groupby("speciesKey").cumcount() < 8].assign(role="query")
    side = pd.concat([sup, q], ignore_index=True)
    side["path"] = [str(ROOT / "data/global/images" / k / f) for k, f in zip(side["speciesKey"], side["filename"])]
    Z20.mkdir(parents=True, exist_ok=True)
    side[["path", "role", "speciesKey"]].to_parquet(Z20 / "index_global.parquet")
    n = len(side)
    json.dump({"path": side["path"].tolist(), "label": [["-1"] * 7] * n, "class": [0] * n, "split": ["test"] * n},
              open(Z20 / "index_global.json", "w"))
    print(f"support {len(sup):,} images of {sup['speciesKey'].nunique():,} species; queries {len(q):,} images of "
          f"{q['speciesKey'].nunique():,} species (clean test-fold images of the vocabulary: "
          f"{int(((meta['set'] == '0') & meta['gbifID'].isin(clean)).sum()):,})")


def _ncm(support: np.ndarray, sy: np.ndarray, query: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """SimpleShot as dev/109 `few_shot`: L2-normalise, subtract the support mean, L2-normalise, class centroids,
    nearest centroid (Euclidean). Returns the predicted class of each query, among `classes`."""
    n2 = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    s, q = n2(support), n2(query)
    mu = s.mean(0)
    s, q = n2(s - mu), n2(q - mu)
    idx = pd.Series(np.arange(len(classes)), index=classes)
    code = idx.reindex(sy).to_numpy()
    keep = ~np.isnan(code)
    centre = np.zeros((len(classes), s.shape[1]), np.float32)
    np.add.at(centre, code[keep].astype(int), s[keep])
    centre /= np.bincount(code[keep].astype(int), minlength=len(classes))[:, None]
    cn = (centre ** 2).sum(1)
    return classes[np.concatenate([(cn[None] - 2 * q[i:i + 2048] @ centre.T).argmin(1) for i in range(0, len(q), 2048)])]


def frozen(args) -> None:
    """Z20, step 2: nearest class mean on frozen embeddings (--embs name=global folder=trap folder,...), 12,041-way on
    the clean test-fold queries and on the probe crops (full vocabulary and the regional list). Beside them, the
    trained heads of E3 on the same images (ToL's own head, VD, OT3)."""
    side = pd.read_parquet(Z20 / "index_global.parquet")
    sup, qry = (side["role"] == "support").to_numpy(), (side["role"] == "query").to_numpy()
    sy, qy = side.loc[sup, "speciesKey"].to_numpy(), side.loc[qry, "speciesKey"].to_numpy()
    classes = np.array(sorted(set(sy)))
    ne = set(open(NE_LIST).read().split())
    in_list = np.array([k in ne for k in classes])
    lab = trap_preds("C3b-shift")[["image_path", "label", "split"]]
    lab["path"] = [str(ROOT / "data/flemming/images" / x) for x in lab["image_path"]]
    lab = lab.set_index("path")
    held = pd.read_parquet(GROUPS, columns=["image_path", "species_heldout"]).set_index("image_path")["species_heldout"]
    rows = []

    def add(name, kind, pq_, pt_full, pt_list, ty, ho):
        f, r, _ = f1_all(qy, pq_, np.ones(len(qy), bool))
        row = {"model": name, "readout": kind, "test fold (clean): accuracy": (pq_ == qy).mean(), "macro-F1": f}
        for tag, pt in (("full vocabulary", pt_full), ("regional list", pt_list)):
            if pt is None:
                continue
            ft, _, _ = f1_all(ty, pt, np.ones(len(ty), bool))
            row.update({f"probe, {tag}: accuracy": (pt == ty).mean(), f"probe, {tag}: macro-F1": ft,
                        f"probe-HO, {tag}: accuracy": (pt == ty)[ho].mean()})
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    ty = ho = tp = None
    for item in args.embs.split(","):
        name, gdir, tdir = item.split("=")
        e = np.load(Path(gdir) / "embeddings.npy").astype(np.float32)
        assert json.load(open(Path(gdir) / "paths.json")) == side["path"].tolist()
        t = np.load(Path(tdir) / "embeddings.npy").astype(np.float32)
        tp = json.load(open(Path(tdir) / "paths.json"))
        probe = (lab["split"].reindex(tp) == "probe").to_numpy()
        ty = lab["label"].reindex(tp).to_numpy()[probe]
        ho = held.reindex([x.split("/images/")[1] for x in tp]).to_numpy()[probe]
        add(name, "nearest class mean, 10 images per species", _ncm(e[sup], sy, e[qry], classes),
            _ncm(e[sup], sy, t[probe], classes), _ncm(e[sup][np.isin(sy, classes[in_list])], sy[np.isin(sy, classes[in_list])],
                                                      t[probe], classes[in_list]), ty, ho)
    # the trained heads on the same images (E3's files; single view)
    qfile = pd.Series(qy, index=[Path(x).name for x in side.loc[qry, "path"]])
    last2 = lambda f: "/".join(Path(f).parts[-2:])  # noqa: E731  (species folder / crop: crop names repeat across folders)
    tfile = [last2(x) for x in np.array(tp)[probe]]
    for name, test_csv, trap_dir in (("ToL", E3 / "tol/e3_testfold/line1/predictions.csv", E3 / "tol/scores"),
                                     ("VD (DINOv3, Lepidoptera only, 320 px)", E3 / "dinov3_bu_s42_test/e3/line1/predictions.csv", E3 / "vd/scores"),
                                     ("OT3 (BioCLIP-2, fine-tuned and self-trained)",
                                      E3 / "otto_bioclip_bu_gate70_trivial_s42_test/e3/line1/predictions.csv", None)):
        d = pd.read_csv(test_csv, usecols=["filename", "level", "prediction"])
        d = d[d["level"] == 0]
        pred = pd.Series(d["prediction"].astype(str).to_numpy(), index=[Path(x).name for x in d["filename"]])
        pq_ = pred.reindex(qfile.index).to_numpy()
        pts = []
        for sub in ("full-none", "ne-none"):
            if trap_dir is None:
                pts.append(None)
                continue
            x = pd.read_csv(trap_dir / sub / "trap/mini_metric.csv", usecols=["filename", "level", "prediction"])
            x = x[x["level"] == 0]
            pts.append(pd.Series(x["prediction"].astype(str).to_numpy(), index=[last2(f) for f in x["filename"]]).reindex(tfile).to_numpy())
        add(name, "its trained head (E3)", pq_, pts[0], pts[1], ty, ho)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(4).T.to_string())
    res.to_csv(OUT / "z20_frozen.csv", index=False)


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
    for fn in (observer, prior, ghosts, presence, siteprior, geometry, siteprior2, siteprior3, observers, congeners, severity,
               calibrated, discrimination, invariance, overlap, newsite, tolgap,
               frozen_index, frozen):
        sp = sub.add_parser(fn.__name__)
        sp.set_defaults(fn=fn)
        if fn is siteprior2:
            sp.add_argument("--models", default="", help="comma-separated subset of Nemo,P5,OT3")
        if fn is siteprior3:
            sp.add_argument("--folders", required=True, help="name=folder,... under the logits root, or absolute")
            sp.add_argument("--out", default="z8_siteprior")
        if fn in (calibrated, discrimination):
            sp.add_argument("--folders", required=True, help="name=folder=views,... (views: r0 or tta3bm)")
            sp.add_argument("--out", default="z9_calibrated" if fn is calibrated else "z10_discrimination")
        if fn is newsite:
            sp.add_argument("--unadapted", required=True)
            sp.add_argument("--selftrained", required=True)
            sp.add_argument("--fold", required=True, help="folder with {site} for lv, oh, ss")
        if fn is overlap:
            sp.add_argument("--pairs", required=True, help="name=unadapted folder=self-trained folder;...")
        if fn is calibrated:
            sp.add_argument("--rules", default="", help="subset of the rules, separated by ';'")
        if fn is discrimination:
            sp.add_argument("--common", action="store_true", help="only species that every folder's model can name")
        if fn is frozen:
            sp.add_argument("--embs", required=True, help="name=folder of the global embed=folder of the trap embed,...")
    a = ap.parse_args()
    a.fn(a)
