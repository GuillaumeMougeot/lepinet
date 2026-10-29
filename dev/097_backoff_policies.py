"""Which back-off policy, stated without anything specific to moths or traps? (Q)

Every policy reads only the per-rank top prediction and its confidence, which a marginalising head
gives for any taxonomy, so a dataset with any number of ranks drops in (`--plantnet`, a mini_metrics
long-format predictions.csv whose levels 0..L-1 run from finest to coarsest).

Policies, on a fit half / evaluation half split (trap images: halves of the (trap, night) groups, seed 0
as in dev/086 plus seeds 1-10; images without groups: random image halves):

* P0      greedy, per-rank threshold at 95 % precision conditional on reaching the rank (dev/083).
* P1-0.80 / P1-0.90   P0 with the intermediate ranks' target lowered to 0.80 / 0.90.
* P2      expected utility: answer argmax_r u_r p_r - c (1 - p_r), abstain if every term is negative;
          c = 1 (a wrong answer costs what a correct species earns). No fitting.
* P3      P2 with c chosen on the fit half so the overall precision of answers reaches 95 %.
Two utility vectors: u = (1, 0.5, 0.25) and (1, 0.8, 0.5) for species, genus, family; with more ranks
the vector continues by halving (resp. x 0.625).

Calibration variants for P2/P3: "shipped" (confidences as the model emits them; B8 and B3rep5x carry
an in-distribution temperature) and "target" (per rank, a one-parameter logistic recalibration of the
top-class confidence, p' = sigmoid(a * logit(p)), a fitted by log-loss against correctness on the fit
half). P0/P1 refit their thresholds per rank, so a per-rank monotone recalibration does not change them.

Transfer (Q-H2): c, or P0's thresholds, fitted on the in-distribution fit half and applied to trap images.

    python dev/097_backoff_policies.py [--plantnet preds.csv] [--seeds 11]

Writes data/backoff/<dataset>-<model>.csv as it goes (one row per policy x split), and a summary.
journal/2026-10-23-what-is-the-best-back-off-policy.md.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "backoff"
HF = ROOT / "data" / "hf_release"
MODELS = {"B8": "b8", "P5": "p5v2", "B3rep5x": "b3"}
NOVEL_FILE = {"b8": "eval_gbif_test_novel_ent.parquet", "b3": "eval_gbif_test_novel_ent.parquet",
              "p5v2": "eval_gbif_test_novel.parquet"}
GRID = np.unique(np.concatenate([np.linspace(0, 0.99, 199), 1 - np.logspace(-2, -5, 31)]).round(6))
UTILS = {"u=1/.5/.25": 0.5, "u=1/.8/.5": 0.625}     # ratio between consecutive ranks' utility


def utilities(L, ratio):
    u = [1.0]
    for _ in range(L - 1):
        u.append(u[-1] * (0.8 if ratio == 0.625 and len(u) == 1 else ratio))
    return np.array(u)


# ---------------------------------------------------------------------------- data, as arrays

class Data:
    """conf, ok: (n, L) arrays, finest rank first; group: (n,) or None; label ok may be False for unknowns."""

    def __init__(self, conf, ok, group=None, ranks=None):
        self.conf, self.ok, self.group = np.asarray(conf, float), np.asarray(ok, bool), group
        self.ranks = ranks or [f"rank{i}" for i in range(self.conf.shape[1])]

    def take(self, m):
        return Data(self.conf[m], self.ok[m], None if self.group is None else self.group[m], self.ranks)

    def __len__(self):
        return len(self.conf)


def from_wide(d, ranks=("species", "genus", "family"), group=True):
    conf = np.stack([d[f"conf_{r}"].to_numpy(float) for r in ranks], 1).clip(0, 1)
    ok = np.stack([(d[f"pred_{r}"].to_numpy() == d[f"label_{r}"].to_numpy()) for r in ranks], 1)
    return Data(conf, ok, d["group"].astype(str).to_numpy() if group and "group" in d else None, list(ranks))


def from_long_csv(path):
    d = pd.read_csv(path)
    d["confidence"] = d.confidence.clip(0, 1)
    w = d.pivot_table(index="instance_id", columns="level", values=["prediction", "label", "confidence"],
                      aggfunc="first")
    L = d.level.nunique()
    conf = np.stack([w["confidence"][i].to_numpy(float) for i in range(L)], 1)
    ok = np.stack([w["prediction"][i].astype(str).to_numpy() == w["label"][i].astype(str).to_numpy()
                   for i in range(L)], 1)
    return Data(conf, ok, None, [f"level{i}" for i in range(L)])


def c3b_trap():
    """C3b on all trap crops (dev/096's tables): known crops, and the near-novel crops."""
    spec = __import__("importlib.util").util.spec_from_file_location("o", ROOT / "dev" / "096_trap_openset.py")
    o = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(o)
    p = next((ROOT / "data/ucloud_preds/C3b-shift").rglob("predictions.csv"))
    d = pd.read_csv(p)
    w = d.pivot_table(index=["instance_id", "filename"], columns="level",
                      values=["prediction", "confidence", "label"], aggfunc="first").reset_index()
    res = pd.DataFrame({"filename": w["filename"]})
    for i, r in enumerate(("species", "genus", "family")):
        res[f"pred_{r}"] = w["prediction"][i].astype(str).values
        res[f"label_{r}"] = w["label"][i].astype(str).values
        res[f"conf_{r}"] = w["confidence"][i].astype(float).values
    s = pd.read_parquet(ROOT / "data/ucloud_preds/trap-openset/trap-openset-C3b.parquet").set_index("image_path")
    novel = res.filename.map(s.novel).astype(bool).to_numpy()
    st = o.strata(res.filename, res.label_species).to_numpy()
    res["group"] = o.group_of(res.filename)
    return from_wide(res[~novel]), from_wide(res[(st == "near") & novel])


# ---------------------------------------------------------------------------- policies

def fit_greedy(D, targets):
    """dev/083 fit_thresholds, any number of ranks, a target per rank."""
    th, left = [], np.ones(len(D), bool)
    for r, t_r in enumerate(targets):
        conf, ok = D.conf[:, r], D.ok[:, r]
        th_r = 1.01
        for t in GRID:
            m = left & (conf >= t)
            if m.sum() >= 30 and ok[m].mean() >= t_r:
                th_r = float(t)
                break
        th.append(th_r)
        left &= ~(conf >= th_r)
    return np.array(th)


def apply_greedy(D, th):
    choice, left = np.full(len(D), -1), np.ones(len(D), bool)
    for r, t in enumerate(th):
        take = left & (D.conf[:, r] >= t)
        choice[take] = r
        left &= ~take
    return choice


def apply_utility(conf, u, c):
    score = u[None, :] * conf - c * (1 - conf)
    best = score.argmax(1)
    return np.where(score.max(1) >= 0, best, -1)


def fit_cost(D, conf, u, target=0.95):
    """Smallest error cost c whose answers on the fit half reach `target` overall precision."""
    for c in np.logspace(-2, 4, 241):        # c / (u + c) is the implied confidence bar: up to ~0.9999
        ch = apply_utility(conf, u, c)
        a = ch >= 0
        if a.sum() >= 30 and D.ok[np.arange(len(D)), np.maximum(ch, 0)][a].mean() >= target:
            return float(c)
    return float("inf")


def fit_temperature(conf, ok):
    """Per rank: a in sigmoid(a * logit(p)) minimising log-loss against correctness (golden section)."""
    eps = 1e-6
    z = np.log(np.clip(conf, eps, 1 - eps)) - np.log1p(-np.clip(conf, eps, 1 - eps))
    y = ok.astype(float)

    def loss(a):
        q = 1 / (1 + np.exp(-a * z))
        q = np.clip(q, eps, 1 - eps)
        return -(y * np.log(q) + (1 - y) * np.log1p(-q)).mean()
    lo, hi = 0.01, 5.0
    g = (5 ** 0.5 - 1) / 2
    for _ in range(60):
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
        if loss(x1) < loss(x2):
            hi = x2
        else:
            lo = x1
    return (lo + hi) / 2


def recalibrate(conf, a):
    eps = 1e-6
    z = np.log(np.clip(conf, eps, 1 - eps)) - np.log1p(-np.clip(conf, eps, 1 - eps))
    return 1 / (1 + np.exp(-a[None, :] * z))


def summary(D, choice, u):
    n, L = len(D), D.conf.shape[1]
    ok = np.where(choice >= 0, D.ok[np.arange(n), np.maximum(choice, 0)], False)
    row = {"n": n, "abstain %": 100 * (choice < 0).mean(),
           "precision": ok[choice >= 0].mean() if (choice >= 0).any() else np.nan,
           "correct %": 100 * ok.mean()}
    for r in range(L):
        m = choice == r
        row[f"{D.ranks[r]} %"] = 100 * m.mean()
        row[f"{D.ranks[r]} prec"] = D.ok[m, r].mean() if m.any() else np.nan
    for name, ratio in UTILS.items():
        uu = utilities(L, ratio)
        row[f"utility {name}"] = float(np.where(ok, uu[np.maximum(choice, 0)], 0).mean())
    return row


def halves(D, seed):
    if D.group is not None:
        groups = np.array(sorted(set(D.group)))
        fit_g = set(np.random.default_rng(seed).permutation(groups)[: len(groups) // 2])
        f = np.isin(D.group, list(fit_g))
    else:
        f = np.random.default_rng(seed).random(len(D)) < 0.5
    return f


def run_policies(D, seed, novel=None, transfer=None):
    """All policies for one split. `novel`: Data scored with the fitted policies (unseen species).
    `transfer`: dict of in-distribution-fitted (thresholds / costs) to apply to D's evaluation half."""
    f = halves(D, seed)
    Fi, Ev = D.take(f), D.take(~f)
    L = D.conf.shape[1]
    rows = []
    a_t = np.array([fit_temperature(Fi.conf[:, r], Fi.ok[:, r]) for r in range(L)])

    def emit(policy, calib, uname, choice, extra=None):
        r = {"seed": seed, "policy": policy, "calibration": calib, "utility vector": uname, **summary(Ev, choice, utilities(L, UTILS.get(uname, 0.5)))}
        rows.append({**r, **(extra or {})})

    targets = {"P0": [0.95] * L, "P1-0.80": [0.95] + [0.80] * (L - 2) + [0.95], "P1-0.90": [0.95] + [0.90] * (L - 2) + [0.95]}
    fitted = {}
    for name, t in targets.items():
        th = fit_greedy(Fi, t)
        fitted[name] = th
        nov = {}
        if novel is not None:
            ch = apply_greedy(novel, th)
            nov = {"novel: species answer % (all wrong)": 100 * (ch == 0).mean(),
                   "novel: correct %": 100 * np.where(ch >= 0, novel.ok[np.arange(len(novel)), np.maximum(ch, 0)], False).mean()}
        emit(name, "-", "u=1/.5/.25", apply_greedy(Ev, th), {"thresholds": json.dumps([round(x, 4) for x in th]), **nov})
    for calib in ("shipped", "target"):
        cf_fit = Fi.conf if calib == "shipped" else recalibrate(Fi.conf, a_t)
        cf_ev = Ev.conf if calib == "shipped" else recalibrate(Ev.conf, a_t)
        cf_nov = None if novel is None else (novel.conf if calib == "shipped" else recalibrate(novel.conf, a_t))
        for uname, ratio in UTILS.items():
            u = utilities(L, ratio)
            for policy, c in (("P2", 1.0), ("P3", fit_cost(Fi, cf_fit, u))):
                fitted[(policy, calib, uname)] = c
                nov = {}
                if novel is not None and np.isfinite(c):
                    ch = apply_utility(cf_nov, u, c)
                    nov = {"novel: species answer % (all wrong)": 100 * (ch == 0).mean(),
                           "novel: correct %": 100 * np.where(ch >= 0, novel.ok[np.arange(len(novel)), np.maximum(ch, 0)], False).mean()}
                ch = apply_utility(cf_ev, u, c) if np.isfinite(c) else np.full(len(Ev), -1)
                emit(policy, calib, uname, ch, {"c": c, "temperature a": json.dumps([round(x, 3) for x in a_t]) if calib == "target" else "", **nov})
    if transfer is not None:   # in-distribution-fitted policy applied to this (shifted) evaluation half
        emit("P0 fitted in-distribution", "-", "u=1/.5/.25", apply_greedy(Ev, transfer["P0"]))
        for calib in ("shipped", "target"):
            cf_ev = Ev.conf if calib == "shipped" else recalibrate(Ev.conf, a_t)
            for uname, ratio in UTILS.items():
                c = transfer[("P3", "shipped", uname)]
                emit("P3 cost fitted in-distribution", calib, uname, apply_utility(cf_ev, utilities(L, ratio), c), {"c": c})
    return rows, fitted


def save(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)


def main(a):
    seeds = range(a.seeds)
    for model, folder in MODELS.items():
        if (OUT / f"trap-{model}.csv").exists() and not a.overwrite:
            print(f"{model}: done, skipping (delete data/backoff/trap-{model}.csv to redo)")
            continue
        ind = pd.read_parquet(HF / folder / "eval_gbif_test_full.parquet",
                              columns=[f"{k}_{r}" for k in ("pred", "conf", "label") for r in ("species", "genus", "family")])
        ID = from_wide(ind, group=False)
        nov = from_wide(pd.read_parquet(HF / folder / NOVEL_FILE[folder]), group=False)
        rows_id, fitted_id = [], {}
        for s in seeds:
            r, fitted = run_policies(ID, s, novel=nov)
            rows_id += r
            fitted_id[s] = fitted
            save(rows_id, f"indist-{model}")
        pr = pd.read_parquet(HF / folder / "eval_probe.parquet")
        TR = from_wide(pr[pr.label_species >= 0])
        rows_tr = []
        for s in seeds:
            r, _ = run_policies(TR, s, transfer=fitted_id[s])
            rows_tr += r
            save(rows_tr, f"trap-{model}")
        print(f"{model}: done")
    if not (OUT / "trap-C3b.csv").exists() or a.overwrite:
        known, near = c3b_trap()
        rows = []
        for s in seeds:
            r, _ = run_policies(known, s, novel=near)
            rows += r
            save(rows, "trap-C3b")
        print("C3b: done")
    if a.plantnet:
        D = from_long_csv(a.plantnet)
        rows = []
        for s in seeds:
            r, _ = run_policies(D, s)
            rows += r
            save(rows, "plantnet")
        print("Pl@ntNet: done")
    report()


def report():
    cols = ["policy", "calibration", "utility vector", "abstain %", "precision", "correct %"]
    for f in sorted(OUT.glob("*.csv")):
        d = pd.read_csv(f)
        ranks = [c[:-2] for c in d.columns if c.endswith(" %") and c not in ("abstain %", "correct %")
                 and not c.startswith("novel")]
        keep = cols + [f"{r} %" for r in ranks] + [f"{r} prec" for r in ranks] + \
            [c for c in d.columns if c.startswith("utility ")] + [c for c in d.columns if c.startswith("novel")]
        g = d.groupby(["policy", "calibration", "utility vector"], sort=False)[[c for c in keep if c not in cols[:3]]]
        m, sd = g.mean(), g.std()
        print(f"\n### {f.stem} ({d.seed.nunique()} splits; mean, sd of correct % in brackets)")
        out = m.round(3)
        out["correct % sd"] = sd["correct %"].round(2)
        print(out.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plantnet", default=None, help="mini_metrics long-format predictions.csv (levels 0..L-1, finest first)")
    ap.add_argument("--seeds", type=int, default=11)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", action="store_true", help="only print the summary of data/backoff/")
    a = ap.parse_args()
    report() if a.report else main(a)
