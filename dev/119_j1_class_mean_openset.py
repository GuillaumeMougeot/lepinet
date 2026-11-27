"""J1 open set, one more embedding score: cosine to the nearest class mean (CPU, from e-j1-os2's saved embeddings).

    CUDA_VISIBLE_DEVICES="" taskset -c 0-3 .venv/bin/python -I dev/119_j1_class_mean_openset.py [--reps 500]
    ... --dir /data/au761367/lepinet/j2/openset --arms ctl,frz,gs     (J2, e-j2-os; same layout)

The head rows of a learned cosine head point away from their images (J1: predicted-class cosine -0.44), so "maximum
cosine to a head row" fails as an open-set score there (AUROC 0.54). A class mean, the renormalised average embedding
of a species' reference images, is a prototype computed from data. It exists for any trained model, including
tol-full. This scores it on J1's three arms against the same knowns and unknowns as mt's `openset_auroc.py`.

Inputs (`/data/au761367/lepinet/j1/openset`, mt's e-j1-os2): `rows.parquet` (manifest: set, key, path),
`<arm>.parquet` (mt's scores, same row order), `emb/<arm>-embeddings.pt` (float16, one row per unique path in
manifest order, L2-normalised head input). Reference set: 10 training images per species (126,310 images, all 12,632
species), disjoint from the queries. The class means use those 10 images, not the full training set.

Checks before scoring: the embedding count equals the number of unique paths; 1-NN to the reference set, recomputed
from the embeddings on 3,000 random queries, matches mt's stored `knn1`; and the AUROC code reproduces mt's `knn1` and
`maxcos` AUROC. AUROC follows mt's method: knowns positive, a bootstrap over species (500 draws, the same draws for
every arm and score), paired intervals for frz - ctl.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

D = Path("/data/au761367/lepinet/j1/openset")
ARMS = ("ctl", "frz", "b")  # J1's; --dir / --arms for another run with the same layout (the first arm is the paired reference)
STRATA = {"near": ["near"], "mid": ["mid"], "far": ["far"], "all": ["near", "mid", "far"]}
SCORES = ("cm_max", "knn1", "maxcos")


def load_embeddings(arm):
    x = torch.load(D / "emb" / f"{arm}-embeddings.pt", map_location="cpu", weights_only=True)
    if isinstance(x, dict):
        keys = [k for k, v in x.items() if torch.is_tensor(v) and v.ndim == 2]
        assert len(keys) == 1, f"{arm}: expected one 2-D tensor, got {list(x)}"
        x = x[keys[0]]
    return x


def auroc(known, unknown, wk=None, wu=None):
    """P(known score > unknown score) + ties / 2, with optional per-image weights (mt's Pair.auroc)."""
    order = np.argsort(unknown, kind="stable")
    u = unknown[order]
    wk = np.ones(len(known)) if wk is None else wk
    wu = np.ones(len(unknown)) if wu is None else wu
    cum = np.concatenate([[0.0], np.cumsum(wu[order])])
    left, right = np.searchsorted(u, known, "left"), np.searchsorted(u, known, "right")
    return float((wk * (cum[left] + (cum[right] - cum[left]) / 2)).sum() / (wk.sum() * cum[-1]))


def main():
    global D, ARMS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=500)
    p.add_argument("--dir", type=Path, default=D)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--out", default="classmean_auroc.json", help="output name in --dir (J4: a new name keeps J2's file)")
    a = p.parse_args()
    D, ARMS = a.dir, tuple(a.arms.split(","))
    REF = ARMS[0]  # the paired reference (ctl for J1/J2; frz for J4b against the frozen head)
    torch.set_num_threads(4)
    rows = pd.read_parquet(D / "rows.parquet")
    uniq = rows.path.drop_duplicates()
    emb_row = pd.Series(np.arange(len(uniq)), index=uniq.to_numpy())[rows.path.to_numpy()].to_numpy()
    tables = {arm: pd.read_parquet(D / f"{arm}.parquet") for arm in ARMS}
    base = tables[REF]
    assert (base.key.to_numpy() == rows.key.to_numpy()).all() and (base.set.to_numpy() == rows.set.to_numpy()).all()
    ref = (base.set == "ref").to_numpy()
    sp_codes, species = pd.factorize(base.speciesKey[ref])
    rng = np.random.default_rng(1)
    check = rng.choice(np.flatnonzero(~ref), 3000, replace=False)
    for arm in ARMS:
        E = load_embeddings(arm)
        assert E.shape[0] == len(uniq), f"{arm}: {E.shape[0]} embeddings for {len(uniq)} unique paths"
        R = E[emb_row[ref]].float()
        M = torch.zeros(len(species), R.shape[1]).index_add_(0, torch.from_numpy(sp_codes), R)
        M = torch.nn.functional.normalize(M, dim=1)
        q = np.flatnonzero(~ref)
        cm = np.full(len(base), np.nan, np.float32)
        for s in range(0, len(q), 8192):
            idx = q[s:s + 8192]
            cm[idx] = (E[emb_row[idx]].float() @ M.T).max(1).values.numpy()
        knn = np.concatenate([(E[emb_row[check[s:s + 1000]]].float() @ R.T).max(1).values.numpy()
                              for s in range(0, len(check), 1000)])
        gap = np.abs(knn - tables[arm].knn1.to_numpy()[check]).max()
        print(f"{arm}: 1-NN recomputed on 3,000 queries, largest gap to mt's knn1 {gap:.4f}", flush=True)
        assert gap < 2e-2, "embedding rows do not map to the manifest"
        tables[arm]["cm_max"] = cm
    out = {}
    for known_set in ("known_bal", "known_unif"):
        known = (base.set == known_set).to_numpy()
        k_codes, k_sp = pd.factorize(base.speciesKey[known])
        boot_rng = np.random.default_rng(0)
        out[known_set] = {}
        for stratum, sets in STRATA.items():
            unknown = base.set.isin(sets).to_numpy()
            u_codes, u_sp = pd.factorize(base.speciesKey[unknown])
            vals = {(arm, s): (tables[arm][s].to_numpy()[known], tables[arm][s].to_numpy()[unknown])
                    for arm in ARMS for s in SCORES}
            point = {key: auroc(*v) for key, v in vals.items()}
            boot = {key: [] for key in vals}
            for _ in range(a.reps):
                wk = np.bincount(boot_rng.integers(0, len(k_sp), len(k_sp)), minlength=len(k_sp))[k_codes].astype(float)
                wu = np.bincount(boot_rng.integers(0, len(u_sp), len(u_sp)), minlength=len(u_sp))[u_codes].astype(float)
                for key, (k, u) in vals.items():
                    boot[key].append(auroc(k, u, wk, wu))
            res = {}
            for s in SCORES:
                res[s] = {arm: {"auroc": point[(arm, s)],
                                "ci": list(np.percentile(boot[(arm, s)], [2.5, 97.5]))} for arm in ARMS}
                for arm in ARMS[1:]:
                    d = np.array(boot[(arm, s)]) - np.array(boot[(REF, s)])
                    res[s][f"{arm}-{REF}"] = {"auroc": point[(arm, s)] - point[(REF, s)],
                                            "ci": list(np.percentile(d, [2.5, 97.5]))}
            d = np.array(boot[(REF, "cm_max")]) - np.array(boot[(REF, "maxcos")])
            res[f"{REF}_cm_minus_maxcos"] = {"auroc": point[(REF, "cm_max")] - point[(REF, "maxcos")],
                                          "ci": list(np.percentile(d, [2.5, 97.5]))}
            out[known_set][stratum] = res
            print(f"{known_set:10s} {stratum:4s} " + "  ".join(
                f"{s} " + " / ".join(f"{point[(arm, s)]:.3f}" for arm in ARMS) for s in SCORES), flush=True)
    (D / a.out).write_text(json.dumps(out, indent=1))
    print("wrote", D / a.out)


if __name__ == "__main__":
    main()
