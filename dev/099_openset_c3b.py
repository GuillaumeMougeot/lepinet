"""Open-set AUROC on C3b's withheld taxa, from mini_trainer's per-image summaries (campaign, 2026-11-05).

Reads, per run, ``summary.parquet`` and ``classes.json`` from mini_trainer's ``dev/lepinet/openset.py``
(per image: ``path``, ``label_<r>``, and per rank ``z1_<r>``, ``z2_<r>``, ``lse_<r>``, ``entropy_<r>``),
and the C3b manifest from ``dev/072``. Labels each image known / near / mid / far by the model's own
vocabulary, as ``dev/059``'s ``stratify`` does, keeps only the manifest's scoreable species among the
novel ones, and reports the AUROC of each stratum against the known images for ``dev/061``'s five
rules at every rank. All rows are used (``dev/059`` sampled 8,000 per stratum for inference cost).
Rules that carry no information for a rank's output are flagged: where every ``logsumexp`` is 0 the
output is already log-probabilities, so energy is constant and msp is monotone in max (``dev/061``).

    python dev/099_openset_c3b.py --run data/ucloud_preds/c-openset/os_v2s_bu_s42 [--run ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

RULES = {  # novelty score, higher = more novel (dev/061's SCORE_RULES on the summary columns)
    "max": lambda d, r: -d[f"z1_{r}"],
    "energy": lambda d, r: -d[f"lse_{r}"],
    "msp": lambda d, r: -np.exp(d[f"z1_{r}"] - d[f"lse_{r}"]),
    "entropy": lambda d, r: d[f"entropy_{r}"],
    "margin": lambda d, r: -(d[f"z1_{r}"] - d[f"z2_{r}"]),
}
STRATA = ("near", "mid", "far")


def auroc(novel: np.ndarray, known: np.ndarray) -> float:
    """P(score_novel > score_known) by Mann-Whitney, ties at half weight."""
    if len(novel) == 0 or len(known) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([known, novel]))
    n_k, n_o = len(known), len(novel)
    return float((ranks[n_k:].sum() - n_o * (n_o + 1) / 2) / (n_k * n_o))


def vocabularies(classes, n_ranks: int) -> list[set[str]]:
    """The model's class names per rank, from ``classes.json`` (one mapping per rank, or one flat one)."""
    if isinstance(classes, dict) and all(isinstance(v, dict) for v in classes.values()):
        classes = list(classes.values())
    if isinstance(classes, dict):
        classes = [classes]
    if len(classes) != n_ranks:
        raise SystemExit(f"classes.json has {len(classes)} ranks, the summary {n_ranks}")
    return [{str(k) for k in c} for c in classes]


def stratify(d: pd.DataFrame, vocab: list[set[str]]) -> pd.Series:
    new = [~d[f"label_{r}"].astype(str).isin(vocab[r]) for r in range(3)]
    out = pd.Series("known", index=d.index, dtype=object)
    out[new[0]] = "near"
    out[new[0] & new[1]] = "mid"
    out[new[0] & new[1] & new[2]] = "far"
    return out


def score_run(run: Path, manifest: dict) -> pd.DataFrame:
    d = pd.read_parquet(run / "summary.parquet")
    n_ranks = sum(c.startswith("z1_") for c in d.columns)
    vocab = vocabularies(json.loads((run / "classes.json").read_text()), n_ranks)
    d["_strat"] = stratify(d, vocab)
    allowed = set().union(*({str(s) for s in v} for v in manifest["score_species"].values()))
    novel = d["_strat"] != "known"
    missing = allowed - set(d.loc[novel, "label_0"].astype(str))
    if len(missing) > len(allowed) / 2:
        raise SystemExit(f"{run.name}: {len(missing)}/{len(allowed)} scoreable species are not novel "
                         "for this model; wrong model, or trained on the full parquet")
    d = d[~novel | d["label_0"].astype(str).isin(allowed)]
    known = d["_strat"] == "known"
    rows = []
    for r in range(n_ranks):
        flat = bool((d[f"lse_{r}"].abs() < 1e-3).all())
        for rule, fn in RULES.items():
            s = fn(d, r).to_numpy(dtype=np.float64)
            for stratum in (*STRATA, "all_novel"):
                pick = (d["_strat"] != "known") if stratum == "all_novel" else (d["_strat"] == stratum)
                rows.append({"run": run.name, "rank": r, "rule": rule, "stratum": stratum,
                             "auroc": auroc(s[pick.to_numpy()], s[known.to_numpy()]),
                             "n_known": int(known.sum()), "n_novel": int(pick.sum()),
                             "degenerate": flat and rule in ("energy", "msp")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--manifest", type=Path,
                    default=Path("/data/au761367/lepinet/ucloud_mirror/data/splits/c3b_holdout.json"))
    ap.add_argument("--out", type=Path, default=Path("data/openset_c3b/auroc.csv"))
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    res = pd.concat([score_run(r, manifest) for r in a.run], ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    sp = res[res["rank"] == 0].pivot_table(index=["run", "rule"], columns="stratum", values="auroc")
    print(sp[[*STRATA, "all_novel"]].round(4).to_string())
    n = res[(res["rank"] == 0) & (res["rule"] == "max")].set_index(["run", "stratum"])["n_novel"]
    print(f"\nnovel images per stratum: {n.to_dict()}")
    bad = res[res["degenerate"]][["run", "rank", "rule"]].drop_duplicates()
    if len(bad):
        print(f"degenerate (log-probability output; not findings):\n{bad.to_string(index=False)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
