"""Re-score prediction files in mini_metrics' current convention, zero and calibrated, next to ours.

The merged paper with Svenning et al. reports macro metrics as mini_metrics computes them since
commit `1132a10` (2026-07-27): classes = species in the truth or in the predictions, out-of-vocabulary
rows kept (`known_only=False`). "zero" scores argmax predictions with no threshold; "calibrated" fits
the macro-F1-optimal confidence threshold per level on a stratified random 10 % split and scores the
other 90 %, repeated over `--seeds` splits (mean and sd). The settings are those of the colleague's
`publication/experiments/statistics/boot_metrics.py`. The column `old` is our convention (rows with an
in-vocabulary label, classes = truth species), for reference.

    PYTHONPATH=data/tools/mini_metrics-61db373 .venv/bin/python dev/093_rescore_mini_metrics.py \
        --set probe --runs data/ucloud_preds/B1-probe data/ucloud_preds/B3rep5x-probe ...
    ... --set probe-ho --heldout --runs ...      # held-out species only, filtered before scoring
    ... --set probe --grouped --runs ...         # calibration split by (trap, night) -> probe-grouped.csv

One row per (run, level, metric) in data/rescore/<set>.csv. A run folder may hold several
predictions.csv; each becomes its own row, named <folder>/<model>/<eval>.
journal/2026-10-22-rescoring-in-the-mini-metrics-convention.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["accuracy", "precision", "recall", "f1", "micro_accuracy", "theilU", "coverage"]


def old_f1(d):
    k = d[d.known_label]
    return f1_score(k.label.astype(str), k.prediction.astype(str), average="macro",
                    labels=sorted(k.label.astype(str).unique()), zero_division=0)


def score(path, heldout, seeds, mm):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroF1, evaluate_file

    d = pd.read_csv(path)
    d["confidence"] = d.confidence.clip(0, 1)   # float32 softmax can exceed 1 by ~1e-7; mini_metrics rejects it
    if heldout is not None:
        species = d.filename.str.split("/").str[0]            # filename is "<species folder>/<crop>"
        keep = d[(d.level == 0) & species.isin(heldout)].instance_id
        d = d[d.instance_id.isin(set(keep))]
    if d.empty:
        raise ValueError("no rows left after the held-out filter")
    tmp = path.parent / ".rescore_tmp.csv"
    d.to_csv(tmp, index=False)
    try:
        df = MetricDF.from_source(str(tmp))
        common = dict(combinations=None, known_only=False, label_filter=None, subsample=None,
                      per_class=False, pattern=None, simple=True, hierarchical=False, verbose=0)
        zero = evaluate_file(source=df, optimal=False, threshold=0.0, **common)
        cal = [evaluate_file(source=df, optimal=True, threshold=None, use_quantiles=True, eps=5e-2,
                             opt_crit=MacroF1, seed=s, **common) for s in range(seeds)]
    finally:
        tmp.unlink(missing_ok=True)
    rows = []
    for lvl in sorted(d.level.unique()):
        dl = d[d.level == lvl]
        for m in METRICS:
            cv = np.array([c[m][lvl] for c in cal if m in c], dtype=float)
            rows.append({"level": int(lvl), "metric": m, "n_rows": len(dl),
                         "old": old_f1(dl) if m == "f1" else np.nan,
                         "zero": zero[m][lvl] if m in zero else np.nan,
                         "cal_mean": cv.mean() if len(cv) else np.nan,
                         "cal_sd": cv.std() if len(cv) else np.nan})
    return rows


TRAP_RE = r"TRAPNAME_([^_]+)_IMAGENAME_(\d{14})"


def trap_night(filenames, groups_parquet):
    """(trap, night) per crop, nights running midday to midday: from the split's groups, else the filename."""
    g = pd.read_parquet(groups_parquet, columns=["image_path", "group"]).set_index("image_path").group
    out = pd.Series(filenames.map(g).values, index=filenames.index)
    miss = out.isna()
    if miss.any():
        m = filenames[miss].str.extract(TRAP_RE)
        ts = pd.to_datetime(m[1], format="%Y%m%d%H%M%S", errors="coerce") - pd.Timedelta(hours=12)
        out[miss] = m[0] + "|" + ts.dt.strftime("%Y-%m-%d")
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} crops without a trap and night")
    return out


def score_grouped(path, heldout, seeds, groups_parquet, cal_frac=0.1):
    """Calibrated scores with whole (trap, night) groups on one side of the calibration split.

    mini_metrics' own split is random over images, stratified by label; on trap images it puts frames of
    the same moth on both sides. Here a seeded ~10 % of the groups calibrates the per-level threshold
    (mini_metrics' OptimalConfidenceThreshold, same settings), and the rest is scored with it.
    """
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroF1, OptimalConfidenceThreshold, evaluate_file

    d = pd.read_csv(path)
    d["confidence"] = d.confidence.clip(0, 1)
    if heldout is not None:
        species = d.filename.str.split("/").str[0]
        keep = d[(d.level == 0) & species.isin(heldout)].instance_id
        d = d[d.instance_id.isin(set(keep))]
    first = d[d.level == 0].set_index("instance_id").filename
    grp = trap_night(first, groups_parquet)
    groups = np.array(sorted(grp.unique()))
    tmp = path.parent / ".rescore_tmp.csv"
    res = []
    try:
        for s in range(seeds):
            rng = np.random.default_rng(s)
            cal_groups = set(rng.permutation(groups)[: max(1, round(cal_frac * len(groups)))])
            cal_ids = set(grp.index[grp.isin(cal_groups)])
            parts = {}
            for side, mask in (("cal", d.instance_id.isin(cal_ids)), ("rep", ~d.instance_id.isin(cal_ids))):
                d[mask].to_csv(tmp, index=False)
                parts[side] = MetricDF.from_source(str(tmp))
            thr = OptimalConfidenceThreshold(crit=MacroF1, use_quantiles=True, eps=5e-2)(parts["cal"], filter=False, verbose=0)
            thr = [float(v) for _, v in sorted(thr.items())] if isinstance(thr, dict) else float(thr)
            m = evaluate_file(source=parts["rep"], optimal=False, threshold=thr, combinations=None, known_only=False,
                              label_filter=None, subsample=None, per_class=False, pattern=None, simple=True,
                              hierarchical=False, verbose=0)
            res.append((m, len(cal_ids)))
    finally:
        tmp.unlink(missing_ok=True)
    rows = []
    for lvl in sorted(d.level.unique()):
        for k in METRICS:
            v = np.array([m[k][lvl] for m, _ in res if k in m], dtype=float)
            rows.append({"level": int(lvl), "metric": k, "n_rows": int((d.level == lvl).sum()),
                         "n_groups": len(groups), "cal_images_mean": float(np.mean([n for _, n in res])),
                         "calg_mean": v.mean() if len(v) else np.nan, "calg_sd": v.std() if len(v) else np.nan})
    return rows


def main(a):
    sys.path.insert(0, str(ROOT / a.mm))
    import mini_metrics
    heldout = None
    if a.heldout:
        heldout = {str(s) for s in json.loads(Path(a.split_summary).read_text())["held_out_species"]}
    out = ROOT / "data" / "rescore" / f"{a.set}{'-grouped' if a.grouped else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = pd.read_csv(out) if out.exists() and not a.overwrite else pd.DataFrame(columns=["run"])
    done = set(prev.run)
    rows = []
    found = sum(1 for r in a.runs for _ in Path(r).rglob("predictions.csv"))
    for r in a.runs:
        for p in sorted(Path(r).rglob("predictions.csv")):
            name = f"{Path(r).name}/{p.parent.parent.name}/{p.parent.name}"
            if name in done:
                continue
            t = time.time()
            try:
                res = (score_grouped(p, heldout, a.seeds, a.groups) if a.grouped
                       else score(p, heldout, a.seeds, mini_metrics))
            except Exception as e:  # report and carry on: one broken file must not stop the sweep
                print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)
                continue
            for x in res:
                x["run"] = name
            rows += res
            f1 = min((x for x in res if x["metric"] == "f1"), key=lambda x: x["level"])
            if a.grouped:
                print(f"{name}: grouped cal {f1['calg_mean']:.4f}±{f1['calg_sd']:.4f} ({time.time() - t:.0f}s)", flush=True)
            else:
                print(f"{name}: old {f1['old']:.4f} zero {f1['zero']:.4f} cal {f1['cal_mean']:.4f}"
                      f"±{f1['cal_sd']:.4f} ({time.time() - t:.0f}s)", flush=True)
            pd.concat([prev, pd.DataFrame(rows)]).to_csv(out, index=False)
            progress(a.set + ("-grouped" if a.grouped else ""), len(done) + len({x["run"] for x in rows}), found)
    progress(a.set + ("-grouped" if a.grouped else ""), len(done) + len({x["run"] for x in rows}), found)
    print(f"-> {out}")


def progress(name, n_done, n_found):
    """One line per evaluation set in data/rescore/progress.md, so a later session sees where it stopped."""
    p = ROOT / "data" / "rescore" / "progress.md"
    lines = [ln for ln in (p.read_text().splitlines() if p.exists() else []) if not ln.startswith(f"| {name} |")]
    if not lines:
        lines = ["| set | runs done / found | updated |", "|---|---|---|"]
    lines.append(f"| {name} | {n_done} / {max(n_done, n_found)} | {time.strftime('%Y-%m-%d %H:%M')} |")
    p.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="name of the evaluation set, e.g. probe, probe-ho, trap, indist")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--heldout", action="store_true", help="keep only the held-out species' rows")
    ap.add_argument("--split-summary", default="data/flemming/splits/split_summary.json")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--mm", default="data/tools/mini_metrics-61db373")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--grouped", action="store_true",
                    help="calibrate on whole (trap, night) groups (trap images only) -> <set>-grouped.csv")
    ap.add_argument("--groups", default="data/flemming/splits/flemming_groups.parquet")
    main(ap.parse_args())
