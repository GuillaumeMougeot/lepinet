"""Is O1's 17-point useful-answer gap between B8 and P5 a model property or a readout artefact?

Runs the release's own back-off policy (`dev/083`: `fit_thresholds` + `cascade`, thresholds fitted
on half the trap nights with seed 0, verified on the other half) on B8 without its temperature
(O1's probe predictions, `lepinet test` output), B8 with it (the published model's probe eval) and
P5. Also reports what a precision-targeted policy actually depends on (ranking: AURC, coverage at
95 % precision with ties kept together) next to calibration (ECE), and the share of saturated
confidences. Writes paper/figures/fig4_deployment.json for the paper figure.

    python dev/086_deployment_gap.py
Inputs: data/paper_figs/B8-probe-predictions.csv (UCloud: ucloud_preds/B8-probe/...),
data/hf_release/{b8,p5}/eval_probe.parquet. journal/2026-10-02-is-the-deployment-gap-a-readout-artefact.md
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("rel", ROOT / "dev" / "083_hf_release.py")
rel = importlib.util.module_from_spec(_s)
sys.modules["rel"] = rel
_s.loader.exec_module(rel)
LEVELS = rel.LEVELS


def from_lepinet_csv(path: Path) -> pd.DataFrame:
    """`lepinet test` predictions.csv (long: one row per image x level) -> the release's wide frame."""
    d = pd.read_csv(path)
    w = {}
    for i, lv in enumerate(LEVELS):
        x = d[d.level == i].sort_values("instance_id")
        w[f"pred_{lv}"] = x.prediction.astype(str).to_numpy()
        w[f"conf_{lv}"] = x.confidence.astype(float).to_numpy()
        w[f"label_{lv}"] = x.label.astype(str).to_numpy()
        fn = x.filename.astype(str)
    res = pd.DataFrame(w)
    grp = fn.str.extract(r"TRAPNAME_(\w+?)_IMAGENAME_(\d{8})")
    res["group"] = (grp[0] + "_" + grp[1]).to_numpy()
    return res


def split_policy(res: pd.DataFrame, target=0.95) -> dict:
    """Exactly dev/083 cmd_thresholds: seed-0 half of the groups to fit, the other half to verify."""
    groups = np.array(sorted(res["group"].unique(), key=str))
    fit_g = set(np.random.default_rng(0).permutation(groups)[: len(groups) // 2])
    fit, held = res[res["group"].isin(fit_g)], res[~res["group"].isin(fit_g)]
    th = rel.fit_thresholds(fit, target)
    return {"thresholds": th, "held_out": rel.cascade(held, th), "n_fit": len(fit), "n_held": len(held)}


def ranking_and_calibration(conf, correct, target=0.95, bins=15) -> dict:
    order = np.argsort(-conf, kind="stable")
    c, cf = correct[order], conf[order]
    ends = np.r_[np.nonzero(np.diff(cf))[0], len(cf) - 1]       # ties stay together
    n = np.arange(1, len(c) + 1)
    prec, cov = np.cumsum(c)[ends] / n[ends], n[ends] / len(c)
    aurc = float(np.trapezoid(np.r_[1 - prec[0], 1 - prec], np.r_[0, cov]))
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    ece = float(sum((idx == b).mean() * abs(correct[idx == b].mean() - conf[idx == b].mean())
                    for b in range(bins) if (idx == b).any()))
    rel_curve = [{"conf": float(conf[idx == b].mean()), "acc": float(correct[idx == b].mean()),
                  "n": int((idx == b).sum())} for b in range(bins) if (idx == b).any()]
    keep = np.unique(np.r_[0, np.linspace(0, len(cov) - 1, 200).astype(int), len(cov) - 1])
    return {"aurc": aurc, "ece": ece, "saturated": float(np.mean(conf >= 1.0)),
            "coverage_at_target": float(cov[prec >= target].max()) if (prec >= target).any() else 0.0,
            "accuracy": float(correct.mean()),
            "risk_coverage": {"coverage": cov[keep].round(5).tolist(), "precision": prec[keep].round(5).tolist()},
            "reliability": rel_curve}


def main():
    arms = {
        "B8, T = 1 (O1's readout)": from_lepinet_csv(ROOT / "data/paper_figs/B8-probe-predictions.csv"),
        "B8, T = 1.91 (published)": pd.read_parquet(ROOT / "data/hf_release/b8/eval_probe.parquet"),
        "P5 (published, T = 1)": pd.read_parquet(ROOT / "data/hf_release/p5/eval_probe.parquet"),
    }
    out = {}
    for name, res in arms.items():
        res = res[res["label_species"].astype(str) != "-1"]
        sp = ranking_and_calibration(res["conf_species"].to_numpy(float),
                                     (res["pred_species"].astype(str) == res["label_species"].astype(str)).to_numpy(float))
        pol = split_policy(res)
        out[name] = {"policy": pol, "species": sp}
        h = pol["held_out"]
        print(f"{name:28s} useful {h['useful_rate']:.3f}  answered {1 - h['abstain']:.3f}  "
              f"prec {h['precision_answered']:.3f}  th {pol['thresholds']}  | species: acc {sp['accuracy']:.4f} "
              f"saturated {sp['saturated']:.3f} ECE {sp['ece']:.4f} AURC {sp['aurc']:.4f} "
              f"cov@95 {sp['coverage_at_target']:.3f}")
    (ROOT / "paper/figures/fig4_deployment.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
