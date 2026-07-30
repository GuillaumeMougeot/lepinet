"""Rank abstention: how much can the model say, and how often is it right? (C1, no GPU)

The product question is not "what is the species macro-F1" but **"when the photo cannot support a
species call, what does the user get?"** Marginalisation makes the answer principled: the genus
posterior is the sum of its species, so backing off is a *threshold*, not a different model.

Policy evaluated here — the obvious one, and the one an app would ship:

    predict species  if  P(species) >= t_species
    else genus       if  P(genus)   >= t_genus
    else family      if  P(family)  >= t_family
    else abstain ("unknown")

Reported as a **coverage/precision profile**: what fraction of images resolve at each rank, and how
often that answer is correct. Precision *at the rank actually returned* is what a user experiences —
a confident genus is a useful answer, a wrong species is not.

Also reports each rank's standalone precision/coverage curve, so thresholds can be chosen for a
target precision (`--target-precision`).

Input is a `predictions.csv` in mini_metrics long format (one row per image × level), i.e. exactly
what `lepinet test` writes.

    python dev/058_rank_abstention.py --predictions preds.csv --out abstention/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVEL_NAMES = {0: "species", 1: "genus", 2: "family"}


def load_wide(path: str) -> pd.DataFrame:
    """Long mini_metrics format -> one row per image with (pred, conf, correct) per level."""
    df = pd.read_csv(path)
    out = {}
    for lvl, name in LEVEL_NAMES.items():
        s = df[df.level == lvl]
        if s.empty:
            continue
        out[f"{name}_conf"] = s["confidence"].to_numpy()
        out[f"{name}_ok"] = (s["prediction"].astype(str).to_numpy() == s["label"].astype(str).to_numpy())
    return pd.DataFrame(out)


def per_rank_curve(conf: np.ndarray, ok: np.ndarray, n_points: int = 101) -> list[dict]:
    """Precision and coverage as the confidence threshold sweeps 0 -> 1 for a single rank."""
    rows = []
    for t in np.linspace(0.0, 1.0, n_points):
        m = conf >= t
        cov = float(m.mean())
        prec = float(ok[m].mean()) if m.any() else float("nan")
        rows.append({"threshold": round(float(t), 3), "coverage": cov, "precision": prec})
    return rows


def threshold_for_precision(curve: list[dict], target: float) -> dict | None:
    """Lowest threshold (i.e. highest coverage) whose precision reaches the target."""
    ok = [r for r in curve if not np.isnan(r["precision"]) and r["precision"] >= target]
    return min(ok, key=lambda r: r["threshold"]) if ok else None


def cascade(w: pd.DataFrame, t_sp: float, t_gn: float, t_fm: float) -> dict:
    """Back-off policy: species -> genus -> family -> abstain. Precision is *at the rank returned*."""
    sp = w["species_conf"].to_numpy() >= t_sp
    gn = (~sp) & (w["genus_conf"].to_numpy() >= t_gn)
    fm = (~sp) & (~gn) & (w["family_conf"].to_numpy() >= t_fm)
    ab = ~(sp | gn | fm)
    n = len(w)
    res = {"thresholds": {"species": t_sp, "genus": t_gn, "family": t_fm}, "n": int(n), "ranks": {}}
    for name, mask in (("species", sp), ("genus", gn), ("family", fm)):
        okc = w[f"{name}_ok"].to_numpy()[mask]
        res["ranks"][name] = {"coverage": float(mask.mean()),
                              "precision": float(okc.mean()) if mask.any() else None,
                              "n": int(mask.sum())}
    res["ranks"]["abstain"] = {"coverage": float(ab.mean()), "precision": None, "n": int(ab.sum())}
    answered = sp | gn | fm
    correct = (w["species_ok"].to_numpy() & sp) | (w["genus_ok"].to_numpy() & gn) | (w["family_ok"].to_numpy() & fm)
    res["answered_coverage"] = float(answered.mean())
    res["answered_precision"] = float(correct[answered].mean()) if answered.any() else None
    # The honest headline: an abstention counts as not-useful, a wrong answer counts as harmful.
    res["useful_rate"] = float(correct.mean())
    return res


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = load_wide(a.predictions)
    print(f"{len(w)} images | ranks: {[c[:-5] for c in w.columns if c.endswith('_conf')]}")

    curves = {name: per_rank_curve(w[f"{name}_conf"].to_numpy(), w[f"{name}_ok"].to_numpy())
              for name in LEVEL_NAMES.values() if f"{name}_conf" in w}

    print("\nStandalone accuracy at threshold 0 (i.e. always answer):")
    for name, c in curves.items():
        print(f"  {name:8s} precision {c[0]['precision']:.4f}")

    print(f"\nThresholds reaching {a.target_precision:.0%} precision, and what they cost:")
    chosen = {}
    for name, c in curves.items():
        hit = threshold_for_precision(c, a.target_precision)
        chosen[name] = hit["threshold"] if hit else 1.0
        if hit:
            print(f"  {name:8s} t={hit['threshold']:.2f} -> precision {hit['precision']:.4f}, "
                  f"coverage {hit['coverage']:.4f}")
        else:
            print(f"  {name:8s} never reaches {a.target_precision:.0%} (max "
                  f"{max(r['precision'] for r in c if not np.isnan(r['precision'])):.4f})")

    # Thresholds for the coarse ranks MUST be calibrated on the subset they will actually serve —
    # the images where species was uncertain — not on the whole set. Genus is ~97% accurate overall
    # but far worse conditioned on "species was unsure", because the marginal inherits the very
    # uncertainty that triggered the back-off. Choosing t_genus from the global curve silently
    # assumes the hard cases behave like the easy ones; they do not.
    t_sp = chosen.get("species", 1.0)
    hard = w[w["species_conf"].to_numpy() < t_sp]
    cond = {}
    if len(hard):
        print(f"\nConditional calibration on the {len(hard)} images where species < {t_sp:.2f}:")
        for name in ("genus", "family"):
            c = per_rank_curve(hard[f"{name}_conf"].to_numpy(), hard[f"{name}_ok"].to_numpy())
            print(f"  {name:8s} precision at t=0 is {c[0]['precision']:.4f} "
                  f"(vs {curves[name][0]['precision']:.4f} overall)")
            hit = threshold_for_precision(c, a.target_precision)
            cond[name] = hit["threshold"] if hit else None
            if hit:
                print(f"           t={hit['threshold']:.2f} -> precision {hit['precision']:.4f}, "
                      f"covers {hit['coverage']:.2%} of the hard subset")
            else:
                best = max(r["precision"] for r in c if not np.isnan(r["precision"]))
                print(f"           never reaches {a.target_precision:.0%} (max {best:.4f})")
    casc = cascade(w, t_sp, cond.get("genus") or 1.0, cond.get("family") or 1.0)
    print(f"\nBack-off policy at those thresholds ({a.target_precision:.0%} target):")
    for name, r in casc["ranks"].items():
        p = f"{r['precision']:.4f}" if r["precision"] is not None else "   -  "
        print(f"  {name:8s} covers {r['coverage']:7.2%}  precision {p}")
    print(f"  -> answers {casc['answered_coverage']:.2%} of images, "
          f"{casc['answered_precision']:.2%} of those correct; "
          f"useful (answered AND correct) {casc['useful_rate']:.2%}")

    # A sweep of species thresholds, holding coarse ranks permissive: the product trade-off curve.
    sweep = []
    for t in np.linspace(0.0, 0.99, 34):
        sweep.append(cascade(w, float(t), chosen.get("genus", 0.5), 0.0))
    json.dump({"curves": curves, "chosen_thresholds": chosen, "conditional_thresholds": cond,
               "cascade": casc, "sweep": sweep},
              open(out / "abstention.json", "w"), indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        for name, c in curves.items():
            cov = [r["coverage"] for r in c]
            pre = [r["precision"] for r in c]
            ax[0].plot(cov, pre, label=name)
        ax[0].set_xlabel("coverage (fraction answered)")
        ax[0].set_ylabel("precision")
        ax[0].set_title("Per-rank precision vs coverage")
        ax[0].legend()
        ax[0].grid(alpha=.3)
        xs = [s["thresholds"]["species"] for s in sweep]
        ax[1].stackplot(xs,
                        [s["ranks"]["species"]["coverage"] for s in sweep],
                        [s["ranks"]["genus"]["coverage"] for s in sweep],
                        [s["ranks"]["family"]["coverage"] for s in sweep],
                        [s["ranks"]["abstain"]["coverage"] for s in sweep],
                        labels=["species", "genus", "family", "abstain"],
                        colors=["#1b6b4a", "#4c9f70", "#a8c8a0", "#d9d9d9"])
        ax[1].set_xlabel("species confidence threshold")
        ax[1].set_ylabel("fraction of images")
        ax[1].set_title("What rank the user gets as the species bar rises")
        ax[1].legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "abstention.png", dpi=150)
        print(f"\nwrote {out / 'abstention.png'}")
    except Exception as e:
        print("plot skipped:", e)
    print(f"wrote {out / 'abstention.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default="abstention")
    ap.add_argument("--target-precision", type=float, default=0.95)
    main(ap.parse_args())
