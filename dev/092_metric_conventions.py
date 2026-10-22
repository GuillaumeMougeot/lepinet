"""One predictions.csv, scored under each macro-F1 convention in use: ours and mini_metrics' (the draft
by Svenning et al. and the MAMBO model cards).

    python dev/092_metric_conventions.py PRED.csv [PRED.csv ...]
    PYTHONPATH=<mini_metrics checkout at 61db373 or later> python dev/092_metric_conventions.py ...

Conventions (species level, macro over classes, every class weighted equally):
  A  ours: rows with a label in the training vocabulary, classes = species in the truth
  B  + out-of-vocabulary rows and their species (mini_metrics known_only=False)
  C  classes = truth species + species only ever predicted (they score F1 = 0)
  D  B and C together: mini_metrics' default without a threshold
With mini_metrics importable, also its optimal-threshold score (boot_metrics.py's settings).

journal/2026-10-22-why-the-trap-scores-of-the-two-drafts-differ.md.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd
from sklearn.metrics import f1_score


def mf1(lab, pred, classes):
    return f1_score(lab, pred, average="macro", labels=sorted(classes), zero_division=0)


def main(paths):
    for f in paths:
        d = pd.read_csv(f)
        d = d[d.level == 0].copy()
        d["label"], d["prediction"] = d.label.astype(str), d.prediction.astype(str)
        k = d[d.known_label]
        print(f"{f}\n  rows {len(d)}, out-of-vocabulary rows {(~d.known_label).sum()}, truth species "
              f"{d.label.nunique()}, species predicted but absent from the truth "
              f"{len(set(d.prediction) - set(d.label))}")
        print(f"  A ours                               {mf1(k.label, k.prediction, set(k.label)):.4f}")
        print(f"  B + out-of-vocabulary                {mf1(d.label, d.prediction, set(d.label)):.4f}")
        print(f"  C + predicted-only classes           {mf1(k.label, k.prediction, set(k.label) | set(k.prediction)):.4f}")
        print(f"  D both (mini_metrics, no threshold)  {mf1(d.label, d.prediction, set(d.label) | set(d.prediction)):.4f}")
        try:
            import mini_metrics
            from mini_metrics.metrics import evaluate_file
        except ImportError:
            continue
        tmp = os.path.join(tempfile.mkdtemp(), "p.csv")
        d.to_csv(tmp, index=False)
        m = evaluate_file(source=tmp, combinations=None, optimal=True, threshold=None, known_only=False,
                          label_filter=None, subsample=None, per_class=False, pattern=None, simple=True,
                          hierarchical=False, seed=42, opt_crit=mini_metrics.metrics.MacroF1,
                          use_quantiles=True, eps=5e-2, verbose=0)
        print("  E mini_metrics, optimal threshold    " + "  ".join(
            f"{n} {m[n][0]:.4f}" for n in ("f1", "accuracy", "precision", "recall", "coverage")))


if __name__ == "__main__":
    main(sys.argv[1:])
