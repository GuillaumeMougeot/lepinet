"""Turn "confidence" into a number that means something, and pick thresholds that defend a claim.

A softmax over 12,041 cosine-derived logits is systematically **over**confident, so a UI that greys
a name below 0.5 is making a reliability claim the model does not support — and greying a correct
answer at 0.49 while highlighting a wrong one at 0.51 is worse than showing no confidence at all.

Two stages replace the guess with a derivation:

1. **Temperature scaling** (Guo et al. 2017). Fit one scalar $T$ minimising NLL of $z/T$. One
   parameter, so it cannot overfit, and it changes no argmax — accuracy is untouched *by
   construction*; only the number attached to the answer becomes honest.
2. **Precision-targeted thresholds.** Per level, the smallest calibrated confidence at which
   precision-among-shown reaches a target: *"when this app shows a species name un-greyed, it is
   right about 95 % of the time"* — a claim a user can understand and the project can defend.

**The split discipline is the point, not a detail.** $T$ and the thresholds are fitted on the
**validation** fold and then *reported* on the held-out **test** fold. A threshold chosen and quoted
on the same data will not survive contact with a user; the gap between the fitted target and the
achieved test precision is the honest error bar, and it is emitted.

Ported from `dev/044` (which predates the package and imports the old `dev/028` chain). The clever
part is preserved and explained in :func:`collect_stats`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: 96 log-spaced candidate temperatures. A grid rather than gradient descent because T is
#: one-dimensional: at this resolution the spacing is far finer than the noise on the estimate,
#: and a grid lets the fit be a *streaming* accumulation (see :func:`collect_stats`).
TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 96))


# --------------------------------------------------------------------------- pure core
# These four take arrays and return numbers. They are the part worth testing, and they are tested
# on synthetic data in tests/test_calibrate.py -- no model, no images, no GPU.

def fit_temperature(stats: dict) -> tuple[float, int, float, float]:
    """Grid temperature minimising NLL ``mean(logsumexp(z/T) - z_y/T)``.

    Returns ``(T, grid_index, nll_at_T, nll_at_T1)`` — the last so a caller can report how much
    calibration actually bought. A T very close to 1, or an NLL that barely moves, is worth knowing
    rather than silently shipping.
    """
    grid = torch.tensor(TEMPERATURE_GRID, dtype=torch.float32)
    nll = (stats["lse_grid"] - stats["true_logit"].unsqueeze(1) / grid.view(1, -1)).mean(dim=0)
    i = int(nll.argmin())
    j = int(np.abs(TEMPERATURE_GRID - 1.0).argmin())
    return float(TEMPERATURE_GRID[i]), i, float(nll[i]), float(nll[j])


def calibrated_confidence(stats: dict, t_index: int) -> np.ndarray:
    """``P(top-1)`` at the chosen grid temperature."""
    t = float(TEMPERATURE_GRID[t_index])
    return (stats["top1_logit"] / t - stats["lse_grid"][:, t_index]).exp().numpy()


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target_precision: float) -> float | None:
    """Lowest confidence cut whose precision-among-shown reaches ``target_precision``.

    *Lowest* rather than any: every increment above it costs coverage — names needlessly greyed —
    for precision the target did not ask for. Returns ``None`` when the target is unreachable at any
    cut, which is a real answer: that level cannot support the claim and the UI should not make it.
    """
    order = np.argsort(-conf)
    c_sorted, correct_sorted = conf[order], correct[order].astype(np.float64)
    running_precision = np.cumsum(correct_sorted) / np.arange(1, len(c_sorted) + 1)
    ok = np.where(running_precision >= target_precision)[0]
    if len(ok) == 0:
        return None
    # Deepest prefix still meeting the target == lowest threshold == maximum coverage.
    return float(c_sorted[ok[-1]])


def report(conf: np.ndarray, correct: np.ndarray, thr: float | None) -> dict:
    """Coverage and achieved precision at a threshold — what the user actually experiences."""
    if thr is None:
        return {"threshold": None, "coverage": 0.0, "precision_among_shown": None}
    shown = conf >= thr
    return {"threshold": thr,
            "coverage": float(shown.mean()),
            "precision_among_shown": float(correct[shown].mean()) if shown.any() else None}


# --------------------------------------------------------------------------- collection

@torch.no_grad()
def collect_stats(model, dls, df, levels, vocabs, sparse_masks, device, num_workers=8):
    """Everything the calibration needs, **without ever holding the logit matrix**.

    Retaining full logits is not an option — 12,041 floats x 20,000 images is ~1 GB per level — but
    the NLL of a temperature-scaled distribution, ``logsumexp(z/T) - z_y/T``, genuinely needs the
    whole row: ``logsumexp(z/T)`` cannot be recovered from ``logsumexp(z)`` and the top-1.

    The way out is that T is one-dimensional. ``logsumexp(z/T)`` is accumulated per image for the
    fixed 96-point grid as batches stream past — exact at every grid point, and **96 floats per
    image instead of 12,041**. The fit is then a lookup rather than an optimisation.

    Coarse levels are the **marginals** of the species posterior, because that is the path the app
    ships; calibrating anything else would calibrate a model nobody runs.
    """
    from .export import scatter_logsumexp
    from .test import dl_num_workers

    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    dl = dls.test_dl(df, num_workers=nw)
    model.to(device).eval()

    idx = {lv: {str(k): i for i, k in enumerate(vocabs[lv])} for lv in levels}
    y_all = {lv: torch.tensor([idx[lv][str(k)] for k in df[lv]]) for lv in levels}
    grid = torch.tensor(TEMPERATURE_GRID, dtype=torch.float32)
    acc = {lv: {"top1_idx": [], "top1_logit": [], "lse_grid": [], "true_logit": []} for lv in levels}

    pos = 0
    for batch in dl:
        out = model(batch[0].to(device))
        z0 = (out[0] if isinstance(out, (list, tuple)) else out).float().cpu()
        log_sp = torch.log_softmax(z0, dim=1)
        per_level = [log_sp]
        cur = log_sp
        for i, mask in enumerate(sparse_masks):
            cur = scatter_logsumexp(cur, mask, len(vocabs[levels[i + 1]]))
            per_level.append(cur)

        bs = z0.shape[0]
        sl = slice(pos, pos + bs)
        pos += bs
        for lv, z in zip(levels, per_level):
            top_logit, top_idx = z.max(dim=1)
            acc[lv]["top1_idx"].append(top_idx)
            acc[lv]["top1_logit"].append(top_logit)
            # [batch, 96] -- accumulated now because z is about to go out of scope and cannot be
            # reconstructed from what we keep.
            acc[lv]["lse_grid"].append(torch.logsumexp(z.unsqueeze(-1) / grid.view(1, 1, -1), dim=1))
            acc[lv]["true_logit"].append(z.gather(1, y_all[lv][sl].unsqueeze(1)).squeeze(1))

    return ({lv: {k: torch.cat(v) for k, v in d.items()} for lv, d in acc.items()},
            {lv: v[:pos] for lv, v in y_all.items()})


# --------------------------------------------------------------------------- orchestration

def write_calibration(out_dir: str | Path, val_stats: dict, val_y: dict,
                      test_stats: dict | None, test_y: dict | None,
                      levels: list[str], target_precision: float = 0.95) -> dict[str, Any]:
    """Fit on validation, report on test, and emit the two JSON files the app reads."""
    out_dir = Path(out_dir)
    calibration, thresholds, achieved = {}, {}, {}

    for lv in levels:
        t, t_idx, nll_t, nll_1 = fit_temperature(val_stats[lv])
        calibration[lv] = {"temperature": t, "val_nll": nll_t, "val_nll_uncalibrated": nll_1}

        conf = calibrated_confidence(val_stats[lv], t_idx)
        correct = (val_stats[lv]["top1_idx"] == val_y[lv]).numpy()
        thr = choose_threshold(conf, correct, target_precision)
        thresholds[lv] = report(conf, correct, thr) | {"target_precision": target_precision}

        if test_stats is not None:
            tconf = calibrated_confidence(test_stats[lv], t_idx)
            tcorrect = (test_stats[lv]["top1_idx"] == test_y[lv]).numpy()
            achieved[lv] = report(tconf, tcorrect, thr)

    (out_dir / "calibration.json").write_text(json.dumps({
        "note": "Divide logits by the level's temperature before softmax. Fitted on the validation "
                "fold; does not change any argmax, only the confidence attached to it.",
        "temperatures": calibration,
    }, indent=2))
    (out_dir / "thresholds.json").write_text(json.dumps({
        "note": "Apply the temperature from calibration.json, then grey any level whose calibrated "
                "confidence is below its threshold. Thresholds were FITTED on validation and "
                "VERIFIED on the held-out test fold ('achieved'); trust 'achieved'.",
        "target_precision": target_precision,
        "levels": thresholds,
        "achieved_on_test": achieved or None,
    }, indent=2))
    return {"temperatures": calibration, "fitted": thresholds, "achieved_on_test": achieved}


def build_names(parquet_path: str | Path, taxonomy_path: str | Path, out_path: str | Path) -> dict:
    """Map taxon keys to display names, **aligned to the taxonomy vocab order**.

    The model is key-based (GBIF taxon keys); the app should show readable names and be able to
    index them by the same logit index it uses for everything else. Species get the clean binomial
    ``Genus epithet`` (dropping GBIF authorship), falling back to ``scientificName``. An empty
    string means unknown, and the app falls back to showing the key.

    Ported from `dev/047`.
    """
    import pandas as pd

    tax = json.loads(Path(taxonomy_path).read_text())
    cols = ["speciesKey", "genusKey", "familyKey", "scientificName", "genus", "family",
            "specificEpithet"]
    df = pd.read_parquet(parquet_path, columns=cols)

    def binomial(row):
        g = str(row["genus"]) if pd.notna(row["genus"]) else ""
        e = str(row["specificEpithet"]) if pd.notna(row["specificEpithet"]) else ""
        return f"{g} {e}".strip() or (str(row["scientificName"]) if pd.notna(row["scientificName"]) else "")

    sp = df.dropna(subset=["speciesKey"]).drop_duplicates("speciesKey")
    maps = {
        "species": {str(k): binomial(r) for k, r in zip(sp["speciesKey"], sp.to_dict("records"))},
        "genus": {str(k): (str(v) if pd.notna(v) else "") for k, v in
                  zip(*[df.dropna(subset=["genusKey"]).drop_duplicates("genusKey")[c]
                        for c in ("genusKey", "genus")])},
        "family": {str(k): (str(v) if pd.notna(v) else "") for k, v in
                   zip(*[df.dropna(subset=["familyKey"]).drop_duplicates("familyKey")[c]
                         for c in ("familyKey", "family")])},
    }
    names = {lv: [maps[lv].get(str(k), "") for k in tax["vocabs"][lv]] for lv in maps}
    missing = {lv: sum(1 for x in v if not x) for lv, v in names.items()}
    Path(out_path).write_text(json.dumps({
        "note": "Display names aligned to taxonomy.json vocab order; empty string = unknown "
                "(the app shows the key).",
        "names": names,
    }, ensure_ascii=False))
    return missing
