"""Phase B4 of the lepi-app plan: turn "confidence" into a number that means something.

The proposal greys a taxon name when its confidence falls below 0.5. On this model that
number does not mean what a user would read it to mean. Softmax over 12,041 cosine-derived
logits is systematically overconfident, so 0.5 is not "the model is 50% sure" -- and greying
a correct answer at 0.49 while highlighting a wrong one at 0.51 is worse than showing no
confidence at all, because the UI is making an explicit reliability claim on the model's
behalf.

This script replaces the guess with a derivation, in two stages:

1. **Temperature scaling** (Guo et al. 2017): fit a single scalar T on the *validation* fold
   by minimizing NLL of `logits / T`. One parameter, so it cannot overfit; it does not change
   any argmax, so accuracy is untouched by construction -- it only makes the probability
   attached to that argmax honest.

2. **Precision-targeted thresholds.** For each level, find the smallest calibrated confidence
   at which precision-among-shown reaches a target (default 0.80): "when this app shows you a
   species name un-greyed, it is right about 80% of the time". That is a claim a user can
   understand and the project can defend, unlike "p > 0.5".

The split discipline matters and is the reason both folds appear here: T and the thresholds
are fitted on the **validation** fold (the run's `fold`, '1'), and then *reported* on the
held-out **test** fold ('0'). A threshold chosen and quoted on the same data is a threshold
that will not survive contact with a user. The gap between the fitted target and the achieved
test precision is the honest error bar, and it is printed.

Emits `calibration.json` and `thresholds.json` for the artifact bundle, so the app never
hardcodes a threshold and every deployed build states the precision it was tuned for.

Usage:
    python dev/044_calibrate.py --onnx <bundle>/model.onnx --out-dir <bundle> \
        --target-precision 0.8 -n 20000
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing

torch.multiprocessing.set_start_method("fork", force=True)  # see dev/030 for why

v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod032 = importlib.import_module("032_hierarchical_heads_test")
mod042 = importlib.import_module("042_marginalize")
mod043 = importlib.import_module("043_quantize")

HIERARCHY_LEVELS = v4.HIERARCHY_LEVELS
LEVEL_NAMES = ["species", "genus", "family"]


# ---------------------------------------------------------------------------
# Logit collection (keeps only the top-1 per level, which is all calibration needs)
# ---------------------------------------------------------------------------

TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 96))


def collect_logits(onnx_path, test_df, img_dir, vocabs, sp2gn, gn2fm, n_gn, n_fm,
                   batch_size, num_workers, img_size, aug_img_size):
    """Return, per level, everything the calibration needs -- without keeping the logit matrix.

    Retaining full logits is not an option (12,041 floats x 20,000 images is ~1 GB per level),
    but the NLL of a temperature-scaled distribution, `logsumexp(z/T) - z_y/T`, genuinely needs
    the whole row: `logsumexp(z/T)` cannot be recovered from `logsumexp(z)` and the top-1.

    The way out is that T is one-dimensional. `logsumexp(z/T)` is accumulated per image for a
    fixed grid of 96 candidate temperatures as the batches stream past, which is exact at every
    grid point and costs 96 floats per image instead of 12,041. The fit is then a lookup over
    the grid rather than a gradient descent -- with one parameter and a log-spaced grid spanning
    0.25-8.0, the grid resolution is far finer than the noise on the estimate.

    Note the inputs here are already log-probabilities. That is harmless:
    `softmax(log_softmax(z)/T) == softmax(z/T)`, since the two differ by a constant shift that
    softmax removes -- so scaling log-probs is the same calibration as scaling raw logits.
    """
    import onnxruntime as ort
    from fastai.vision.all import imagenet_stats

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    img_dir = Path(img_dir)  # see dev/043: a str prefix silently mis-joins
    dls = v4.make_dls(test_df, vocabs, img_dir, aug_img_size, img_size, batch_size,
                      num_workers, lowmem=False)
    dl = dls.test_dl(test_df, num_workers=num_workers)
    mean = torch.tensor(imagenet_stats[0]).view(1, 3, 1, 1)
    std = torch.tensor(imagenet_stats[1]).view(1, 3, 1, 1)

    idx = {level: {str(k): i for i, k in enumerate(vocabs[level])} for level in HIERARCHY_LEVELS}
    y_all = {name: torch.tensor([idx[level][k] for k in test_df[level]])
             for name, level in zip(LEVEL_NAMES, HIERARCHY_LEVELS)}

    grid = torch.tensor(TEMPERATURE_GRID, dtype=torch.float32)
    acc = {name: {"top1_idx": [], "top1_logit": [], "lse_grid": [], "true_logit": []}
           for name in LEVEL_NAMES}
    pos = 0
    for batch in dl:
        x01 = (batch[0].float().cpu() * std + mean).numpy()
        out = sess.run(None, {"image": np.ascontiguousarray(x01, dtype=np.float32)})
        bs = out[0].shape[0]
        sl = slice(pos, pos + bs)
        pos += bs

        log_sp = torch.log_softmax(torch.from_numpy(out[0]).float(), dim=1)
        log_gn = mod042.scatter_logsumexp(log_sp, sp2gn, n_gn)
        log_fm = mod042.scatter_logsumexp(log_gn, gn2fm, n_fm)
        # The marginal path is the one the app will ship (dev/042), so it is the one calibrated.
        per_level = {"species": log_sp, "genus": log_gn, "family": log_fm}

        for name, z in per_level.items():
            top_logit, top_idx = z.max(dim=1)
            acc[name]["top1_idx"].append(top_idx)
            acc[name]["top1_logit"].append(top_logit)
            # [batch, n_temperatures]: logsumexp(z/T) at every candidate T, accumulated now
            # because z is about to go out of scope and cannot be reconstructed later.
            acc[name]["lse_grid"].append(
                torch.logsumexp(z.unsqueeze(-1) / grid.view(1, 1, -1), dim=1)
            )
            acc[name]["true_logit"].append(z.gather(1, y_all[name][sl].unsqueeze(1)).squeeze(1))

    return ({name: {k: torch.cat(v) for k, v in d.items()} for name, d in acc.items()},
            {k: v[:pos] for k, v in y_all.items()})


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(stats):
    """Pick the grid temperature minimizing NLL: `mean(logsumexp(z/T) - z_y/T)`.

    Returns (T, index into TEMPERATURE_GRID, nll_at_T, nll_at_T=1) so the caller can report how
    much calibration actually bought -- a temperature very close to 1, or an NLL that barely
    moves, is worth knowing rather than silently shipping.
    """
    grid = torch.tensor(TEMPERATURE_GRID, dtype=torch.float32)
    nll = (stats["lse_grid"] - stats["true_logit"].unsqueeze(1) / grid.view(1, -1)).mean(dim=0)
    i = int(nll.argmin())
    j = int(np.abs(TEMPERATURE_GRID - 1.0).argmin())
    return float(TEMPERATURE_GRID[i]), i, float(nll[i]), float(nll[j])


def calibrated_confidence(stats, t_index):
    """P(top-1) at the chosen grid temperature."""
    t = float(TEMPERATURE_GRID[t_index])
    return (stats["top1_logit"] / t - stats["lse_grid"][:, t_index]).exp().numpy()


# ---------------------------------------------------------------------------
# Precision-targeted thresholds
# ---------------------------------------------------------------------------

def choose_threshold(conf, correct, target_precision):
    """Lowest confidence cut whose precision-among-shown reaches `target_precision`.

    Lowest rather than any: every increment above it costs coverage (names needlessly greyed)
    for precision the target did not ask for. Returns None if the target is unreachable at any
    cut, which is a real answer -- it means this level cannot support the claim, and the UI
    should not make it.
    """
    order = np.argsort(-conf)
    c_sorted, correct_sorted = conf[order], correct[order].astype(np.float64)
    running_precision = np.cumsum(correct_sorted) / np.arange(1, len(c_sorted) + 1)
    ok = np.where(running_precision >= target_precision)[0]
    if len(ok) == 0:
        return None
    # Deepest prefix still meeting the target = lowest threshold = maximum coverage.
    return float(c_sorted[ok[-1]])


def report(conf, correct, thr):
    if thr is None:
        return {"threshold": None, "coverage": 0.0, "precision_among_shown": None}
    shown = conf >= thr
    return {
        "threshold": thr,
        "coverage": float(shown.mean()),
        "precision_among_shown": float(correct[shown].mean()) if shown.any() else None,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(onnx_path, out_dir, parquet_path, img_dir, n, val_set, test_set, min_img_per_spc,
        batch_size, num_workers, img_size, aug_img_size, seed, target_precision):
    onnx_path, out_dir = Path(onnx_path), Path(out_dir)
    tax = json.loads((onnx_path.parent / "taxonomy.json").read_text())
    vocabs = {level: tax["vocabs"][name] for name, level in zip(LEVEL_NAMES, HIERARCHY_LEVELS)}
    sp2gn = torch.tensor(tax["parents"]["species_to_genus"])
    gn2fm = torch.tensor(tax["parents"]["genus_to_family"])
    n_gn, n_fm = len(vocabs[HIERARCHY_LEVELS[1]]), len(vocabs[HIERARCHY_LEVELS[2]])

    out = {}
    for tag, fold in [("val", val_set), ("test", test_set)]:
        print(f"\n--- {tag} fold '{fold}' ---")
        df = mod043.build_test_df(parquet_path, img_dir, n, fold, min_img_per_spc, vocabs, seed)
        print(f"{len(df)} images.")
        stats, y = collect_logits(onnx_path, df, img_dir, vocabs, sp2gn, gn2fm, n_gn, n_fm,
                                  batch_size, num_workers, img_size, aug_img_size)
        out[tag] = (stats, y)

    val_stats, val_y = out["val"]
    test_stats, test_y = out["test"]

    calibration, thresholds, summary, sweep = {}, {}, [], []
    for name in LEVEL_NAMES:
        t, t_i, nll_t, nll_1 = fit_temperature(val_stats[name])
        calibration[name] = {"temperature": t, "val_nll": nll_t, "val_nll_uncalibrated": nll_1}

        v_conf = calibrated_confidence(val_stats[name], t_i)
        v_correct = (val_stats[name]["top1_idx"].numpy() == val_y[name].numpy())
        thr = choose_threshold(v_conf, v_correct, target_precision)

        t_conf = calibrated_confidence(test_stats[name], t_i)
        t_correct = (test_stats[name]["top1_idx"].numpy() == test_y[name].numpy())

        # The sweep is free -- inference is already done and a threshold is a sort plus a
        # cumulative mean -- and it is the part that tells you whether the *target* was
        # sensible. A target below the model's unconditional accuracy produces a threshold
        # nothing ever falls under, i.e. a confidence display that never greys anything.
        for tp in (0.80, 0.90, 0.95, 0.98, 0.99):
            t_thr = choose_threshold(v_conf, v_correct, tp)
            r = report(t_conf, t_correct, t_thr)
            sweep.append({
                "level": name, "target": tp, "threshold": t_thr,
                "test_precision": r["precision_among_shown"], "test_coverage": r["coverage"],
            })

        val_r, test_r = report(v_conf, v_correct, thr), report(t_conf, t_correct, thr)
        thresholds[name] = {
            "threshold": thr,
            "target_precision": target_precision,
            "fitted_on": f"fold {val_set}",
            "achieved_on_test": test_r["precision_among_shown"],
            "coverage_on_test": test_r["coverage"],
        }
        summary.append({
            "level": name, "T": t, "nll_T": nll_t, "nll_T1": nll_1, "threshold": thr,
            "val_prec": val_r["precision_among_shown"], "val_cov": val_r["coverage"],
            "test_prec": test_r["precision_among_shown"], "test_cov": test_r["coverage"],
            "raw_top1_acc_test": float(t_correct.mean()),
        })

    print("\n" + "=" * 88)
    print(f"Calibration + thresholds (target precision {target_precision:.2f})")
    print("=" * 88)
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n'coverage' = fraction of images whose name is shown un-greyed.")
    print("'test_prec' is the claim the UI can honestly make; the gap to target is the "
          "generalization cost of fitting the cut on the validation fold.")
    print("\nThreshold vs target precision (threshold fitted on val, measured on test):")
    print(pd.DataFrame(sweep).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("A target at or below the model's unconditional accuracy yields coverage ~1.0: "
          "nothing is ever greyed, and the confidence display carries no information.")
    print("=" * 88)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps({
        "method": "temperature scaling (Guo et al. 2017), fitted per level on the validation fold",
        "fitted_on": f"fold {val_set}",
        "n_images": int(len(val_y["species"])),
        "temperatures": calibration,
    }, indent=2))
    (out_dir / "thresholds.json").write_text(json.dumps({
        "note": "Apply temperature from calibration.json, then grey any level whose calibrated "
                "top-1 probability is below its threshold. Thresholds are precision-targeted, "
                "not round numbers -- see journal/2026-07-lepi-app-claude.md D2.",
        "path": "marginal (genus/family derived from species; see dev/042)",
        "levels": thresholds,
        "sweep": sweep,
    }, indent=2))
    print(f"Wrote {out_dir/'calibration.json'} and {out_dir/'thresholds.json'}")
    return summary


def cli():
    p = argparse.ArgumentParser(description="Temperature calibration + precision-targeted thresholds (Phase B4).")
    p.add_argument("--onnx", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--parquet", default="../data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet")
    p.add_argument("--img-dir", default="../data/global/images")
    p.add_argument("-n", type=int, default=20000, help="Images per fold.")
    p.add_argument("--val-set", default="1", help="Validation fold (the run's `fold`).")
    p.add_argument("--test-set", default="0", help="Held-out test fold.")
    # 0, matching dev/032's test path -- the one that produced the project's 0.9148 headline.
    # A non-zero value here silently restricts the test fold to well-represented species (50
    # gives 3,696 species / 484k images instead of 12,632 / 633k) and inflates every number.
    p.add_argument("--min-img-per-spc", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--aug-img-size", type=int, default=460)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-precision", type=float, default=0.8)
    a = p.parse_args()
    run(a.onnx, a.out_dir, a.parquet, a.img_dir, a.n, a.val_set, a.test_set, a.min_img_per_spc,
        a.batch_size, a.num_workers, a.img_size, a.aug_img_size, a.seed, a.target_precision)


if __name__ == "__main__":
    cli()
