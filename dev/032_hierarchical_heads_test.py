"""Testing / prediction counterpart to dev/030 (hierarchical heads x fastai).

Loads a model trained by `dev/030_hierarchical_heads_benchmark.py` (its `.pt` checkpoint
bundle: model_state_dict + head/arch/vocabs/cls2idx/hierarchy), runs inference over a test
set, writes predictions in the `mini_metrics` long CSV schema, then invokes `mini_metrics`
to compute and save a metric report.

Two input modes:
  - parquet mode: evaluate the held-out fold of a parquet (global test set = fold '0', which
    dev/030 removes from training). Ground-truth labels come from the parquet. This is the
    real test path, and (via the small parquet) the local placeholder for the not-yet-present
    independent test set.
  - (folder mode for a truly external image-folder set can be added later; the independent
    set isn't on this server yet, so the small parquet stands in as the placeholder.)

Output layout (self-contained per model x eval-set, so re-evals never clobber):
    <dataset>/preds/<model_name>/<eval_name>/
        predictions.csv     # mini_metrics input (instance_id, filename, level, label,
                            #   prediction, confidence, threshold, known_label)
        metrics.csv         # mini_metrics metric table
        metrics.json        # mini_metrics full metrics (--all)
        combinations.csv    # the species->genus->family hierarchy used (reproducibility)
        test_config.yaml    # copy of the config used

Usage: python dev/032_hierarchical_heads_test.py --config <test_config.yaml>
"""

import argparse
import glob
import importlib
import json
from collections import OrderedDict
from os.path import exists, join
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing

torch.multiprocessing.set_start_method("fork", force=True)  # see dev/030 for why

import yaml

from fastai.vision.all import vision_learner
from mini_trainer.hierarchical.integration import sparse_masks_from_labels
from mini_metrics.metrics import main as mini_metrics_main

v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod030 = importlib.import_module("030_hierarchical_heads_benchmark")

HIERARCHY_LEVELS = v4.HIERARCHY_LEVELS  # [speciesKey, genusKey, familyKey], finest -> coarsest
VALID_CONFIG_VERSIONS = [1.0]


# ---------------------------------------------------------------------------
# Model reconstruction from a dev/030 checkpoint
# ---------------------------------------------------------------------------

def class_spec_from_hierarchy(hierarchy_df, vocabs, cls2idx):
    """Rebuild mini_trainer's `sparse_masks` from the saved hierarchy + vocabs, matching what
    dev/030.build_class_spec did at train time (same cls2idx, so head masks line up).

    Restricted to species the model actually knows. The two sides derive this from different
    places: dev/030 builds its masks from the *training dataframe*, but saves `hierarchy` --
    which `gen_df` reads from a pre-existing `hierarchy.csv` covering the whole dataset. Those
    agree only when no filtering removed species relative to that file. Under `family_filter`
    (or a raised `min_img_per_spc`, or simply a stale hierarchy.csv) the saved hierarchy names
    species that were never trained and have no cls2idx entry, and this raised
    `KeyError: '<speciesKey>'` from inside sparse_masks_from_labels -- after training had
    finished and the checkpoint was written.

    Filtering here is the correct reading rather than a workaround: a mask over classes the
    head has no output for is meaningless, and cls2idx is authoritative about what the model
    can predict.
    """
    known = set(cls2idx["0"])  # level 0 = species; keys are the vocab's string labels
    labels = OrderedDict(
        (row.speciesKey, tuple(str(getattr(row, level)) for level in HIERARCHY_LEVELS))
        for row in hierarchy_df.itertuples(index=False)
        if str(row.speciesKey) in known
    )
    dropped = len(hierarchy_df) - len(labels)
    if dropped:
        print(f"Hierarchy: kept {len(labels)} of {len(hierarchy_df)} rows "
              f"({dropped} species in hierarchy.csv were not in the model's vocab -- "
              f"filtered out at training time).")
    if not labels:
        raise ValueError(
            "No species in the saved hierarchy are in the model's vocab. The checkpoint's "
            "hierarchy and cls2idx disagree entirely -- check that hierarchy.csv matches the "
            "dataset this model was trained on."
        )
    return sparse_masks_from_labels(labels, cls2idx)


def load_model(checkpoint, dls, decoder_num_layers, decoder_nhead):
    """Reconstruct the exact dev/030 model (backbone + MT head) and load its weights.

    Uses `vision_learner` with the same args dev/030 used so the module structure matches
    the saved state_dict (`nn.Sequential(body, MTHeadAdapter)`), then loads the state_dict.
    """
    vocabs = checkpoint["vocabs"]
    cls2idx = checkpoint["cls2idx"]
    head = checkpoint["head"]
    hierarchy_df = pd.DataFrame(checkpoint["hierarchy"])
    n_classes = [len(vocabs[level]) for level in HIERARCHY_LEVELS]

    sparse_masks = class_spec_from_hierarchy(hierarchy_df, vocabs, cls2idx)

    arch = getattr(importlib.import_module("fastai.vision.all"), checkpoint["model_arch_name"])
    nf = v4.body_out_features(arch)
    custom_head = mod030.build_head(
        head, nf, n_classes, cls2idx, sparse_masks,
        decoder_kwargs={"num_layers": decoder_num_layers, "nhead": decoder_nhead},
    )
    learn = vision_learner(
        dls, arch, n_out=1, custom_head=custom_head, init=None,
        loss_func=lambda *a, **k: torch.tensor(0.0),  # unused at inference
    )
    learn.model.load_state_dict(checkpoint["model_state_dict"])
    learn.model.eval()
    return learn


def filter_known_species(df, vocabs):
    """Drop test rows whose speciesKey wasn't in the training vocab.

    The global test fold ('0') is a fixed 10-fold split taken before any species-level
    filtering, so it can contain species that `min_img_per_spc` (or an absent family)
    excluded from training entirely. Those species have no vocab index -- the model can
    never predict them -- so leaving them in inflates the effective class count and drags
    down macro metrics on classes the model was never given a chance to learn.
    """
    known = set(vocabs["speciesKey"])
    n_before, s_before = len(df), df["speciesKey"].nunique()
    df = df[df["speciesKey"].astype(str).isin(known)]
    s_after = df["speciesKey"].nunique()
    print(f"Filtered to species seen in training: {n_before} -> {len(df)} rows, "
          f"{s_before} -> {s_after} species.")
    return df


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(learn, test_df, img_dir, vocabs, device):
    """Run the model over `test_df` (order-preserving) and return per-level top-1 label +
    softmax confidence arrays of shape (n_images, n_levels).

    Reduces to top-1 label/confidence per batch instead of accumulating raw logits for the
    whole test set: at global scale (632k images x 12041 species) the full species logit
    matrix alone is ~30GB (float32), ~42GB across all three levels -- unnecessary and
    fragile when the compact per-image result is a few bytes. Streaming keeps peak memory
    at one batch's logits regardless of test-set size.
    """
    test_dl = learn.dls.test_dl(test_df, num_workers=learn.dls.train.num_workers)
    learn.model.to(device)
    vocab_arrays = [np.array([str(v) for v in vocabs[level]]) for level in HIERARCHY_LEVELS]
    pred_chunks = [[] for _ in HIERARCHY_LEVELS]
    conf_chunks = [[] for _ in HIERARCHY_LEVELS]
    for batch in test_dl:
        xb = batch[0].to(device)
        out = learn.model(xb)  # list of per-level logit tensors
        for i in range(len(HIERARCHY_LEVELS)):
            probs = torch.softmax(out[i].float(), dim=1)
            conf, idx = probs.max(dim=1)
            pred_chunks[i].append(vocab_arrays[i][idx.cpu().numpy()])
            conf_chunks[i].append(conf.cpu().numpy())

    preds = [np.concatenate(chunks) for chunks in pred_chunks]
    confs = [np.concatenate(chunks) for chunks in conf_chunks]
    return np.stack(preds, axis=1), np.stack(confs, axis=1)


# ---------------------------------------------------------------------------
# Checkpoint lookup
# ---------------------------------------------------------------------------

def resolve_model_path(model_path):
    """Resolve `model_path`, which may be a glob, to a single checkpoint.

    dev/030 writes to `<out_dir>/<timestamp>-<model_name>/<model_name>.pt`, and the timestamp
    is minted when the run *starts* -- so a test config written in advance cannot name the
    checkpoint a training run is about to produce. That matters for a chained train-then-test
    job (e.g. UCloud batch), where both configs are committed before either runs. A glob lets
    the config say "whichever run of this model_name", resolved at test time:

        model_path: /work/lepinet/data/ucloud_models/*-heads-...-ucloud/heads-...-ucloud.pt

    Newest match wins (by mtime), so a re-run supersedes its predecessor rather than picking
    an arbitrary one. Plain paths pass through untouched.
    """
    path = Path(model_path)
    if not glob.has_magic(str(model_path)):
        return path
    matches = [Path(p) for p in glob.glob(str(model_path))]
    if not matches:
        raise FileNotFoundError(f"No checkpoint matches the glob: {model_path}")
    newest = max(matches, key=lambda p: p.stat().st_mtime)
    if len(matches) > 1:
        others = ", ".join(p.parent.name for p in sorted(matches) if p != newest)
        print(f"Glob matched {len(matches)} checkpoints; using newest ({newest.parent.name}); "
              f"ignoring: {others}")
    return newest


# ---------------------------------------------------------------------------
# mini_metrics-format prediction CSV (long: one row per image x level)
# ---------------------------------------------------------------------------

def save_predictions_csv(path, filenames, preds, confs, labels, vocabs, threshold=0.5):
    """`threshold` is the abstention threshold recorded per row: mini_metrics counts a
    prediction as made only where `confidence >= threshold`. Accepts a scalar (same for every
    level) or a per-level list, matching the config's `threshold` key.

    Note this column is only what the CSV *records*: `mini_metrics.main(threshold=...)`
    overwrites it in memory before computing anything. It is written faithfully anyway so the
    saved predictions state the threshold they were actually evaluated at, rather than a stale
    default that contradicts metrics.json sitting next to it.
    """
    n, p = preds.shape
    known = {i: {str(v) for v in vocabs[level]} for i, level in enumerate(HIERARCHY_LEVELS)}
    if isinstance(threshold, (list, tuple)):
        if len(threshold) == 1:
            thr_col = np.full(n * p, threshold[0], dtype=float)
        elif len(threshold) == p:
            thr_col = np.tile(np.asarray(threshold, dtype=float), n)
        else:
            raise ValueError(f"threshold has {len(threshold)} entries; expected 1 or {p} "
                             f"(one per level: {HIERARCHY_LEVELS}).")
    else:
        thr_col = float(threshold)
    df = pd.DataFrame({
        "instance_id": np.repeat(np.arange(n), p),
        "filename": np.repeat(filenames, p),
        "level": np.tile(np.arange(p), n),
        "label": labels.flatten(order="C").astype(str),
        "prediction": preds.flatten(order="C").astype(str),
        "confidence": confs.flatten(order="C"),
        "threshold": thr_col,
    })
    df["known_label"] = [lbl in known[lvl] for lvl, lbl in zip(df["level"], df["label"])]
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def test(
    model_path: str,
    parquet_path: str,
    img_dir: str,
    out_dir: str,
    eval_name: str,
    test_set: str = "0",
    min_img_per_spc: int = 0,
    family_filter: list = [],
    batch_size: int = 64,
    aug_img_size: int = 460,
    img_size: int = 256,
    num_workers: int = None,
    optimal_threshold: bool = True,
    threshold: list = None,
    decoder_num_layers: int = 4,
    decoder_nhead: int = 1,
):
    model_path = resolve_model_path(model_path)
    img_dir, parquet_path = Path(img_dir), Path(parquet_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    vocabs = checkpoint["vocabs"]
    model_name = model_path.stem

    # --- build the test dataframe (held-out fold of the parquet) ---
    print(f"Loading test set (fold '{test_set}') from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = v4.filter_df(df, keep_in=[test_set], min_img_per_spc=min_img_per_spc, family_filter=family_filter)
    df = filter_known_species(df, vocabs)
    if len(df) == 0:
        raise ValueError(f"No rows in test fold '{test_set}' of {parquet_path}.")
    df["image_path"] = df["speciesKey"].astype(str) + "/" + df["filename"]
    for level in HIERARCHY_LEVELS:
        df[level] = df[level].astype(str)
    # split-agnostic dummy is_valid so `make_dls` builds cleanly; we use test_dl regardless.
    df["is_valid"] = np.arange(len(df)) % 5 == 0
    test_df = df[["image_path", "is_valid", *HIERARCHY_LEVELS]].reset_index(drop=True)
    print(f"Test images: {len(test_df)} | species present: {test_df['speciesKey'].nunique()}")

    # --- reconstruct model (dls only used for its transform pipeline + vocab) ---
    # lowmem=False here on purpose: inference feeds `learn.dls.test_dl(test_df)` a DataFrame,
    # but the lowmem DataBlock's getters index numpy arrays by integer, so a DataFrame row
    # reaches them as a pandas.Series and fastai's type dispatch aborts ("got pandas.Series").
    # The test is a single pass over one fold, so the per-worker memory the lowmem path saves
    # does not matter here; the DataFrame path is the one test_dl was built for.
    dls = v4.make_dls(test_df, vocabs, img_dir, aug_img_size, img_size, batch_size, num_workers,
                      lowmem=False)
    learn = load_model(checkpoint, dls, decoder_num_layers, decoder_nhead)
    print(f"Model loaded (head={checkpoint['head']}, arch={checkpoint['model_arch_name']}).")

    # --- inference ---
    print("Running inference...")
    preds, confs = predict(learn, test_df, img_dir, vocabs, device)
    labels = test_df[HIERARCHY_LEVELS].to_numpy().astype(str)
    filenames = test_df["image_path"].to_numpy()

    # --- output layout: <out_dir>/<model_name>/<eval_name>/ ---
    out_base = Path(out_dir) / model_name / eval_name
    out_base.mkdir(parents=True, exist_ok=True)

    combos_path = out_base / "combinations.csv"
    pd.DataFrame(checkpoint["hierarchy"])[HIERARCHY_LEVELS].astype(str).to_csv(combos_path, index=False)

    preds_path = out_base / "predictions.csv"
    # A fixed `threshold` from the config is the one actually applied downstream; with
    # optimal_threshold the value isn't known until mini_metrics has probed for it, so the
    # 0.5 default stands in and pass 2 overwrites it in memory anyway.
    save_predictions_csv(preds_path, filenames, preds, confs, labels, vocabs,
                         threshold=threshold if threshold is not None else 0.5)
    print(f"Predictions written: {preds_path}")

    # --- metrics via mini_metrics ---
    metrics_base = out_base / "metrics"
    compute_metrics(preds_path, metrics_base, combos_path, optimal_threshold, threshold)
    print(f"\nMetric report saved to: {out_base}/metrics.csv (+ .json)")
    return out_base


def compute_metrics(preds_path, metrics_base, combos_path, optimal_threshold, threshold):
    """Run mini_metrics and save `metrics.csv`/`metrics.json`.

    mini_metrics' own `--optimal` flag path is broken in the installed version (it calls
    `OptimalConfidenceThreshold(df, progress=...)` with a kwarg `compute()` doesn't accept),
    so we don't use it. Instead: if per-level optimal thresholds are requested, do a cheap
    first pass at threshold 0.5 (the optimal threshold is reported as a metric regardless),
    read the reported `optimal_confidence_threshold`, then re-evaluate at those thresholds."""
    def run(thr):
        mini_metrics_main(file=str(preds_path), output=str(metrics_base),
                          combinations=str(combos_path), optimal=False,
                          threshold=list(thr), all=True, verbose=1)

    if threshold is not None:
        print("Computing metrics with mini_metrics (fixed threshold)...")
        run(threshold)
        return
    if not optimal_threshold:
        print("Computing metrics with mini_metrics (threshold 0.5)...")
        run([0.5])
        return

    print("Computing metrics with mini_metrics (pass 1: probe optimal thresholds)...")
    mini_metrics_main(file=str(preds_path), output=str(metrics_base), combinations=str(combos_path),
                      optimal=False, threshold=[0.5], all=True, verbose=0)
    opt = json.load(open(f"{metrics_base}.json")).get("optimal_confidence_threshold", {})
    thr = [opt[k] for k in sorted(opt, key=int)]
    print(f"Optimal per-level thresholds: {thr}")
    print("Computing metrics with mini_metrics (pass 2: at optimal thresholds)...")
    run(thr)


def create_eval_name(config):
    """Descriptive, collision-resistant eval-set folder name."""
    from datetime import datetime
    base = config.get("eval_name") or f"{Path(config['parquet_path']).stem[:24]}-set{config.get('test_set', '0')}"
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{base}"


def cli(config_path: str = None):
    if config_path is None:
        parser = argparse.ArgumentParser(description="Test a dev/030 hierarchical-head model + mini_metrics report.")
        parser.add_argument("-c", "--config", type=str, help="Path to test config file.")
        config_path = parser.parse_args().config
    if not exists(config_path):
        raise FileNotFoundError(f"Path to config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert float(config["version"]) in VALID_CONFIG_VERSIONS, (
        f"Wrong config version: {config['version']}. Must be in {VALID_CONFIG_VERSIONS}."
    )
    cfg = dict(config["test"])
    cfg["eval_name"] = create_eval_name(cfg)
    out_base = test(**cfg)
    copyfile(config_path, join(out_base, "test_config.yaml"))


if __name__ == "__main__":
    cli()
