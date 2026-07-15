"""Testing / prediction counterpart to dev/028 (plain-softmax multi-head x fastai).

dev/028 exports a self-contained fastai `.pkl` (via `learn.export`), unlike dev/030's raw
`.pt` state-dict bundle, so it needs its own loader -- but shares dev/032's inference,
mini_metrics-CSV and metrics-report logic verbatim (imported from it), so the two test
scripts produce byte-for-byte comparable output.

`MultiHead`/`MultiHeadLoss` are redefined here (structurally identical to dev/028's) rather
than imported: fastai's `.pkl` records classes by their pickling-time module, which was
`__main__` (dev/028 always runs as a script), so unpickling from a *different* module name
fails. Redefining them under this script's own `__main__` makes them resolve correctly.

Usage: python dev/033_multihead_v4_test.py --config <test_config.yaml>
"""

import argparse
import importlib
from os.path import exists, join
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing

torch.multiprocessing.set_start_method("fork", force=True)  # see dev/030 for why

import yaml
from torch import nn

from fastai.vision.all import load_learner

v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod032 = importlib.import_module("032_hierarchical_heads_test")

HIERARCHY_LEVELS = v4.HIERARCHY_LEVELS
VALID_CONFIG_VERSIONS = [1.0]


class MultiHead(nn.Module):
    """Structurally identical to dev/028's `MultiHead`; see module docstring for why this
    is redefined here instead of imported."""

    def __init__(self, nf, n_classes_per_level):
        super().__init__()
        from fastai.vision.all import create_head

        self.heads = nn.ModuleList([create_head(nf, n) for n in n_classes_per_level])

    def forward(self, x):
        return [head(x) for head in self.heads]


class MultiHeadLoss(nn.Module):
    def __init__(self, level_weights=None, label_smoothing=0.0):
        super().__init__()
        self.weights = level_weights or [1.0] * len(HIERARCHY_LEVELS)
        self.loss_fns = nn.ModuleList([
            nn.CrossEntropyLoss(label_smoothing=label_smoothing) for _ in HIERARCHY_LEVELS
        ])

    def forward(self, preds, *targs):
        return sum(w * fn(p, t) for w, fn, p, t in zip(self.weights, self.loss_fns, preds, targs))


def test(
    model_path: str,
    parquet_path: str,
    img_dir: str,
    out_dir: str,
    eval_name: str,
    test_set: str = "0",
    min_img_per_spc: int = 0,
    family_filter: list = [],
    optimal_threshold: bool = True,
    threshold: list = None,
):
    model_path, img_dir, parquet_path = Path(model_path), Path(img_dir), Path(parquet_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_path.stem

    print(f"Loading model: {model_path}")
    learn = load_learner(model_path, cpu=not torch.cuda.is_available())
    learn.model.eval()
    vocabs = learn.vocabs

    print(f"Loading test set (fold '{test_set}') from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = v4.filter_df(df, keep_in=[test_set], min_img_per_spc=min_img_per_spc, family_filter=family_filter)
    df = mod032.filter_known_species(df, vocabs)
    if len(df) == 0:
        raise ValueError(f"No rows in test fold '{test_set}' of {parquet_path}.")
    df["image_path"] = df["speciesKey"].astype(str) + "/" + df["filename"]
    for level in HIERARCHY_LEVELS:
        df[level] = df[level].astype(str)
    test_df = df[["image_path", *HIERARCHY_LEVELS]].reset_index(drop=True)
    print(f"Test images: {len(test_df)} | species present: {test_df['speciesKey'].nunique()}")

    print("Running inference...")
    preds, confs = mod032.predict(learn, test_df, img_dir, vocabs, device)
    labels = test_df[HIERARCHY_LEVELS].to_numpy().astype(str)
    filenames = test_df["image_path"].to_numpy()

    out_base = Path(out_dir) / model_name / eval_name
    out_base.mkdir(parents=True, exist_ok=True)

    combos_path = out_base / "combinations.csv"
    learn.hierarchy[HIERARCHY_LEVELS].astype(str).to_csv(combos_path, index=False)

    preds_path = out_base / "predictions.csv"
    mod032.save_predictions_csv(preds_path, filenames, preds, confs, labels, vocabs)
    print(f"Predictions written: {preds_path}")

    metrics_base = out_base / "metrics"
    mod032.compute_metrics(preds_path, metrics_base, combos_path, optimal_threshold, threshold)
    print(f"\nMetric report saved to: {out_base}/metrics.csv (+ .json)")
    return out_base


def cli(config_path: str = None):
    if config_path is None:
        parser = argparse.ArgumentParser(description="Test a dev/028 multi-head model + mini_metrics report.")
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
    cfg["eval_name"] = mod032.create_eval_name(cfg)
    out_base = test(**cfg)
    copyfile(config_path, join(out_base, "test_config.yaml"))


if __name__ == "__main__":
    cli()
