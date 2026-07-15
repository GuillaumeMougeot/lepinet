"""v4 hierarchical Lepidoptera classifier: species / genus / family.

Earlier attempts:
  - 011/014: flattened the hierarchy into one multi-label head (correct fastai
    idioms via `vision_learner`, but the three levels share one sigmoid output
    and can't be trained/weighted independently).
  - 022/025: introduced real per-level heads, but built the model by hand
    (`create_body` + a raw `Learner`), which silently dropped pretrained
    weights and the body/head param-group split that `fine_tune()` needs to
    freeze anything.

This version keeps the real multi-head architecture from 022 but builds it
through `vision_learner(..., custom_head=...)` instead of by hand, so
pretrained weights, ImageNet normalization and the body/head split all come
from fastai itself and `fine_tune()` behaves as documented.
"""

import argparse
import asyncio
import importlib
from datetime import datetime
from os.path import exists, join
from pathlib import Path
from shutil import copyfile

import aiohttp
import pandas as pd
import yaml
import torch.multiprocessing

# Python 3.14 defaults the Linux multiprocessing start method to 'forkserver', which must
# pickle the DataLoader worker state to hand it to the worker -- and that state holds a
# GPU-resident tensor from aug_transforms' warm-up batch, which CUDA can't pickle. fastai
# only forces 'fork' on macOS, so do it here (before any CUDA/DataLoader is created).
torch.multiprocessing.set_start_method("fork", force=True)

from torch import nn

from fastai.callback.core import Callback
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.metrics import Metric
from fastai.vision.all import (
    CategoryBlock,
    ColReader,
    ColSplitter,
    CSVLogger,
    DataBlock,
    GradientClip,
    ImageBlock,
    Normalize,
    Resize,
    SaveModelCallback,
    aug_transforms,
    create_body,
    create_head,
    imagenet_stats,
    num_features_model,
    vision_learner,
)

VALID_CONFIG_VERSIONS = [1.0]

# Fine -> coarse. Order is load-bearing: it's the order of the model's heads,
# of `get_y`, and of every per-level list below.
HIERARCHY_LEVELS = ["speciesKey", "genusKey", "familyKey"]


class GCCallback(Callback):
    """Force a gen-0 garbage collection after each batch. fastai's Learner holds each batch's
    prediction/loss (with its autograd graph) as attributes, and the Learner<->callback
    references form a reference cycle -- so the batch's graph is only freed by Python's cyclic
    GC, which at fp16/high-alloc rates doesn't run often enough. Without this, GPU memory
    climbs every batch (batch-size-independently) until CUDA OOMs (~25GB on resnet18 here).
    gen-0 only: the cycle dies within one batch, so it's always young; costs ~ms."""

    def after_batch(self):
        import gc

        gc.collect(0)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def filter_df(df, remove_in=(), keep_in=(), min_img_per_spc=0, family_filter=()):
    df = df.copy()
    if remove_in:
        df = df[~df["set"].isin(remove_in)]
    if keep_in:
        df = df[df["set"].isin(keep_in)]
    if family_filter:
        df = df[df["familyKey"].astype(str).isin(family_filter)]
    if min_img_per_spc > 0:
        df = df[df.groupby("speciesKey")["speciesKey"].transform("count") >= min_img_per_spc]
    print(f"Filtered DataFrame: {len(df)} rows, {df['speciesKey'].nunique()} species.")
    return df


def prepare_df(df, valid_set="1"):
    df = df.copy()
    df["image_path"] = df["speciesKey"].astype(str) + "/" + df["filename"]
    df["is_valid"] = df["set"].eq(valid_set)
    for level in HIERARCHY_LEVELS:
        df[level] = df[level].astype(str)
    return df[["image_path", "is_valid", *HIERARCHY_LEVELS]]


def build_hierarchy(df):
    """One row per species, giving its genus/family ancestry (for readable exports)."""
    return (
        df.groupby("speciesKey")[HIERARCHY_LEVELS]
        .first()
        .reset_index(drop=True)
        .sort_values(HIERARCHY_LEVELS[::-1])
    )


def gen_df(parquet_path, out_dir, min_img_per_spc, fold, hierarchy_path, family_filter):
    """Load+preprocess the metadata parquet, caching the result next to `out_dir`."""
    cache_path = out_dir.parent / parquet_path.with_suffix(".lepinet.parquet").name

    if cache_path.exists() and hierarchy_path.exists():
        print(f"Loading cached preprocessed df: {cache_path}")
        return pd.read_parquet(cache_path), pd.read_csv(hierarchy_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet path not found: {parquet_path}")

    print(f"Loading parquet file {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = filter_df(df, remove_in=["0"], min_img_per_spc=min_img_per_spc, family_filter=family_filter)
    df = prepare_df(df, valid_set=fold)

    if not hierarchy_path.exists():
        build_hierarchy(df).to_csv(hierarchy_path, index=False)
        print(f"Hierarchy saved to {hierarchy_path}")
    hierarchy = pd.read_csv(hierarchy_path)

    df.to_parquet(cache_path, index=False)
    print(f"Preprocessed df cached to {cache_path}")
    return df, hierarchy


def make_dls(df, vocabs, img_dir, aug_img_size, img_size, batch_size, num_workers=None,
             aug_kwargs=None):
    # `aug_kwargs` overrides fastai's `aug_transforms` defaults. fastai's defaults are
    # fairly heavy (perspective warp 0.2, lighting 0.2, zoom 1.1) -- fine for many-epoch
    # runs on smaller data, but on this dataset (millions of images, only a handful of
    # epochs) each image is seen ~nb_epochs times, so heavy distortion trades away clean
    # signal the model never gets enough passes to recover, and mini_trainer's own loop
    # (the comparison target) uses lighter geometric aug + no warp/lighting. Pass e.g.
    # {"max_warp": 0.0, "max_lighting": 0.0, "flip_vert": True, "max_rotate": 15.0} to match.
    aug_kwargs = aug_kwargs or {}
    dblock = DataBlock(
        blocks=(ImageBlock, *(CategoryBlock(vocab=vocabs[level]) for level in HIERARCHY_LEVELS)),
        n_inp=1,
        splitter=ColSplitter(),
        get_x=ColReader("image_path", pref=img_dir),
        get_y=[ColReader(level) for level in HIERARCHY_LEVELS],
        item_tfms=Resize(aug_img_size),
        # `vision_learner`'s automatic ImageNet normalization only fires for
        # archs registered in fastai's `model_meta` (resnet/vgg/densenet/...);
        # efficientnet is not, so add it ourselves to cover every arch.
        batch_tfms=[*aug_transforms(size=img_size, **aug_kwargs), Normalize.from_stats(*imagenet_stats)],
    )
    dl_kwargs = {} if num_workers is None else {"num_workers": num_workers}
    return dblock.dataloaders(df, bs=batch_size, **dl_kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def body_out_features(arch, n_in=3):
    """Channel count of `arch`'s body after cutting, without downloading pretrained weights."""
    body = create_body(arch(weights=None), n_in=n_in, pretrained=False)
    return num_features_model(body)


class MultiHead(nn.Module):
    """One standard fastai classification head per taxonomic level, all reading
    from the same backbone feature map."""

    def __init__(self, nf, n_classes_per_level):
        super().__init__()
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class LevelAccuracy(Metric):
    """Accuracy for a single head, so per-level difficulty (species is much
    harder than family) stays visible instead of being averaged away."""

    def __init__(self, level_idx, name):
        self.level_idx, self._name = level_idx, name
        self.reset()

    def reset(self):
        self.correct = self.total = 0

    def accumulate(self, learn):
        preds, targs = learn.pred[self.level_idx], learn.y[self.level_idx]
        self.correct += (preds.argmax(dim=1) == targs).sum().item()
        self.total += targs.shape[0]

    @property
    def value(self):
        return self.correct / self.total if self.total else None

    @property
    def name(self):
        return self._name


class LevelMacroF1(Metric):
    """Per-level macro-F1 that matches `mini_metrics`' `MacroF1` (the metric the test-set
    report -- and the mini_trainer comparison target of ~0.896 species -- is measured with).

    mini_metrics computes, per class: precision (over rows predicted as that class), recall
    (over rows whose true label is that class), F1 = their harmonic mean (defined as 0 when
    either is 0, e.g. a class that is never predicted), then an unweighted mean over the
    classes that appear as a ground-truth label. This reproduces that at plain argmax (no
    confidence threshold -- training has none), streaming TP/FP/FN so it costs one vector
    per class. Reported per level (NOT averaged across heads like `StreamingF1MultiHead`), so
    species macro-F1 -- which the 3-level `F1(macro)` hides behind the easy genus/family
    heads -- is visible in the CSV every epoch and directly comparable to the test report."""

    def __init__(self, level_idx, name):
        self.level_idx, self._name = level_idx, name
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = None

    def accumulate(self, learn):
        p, y = learn.pred[self.level_idx], learn.y[self.level_idx]
        n = p.shape[1]
        if self.tp is None:
            self.tp, self.fp, self.fn = (p.new_zeros(n) for _ in range(3))
        pred_oh = nn.functional.one_hot(p.argmax(dim=1), n).bool()
        true_oh = nn.functional.one_hot(y, n).bool()
        self.tp += (pred_oh & true_oh).sum(dim=0)
        self.fp += (pred_oh & ~true_oh).sum(dim=0)
        self.fn += (~pred_oh & true_oh).sum(dim=0)

    @property
    def value(self):
        if self.tp is None:
            return None
        precision = self.tp / (self.tp + self.fp)  # 0/0 -> nan for never-predicted classes
        recall = self.tp / (self.tp + self.fn)      # defined for every present class
        denom = precision + recall
        # harmonic mean, but 0 when precision or recall is 0 (matches mini_metrics; also folds
        # the never-predicted nan into 0 since its recall is 0 -> denom is nan -> not > 0).
        f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom))
        present = (self.tp + self.fn) > 0            # classes appearing as a ground-truth label
        return f1[present].mean().item() if present.any() else None

    @property
    def name(self):
        return self._name


class StreamingF1MultiHead(Metric):
    """Streaming (TP/FP/FN-accumulating) macro/micro F1, averaged across heads."""

    def __init__(self, average="macro", name=None):
        assert average in {"macro", "micro"}
        self.average = average
        self._name = name or f"F1_{average}_multihead"
        self.reset()

    def reset(self):
        self.tp, self.fp, self.fn = {}, {}, {}

    def accumulate(self, learn):
        for h, (p, y) in enumerate(zip(learn.pred, learn.y)):
            pred_cls = p.argmax(dim=1)
            n_classes = p.shape[1]
            if h not in self.tp:
                self.tp[h] = p.new_zeros(n_classes)
                self.fp[h] = p.new_zeros(n_classes)
                self.fn[h] = p.new_zeros(n_classes)
            pred_oh = nn.functional.one_hot(pred_cls, n_classes).bool()
            true_oh = nn.functional.one_hot(y, n_classes).bool()
            self.tp[h] += (pred_oh & true_oh).sum(dim=0)
            self.fp[h] += (pred_oh & ~true_oh).sum(dim=0)
            self.fn[h] += (~pred_oh & true_oh).sum(dim=0)

    @property
    def value(self):
        eps = 1e-8
        if self.average == "macro":
            f1s = []
            for h in self.tp:
                precision = self.tp[h] / (self.tp[h] + self.fp[h] + eps)
                recall = self.tp[h] / (self.tp[h] + self.fn[h] + eps)
                f1s.append((2 * precision * recall / (precision + recall + eps)).mean())
            return sum(f1s).item() / len(f1s)
        # Heads have different class counts, so sum each head down to a scalar
        # before combining -- summing the per-class vectors directly would
        # fail (they're different lengths per level).
        tp, fp, fn = (sum(v.sum() for v in d.values()) for d in (self.tp, self.fp, self.fn))
        precision, recall = tp / (tp + fp + eps), tp / (tp + fn + eps)
        return (2 * precision * recall / (precision + recall + eps)).item()

    @property
    def name(self):
        return self._name


# ---------------------------------------------------------------------------
# GBIF name lookup
# ---------------------------------------------------------------------------

async def fetch_name(session, key):
    async with session.get(f"https://api.gbif.org/v1/species/{key}/name") as resp:
        if resp.status != 200:
            return key, None
        r = await resp.json()
        return key, r.get("canonicalName") or r.get("scientificName")


async def fetch_all_names(vocabs):
    async with aiohttp.ClientSession() as session:
        out = {}
        for level, keys in vocabs.items():
            out[level] = dict(await asyncio.gather(*[fetch_name(session, k) for k in keys]))
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    parquet_path: str,
    img_dir: str,
    out_dir: str,
    fold: str,
    model_name: str,
    model_arch_name: str,
    nb_epochs: int,
    batch_size: int,
    aug_img_size: int,
    img_size: int,
    min_img_per_spc: int = 0,
    family_filter: list = [],
    base_lr: float = 1e-3,
    freeze_epochs: int = 1,
    level_weights: list = None,
    label_smoothing: float = 0.0,
    hierarchy_path: str = None,
    num_workers: int = None,
    fp16: bool = True,
    grad_clip: float = 1.0,
    schedule: str = "fine_tune",
):
    parquet_path, img_dir, out_dir = Path(parquet_path), Path(img_dir), Path(out_dir)
    hierarchy_path = Path(hierarchy_path) if hierarchy_path else parquet_path.parent / "hierarchy.csv"

    df, hierarchy = gen_df(parquet_path, out_dir, min_img_per_spc, fold, hierarchy_path, family_filter)

    vocabs = {level: sorted(df[level].unique().tolist()) for level in HIERARCHY_LEVELS}
    n_classes = [len(vocabs[level]) for level in HIERARCHY_LEVELS]
    print(f"Classes per level -- species: {n_classes[0]}, genus: {n_classes[1]}, family: {n_classes[2]}")

    dls = make_dls(df, vocabs, img_dir, aug_img_size, img_size, batch_size, num_workers)

    arch = getattr(importlib.import_module("fastai.vision.all"), model_arch_name)
    nf = body_out_features(arch)

    metrics = [
        *(LevelAccuracy(i, f"acc_{level}") for i, level in enumerate(HIERARCHY_LEVELS)),
        *(LevelMacroF1(i, f"f1_{level}") for i, level in enumerate(HIERARCHY_LEVELS)),
        StreamingF1MultiHead(average="macro", name="F1(macro)"),
        StreamingF1MultiHead(average="micro", name="F1(micro)"),
    ]

    learn = vision_learner(
        dls, arch,
        n_out=1,  # unused: `custom_head` builds the real per-level heads
        custom_head=MultiHead(nf, n_classes),
        loss_func=MultiHeadLoss(level_weights, label_smoothing),
        metrics=metrics,
        model_dir=out_dir / "models",
        cbs=[
            GCCallback(),
            *([GradientClip(grad_clip)] if grad_clip and grad_clip > 0 else []),
            CSVLogger(out_dir / f"{model_name}.csv", append=True),
            TensorBoardCallback(log_dir=out_dir / "tensorboard", trace_model=False, log_preds=False),
            SaveModelCallback(fname=model_name, every_epoch=True),
        ],
    )

    # Mixed precision: the plain softmax heads here are BatchNorm-bounded and use log_softmax,
    # so (unlike mini_trainer's cosine head) they don't overflow fp16 -- safe and ~1.7x faster.
    if fp16:
        learn = learn.to_fp16()

    if schedule in ("one_cycle", "flat_cos"):
        # Unfrozen from step 0 at a uniform base_lr (no freeze phase, no discriminative LR).
        # `fine_tune`'s defaults (1 frozen epoch + backbone LR = base_lr/100) train the
        # backbone ~100x too gently for this far-from-ImageNet, 12k-class problem -- it barely
        # adapts in a few epochs. `one_cycle` keeps one-cycle's warmup+anneal (safer, but its
        # ramp wastes early epochs on a short budget); `flat_cos` holds a high LR right after
        # a short warmup, front-loading convergence -- much better for a fixed 5-epoch budget.
        learn.unfreeze()
        if schedule == "one_cycle":
            learn.fit_one_cycle(nb_epochs, base_lr)
        else:
            learn.fit_flat_cos(nb_epochs, base_lr)
    else:
        # `fine_tune`: train the heads on the frozen backbone first, then unfreeze with a
        # discriminative LR. Fine for ImageNet-like domains; too conservative here.
        learn.fine_tune(nb_epochs, base_lr, freeze_epochs=freeze_epochs)

    # Rebuild a bare learner shell to export: keeps the trained weights but
    # drops the dls/callbacks/metric state from the pickle (same trick 011
    # uses for the flat multi-label model).
    slim_learn = vision_learner(learn.dls, arch, n_out=1, custom_head=MultiHead(nf, n_classes))
    slim_learn.model = learn.model
    slim_learn.hierarchy = hierarchy
    slim_learn.vocabs = vocabs
    slim_learn.hierarchy_levels = HIERARCHY_LEVELS
    # id2name (GBIF key -> readable name) is a convenience only: the model, testing and
    # metrics all key on the numeric GBIF ids, not names. The GBIF species API is unreliable
    # (indexing changes have broken this lookup), so never let it block the actual export --
    # the trained weights are the deliverable. SaveModelCallback has also already written a
    # per-epoch `.pth` checkpoint, so the weights are on disk regardless of what happens here.
    try:
        slim_learn.id2name = asyncio.run(fetch_all_names(vocabs))
    except Exception as e:
        print(f"GBIF name lookup failed ({type(e).__name__}: {e}); exporting without id2name.")
        slim_learn.id2name = {}

    model_path = out_dir / f"{model_name}.pkl"
    slim_learn.export(model_path)
    print(f"Model exported to {model_path}")


def create_out_dir(out_dir, desc):
    """Create the output directory named after datetime-desc."""
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dirname = join(out_dir, current_time + "-" + desc)
    Path(out_dirname).mkdir(parents=True, exist_ok=True)
    return out_dirname


def cli(config_path: str = None):
    if config_path is None:
        parser = argparse.ArgumentParser(description="Hierarchical multi-head training (v4).")
        parser.add_argument("-c", "--config", type=str, help="Path to config file.")
        config_path = parser.parse_args().config

    if not exists(config_path):
        raise FileNotFoundError(f"Path to config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert float(config["version"]) in VALID_CONFIG_VERSIONS, (
        f"Wrong config version: {config['version']}. Must be in {VALID_CONFIG_VERSIONS}."
    )

    config["train"]["out_dir"] = create_out_dir(config["train"]["out_dir"], config["desc"])
    copyfile(config_path, join(config["train"]["out_dir"], "config.yaml"))

    train(**config["train"])


if __name__ == "__main__":
    cli()
