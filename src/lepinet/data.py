"""Data: parquet → filtered/cached dataframe, and dataframe → fastai ``DataLoaders``.

Ported from ``dev/028`` (dataframe + ``make_dls``) and ``dev/034`` (rare-class oversampling),
generalized to a **configurable list of hierarchy levels** (``levels``, fine→coarse). The
Lepidoptera default is ``["speciesKey", "genusKey", "familyKey"]``; the image folder is keyed on
the finest level. Two lessons are baked in:

* **Low-memory item access** (``lowmem=True``): the DataBlock iterates integer indices into
  fixed-width numpy arrays instead of DataFrame rows, so forked workers don't dirty a
  copy-on-write page per row read. At many workers this is the difference between fitting and
  OOM-killing the node (``journal/2026-07-17-ucloud-benchmark-oom.md``).
* **Square-root oversampling** (:func:`sample_weights`): reweight *which* rare-species images are
  seen, via fastai's ``WeightedDL`` (samples with replacement, so epoch length and LR-schedule
  timing are preserved). This was the single biggest accuracy lever (+1.7pt,
  ``journal/2026-07-17-does-longtail-help.md``).
"""
from __future__ import annotations

import json
import os
from hashlib import sha1
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_LEVELS = ["speciesKey", "genusKey", "familyKey"]  # fine -> coarse


def ensure_fork_start_method() -> None:
    """Force the ``fork`` multiprocessing start method (idempotent).

    fastai only forces ``fork`` on macOS, but Python 3.14 defaults Linux to ``forkserver``, which
    must pickle DataLoader worker state — and that state can hold a CUDA tensor from the aug
    warm-up batch, which is unpicklable, crashing every ``num_workers>0`` run. Must run before any
    CUDA context / DataLoader is created.
    """
    import torch.multiprocessing as mp

    try:
        if mp.get_start_method(allow_none=True) != "fork":
            mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # context already established; nothing we can do, and usually already fork


# ---------------------------------------------------------------------------
# Dataframe preparation
# ---------------------------------------------------------------------------

def filter_df(df, remove_in=(), keep_in=(), min_img_per_spc=0, family_filter=(), levels=DEFAULT_LEVELS):
    """Filter by ``set`` membership, family, and a minimum image count on the finest level."""
    df = df.copy()
    if remove_in:
        df = df[~df["set"].isin(remove_in)]
    if keep_in:
        df = df[df["set"].isin(keep_in)]
    if family_filter:
        df = df[df["familyKey"].astype(str).isin(family_filter)]
    if min_img_per_spc > 0:
        leaf = levels[0]
        df = df[df.groupby(leaf)[leaf].transform("count") >= min_img_per_spc]
    print(f"Filtered DataFrame: {len(df)} rows, {df[levels[0]].nunique()} {levels[0]}.")
    return df


def prepare_df(df, valid_set="1", levels=DEFAULT_LEVELS):
    """Add ``image_path`` (``<finest_level>/<filename>``), ``is_valid``, and stringified levels.

    A **pre-supplied ``image_path`` column is preserved**, matching what :func:`lepinet.test.evaluate`
    already does. Two cases need it: datasets whose folders are not named by the finest-level key
    (the trap set uses species *names*), and training sets that mix image trees — a row can carry a
    relative escape such as ``../../flemming/images/<...>`` so a single ``img_dir`` reaches both.
    Rebuilding the column unconditionally silently discarded both, which is a data bug that shows up
    only as missing files.
    """
    df = df.copy()
    if "image_path" not in df.columns:
        df["image_path"] = df[levels[0]].astype(str) + "/" + df["filename"]
    df["is_valid"] = df["set"].eq(valid_set)
    for level in levels:
        df[level] = df[level].astype(str)
    return df[["image_path", "is_valid", *levels]]


def build_hierarchy(df, levels=DEFAULT_LEVELS):
    """One row per finest-level class, with its ancestry (for readable exports/checkpoints)."""
    return (
        df.groupby(levels[0])[levels]
        .first()
        .reset_index(drop=True)
        .sort_values(levels[::-1])
    )


def _cache_key(min_img_per_spc, fold, family_filter, levels):
    spec = json.dumps(
        {
            "min_img_per_spc": min_img_per_spc,
            "fold": str(fold),
            "family_filter": sorted(str(f) for f in (family_filter or ())),
            "levels": list(levels),
        },
        sort_keys=True,
    )
    return sha1(spec.encode()).hexdigest()[:8]


def gen_df(parquet_path, out_dir, min_img_per_spc, fold, hierarchy_path, family_filter, levels=DEFAULT_LEVELS):
    """Load + filter + prepare the metadata parquet, caching the result next to ``out_dir``.

    The cache key includes every argument that changes the rows (fold, min count, family filter,
    levels), so changing the filtering never silently reuses a stale cache.
    """
    parquet_path, out_dir, hierarchy_path = Path(parquet_path), Path(out_dir), Path(hierarchy_path)
    key = _cache_key(min_img_per_spc, fold, family_filter, levels)
    cache_path = out_dir.parent / f"{parquet_path.stem}.lepinet.{key}.parquet"

    if cache_path.exists() and hierarchy_path.exists():
        print(f"Loading cached preprocessed df: {cache_path}")
        return pd.read_parquet(cache_path), pd.read_csv(hierarchy_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet path not found: {parquet_path}")
    print(f"Loading parquet file {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = filter_df(df, remove_in=["0"], min_img_per_spc=min_img_per_spc, family_filter=family_filter, levels=levels)
    df = prepare_df(df, valid_set=fold, levels=levels)

    if not hierarchy_path.exists():
        build_hierarchy(df, levels).to_csv(hierarchy_path, index=False)
        print(f"Hierarchy saved to {hierarchy_path}")
    hierarchy = pd.read_csv(hierarchy_path)
    df.to_parquet(cache_path, index=False)
    print(f"Preprocessed df cached to {cache_path}")
    return df, hierarchy


# ---------------------------------------------------------------------------
# Long-tail rare-class oversampling (dev/034)
# ---------------------------------------------------------------------------

def sample_weights(df, level=None, power=0.5, levels=DEFAULT_LEVELS):
    """Per-training-row sampling weights for a fastai ``WeightedDL`` (square-root oversampling).

    Per-row weight = ``count(row's class) ** (-power)``, so a class with ``n`` rows carries total
    weight ``n ** (1-power)``: ``power=0`` natural (off, returns ``None``), ``0.5`` square-root
    (the safe sweet spot, Mahajan et al.), ``1`` fully class-balanced. Computed on the **training
    split only** (``is_valid == False``), aligned to that subset in df order.
    """
    if not power:
        return None
    level = level or levels[0]
    train_mask = ~df["is_valid"].to_numpy()
    vc = df.loc[train_mask, level].value_counts()
    counts = df.loc[train_mask, level].map(vc).to_numpy().astype(np.float64)
    return counts ** (-float(power))


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def make_dls(df, vocabs, img_dir, aug_img_size, img_size, batch_size, num_workers=None,
             aug_kwargs=None, sample_wgts=None, lowmem=True, levels=DEFAULT_LEVELS,
             domain_aug=None):
    """Build fastai ``DataLoaders`` from the prepared df.

    ``aug_kwargs`` overrides fastai's ``aug_transforms`` defaults (the baseline uses light aug:
    no warp, no lighting — heavy distortion hurts on a few-epoch, millions-of-images run).
    ``sample_wgts`` (from :func:`sample_weights`) turns the *train* loader into a ``WeightedDL``;
    the valid loader stays a plain sequential pass over the natural distribution.
    """
    # An all-True (or all-False) `is_valid` leaves one side of ColSplitter empty, and fastai then
    # dies inside DataBlock.setup with `IndexError: single positional indexer is out-of-bounds` --
    # a message that points at pandas indexing and says nothing about the split. It has cost debugging
    # time twice: once building an eval loader, once building an inference-only loader where every
    # row was marked valid because the labels were placeholders. Fail here instead, where the fix is.
    n_valid = int(df["is_valid"].sum())
    if n_valid in (0, len(df)):
        raise ValueError(
            f"`is_valid` is all-{bool(n_valid)}: ColSplitter needs a non-empty split on BOTH sides "
            f"({len(df)} rows, {n_valid} valid). For an inference-only loader the column is not read "
            f"-- it only has to make the block buildable -- so use a dummy split such as "
            f"`df['is_valid'] = np.arange(len(df)) % 5 == 0`."
        )
    from fastai.vision.all import (
        CategoryBlock,
        ColReader,
        ColSplitter,
        DataBlock,
        ImageBlock,
        Normalize,
        Resize,
        aug_transforms,
        imagenet_stats,
    )

    ensure_fork_start_method()
    aug_kwargs = aug_kwargs or {}
    # efficientnet/timm nets aren't in fastai's model_meta, so add ImageNet normalization here.
    # domain_aug (default None) appends OPT-IN domain-mimicking transforms after the standard
    # pipeline and before normalization; with it unset the list is byte-identical to every previous
    # run, so published numbers stay reproducible (lepinet.augment).
    from .augment import build_domain_aug

    batch_tfms = [*aug_transforms(size=img_size, **aug_kwargs), *build_domain_aug(domain_aug),
                  Normalize.from_stats(*imagenet_stats)]

    if lowmem:
        dblock = _lowmem_datablock(df, vocabs, img_dir, aug_img_size, batch_tfms, levels)
        source = np.arange(len(df))
    else:
        dblock = DataBlock(
            blocks=(ImageBlock, *(CategoryBlock(vocab=vocabs[level]) for level in levels)),
            n_inp=1,
            splitter=ColSplitter(),
            get_x=ColReader("image_path", pref=Path(img_dir)),
            get_y=[ColReader(level) for level in levels],
            item_tfms=Resize(aug_img_size),
            batch_tfms=batch_tfms,
        )
        source = df

    dl_kwargs = {} if num_workers is None else {"num_workers": num_workers}
    if sample_wgts is not None:
        from fastai.callback.data import WeightedDL
        dl_kwargs.update(dl_type=WeightedDL, dl_kwargs=({"wgts": sample_wgts}, {}))
    return dblock.dataloaders(source, bs=batch_size, **dl_kwargs)


def _lowmem_datablock(df, vocabs, img_dir, aug_img_size, batch_tfms, levels):
    """DataBlock over integer indices into fixed-width numpy arrays (COW-safe; see module docstring)."""
    from fastai.vision.all import CategoryBlock, DataBlock, ImageBlock, Resize

    paths = df["image_path"].to_numpy(dtype="U")
    is_valid = df["is_valid"].to_numpy(dtype=bool)
    label_arrs = {level: df[level].to_numpy(dtype="U") for level in levels}
    pref = str(img_dir) + os.path.sep  # matches ColReader(pref=Path(img_dir))

    def get_x(i):
        return pref + str(paths[i])

    def make_get_y(level):
        arr = label_arrs[level]
        return lambda i: str(arr[i])

    def splitter(items):
        return np.where(~is_valid)[0], np.where(is_valid)[0]

    return DataBlock(
        blocks=(ImageBlock, *(CategoryBlock(vocab=vocabs[level]) for level in levels)),
        n_inp=1,
        get_items=lambda _: np.arange(len(df)),
        splitter=splitter,
        get_x=get_x,
        get_y=[make_get_y(level) for level in levels],
        item_tfms=Resize(aug_img_size),
        batch_tfms=batch_tfms,
    )
