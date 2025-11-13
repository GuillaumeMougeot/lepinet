#------------------------------------------------------------------------------
# Requires timm to be installed: pip install timm
#------------------------------------------------------------------------------

# Production model.
# No cross-validation.

import pandas as pd
from pathlib import Path
import os
from os.path import exists, join
import importlib
from fastai.vision.all import (
    DataBlock,
    ImageBlock,
    MultiCategoryBlock,
    ColSplitter,
    ColReader,
    Resize,
    aug_transforms,
    vision_learner,
    F1ScoreMulti,
    accuracy_multi,
    ShowGraphCallback,
    CSVLogger,
    EarlyStoppingCallback,
    ImageDataLoaders,
    SaveModelCallback,
)
from fastai.callback.tensorboard import TensorBoardCallback
# from fastai.distributed import *
import json
from collections import defaultdict
import argparse
from datetime import datetime 
import yaml
import aiohttp, asyncio
from shutil import copyfile
import pickle
from sklearn.metrics import f1_score
from fastai.metrics import Metric
import torch
from timm.loss import AsymmetricLossMultiLabel
from timm.utils import ModelEmaV2
from timm.optim import AdamW
from fastai.vision.augment import RandTransform
from functools import partial

VALID_CONFIG_VERSIONS = [1.0]

LEVELS = {
    "speciesKey":"scientificName",
    "genusKey":"genus",
    "familyKey":"family",
}
HIERARCHY_LEVELS = list(LEVELS.keys())

def build_hierarchy(df: pd.DataFrame, hierarchy_levels: list):
    """
    Build a hierarchical
    """
    return (
    df
    .groupby('speciesKey')[hierarchy_levels]
    .take([0])
    .reset_index(drop=True)
    .sort_values(hierarchy_levels[::-1])
    )

def save_hierarchy(hierarchy: pd.DataFrame, filename: str|Path):
    """
    Save the hierarchy dictionary to a JSON file.
    """
    hierarchy.to_csv(filename, index=False)

def load_hierarchy(filename: str|Path):
    try:
        return pd.read_csv(filename)
    except Exception as e:
        print(f"Error while loading the hierarchy: {filename}")

def flatten_hierarchy(hierarchy: pd.DataFrame):
    """
    Flatten the hierarchy into a sequential list.
    """
    flat_hierarchy = []
    for c in hierarchy.columns:
        flat_hierarchy.extend(hierarchy[c].unique().astype(str).tolist())
    return flat_hierarchy

def filter_df(df, remove_in=[], keep_in=[],  img_per_spc=0):
    """
    Parameters
    ----------
    img_per_spc : int, default=0
        Number of images per species to select. If 0, then select them all.
    """
    df=df.copy()
    # Filter out 'test_ood' rows and 'test_in' rows
    if len(remove_in)>0:
        df = df[~df['set'].isin(remove_in)]
    if len(keep_in)>0:
        df = df[df['set'].isin(keep_in)]

    # Filter rows if too few images per species
    if img_per_spc > 0:
        print(f"Selecting {img_per_spc} images per species.")
        df=df[(df
            .groupby('speciesKey')['speciesKey']
            .transform('count') > img_per_spc)]
    
    print(f"Length of the filtered DataFrame: {len(df)}.")
    return df

def prepare_df_v1(df, valid_set='1'):
    def generate_image_path(row):
        return Path(str(row['speciesKey'])) / row['filename']

    # Apply the function to create the image paths
    df['image_path'] = df.apply(generate_image_path, axis=1)
    # Add a column to specify whether the row is for training or validation
    df['is_valid'] = df['set'] == valid_set

    # Create a function to extract the labels at different hierarchy levels
    def get_hierarchy_labels(row):
        return ' '.join(map(str, [row[level] for level in HIERARCHY_LEVELS]))

    # Add a column with hierarchy labels
    df['hierarchy_labels'] = df.apply(get_hierarchy_labels, axis=1)
    # Keep only the columns needed for ImageDataLoaders
    df = df[['image_path', 'hierarchy_labels', 'is_valid']]
    return df

def prepare_df(df, valid_set='1'):
    # Vectorized image path creation (no apply)
    df = df.copy()
    df['image_path'] = df['speciesKey'].astype(str) + '/' + df['filename']

    # Vectorized is_valid flag
    df['is_valid'] = df['set'].eq(valid_set)

    # Vectorized hierarchy label creation
    df['hierarchy_labels'] = df[HIERARCHY_LEVELS].astype(str).agg(' '.join, axis=1)

    # Convert image_path to pathlib.Path objects if fastai requires Path type
    # df['image_path'] = df['image_path'].map(Path)

    return df[['image_path', 'hierarchy_labels', 'is_valid']]

class StreamingF1(Metric):
    "Non-accumulating, streaming F1 metric for multi-label tasks."
    def __init__(self, average='macro', thresh=0.5, sigmoid=True):
        self.average, self.thresh, self.sigmoid = average, thresh, sigmoid
        self.reset()

    def reset(self):
        self.total, self.count = 0.0, 0

    def accumulate(self, learn):
        preds, targs = learn.pred, learn.y
        if self.sigmoid:
            preds = torch.sigmoid(preds)
        preds = (preds >= self.thresh).float()

        # Move to CPU + numpy
        preds = preds.detach().cpu().numpy()
        targs = targs.detach().cpu().numpy()

        batch_f1 = f1_score(targs, preds, average=self.average, zero_division=0)
        self.total += batch_f1
        self.count += 1

    @property
    def value(self):
        if self.count == 0: return None
        return self.total / self.count
    
    @property
    def name(self):  return self._name

    @name.setter
    def name(self, value): self._name = value

class FastStreamingF1(Metric):
    """
    Efficient, streaming F1 metric for multi-label problems.
    Keeps only running TP/FP/FN counts.
    """
    def __init__(self, average='macro', thresh=0.5, sigmoid=True, name=None):
        self.average, self.thresh, self.sigmoid = average, thresh, sigmoid
        self.name = name or f"F1({average})"
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = None
        self.count = 0  # for 'macro' averaging

    def accumulate(self, learn):
        preds, targs = learn.pred, learn.y
        if self.sigmoid: preds = torch.sigmoid(preds)
        preds = (preds >= self.thresh).float()

        # Compute true positives, false positives, false negatives per class
        tp = (preds * targs).sum(dim=0)
        fp = (preds * (1 - targs)).sum(dim=0)
        fn = ((1 - preds) * targs).sum(dim=0)

        if self.tp is None:
            self.tp, self.fp, self.fn = tp, fp, fn
        else:
            self.tp += tp
            self.fp += fp
            self.fn += fn
        self.count += 1

    @property
    def value(self):
        # Compute per-class precision, recall, f1
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if self.average == 'macro':
            return f1.mean().item()
        elif self.average == 'micro':
            # Micro average: sum over all classes
            tp, fp, fn = self.tp.sum(), self.fp.sum(), self.fn.sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            return (2 * precision * recall / (precision + recall + 1e-8)).item()
        elif self.average == 'samples':
            # optional extension — per-sample averaging
            # could be added easily if you need it later
            return f1.mean().item()
        else:
            raise ValueError(f"Unsupported average: {self.average}")
        
    @property
    def name(self):  return self._name

    @name.setter
    def name(self, value): self._name = value

async def get_key(session, scientificName=None, usageKey=None, rank='SPECIES', order='Lepidoptera'):
    url = "https://api.gbif.org/v1/species/match?"
    assert usageKey is not None or scientificName is not None, "One of scientificName or usageKey must be defined."

    if usageKey is not None:
        url += f"usageKey={usageKey}&"
    if scientificName is not None:
        url += f"scientificName={scientificName}&"
    if rank is not None:
        url += f"rank={rank}&"
    if order is not None:
        url += f"order={order}"

    async with session.get(url) as response:
        r = await response.json()
        return r['canonicalName']

async def get_all_keys(vocab):
    async with aiohttp.ClientSession() as session:
        tasks = [get_key(session, usageKey=k, rank=None) for k in vocab]
        return await asyncio.gather(*tasks)

def gen_dls(
    parquet_path: str|Path,
    img_dir: str|Path,
    out_dir: str|Path,
    img_per_spc: int,
    fold: str,
    model_name: str,
    nb_epochs: int,
    batch_size: int,
    aug_img_size: int,
    img_size: int,
    model_arch_name: str,
    hierarchy_path: str|Path = None,
    ):
    # First check if an existing preprocessed df exists, and if so, load it
    parquet_name = Path(parquet_path.name)
    df_path = out_dir.parent / parquet_name.with_suffix(".lepinet.parquet")
    if df_path.exists() and hierarchy_path.exists():
        print(f"Found existing preprocessed df: {df_path}")
        print("Loading it...")
        df = pd.read_parquet(df_path)

        hierarchy=load_hierarchy(filename=hierarchy_path)
        vocab=flatten_hierarchy(hierarchy)
 
        print("Df and vocab loaded.")
    # Else, preprocessed the DataFrame
    elif parquet_path.exists():
        print(f"Loading parquet file {parquet_path}")
        # Read parquet 
        df=pd.read_parquet(parquet_path)

        # Filter rows
        print("Filtering rows...")
        df=filter_df(df, remove_in=["0"], img_per_spc=img_per_spc)
        print("DataFrame filtered.")

        # Read or create hierarchy path
        if not hierarchy_path.exists():
            print(f"Hierarchy not found in {hierarchy_path}. Creating it...")
            hierarchy=build_hierarchy(df, hierarchy_levels = HIERARCHY_LEVELS)
            save_hierarchy(hierarchy, filename=hierarchy_path)
            print(f"Hierarchy saved in {hierarchy_path}.")
        
        # Read hierarchy file
        hierarchy=load_hierarchy(filename=hierarchy_path)
        vocab=flatten_hierarchy(hierarchy)

        # Remove test_ood and test_in data
        print("Preparing DataFrame...")
        df = prepare_df(df, valid_set=fold)
        print("DataFrame ready.")

        # Save the preprocessed DataFrame for later use
        print(f"Saving the DataFrame to {df_path}")
        df.to_parquet(df_path, index=False)
    else:
        raise FileNotFoundError(f"Parquet path not found: {parquet_path}")

    datablock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(vocab=vocab)),
        splitter=ColSplitter(),
        get_x=ColReader(0, pref=img_dir),
        get_y=ColReader(1, label_delim=' '),
        item_tfms=Resize(aug_img_size),
        batch_tfms=aug_transforms(size=img_size)
    )
    dls = datablock.dataloaders(df, bs=batch_size)
    # dls.train.num_workers = 16
    # dls.valid.num_workers = 8
    print(f"Number of workers: {dls.num_workers}, {dls.train.num_workers}, {dls.valid.num_workers}")

    return dls, hierarchy

def train(
    parquet_path: str|Path,
    img_dir: str|Path,
    out_dir: str|Path,
    img_per_spc: int,
    fold: str,
    model_name: str,
    nb_epochs: int,
    batch_size: int,
    aug_img_size: int,
    img_size: int,
    model_arch_name: str,
    hierarchy_path: str|Path = None,
):
    if isinstance(parquet_path, str): parquet_path = Path(parquet_path)
    if isinstance(img_dir, str): img_dir = Path(img_dir)
    if isinstance(out_dir, str): out_dir = Path(out_dir)
    if hierarchy_path is None:
        hierarchy_path = parquet_path.parent / "hierarchy.csv"

    # -------------------------------------------------------------------------
    # DataLoaders with stronger augmentations
    # -------------------------------------------------------------------------
    dls, hierarchy = gen_dls(
        parquet_path=parquet_path,
        img_dir=img_dir,
        out_dir=out_dir,
        img_per_spc=img_per_spc,
        fold=fold,
        model_name=model_name,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        aug_img_size=aug_img_size,
        img_size=img_size,
        model_arch_name=model_arch_name,
        hierarchy_path=hierarchy_path,
    )

    # Slightly stronger but class-safe augmentations
    dls.after_batch.add(RandTransform(tfm_y=False, p=0.8))
    dls.after_batch.add(aug_transforms(
        size=img_size,
        max_rotate=15,
        max_warp=0.1,
        max_zoom=1.1,
        p_affine=0.8,
        p_lighting=0.8,
    ))

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    f1_macro = FastStreamingF1(average='macro', thresh=0.5)
    f1_macro.name = 'F1(macro)'
    f1_micro = FastStreamingF1(average='micro', thresh=0.5)
    f1_micro.name = 'F1(micro)'

    # -------------------------------------------------------------------------
    # Model + optimizer + loss
    # -------------------------------------------------------------------------
    model_arch = getattr(importlib.import_module('fastai.vision.all'), model_arch_name)

    learn = vision_learner(
        dls,
        model_arch,
        metrics=[partial(accuracy_multi, thresh=0.5), f1_macro, f1_micro],
        loss_func=AsymmetricLossMultiLabel(gamma_pos=0, gamma_neg=4, clip=0.05),
        opt_func=partial(AdamW, lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999)),
        model_dir=out_dir / "models",
        cbs=[
            CSVLogger(out_dir / f"{model_name}.csv", append=True),
            SaveModelCallback(fname=f"{model_name}", monitor='F1(micro)', comp=np.greater),
            TensorBoardCallback(log_dir=out_dir / 'tensorboard', trace_model=False),
        ]
    )

    # -------------------------------------------------------------------------
    # EMA tracking for smoother validation and generalization
    # -------------------------------------------------------------------------
    model_ema = ModelEmaV2(learn.model, decay=0.9998)

    # Patch learner to update EMA each step
    def after_step_cb():
        model_ema.update(learn.model)
    learn.add_cb(Callback(after_step=after_step_cb))

    # -------------------------------------------------------------------------
    # Mixed precision (for speed and stability)
    # -------------------------------------------------------------------------
    learn.to_fp16()

    # -------------------------------------------------------------------------
    # Train with cosine LR schedule
    # -------------------------------------------------------------------------
    learn.fit_one_cycle(
        nb_epochs,
        lr_max=1e-3,
        wd=0.05,
        pct_start=0.1,   # warmup
        div_final=1e3,   # cosine decay end
    )

    # -------------------------------------------------------------------------
    # Save EMA model instead of raw weights
    # -------------------------------------------------------------------------
    learn.model.load_state_dict(model_ema.module.state_dict())
    slim_learn = vision_learner(learn.dls, model_arch)
    slim_learn.model = learn.model
    slim_learn.hierarchy = hierarchy

    # Optionally map vocab → canonical names via GBIF API
    id2name = asyncio.run(get_all_keys(slim_learn.dls.vocab))
    id2name = {v: n for (v, n) in zip(slim_learn.dls.vocab, id2name)}
    slim_learn.id2name = id2name

    model_path = out_dir / f"{model_name}.pkl"
    slim_learn.export(model_path)
    print(f"Model saved to {model_path}")

def create_out_dir(out_dir, desc):
    """Create the output directory named after datetime-desc"""
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dirname = join(out_dir, current_time + '-' + desc)
    os.makedirs(out_dirname, exist_ok=True)
    return out_dirname

def cli(config_path:str|Path=None):
    if config_path is None:
        parser = argparse.ArgumentParser(description="Main training file.")
        parser.add_argument("-c", "--config", type=str,
            help="Path to config file.")
        args = parser.parse_args()
        config_path = args.config
    
    if exists(config_path):
        # Load config file
        with open(config_path) as f:
            config=yaml.safe_load(f)

        # Check config version
        assert float(config['version']) in VALID_CONFIG_VERSIONS, (
            f"Wrong config version: {config['version']}. "
            f"Must be in {VALID_CONFIG_VERSIONS}.") 
        
        # Create and edit the output directory
        config['train']['out_dir'] = create_out_dir(
            config['train']['out_dir'], config['desc'])
        
        # Put the config file inside the output dir
        copyfile(config_path, join(config['train']['out_dir'], 'config.yaml'))

        # TODO: if a 'test' key exists in the config dict, then modify the 
        # value of 'model_path' key to be set as the above folder.
        
        # Start the training
        train(**config['train'])
    else:
        raise FileNotFoundError(f"Path to config not found: {config_path}")

if __name__=='__main__':
    cli()