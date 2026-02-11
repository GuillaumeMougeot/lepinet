from itertools import chain

import torch
from torch import nn as nn
import torchvision
import importlib
import os
import argparse
from os.path import exists, join
from datetime import datetime
from shutil import copyfile
import aiohttp, asyncio 

# from mini_trainer.classifier import Classifier
from mini_trainer.hierarchical.integration import sparse_masks_from_labels, HierarchicalBuilder
from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.hierarchical.loss import MultiLevelWeightedCrossEntropyLoss, MultiLevelLoss

from pathlib import Path
import pandas as pd
from collections import OrderedDict
import yaml

from fastai.learner import Learner
from fastai.vision.all import (
    DataBlock,
    ImageBlock,
    MultiCategoryBlock,
    CategoryBlock,
    ColSplitter,
    ColReader,
    Pipeline,
    Resize,
    aug_transforms,
    vision_learner,
    partial,
    accuracy_multi,
    CSVLogger,
    EarlyStoppingCallback,
    ImageDataLoaders,
    SaveModelCallback,
    Normalize,
    accuracy,
    create_body,
    create_head,
    num_features_model
)

from fastai.callback.tensorboard import TensorBoardCallback
from fastai.metrics import Metric
from accelerate.utils import write_basic_config
write_basic_config()

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
    model_preprocessor=None,
    ):
    # Assert types
    if isinstance(parquet_path, str): parquet_path = Path(parquet_path)
    if isinstance(img_dir, str): img_dir = Path(img_dir)
    if isinstance(out_dir, str): out_dir = Path(out_dir)
    if hierarchy_path is None:
        hierarchy_path = parquet_path.parent / "hierarchy.csv"

    # First check if an existing preprocessed df exists, and if so, load it
    parquet_name = Path(parquet_path.name)
    df_path = out_dir.parent / parquet_name.with_suffix(".lepinet.parquet")
    if df_path.exists() and hierarchy_path.exists():
        print(f"Found existing preprocessed df: {df_path}")
        print("Loading it...")
        df = pd.read_parquet(df_path)

        hierarchy=load_hierarchy(filename=hierarchy_path)
 
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

    df[['speciesKey','genusKey','familyKey']] = df['hierarchy_labels'].str.split(' ', expand=True)

    vocab_3 = df['speciesKey'].unique().tolist()
    vocab_4 = df['genusKey'].unique().tolist()
    vocab_5 = df['familyKey'].unique().tolist()

    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=vocab_3), CategoryBlock(vocab=vocab_4), CategoryBlock(vocab=vocab_5)),
        n_inp=1,
        splitter=ColSplitter(),
        get_x=ColReader(0, pref=img_dir),
        get_y=[ColReader(3), ColReader(4), ColReader(5)],
        item_tfms=Resize(aug_img_size),
        batch_tfms=[*aug_transforms(size=img_size), model_preprocessor]
    )
    dls = datablock.dataloaders(df, bs=batch_size)
    # dls.train.num_workers = 16
    # dls.valid.num_workers = 8
    print(f"Number of workers: {dls.num_workers}, {dls.train.num_workers}, {dls.valid.num_workers}")

    return dls, hierarchy

class FastStreamingF1MultiHead(Metric):
    def __init__(self, average="macro", name=None):
        assert average in {"macro", "micro"}
        self.average = average
        self._name = name or f"F1_{average}_multihead"
        self.reset()

    def reset(self):
        self.tp = {}
        self.fp = {}
        self.fn = {}

    def accumulate(self, learn):
        preds, targs = learn.pred, learn.y
        if not isinstance(preds, (tuple, list)):
            preds, targs = (preds,), (targs,)

        for h, (p, y) in enumerate(zip(preds, targs)):
            pred_cls = p.argmax(dim=1)

            num_classes = p.shape[1]
            device = p.device

            if h not in self.tp:
                self.tp[h] = torch.zeros(num_classes, device=device)
                self.fp[h] = torch.zeros(num_classes, device=device)
                self.fn[h] = torch.zeros(num_classes, device=device)

            # One-hot (bs, C)
            pred_oh = torch.nn.functional.one_hot(pred_cls, num_classes)
            true_oh = torch.nn.functional.one_hot(y, num_classes)

            self.tp[h] += (pred_oh & true_oh).sum(dim=0)
            self.fp[h] += (pred_oh & ~true_oh).sum(dim=0)
            self.fn[h] += (~pred_oh & true_oh).sum(dim=0)

    @property
    def value(self):
        eps = 1e-8

        if self.average == "macro":
            f1s = []
            for h in self.tp:
                tp, fp, fn = self.tp[h], self.fp[h], self.fn[h]
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)
                f1s.append((2 * precision * recall / (precision + recall + eps)).mean())
            return torch.stack(f1s).mean().item()

        else:  # micro
            tp = sum(v.sum() for v in self.tp.values())
            fp = sum(v.sum() for v in self.fp.values())
            fn = sum(v.sum() for v in self.fn.values())
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            return (2 * precision * recall / (precision + recall + eps)).item()

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value): self._name = value

class SumMultiLevelWeightedCrossEntropyLoss(torch.nn.modules.loss._Loss):
    def __init__(
            self, 
            weights : list[float | int] | torch.Tensor,
            device : torch._prims_common.DeviceLikeType, 
            dtype : torch.types._dtype, 
            class_weights : list[torch.Tensor] | None=None,
            label_smoothing : float = 0.0
        ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.weights = torch.tensor(weights).to(device=device, dtype=dtype)
        self.n_levels = len(weights)
        self.label_smoothing = [1 - (1 - label_smoothing)**(1/(i+1)) for i in range(self.n_levels)]
        
        self._loss_fns = [
            nn.CrossEntropyLoss(
                # weight=None, #self.class_weights[i], 
                # reduction="none", 
                label_smoothing=label_smoothing
            ) for _ in range(self.n_levels)
        ]

    def __call__(
            self, 
            preds : torch.Tensor, 
            *targets
        ) -> "MultiLevelLoss":
        return sum(list(MultiLevelLoss(
            [
                self._loss_fns[i](preds[i], targets[i])
                for i in range(self.n_levels)
            ], 
             self.weights
        )))

async def get_sn(session, usageKey):
    url = "https://api.gbif.org/v1/species/{}/name"
    if usageKey is not None:
        url = url.format(usageKey)
        
    async with session.get(url) as response:
        r = await response.json()
        return r if not 'scientificName' in r.keys() else " ".join(r['scientificName'].split(' ')[:2])
        

async def get_all_sn(vocab):
    async with aiohttp.ClientSession() as session:
        tasks = [get_sn(session, usageKey=k) for k in vocab]
        return await asyncio.gather(*tasks)

class MultiHead(nn.Module):
    def __init__(self, body, heads):
        super().__init__()
        self.body = body
        self.heads = heads

    def forward(self, x):
        feats = self.body(x)
        return [h(feats) for h in self.heads]

class MultiHeadLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, preds, targs_species, targs_genus, targs_family):
        targs = [targs_species, targs_genus, targs_family]
        return sum(self.loss(p, t) for p, t in zip(preds, targs))

def accuracy_multi(preds, targs_species, targs_genus, targs_family):
    accs = []
    targs = [targs_species, targs_genus, targs_family]
    for p, t in zip(preds, targs):
        accs.append(accuracy(p, t))
    return torch.stack(accs).mean()

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
    # Assert types
    if isinstance(parquet_path, str): parquet_path = Path(parquet_path)
    if isinstance(img_dir, str): img_dir = Path(img_dir)
    if isinstance(out_dir, str): out_dir = Path(out_dir)
    if hierarchy_path is None:
        hierarchy_path = parquet_path.parent / "hierarchy.csv"

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

    backbone = getattr(importlib.import_module('fastai.vision.all'), model_arch_name)
    body = create_body(backbone(), pretrained=True)
    nf = num_features_model(body)

    heads = nn.ModuleList([
        create_head(nf, len(v)) for v in dls.vocab
    ])
    model_arch = MultiHead(body, heads)

    learn = Learner(
        dls, 
        model_arch, 
        loss_func=MultiHeadLoss(),
        metrics=[accuracy_multi],
        model_dir=out_dir / "models",
        cbs=[
            CSVLogger(out_dir/f"{model_name}.csv", append=True),
            # TensorBoard logging
            TensorBoardCallback(
                log_dir=out_dir/'tensorboard',  # where to store logs
                trace_model=False,              # disable tracing to save memory
                log_preds=False,                # optional: skip predictions logging
            ),
            
            # Automatically save best model and optionally every epoch
            SaveModelCallback(
                fname=f"{model_name}",
                every_epoch=True
            ),

            # EarlyStoppingCallback(patience=10),
            ])
    
    # res = learn.validate(cbs=[])
    # print(res)  # loss + metrics
    # with learn.distrib_ctx():
    learn.fine_tune(nb_epochs, 1e-3)
    # learn.fit_one_cycle(nb_epochs, lr_max=5e-3)

    # --- Debug mode: run validation only ---
    # Run validation directly to test metrics and memory
    # val_loss, val_metrics = learn.validate()
    # print(f"Validation results:\nLoss: {val_loss}\nMetrics: {val_metrics}")
    # return

    # Save the model
    # ... remove cbs first
    learn.recorder = None
    learn.remove_cbs((CSVLogger,EarlyStoppingCallback))

    # Integrate hierarchy
    learn.hierarchy = hierarchy

    # Integrate id2name
    id2name = asyncio.run(get_all_sn(learn.dls.vocab))
    id2name = {v:n for (v,n) in zip(learn.dls.vocab, id2name)}
    learn.id2name=id2name

    model_path = out_dir / f"{model_name}.pkl"
    learn.export(model_path)

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