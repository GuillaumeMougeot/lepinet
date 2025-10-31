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
    partial,
    F1ScoreMulti,
    accuracy_multi,
    ShowGraphCallback,
    CSVLogger,
    EarlyStoppingCallback,
    ImageDataLoaders
)
# from fastai.distributed import *
import json
from collections import defaultdict
import argparse
from datetime import datetime 
import yaml
import aiohttp, asyncio
from shutil import copyfile

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

    # Read parquet 
    df=pd.read_parquet(parquet_path)

    # Filter rows
    print("Filtering rows...")
    df=filter_df(df, remove_in=["0"], img_per_spc=img_per_spc)
    print("DataFrame filtered.")

    # Read or create hierarchy path
    if hierarchy_path is None:
        hierarchy_path = parquet_path.parent / "hierarchy.csv"
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

    datablock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(vocab=vocab)),
        splitter=ColSplitter(),
        get_x=ColReader(0, pref=img_dir),
        get_y=ColReader(1, label_delim=' '),
        item_tfms=Resize(aug_img_size),
        batch_tfms=aug_transforms(size=img_size)
    )
    dls = datablock.dataloaders(df, bs=batch_size)

    # TODO: To use or to remove:
    # dls = ImageDataLoaders.from_df(
    #     df,
    #     img_dir,
    #     valid_col='is_valid',
    #     label_delim=' ',
    #     bs=batch_size,
    #     item_tfms=Resize(aug_img_size),
    #     batch_tfms=aug_transforms(size=img_size))

    f1_macro = F1ScoreMulti(thresh=0.5, average='macro')
    f1_macro.name = 'F1(macro)'
    f1_samples = F1ScoreMulti(thresh=0.5, average='samples')
    f1_samples.name = 'F1(samples)'

    model_arch = getattr(importlib.import_module('fastai.vision.all'), model_arch_name)

    learn = vision_learner(
        dls, 
        model_arch, 
        metrics=[partial(accuracy_multi, thresh=0.5), f1_macro, f1_samples],
        cbs=[
            ShowGraphCallback(),
            CSVLogger(out_dir/f"{model_name}.csv"),
            # EarlyStoppingCallback(patience=10),
            ])
    
    # with learn.distrib_ctx():
    learn.fine_tune(nb_epochs, 2e-2)

    # Save the model
    # ... remove cbs first
    # learn.remove_cbs((CSVLogger,EarlyStoppingCallback))

    # ...recreate a vision learner to remove large files that lives inside learner
    slim_learn = vision_learner(learn.dls, model_arch)
    slim_learn.model = learn.model

    # Integrate hierarchy
    slim_learn.hierarchy = hierarchy

    # Integrate id2name
    id2name = asyncio.run(get_all_keys(slim_learn.dls.vocab))
    id2name = {v:n for (v,n) in zip(slim_learn.dls.vocab, id2name)}
    slim_learn.id2name=id2name

    model_path = out_dir / f"{model_name}.pkl"
    slim_learn.export(model_path)

def create_out_dir(out_dir, desc):
    """Create the output directory named after datetime-desc"""
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dirname = join(out_dir, current_time + '-' + desc)
    os.makedirs(out_dirname, exist_ok=True)
    return out_dirname

def cli():
    parser = argparse.ArgumentParser(description="Main training file.")
    parser.add_argument("--config", type=str,
        help="Path to config file.")
    args = parser.parse_args()
    
    if exists(args.config):
        # Load config file
        with open(args.config) as f:
            config=yaml.safe_load(f)

        # Check config version
        assert float(config['version']) in VALID_CONFIG_VERSIONS, (
            f"Wrong config version: {config['version']}. "
            f"Must be in {VALID_CONFIG_VERSIONS}.") 
        
        # Create and edit the output directory
        config['train']['out_dir'] = create_out_dir(
            config['train']['out_dir'], config['desc'])
        
        # Put the config file inside the output dir
        copyfile(args.config, join(config['train']['out_dir'], 'config.yaml'))
        
        # Start the training
        train(**config['train'])
    else:
        raise FileNotFoundError(f"Path to config not found: {args.config}")

if __name__=='__main__':
    cli()