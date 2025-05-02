# Production model.
# No cross-validation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
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
    resnet50,
    accuracy_multi,
    ShowGraphCallback,
    CSVLogger,
    EarlyStoppingCallback
)
# from fastai.distributed import *
import json
from collections import defaultdict
import argparse
from datetime import datetime 

LEVELS = {
    "speciesKey":"scientificName",
    "genusKey":"genus",
    "familyKey":"family",
}
HIERARCHY_LEVELS = list(LEVELS.keys())

def build_hierarchy(df: pd.DataFrame, hierarchy_levels: list):
    """
    Build a hierarchical tree where the penultimate level holds a unique list of the lowest-level values.
    """
    hierarchy = defaultdict(lambda: defaultdict(set))  # Use set to avoid duplicates
    if hierarchy_levels[0] == "speciesKey":
        hierarchy_levels = hierarchy_levels[::-1]

    for _, row in df.iterrows():
        current_level = hierarchy
        for level in hierarchy_levels:
            key = row[level]

            if level == "speciesKey":  # Lowest level (store unique values)
                current_level.add(key)
            else:
                if key not in current_level:
                    current_level[key] = set() if level == "genusKey" else defaultdict(set)  # Penultimate level
                current_level = current_level[key] # Goes deeper in the hierarchy

    # Convert sets to lists for the final output
    def convert_sets_to_lists(node):
        if isinstance(node, dict):
            return {k: convert_sets_to_lists(v) for k, v in node.items()}
        elif isinstance(node, set):
            return list(node)
        return node
    
    return convert_sets_to_lists(hierarchy)


def save_hierarchy_to_file(hierarchy: dict, filename: str):
    """
    Save the hierarchy dictionary to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(hierarchy, f, indent=4)


def flatten_hierarchy(hierarchy: dict):
    """
    Flatten the hierarchy into a sequential list.
    """
    flat_list = []
    
    def traverse(node):
        if isinstance(node, dict):  # Regular nested dictionary structure
            for key in node.keys():
                flat_list.append(key)
            for subnode in node.values():
                traverse(subnode)
        elif isinstance(node, list):  # Leaf level is a list
            for item in node:
                flat_list.append(item)

    traverse(hierarchy)
    return flat_list

def prepare_df(df, remove_in=[], keep_in=[], valid_set='1'):
    # Filter out 'test_ood' rows and 'test_in' rows
    if len(remove_in)>0:
        df = df[~df['set'].isin(remove_in)]
    if len(keep_in)>0:
        df = df[df['set'].isin(keep_in)]
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

def train(
    hierarchy_path: str|Path,
    parquet_path: str|Path,
    images_path: str|Path,
    export_path: str|Path,
    model_name: str,
    history_csv_name: str,
    ):

    df=pd.read_parquet(parquet_path)
    hierarchy=build_hierarchy(df, hierarchy_levels = HIERARCHY_LEVELS)
    save_hierarchy_to_file(hierarchy, filename=root_path/"hierarchy_all.json")
    vocab=flatten_hierarchy(hierarchy)
    print(vocab)
    print(len(vocab))

    # Remove test_ood and test_in data
    # df = prepare_df(df.copy(), remove_in=["0", "test_ood"])
    df = prepare_df(df.copy(), remove_in=["0"])

    datablock = DataBlock(
        blocks=(ImageBlock, MultiCategoryBlock(vocab=vocab)),
        splitter=ColSplitter(),
        get_x=ColReader(0, pref=images_path),
        get_y=ColReader(1, label_delim=' '),
        item_tfms=Resize(460),
        batch_tfms=aug_transforms(size=224)
    )
    dls = datablock.dataloaders(df, bs=256)

    f1_macro = F1ScoreMulti(thresh=0.5, average='macro')
    f1_macro.name = 'F1(macro)'
    f1_samples = F1ScoreMulti(thresh=0.5, average='samples')
    f1_samples.name = 'F1(samples)'
    os.makedirs(export_path, exist_ok=True)
    learn = vision_learner(
        dls, 
        resnet50, 
        metrics=[partial(accuracy_multi, thresh=0.5), f1_macro, f1_samples],
        cbs=[
            ShowGraphCallback(),
            CSVLogger(export_path/history_csv_name),
            EarlyStoppingCallback(patience=10),
            ])
    
    # with learn.distrib_ctx():
    learn.fine_tune(100, 2e-2)

    # Save the model
    # ... remove cbs first
    # learn.remove_cbs((CSVLogger,EarlyStoppingCallback))

    # ...recreate a vision learner to remove large files that lives inside learner
    slim_learn = vision_learner(learn.dls, resnet50)
    slim_learn.model = learn.model

    model_path = export_path / model_name
    slim_learn.export(model_path)

if __name__=='__main__':
    # parser = argparse.ArgumentParser(description="Main training file.")
    # parser.add_argument("-i", "--img_dir", type=str,
    #     help="Image folder.")
    # parser.add_argument("-p", "--parquet", type=str,
    #     help="Parquet path.")
    # parser.add_argument("-o", "--out_dir", type=str,
    #     help="Output dir, stores models and logs.")
    # parser.add_argument("-n", "--name", type=str,
    #     help="Model name.")

    # Large Lepi dataset
    # Add datetime to output folder name
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    root_path=Path("data/lepi")
    parquet_path=root_path/"0061420-241126133413365_sampled_processing_metadata_postprocessed.parquet"

    images_path=root_path/"images"
    export_path=root_path/"models"
    model_name="20250424-lepi-prod_model2"
    train(
        None,
        # hierarchy_path,
        parquet_path,
        images_path,
        export_path,
        model_name=model_name,
        history_csv_name=model_name+"-h1.csv",
    )