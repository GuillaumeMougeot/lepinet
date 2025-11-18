from itertools import chain

import torch
from torch import nn as nn
import torchvision

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
)


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

    labels = {str(r['speciesKey']):r.values.astype(str).tolist() for i,r in hierarchy.iterrows()}
    cls2idx = {str(i):{str(e):j for j,e in enumerate(s.unique())} for i,(n,s) in enumerate(hierarchy.items())}
    sparse_masks=sparse_masks_from_labels(labels, cls2idx)

    # TODO: To use or to remove:
    # dls = ImageDataLoaders.from_df(
    #     df,
    #     img_dir,
    #     valid_col='is_valid',
    #     label_delim=' ',
    #     bs=batch_size,
    #     item_tfms=Resize(aug_img_size),
    #     batch_tfms=aug_transforms(size=img_size))

    # f1_macro = F1ScoreMulti(thresh=0.5, average='macro')
    # f1_macro.name = 'F1(macro)'
    # f1_micro = F1ScoreMulti(thresh=0.5, average='micro')
    # f1_micro.name = 'F1(micro)'
    f1_macro = FastStreamingF1(average='macro', thresh=0.5)
    f1_macro.name = 'F1(macro)'
    f1_micro = FastStreamingF1(average='micro', thresh=0.5)
    f1_micro.name = 'F1(micro)'

    model_arch = getattr(importlib.import_module('fastai.vision.all'), model_arch_name)

    learn = vision_learner(
        dls, 
        model_arch, 
        metrics=[partial(accuracy_multi, thresh=0.5), f1_macro, f1_micro],
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

    
    # with learn.distrib_ctx():
    # learn.fine_tune(nb_epochs, 2e-2, freeze_epochs=0)
    learn.unfreeze()
    learn.fit_one_cycle(nb_epochs, lr_max=slice(5e-3, 1.6e-2))

    # --- Debug mode: run validation only ---
    # Run validation directly to test metrics and memory
    # val_loss, val_metrics = learn.validate()
    # print(f"Validation results:\nLoss: {val_loss}\nMetrics: {val_metrics}")
    # return

    # Save the model
    # ... remove cbs first
    # learn.recorder = None
    # learn.remove_cbs((CSVLogger,EarlyStoppingCallback))
    # import gc; gc.collect()

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