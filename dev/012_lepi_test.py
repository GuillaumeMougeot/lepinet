#-------------------------------------------------------------------------------
# Testing of the trained model
#-------------------------------------------------------------------------------

import argparse
import yaml
import os
import requests
from os.path import exists, join, isdir
from pathlib import Path
import torch
import importlib
import numpy as np
import pandas as pd
from fastai.vision.all import load_learner, CategoryMap, vision_learner

gen_dls = getattr(importlib.import_module('011_lepi_large_prod_v2'), 'gen_dls')

VALID_CONFIG_VERSIONS = [1.0]
VALID_IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.gif', '.webp')

def get_key(scientificName=None, usageKey=None, rank='SPECIES', order='Lepidoptera'):
    """Returns taxon key from scientific name.

    Notes
    -----

    Requests GBIF API. 

    If GBIF API returns more than one element, display a warning and return the first element from the list.
    """

    url = "https://api.gbif.org/v1/species/match?"

    assert usageKey is not None or scientificName is not None, "One of scientificRank or usageKey must be defined."

    if usageKey is not None:
        url += f"usageKey={usageKey}&"
    if scientificName is not None:
        url += f"scientificName={scientificName}&"
    if rank is not None:
        url += f"rank={rank}&"
    if order is not None:
        url += f"order={order}"

    x=requests.get(url)
    return x.json()

def gen_name2id(img_dir):

    # list folder names
    foldernames = os.listdir(img_dir)

    name2id = {
        'verbatimScientificName':[],
        'familyKey':[],
        'genusKey':[],
        'speciesKey':[]
    }

    for i, f in enumerate(foldernames):
        k=get_key(scientificName=f)
        if k['rank']!='SPECIES':
            print(f"Wrong rank for {f} : {k}")
        if f == 'Tethea or': # Bug fix with GBIF Species API, Tethea or gives the order
            k=get_key(usageKey="5142971")
        name2id['verbatimScientificName'].append(f)
        name2id['familyKey'].append(k['familyKey'])
        name2id['genusKey'].append(k['genusKey'])
        speciesKey = k['usageKey'] if 'acceptedUsageKey' not in k.keys() else k['acceptedUsageKey']
        name2id['speciesKey'].append(speciesKey)
    
    return name2id

def nested_dict_to_df(data):
    """
    Convert a nested dict {family: {genus: [species, ...]}} 
    into a pandas DataFrame with columns: speciesKey, genusKey, familyKey.
    """
    rows = []
    for family_key, genera in data.items():
        for genus_key, species_list in genera.items():
            for species_key in species_list:
                rows.append({
                    "speciesKey": int(species_key),
                    "genusKey": int(genus_key),
                    "familyKey": int(family_key)
                })
    return pd.DataFrame(rows, columns=["speciesKey", "genusKey", "familyKey"])

def gen_level_idx(vocab, hierarchy):
    """
    Returns a list of integers of the size of vocab indicating the hierarchical level of the taxa at index i.
    - Species is level 0, Genus 1, Family 2, etc.
    - Missing values are noted with -1.

    Args:
    - vocab (list): List of taxa names to find levels for.
    - hierarchy (pandas.DataFrame): with columns 'speciesKey, genusKey, familyKey'

    Returns:
    - np.ndarray: Array of level indices for each taxa in vocab.
    """
    # Ensure required columns exist
    required_cols = ['speciesKey', 'genusKey', 'familyKey']
    if not all(col in hierarchy.columns for col in required_cols):
        raise ValueError(f"hierarchy DataFrame must contain columns: {required_cols}")

    # Build mapping from key -> level
    level_map = {}
    for col, level in zip(required_cols, range(len(required_cols))):
        # dropna ensures we don’t include NaN values
        level_map.update({str(k): level for k in hierarchy[col].dropna().unique()})

    # Convert vocab to array and map to levels
    levels = [level_map.get(str(taxon), -1) for taxon in vocab]

    return np.array(levels)

def get_pred_conf(preds:torch.Tensor, vocab:CategoryMap, indices:np.ndarray):
    """Returns predicted labels and confidence for each pred and for each 
    hierarchy level.

    `preds` is a batch of predictions.
    """
    out_preds = []
    out_confs = []
    indices = torch.from_numpy(indices)
    for i in range(indices.max()+1):
        one_level_pred = preds[:,indices==i].cpu().numpy()
        one_level_prd = vocab[indices==i][one_level_pred.argmax(axis=1)]
        one_level_cnf = one_level_pred.max(axis=1)
        out_preds += [one_level_prd]
        out_confs += [one_level_cnf]
    return np.array(out_preds).swapaxes(0,1), np.array(out_confs).swapaxes(0,1)

def save_csv(
    fname:str,
    filenames:list,
    prds:np.ndarray,
    cnfs:np.ndarray, 
    thr:int=0.5,
    lbls:np.ndarray=None,
    vocab:list|np.ndarray=None,
    ):

    # Flatten the predictions and confidences
    n, p = prds.shape
    lvls = np.tile(np.arange(p), n)
    prds = prds.flatten(order='C') # C-type 
    cnfs = cnfs.flatten(order='C')
    flns = np.repeat(filenames, p)

    df=pd.DataFrame({
        'filename':flns,
        'level':lvls,
        'prediction':prds,
        'confidence':cnfs
    })
    
    if lbls is not None:
        df["label"] = lbls.flatten(order='C').astype(str)
    if vocab is not None:
        df['known_label'] = df['label'].isin(vocab)

    # Add instance_id
    df['instance_id'] = df.index // 3
    df['threshold'] = thr

    # Reorganize columns
    new_order = ["instance_id","filename","level","prediction","confidence","threshold"]
    if lbls is not None:
        new_order = new_order[:3] + ["label"] + new_order[3:]
    if vocab is not None:
        new_order = new_order + ["known_label"]
    df = df[new_order]

    df.to_csv(fname, index=False)

def test(
    img_dir:str|Path,
    model_path:str|Path,
    out_dir:str|Path,
    hierarchy_path:str|Path=None,
    name2id_path:str|Path=None, # Needed if labels are scientific names instead of GBIF ids.
    parquet_path:str|Path=None,
    cpu:bool=False,
    config:dict=None
    ):

    if isinstance(model_path, str): model_path = Path(model_path)
    if isinstance(img_dir, str): img_dir = Path(img_dir)
    if isinstance(out_dir, str): out_dir = Path(out_dir)
    if hierarchy_path is not None and isinstance(hierarchy_path, str):
        hierarchy_path = Path(hierarchy_path)
    if name2id_path is not None and isinstance(name2id_path, str):
        name2id_path = Path(name2id_path)
    if parquet_path is not None and isinstance(parquet_path, str):
        parquet_path = Path(parquet_path)
        df = pd.read_parquet(parquet_path)
        # Only keep test set images
        df = df[df['set'].isin(['0'])]
        assert len(df)>0, "No file found in df!"
    else: df = None
    
    print("Predicting...")
    print("Loading model...")
    if config is not None and model_path.exists() and model_path.suffix == '.pth':
        dls,hierarchy=gen_dls(**config['train'])
        model_arch = getattr(importlib.import_module('fastai.vision.all'), config['train']['model_arch_name'])
        learn = vision_learner(dls, model_arch)
        learn.load(model_path.with_suffix(''))
        learn.to('cpu' if cpu else 'cuda')
    elif model_path.exists():
        learn = load_learner(model_path, cpu=cpu)
    else:
        raise FileNotFoundError(f"Model not found {model_path}")
    learn.model = learn.model.eval()
    print("Model loaded.")

    print("Reading hierarchy...")
    if hierarchy_path is None and hasattr(learn,'hierarchy'):
        hierarchy = nested_dict_to_df(learn.hierarchy)
    elif hierarchy_path is not None and exists(hierarchy_path):
        hierarchy=pd.read_csv(hierarchy_path)
    elif 'hierarchy' not in locals():
        raise ValueError("Variable 'hierarchy' could not be defined.")
    print("Hierarchy loaded.")

    # Optionally load name2id
    if name2id_path is not None and exists(name2id_path):
        print(f"Found name2id file: {name2id_path}. Loading it...")
        name2id_test = pd.read_csv(name2id_path)
    elif name2id_path is not None:
        print("name2id specified, but not found, trying to generate it...")
        name2id_test = pd.DataFrame(gen_name2id(img_dir=img_dir))
        name2id_test.to_csv(name2id_path, index=False)
        print("name2id created.")

    print("Reading image filenames...")
    if parquet_path is None:
        filenames = list(img_dir.rglob("*"))
        filenames = [f for f in filenames if not isdir(f) and f.suffix.lower() in VALID_IMAGE_EXT]
    else:
        filenames = [(img_dir / r['speciesKey']) / r['filename'] for _,r in df.iterrows()]
    print(f"Found {len(filenames)} images.")

    # TEST
    # filenames = filenames[:65]
    # if df is not None: df = df.head(65)

    print("Creating test DataLoader...")
    test_dl = learn.dls.test_dl(filenames)
    print("Test DataLoader created.")

    print("Get predictions...")
    preds, _ = learn.get_preds(dl=test_dl)
    print(f"Obtained {len(preds)} predictions.")

    print("Format predictions...")
    indices=gen_level_idx(learn.dls.vocab, hierarchy)
    prds, cnfs = get_pred_conf(preds, learn.dls.vocab, indices)
    print("Prediction formatted.")

    print("Getting labels...")
    if name2id_path is not None:
        print("...from name2id...")
        name2id_test_dict = {
            r['verbatimScientificName']:[
                r['speciesKey'],
                r['genusKey'],
                r['familyKey']]
            for _, r in name2id_test.iterrows()}
        lbls = np.array([name2id_test_dict[Path(f).parent.name] for f in filenames])
    elif parquet_path is not None:
        print("...from parquet...")
        lbls = np.array([[r['speciesKey'], r['genusKey'], r['familyKey']] for _,r in df.iterrows()])
    else:
        print("No parquet or name2id specified, supposing that the class names are the folder names.")
        name2id_test_dict = {
            r['speciesKey']:[
                r['speciesKey'],
                r['genusKey'],
                r['familyKey']]
            for _, r in hierarchy.iterrows()}
        lbls = np.array([name2id_test_dict[Path(f).parent.name] for f in filenames])

    print("Saving CSV...")
    out_path = out_dir / model_path.with_suffix('.csv').name
    save_csv(
        out_path,
        filenames=[Path(f).name for f in filenames],
        prds=prds,
        cnfs=cnfs,
        lbls=lbls,
        vocab=learn.dls.vocab
        )
    print(f"CSV saved in {out_path}.")
    print("Prediction done.")

def cli():
    parser = argparse.ArgumentParser(description="Main testing script.")
    parser.add_argument("-c", "--config", type=str,
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
        
        assert 'test' in config.keys(), (
            f"Wrong config format: {config.keys()} "
            f"must includes 'test' key and {test.__code__.co_varnames} "
            "sub-keys.")

        # Start the training
        test(**config['test'], config=config)
    else:
        raise FileNotFoundError(f"Path to config not found: {args.config}")

if __name__=='__main__':
    cli()