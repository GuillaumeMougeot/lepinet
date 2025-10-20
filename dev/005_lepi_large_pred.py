import pandas as pd
import numpy as np
from pathlib import Path
from os.path import isdir
from fastai.vision.all import *
import json
import torch
import argparse

def gen_level_idx(vocab, hierarchy):
    """
    Returns a list of integers of the size of vocab indicating the hierarchical level of the taxa at index i.
    - Species is level 0, Genus 1, Family 2, etc.
    - Missing values are noted with -1.

    Args:
    - vocab (list): List of taxa names to find levels for.
    - hierarchy (dict): Nested dictionary representing taxonomic hierarchy.

    Returns:
    - np.ndarray: Array of level indices for each taxa in vocab.
    """
    level_lookup = {}

    def traverse(node, level=0):
        """Recursively traverse the hierarchy and store levels."""
        for key, subnode in node.items():
            level_lookup[key] = level  # Assign level to the taxon
            if isinstance(subnode, dict):
                traverse(subnode, level + 1)
            elif isinstance(subnode, list):  # Leaf nodes (species level)
                for species in subnode:
                    level_lookup[species] = level + 1

    # Build the level lookup dictionary
    traverse(hierarchy)  # Start from -1 so species end up at level 0

    # Assign levels to vocab, default to -1 if missing
    indices = np.array([level_lookup.get(v, -1) for v in vocab], dtype=int)

    # Invert the indices, so species is 0, genus is 1 etc
    indices = np.where(indices < 0, indices, indices.max()-indices)

    # Warning for missing values
    missing_count = np.sum(indices == -1)
    if missing_count > 0:
        print(f"[Warning] Missing values in taxa dictionary: {missing_count}.")

    return indices

def split_preds(preds:torch.Tensor, indices:np.ndarray):
    """Returns split preds using indices.

    `preds` is a batch of predictions.
    """
    out_preds = []
    indices = torch.from_numpy(indices)
    for i in range(indices.max()+1):
        out_preds += [preds[:,indices==i].cpu().numpy()]
    return out_preds

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

def save_npy(
    fname:str,
    prds:np.ndarray,
    cnfs:np.ndarray, 
    lbls:np.ndarray=None):
    if lbls is not None:
        output = np.stack((prds,cnfs,lbls), axis=-1)
    else:
        output = np.stack((prds,cnfs), axis=-1)
    np.save(fname, output)

def save_csv(
    fname:str,
    filenames:list,
    prds:np.ndarray,
    cnfs:np.ndarray, 
    lbls:np.ndarray=None):

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
        df["label"] = lbls.flatten(order='C')

    df.to_csv(fname, index=False)

def pred(
    img_dir:str,
    model_path:str,
    output_path:str,
    hierarchy_path:str=None,
    cpu:bool=False,
    ):

    print("Predicting...")
    print("Loading model...")
    learn = load_learner(model_path, cpu=cpu)
    learn.model = learn.model.eval()
    print("Model loaded.")

    print("Reading hierarchy...")
    if hierarchy_path is None:
        hierarchy = learn.hierarchy
    else:
        with open(hierarchy_path, "r") as f:
            hierarchy=json.load(f)
    print("Hierarchy loaded.")

    print("Reading image filenames...")
    filenames = list(Path(img_dir).rglob("*"))
    
    filenames = [f for f in filenames if not isdir(f) and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.gif', '.webp')]
    print(f"Found {len(filenames)} images.")

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

    print("Save CSV...")
    save_csv(
        output_path,
        filenames=filenames,
        prds=prds,
        cnfs=cnfs,
        )
    print(f"CSV saved in {output_path}.")
    print("Prediction done.")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Main training file.")
    parser.add_argument("-i", "--img_dir", type=str,
        help="Image folder.")
    parser.add_argument("-m", "--model_path", type=str,
        help="Model path.")
    parser.add_argument("-hp", "--hierarchy_path", type=str, default=None,
        help="Hierarchy path.")
    parser.add_argument("-o", "--output_path", type=str,
        help="CSV output path.")
    parser.add_argument("--cpu", default=False,  action='store_true', dest='cpu',
        help="Whether to use the cpu or not.")  
    args = parser.parse_args()

    pred(
    img_dir=args.img_dir,
    model_path=args.model_path,
    output_path=args.output_path,
    hierarchy_path=args.hierarchy_path,
    cpu=args.cpu,
    )