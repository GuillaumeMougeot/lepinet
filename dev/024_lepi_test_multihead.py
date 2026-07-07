#-------------------------------------------------------------------------------
# Testing of the trained multi-head model
#-------------------------------------------------------------------------------

import argparse
import yaml
import os
import requests
from os.path import exists, join, isdir
from pathlib import Path
import torch
import torch.nn as nn
import importlib
import numpy as np
import pandas as pd
import aiohttp, asyncio
from fastai.vision.all import load_learner, Learner, create_body, create_head, num_features_model

training_module = importlib.import_module('022_lepi_large_prod_v3_multihead')
gen_dls = getattr(training_module, 'gen_dls')

VALID_CONFIG_VERSIONS = [1.0, 1.1]
VALID_IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.gif', '.webp')

# Explicitly defining custom architecture containers within local scope for Pickle resolution
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

def multi_head_tta(learn, dl, n=4, beta=0.25):
    """
    Custom Test-Time Augmentation (TTA) wrapper for Multi-Head / Multi-Output models.
    Combines baseline unaugmented predictions with 'n' randomized augmented passes.
    """
    import torch
    from fastai.data.core import Pipeline

    # 1. Get baseline (unaugmented) predictions
    print("Running baseline unaugmented pass...")
    with learn.no_bar():
        base_preds, _ = learn.get_preds(dl=dl)
    
    # 2. Extract stochastic (random) augmentations from the training dataloader
    stochastic_tfms = [
        tfm for tfm in learn.dls.train.after_batch 
        if getattr(tfm, 'p', 0) > 0 or tfm.__class__.__name__ in ['Flip', 'Rotate', 'Zoom', 'Warp', 'LightingTfms', 'Brightness', 'Contrast']
    ]
    
    if not stochastic_tfms:
        print("Warning: No random augmentations found in your training configurations. TTA will just repeat the baseline.")
    
    # Save the original test dataloader transforms to restore later
    orig_tfms = list(dl.after_batch)
    
    # Initialize trackers to accumulate augmented probabilities for each head
    aug_preds_sum = [torch.zeros_like(p) for p in base_preds]
    
    # 3. Run augmented passes
    print(f"Running {n} augmented TTA passes...")
    for i in range(n):
        # Temporarily inject random training augmentations ahead of the test pipeline
        dl.after_batch = Pipeline(stochastic_tfms + orig_tfms)
        
        with learn.no_bar():
            preds, _ = learn.get_preds(dl=dl)
        
        # Accumulate predictions across each head safely
        for head_idx in range(len(base_preds)):
            aug_preds_sum[head_idx] += preds[head_idx]
            
    # Restore the original unaugmented test pipeline
    dl.after_batch = Pipeline(orig_tfms)
    
    # 4. Blend baseline and augmented averages using the beta weight
    final_preds = []
    for head_idx in range(len(base_preds)):
        aug_preds_avg = aug_preds_sum[head_idx] / n
        combined = base_preds[head_idx] * (1 - beta) + aug_preds_avg * beta
        final_preds.append(combined)
        
    return final_preds

def get_key(scientificName=None, usageKey=None, rank='SPECIES', order='Lepidoptera'):
    url = "https://api.gbif.org/v1/species/match?"
    assert usageKey is not None or scientificName is not None, "One of scientificRank or usageKey must be defined."

    if usageKey is not None: url += f"usageKey={usageKey}&"
    if scientificName is not None: url += f"scientificName={scientificName}&"
    if rank is not None: url += f"rank={rank}&"
    if order is not None: url += f"order={order}"

    x = requests.get(url)
    return x.json()

def gen_name2id(img_dir):
    foldernames = os.listdir(img_dir)
    name2id = {
        'verbatimScientificName':[],
        'familyKey':[],
        'genusKey':[],
        'speciesKey':[]
    }

    for i, f in enumerate(foldernames):
        k = get_key(scientificName=f)
        if k['rank'] != 'SPECIES':
            print(f"Wrong rank for {f} : {k}")
        if f == 'Tethea or': 
            k = get_key(usageKey="5142971")
        name2id['verbatimScientificName'].append(f)
        name2id['familyKey'].append(k['familyKey'])
        name2id['genusKey'].append(k['genusKey'])
        speciesKey = k['usageKey'] if 'acceptedUsageKey' not in k.keys() else k['acceptedUsageKey']
        name2id['speciesKey'].append(speciesKey)
    
    return name2id

def get_pred_conf_multihead(preds_list, vocabs):
    """
    Extracts labels and confidences from a list of multi-head predictions,
    applying Softmax to convert raw logits into 0-1 probabilities.
    """
    import torch  # Ensure torch is imported inside or at the top of the file
    
    out_preds = []
    out_confs = []
    
    # Iterate through each head (Species, Genus, Family)
    for p, v in zip(preds_list, vocabs):
        # p shape: (batch_size, num_classes_at_this_level)
        
        # FIX: Convert raw logits to probabilities along the class dimension
        probs = torch.softmax(p, dim=1)
        
        # Now take the max value from the normalized probabilities
        conf, idx = probs.max(dim=1)
        
        # Map indices to actual labels
        out_preds.append(np.array(v[idx]))
        out_confs.append(conf.cpu().numpy())
        
    # Stack to get (batch_size, 3)
    return np.stack(out_preds, axis=1), np.stack(out_confs, axis=1)

async def fetch_parents(session, usage_key):
    url = f"https://api.gbif.org/v1/species/{usage_key}/parents"
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return usage_key, []
            data = await resp.json()
            parents = [str(d['key']) for d in data] + [usage_key]
            return usage_key, parents[::-1][:3]
    except Exception:
        return usage_key, []

async def get_all_parents(keys, max_concurrent=20):
    connector = aiohttp.TCPConnector(limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_parents(session, k) for k in keys]
        results = await asyncio.gather(*tasks)
        return dict(results)

def save_csv(
    fname: str,
    filenames: list,
    prds: np.ndarray,
    cnfs: np.ndarray, 
    thr: float = 0.5,
    lbls: np.ndarray = None,
    vocabs: list = None,
    ):

    n, p = prds.shape
    lvls = np.tile(np.arange(p), n)
    prds_flat = prds.flatten(order='C') 
    cnfs_flat = cnfs.flatten(order='C')
    flns = np.repeat(filenames, p)

    df = pd.DataFrame({
        'filename': flns,
        'level': lvls,
        'prediction': prds_flat,
        'confidence': cnfs_flat
    })
    
    if lbls is not None:
        df["label"] = lbls.flatten(order='C').astype(str)
        
    if vocabs is not None:
        known_labels = []
        for _, row in df.iterrows():
            lvl = int(row['level'])
            lbl = row['label'] if lbls is not None else row['prediction']
            current_vocab = [str(v) for v in vocabs[lvl]]
            known_labels.append(str(lbl) in current_vocab)
        df['known_label'] = known_labels

    df['instance_id'] = df.index // p
    df['threshold'] = thr

    new_order = ["instance_id", "filename", "level", "prediction", "confidence", "threshold"]
    if lbls is not None:
        new_order = new_order[:3] + ["label"] + new_order[3:]
    if vocabs is not None:
        new_order = new_order + ["known_label"]
    df = df[new_order]

    df.to_csv(fname, index=False)

def test(
    img_dir: str|Path,
    model_path: str|Path,
    out_dir: str|Path,
    hierarchy_path: str|Path = None,
    name2id_path: str|Path = None,
    parquet_path: str|Path = None,
    cpu: bool = False,
    config: dict = None,
    family_filter: list = []
    ):

    if isinstance(model_path, str): model_path = Path(model_path)
    if isinstance(img_dir, str): img_dir = Path(img_dir)
    if isinstance(out_dir, str): out_dir = Path(out_dir)
    if hierarchy_path is not None and isinstance(hierarchy_path, str): hierarchy_path = Path(hierarchy_path)
    if name2id_path is not None and isinstance(name2id_path, str): name2id_path = Path(name2id_path)
    
    if parquet_path is not None and isinstance(parquet_path, str):
        parquet_path = Path(parquet_path)
        df = pd.read_parquet(parquet_path)
        df = df[df['set'].isin(['0'])]
        
        if family_filter:
            df = df[df['familyKey'].astype(str).isin([str(f) for f in family_filter])]
            
        assert len(df) > 0, "No valid records left in dataframe after processing filters!"
    else: 
        df = None
    
    print("Predicting...")
    print("Loading model context...")
    
    if config is not None and model_path.exists() and model_path.suffix == '.pth':
        dls, hierarchy = gen_dls(**config['train'])
        backbone_fn = getattr(importlib.import_module('fastai.vision.all'), config['train']['model_arch_name'])
        body = create_body(backbone_fn(), pretrained=False)
        nf = num_features_model(body)

        heads = nn.ModuleList([
            create_head(nf, len(v)) for v in dls.vocab
        ])
        model_arch = MultiHead(body, heads)
        
        learn = Learner(dls, model_arch, loss_func=MultiHeadLoss())
        learn.load(model_path.with_suffix(''))
        learn.to('cpu' if cpu else 'cuda')
    elif model_path.exists():
        learn = load_learner(model_path, cpu=cpu)
    else:
        raise FileNotFoundError(f"Target model file not detected: {model_path}")
        
    learn.model = learn.model.eval()
    print("Model completely configured.")

    print("Reading hierarchy definitions...")
    if hierarchy_path is None and hasattr(learn, 'hierarchy'):
        hierarchy = learn.hierarchy
    elif hierarchy_path is not None and exists(hierarchy_path):
        hierarchy = pd.read_csv(hierarchy_path)
    elif 'hierarchy' not in locals():
        raise ValueError("Hierarchy configurations could not be parsed.")
    print("Hierarchy matrix active.")

    if name2id_path is not None and exists(name2id_path):
        print(f"Found existing name2id mapping: {name2id_path}.")
        name2id_test = pd.read_csv(name2id_path)
    elif name2id_path is not None:
        print("Name2id configured but missing, attempting programmatic download...")
        name2id_test = pd.DataFrame(gen_name2id(img_dir=img_dir))
        name2id_test.to_csv(name2id_path, index=False)

    print("Gathering testing images target path entries...")
    if parquet_path is None:
        filenames = list(img_dir.rglob("*"))
        filenames = [f for f in filenames if not isdir(f) and f.suffix.lower() in VALID_IMAGE_EXT]
    else:
        filenames = [(img_dir / str(r['speciesKey'])) / r['filename'] for _, r in df.iterrows()]
    print(f"Processing evaluation cycle for {len(filenames)} assets.")

    print("Creating testing dynamic data loader stack...")
    test_dl = learn.dls.test_dl(filenames)
    
    print("Executing predictive test time augmentations (TTA)...")
    # Use our custom multi-head TTA function instead of the default fastai one
    preds = multi_head_tta(learn, dl=test_dl, n=4, beta=0.25)
    print(f"Obtained {len(preds[0])} predictions across {len(preds)} classification heads.")

    print("Restructuring multi-head classification targets...")
    prds, cnfs = get_pred_conf_multihead(preds, learn.dls.vocab)

    print("Aligning validation target grounds labels...")
    if name2id_path is not None:
        name2id_test_dict = {
            r['verbatimScientificName']: [
                str(r['speciesKey']),
                str(r['genusKey']),
                str(r['familyKey'])]
            for _, r in name2id_test.iterrows()}
        lbls = np.array([name2id_test_dict[Path(f).parent.name] for f in filenames])
    elif parquet_path is not None:
        lbls = np.array([[str(r['speciesKey']), str(r['genusKey']), str(r['familyKey'])] for _, r in df.iterrows()])
    else:
        print("Defaulting validation checks against absolute local path names...")
        species = np.unique([Path(f).parent.name for f in filenames])
        name2id_test_dict = asyncio.run(get_all_parents(species))
        lbls = np.array([name2id_test_dict[Path(f).parent.name] for f in filenames])

    print("Generating comprehensive target metrics CSV report...")
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / model_path.with_suffix('.csv').name
    
    save_csv(
        out_path,
        filenames=[Path(f).name for f in filenames],
        prds=prds,
        cnfs=cnfs,
        lbls=lbls,
        vocabs=learn.dls.vocab
    )
    print(f"Metrics written down safely in: {out_path}")

def cli():
    parser = argparse.ArgumentParser(description="Multi-head predictive testing suite interface.")
    parser.add_argument("-c", "--config", type=str, help="Target application configurations YAML track file path.")
    args = parser.parse_args()
    
    if exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)

        assert float(config['version']) in VALID_CONFIG_VERSIONS, (
            f"Unsupported configuration version scheme: {config['version']}. Must conform to {VALID_CONFIG_VERSIONS}"
        ) 
        
        assert 'test' in config.keys(), "Configuration schema requires a robust root 'test' key."
        test(**config['test'], config=config)
    else:
        raise FileNotFoundError(f"Configuration profile target location unreachable: {args.config}")

if __name__ == '__main__':
    cli()