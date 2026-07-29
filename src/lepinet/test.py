"""Evaluation: a trained ``.pt`` → predictions + a native per-level metric report.

Reconstructs the model from a checkpoint alone (no DataLoaders / no parquet needed for the model
itself), streams top-1 predictions over a test fold, computes the metric report **natively**
(dropping the ``mini_metrics`` dependency — :class:`~lepinet.metrics.LevelMacroF1` reproduces its
macro-F1), and — for interoperability — writes ``predictions.csv`` in ``mini_metrics``' long input
format so the external tool can still be run on the same predictions if desired.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import DEFAULT_LEVELS, ensure_fork_start_method, make_dls
from .heads import build_head, infer_hidden_from_state_dict
from .model import arch_body_features, build_backbone_model, resolve_arch

# ---------------------------------------------------------------------------
# Model reconstruction from a checkpoint
# ---------------------------------------------------------------------------

def load_model(checkpoint: dict, img_size: int = 256):
    """Rebuild ``nn.Sequential(body, PooledHead)`` from a checkpoint dict and load its weights.

    Returns ``(model, meta)`` where ``meta`` has ``vocabs``, ``levels``, ``hierarchy``. Geometry
    (hidden width) is read from the weights, so a stale ``hidden`` metadata field is harmless.
    """
    levels = checkpoint.get("levels", DEFAULT_LEVELS)
    vocabs = checkpoint["vocabs"]
    n_classes = [len(vocabs[level]) for level in levels]
    hierarchy_df = pd.DataFrame(checkpoint["hierarchy"])

    sd = checkpoint["model_state_dict"]
    head_sd = {k[len("1.head."):]: v for k, v in sd.items() if k.startswith("1.head.")}
    hidden = infer_hidden_from_state_dict(head_sd, prefix="")

    arch = resolve_arch(checkpoint["model_arch_name"])
    vit = checkpoint.get("vit", False)              # ViT/DINOv3 backbone → FlatHead + ViTBody
    nf = arch_body_features(arch, img_size=img_size)
    # arcface only needs `scale` at inference (the margin is training-only); default 30 for the
    # cosine baseline and pre-existing checkpoints (which lack the key).
    head_kwargs = {"scale": checkpoint.get("arcface_scale", 30.0)} if checkpoint["head"] == "arcface" else {}
    # Taxonomy heads (hierarchical / autoregressive) take `sparse_masks`; the real masks are saved
    # as persistent buffers (mask_i, shape [n_children_i]), so pass correctly-shaped dummies and let
    # the strict load overwrite them. Signature-driven, so non-taxonomy heads are unaffected. (These
    # heads live in dev/ — import the dev module that registers them before loading.)
    import inspect

    from .heads import HEAD_REGISTRY
    if checkpoint["head"] in HEAD_REGISTRY and "sparse_masks" in inspect.signature(HEAD_REGISTRY[checkpoint["head"]]).parameters:
        head_kwargs["sparse_masks"] = [torch.zeros(n_classes[i], dtype=torch.long) for i in range(len(n_classes) - 1)]
    head = build_head(checkpoint["head"], nf, n_classes, hidden=hidden, pool=not vit, **head_kwargs)
    model = build_backbone_model(arch, head, vit=vit)
    model.load_state_dict(sd, strict=True)
    model.eval()
    meta = {"vocabs": vocabs, "levels": levels, "hierarchy": hierarchy_df}
    return model, meta


def resolve_checkpoint_path(model_path: str | Path) -> Path:
    """Resolve a checkpoint path that may be a glob (newest match wins)."""
    if not glob.has_magic(str(model_path)):
        return Path(model_path)
    matches = [Path(p) for p in glob.glob(str(model_path))]
    if not matches:
        raise FileNotFoundError(f"No checkpoint matches: {model_path}")
    return max(matches, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def dl_num_workers(dl, default: int = 0) -> int:
    """The *real* worker count of a fastai ``DataLoader``.

    **Do not read ``dl.num_workers``** — fastai hardcodes that attribute to ``1`` in
    ``DataLoader.__init__`` (``self.rng, self.num_workers, self.offs = ..., 1, 0``) regardless of what
    was requested; the effective count lives on the internal ``fake_l``. Reading the public attribute
    silently pinned evaluation to a **single** worker, which on a ~90 ms/file network mount meant
    ~1 img/s — a 630 k-image eval would have taken a week. This helper is the guard against that trap.
    """
    fake = getattr(dl, "fake_l", None)
    n = getattr(fake, "num_workers", None)
    return int(n) if n is not None else default



def _paths_exist(root: Path, rel_paths) -> pd.Series:
    """Vectorised existence check for many files under ``root`` — one listing per directory.

    A naive ``(root / p).is_file()`` per row is one network round-trip **per image**: at ~90 ms/file
    on the ``/work`` mount, 630 k images is >15 h before inference even starts. Instead we list each
    *parent directory once* (one round-trip returns every name in it) and do the membership test in
    memory, with a thread pool because the cost is latency, not CPU. ~12 k directories at 32 threads
    is well under a minute.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    rel = pd.Series(rel_paths).astype(str)
    parents = rel.map(lambda p: p.rsplit("/", 1)[0] if "/" in p else "")
    unique_dirs = sorted(parents.unique())

    def listing(d: str) -> tuple[str, set[str]]:
        try:
            return d, set(os.listdir(root / d if d else root))
        except OSError:
            return d, set()

    with ThreadPoolExecutor(max_workers=32) as ex:
        contents = dict(ex.map(listing, unique_dirs))
    names = rel.map(lambda p: p.rsplit("/", 1)[-1])
    return pd.Series([n in contents.get(d, ()) for d, n in zip(parents, names)], index=rel.index)


@torch.no_grad()
def predict_df(model, dls, test_df, vocabs, levels, device, tta: bool = False,
               num_workers: int | None = None):
    """Stream the model over ``test_df`` → per-level top-1 label + softmax confidence arrays.

    Reduces to top-1 per batch (not accumulating full logits): at global scale the species logit
    matrix alone is ~30 GB, so streaming keeps peak memory at one batch. ``tta=True`` averages the
    softmax over the 4 dihedral flips (identity + h-flip + v-flip + both) — the same test-time
    augmentation :func:`lepinet.infer.predict` uses; ~4x slower, usually +a fraction of a point.
    """
    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    test_dl = dls.test_dl(test_df, num_workers=nw)
    print(f"Inference dataloader: num_workers={nw}, batches={len(test_dl)}")
    model.to(device)
    vocab_arrays = [np.array([str(v) for v in vocabs[level]]) for level in levels]
    pred_chunks = [[] for _ in levels]
    conf_chunks = [[] for _ in levels]
    for batch in test_dl:
        x = batch[0].to(device)
        views = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)] if tta else [x]
        probs_sum = None
        for v in views:
            out = model(v)
            p = [torch.softmax(out[i].float(), dim=1) for i in range(len(levels))]
            probs_sum = p if probs_sum is None else [probs_sum[i] + p[i] for i in range(len(levels))]
        for i in range(len(levels)):
            probs = probs_sum[i] / len(views)
            conf, idx = probs.max(dim=1)
            pred_chunks[i].append(vocab_arrays[i][idx.cpu().numpy()])
            conf_chunks[i].append(conf.cpu().numpy())
    preds = np.stack([np.concatenate(c) for c in pred_chunks], axis=1)
    confs = np.stack([np.concatenate(c) for c in conf_chunks], axis=1)
    return preds, confs


# ---------------------------------------------------------------------------
# Native metric report (matches mini_metrics; no dependency on it)
# ---------------------------------------------------------------------------

def macro_f1(pred: np.ndarray, true: np.ndarray, classes) -> float:
    """Unweighted mean per-class F1 over classes present as a ground-truth label (mini_metrics' MacroF1).

    F1 is the harmonic mean of precision and recall, defined as 0 when either is 0 (e.g. a
    never-predicted class). Only classes that appear in ``true`` are averaged.
    """
    classes = [str(c) for c in classes]
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    tp = np.zeros(n)
    fp = np.zeros(n)
    fn = np.zeros(n)
    for p, t in zip(pred.astype(str), true.astype(str)):
        pi, ti = idx.get(p), idx.get(t)
        if pi is not None and pi == ti:
            tp[pi] += 1
        else:
            if pi is not None:
                fp[pi] += 1
            if ti is not None:
                fn[ti] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    present = (tp + fn) > 0
    return float(f1[present].mean()) if present.any() else float("nan")


def metric_report(preds: np.ndarray, labels: np.ndarray, vocabs, levels) -> dict:
    """Per-level macro-F1 + micro-accuracy, plus a species-level headline."""
    report = {}
    for i, level in enumerate(levels):
        p, t = preds[:, i], labels[:, i]
        report[level] = {
            "macro_f1": macro_f1(p, t, vocabs[level]),
            "micro_acc": float((p.astype(str) == t.astype(str)).mean()),
            "n": int(len(t)),
        }
    report["species_macro_f1"] = report[levels[0]]["macro_f1"]
    return report


# ---------------------------------------------------------------------------
# mini_metrics-format prediction CSV (interop)
# ---------------------------------------------------------------------------

def save_predictions_csv(path, filenames, preds, confs, labels, vocabs, levels, threshold=0.5):
    """Write predictions in ``mini_metrics``' long input format (one row per image × level).

    Columns: ``instance_id, filename, level, label, prediction, confidence, threshold,
    known_label``. This is the interoperability seam (D2): the external ``mini_metrics`` tool can
    be run on this file unchanged.
    """
    n, p = preds.shape
    known = {i: {str(v) for v in vocabs[level]} for i, level in enumerate(levels)}
    if isinstance(threshold, (list, tuple)):
        thr_col = np.tile(np.asarray(threshold, float), n) if len(threshold) == p else np.full(n * p, threshold[0], float)
    else:
        thr_col = float(threshold)
    df = pd.DataFrame({
        "instance_id": np.repeat(np.arange(n), p),
        "filename": np.repeat(filenames, p),
        "level": np.tile(np.arange(p), n),
        "label": labels.flatten(order="C").astype(str),
        "prediction": preds.flatten(order="C").astype(str),
        "confidence": confs.flatten(order="C"),
        "threshold": thr_col,
    })
    df["known_label"] = [lbl in known[lvl] for lvl, lbl in zip(df["level"], df["label"])]
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    parquet_path: str,
    img_dir: str,
    out_dir: str,
    eval_name: str = "eval",
    test_set: str = "0",
    min_img_per_spc: int = 0,
    family_filter: list | None = None,
    batch_size: int = 64,
    aug_img_size: int = 460,
    img_size: int = 256,
    num_workers: int | None = None,
    drop_unknown_species: bool = True,
    tta: bool = False,
    skip_missing: bool = True,
    limit: int | None = None,
) -> Path:
    """Evaluate a checkpoint on a held-out fold; write predictions + native metric report.

    Output: ``<out_dir>/<model_name>/<eval_name>/`` with ``predictions.csv`` (mini_metrics format),
    ``metrics.json`` (native report), ``combinations.csv`` (the hierarchy used).

    ``drop_unknown_species=True`` (default) keeps only species in the model's vocabulary — the right
    behaviour for a held-out fold of the *training* distribution. Set it ``False`` for an **open-set /
    OOD** test (e.g. an external dataset): out-of-vocabulary species are kept, the model still predicts
    an in-vocab label for them, and each prediction row carries ``known_label=False`` so
    ``mini_metrics`` (and the native report) can separate known from OOD. The model can never score an
    OOD species correct, so its per-class F1 is 0 by construction — that is the honest open-set number.
    """
    ensure_fork_start_method()
    from .data import filter_df  # local: keeps import graph obvious

    family_filter = family_filter or []
    model_path = resolve_checkpoint_path(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    levels = checkpoint.get("levels", DEFAULT_LEVELS)
    vocabs = checkpoint["vocabs"]
    model_name = model_path.stem

    # --- build test df (held-out fold) ---
    df = pd.read_parquet(parquet_path)
    df = filter_df(df, keep_in=[test_set], min_img_per_spc=min_img_per_spc, family_filter=family_filter, levels=levels)
    known = set(str(v) for v in vocabs[levels[0]])
    n_all = len(df)
    if drop_unknown_species:
        df = df[df[levels[0]].astype(str).isin(known)]
    else:
        n_ood = int((~df[levels[0]].astype(str).isin(known)).sum())
        n_ood_spc = int(df.loc[~df[levels[0]].astype(str).isin(known), levels[0]].nunique())
        print(f"Open-set eval: keeping {n_ood} OOD-species images ({n_ood_spc} species) of {n_all} total.")
    if len(df) == 0:
        raise ValueError(f"No rows in test fold '{test_set}' of {parquet_path} for known classes.")
    # Image path: use a pre-supplied ``image_path`` column if the parquet has one (datasets whose
    # folders are named by something other than the speciesKey, e.g. species *name*); otherwise
    # reconstruct the standard ``<speciesKey>/<filename>`` layout.
    if "image_path" not in df.columns:
        df["image_path"] = df[levels[0]].astype(str) + "/" + df["filename"]
    for level in levels:
        df[level] = df[level].astype(str)
    # Drop rows whose image file is absent. The parquet is the *catalogue*; an image mirror can be
    # incomplete (a handful of the 630 k global_lepi files are missing). Without this a multi-hour
    # eval dies at the first gap with a bare FileNotFoundError deep inside the DataLoader — the whole
    # run lost for one file. Skipping is the honest behaviour (the metric is over what exists) and
    # the count is reported so a *large* number of misses is visible rather than silent.
    if skip_missing:
        root = Path(img_dir)
        exists = _paths_exist(root, df["image_path"])
        n_missing = int((~exists).sum())
        if n_missing:
            print(f"WARNING: {n_missing} of {len(df)} images are missing from {img_dir} — skipping them.")
            df = df[exists]
        if len(df) == 0:
            raise ValueError(f"Every image listed in {parquet_path} is missing under {img_dir}.")
    if limit:
        # Throughput probe / smoke: evaluate a random subset. Metrics on a subset are NOT comparable
        # to a full-fold number -- this exists to measure img/s and validate a config cheaply.
        df = df.sample(n=min(int(limit), len(df)), random_state=0)
        print(f"LIMIT: evaluating a random subset of {len(df)} images (metrics not comparable to a full run).")
    df["is_valid"] = np.arange(len(df)) % 5 == 0  # dummy split; we use test_dl regardless
    test_df = df[["image_path", "is_valid", *levels]].reset_index(drop=True)
    print(f"Test images: {len(test_df)} | {levels[0]} present: {test_df[levels[0]].nunique()}")

    # lowmem=False: test_dl feeds a DataFrame, incompatible with the numpy-indexed lowmem getters.
    dls = make_dls(test_df, vocabs, img_dir, aug_img_size, img_size, batch_size, num_workers,
                   lowmem=False, levels=levels)
    model, _meta = load_model(checkpoint, img_size=img_size)

    import time as _time
    print(f"Running inference{' with TTA (4-flip)' if tta else ''}...")
    _t0 = _time.perf_counter()
    preds, confs = predict_df(model, dls, test_df, vocabs, levels, device, tta=tta,
                              num_workers=num_workers)
    _dt = _time.perf_counter() - _t0
    print(f"Inference: {len(test_df)} images in {_dt:.1f}s = {len(test_df)/max(_dt,1e-9):.1f} img/s")
    labels = test_df[levels].to_numpy().astype(str)
    filenames = test_df["image_path"].to_numpy()

    out_base = Path(out_dir) / model_name / eval_name
    out_base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checkpoint["hierarchy"])[levels].astype(str).to_csv(out_base / "combinations.csv", index=False)
    save_predictions_csv(out_base / "predictions.csv", filenames, preds, confs, labels, vocabs, levels)

    report = metric_report(preds, labels, vocabs, levels)
    with open(out_base / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nspecies macro-F1: {report['species_macro_f1']:.4f}")
    for level in levels:
        r = report[level]
        print(f"  {level:12s} macro-F1 {r['macro_f1']:.4f}  micro-acc {r['micro_acc']:.4f}")
    print(f"Report saved to: {out_base}")
    return out_base
