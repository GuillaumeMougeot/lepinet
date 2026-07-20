"""Phase A1 of the lepi-app plan: export a dev/030 checkpoint to ONNX, unmodified.

The point of this script is *de-risking*, not compression. 165 MB is unshippable, but it is
perfectly testable, and exporting it first answers the three questions that would force an
app redesign if discovered late (see journal/2026-07-lepi-app-claude.md, Phase A):

  1. Does the mini_trainer cosine head trace at all? It contains `acos`, a clamp, weight-norm
     caching and `_weight_bias` buffers -- none of it obviously ONNX-friendly until tried.
  2. Do ONNX Runtime and PyTorch agree numerically on the same input?
  3. What does the graph actually cost, in bytes and in latency?

Two deliberate choices, both of which reduce work in the browser:

* **Normalization is baked into the graph.** The exported model takes RGB float32 in [0,1]
  -- exactly what `canvas.getImageData()/255` yields -- and applies `imagenet_stats` inside
  ONNX. Every constant that lives in the graph is a constant the TypeScript cannot get
  wrong, and preprocessing mismatch is the classic silent accuracy loss in browser
  deployment (dev/041 measures what is left).
* **Raw logits out, no softmax.** Calibration (a temperature) and thresholds are not known
  at export time and are decided per-artifact in dev/044. Baking a softmax in would freeze a
  choice that is meant to travel in `thresholds.json` next to the weights.

Also emits `taxonomy.json`: the per-level vocabs in *head index order* plus the
species->genus->family parent arrays. The app needs it for labels and GBIF links; dev/042
needs the parent arrays to marginalize. Deriving it here, from the same checkpoint as the
weights, is what keeps the two from drifting apart.

Usage:
    python dev/040_onnx_export.py --checkpoint <path/to/*.pt> --out-dir <dir> [--img-size 256]
    python dev/040_onnx_export.py --checkpoint ... --opset 17 --no-check
"""

import argparse
import importlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fastai.vision.learner import create_body
from fastai.vision.all import imagenet_stats
from mini_trainer.hierarchical.integration import sparse_masks_from_labels

v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod030 = importlib.import_module("030_hierarchical_heads_benchmark")
mod032 = importlib.import_module("032_hierarchical_heads_test")

HIERARCHY_LEVELS = v4.HIERARCHY_LEVELS  # [speciesKey, genusKey, familyKey], finest -> coarsest
LEVEL_NAMES = ["species", "genus", "family"]


# ---------------------------------------------------------------------------
# Model reconstruction (no fastai DataLoaders)
# ---------------------------------------------------------------------------

def build_model(checkpoint):
    """Rebuild the trained `nn.Sequential(body, MTHeadAdapter)` from a checkpoint alone.

    dev/032 reconstructs the same model via `vision_learner`, which needs a `DataLoaders`,
    which needs the image directory and the parquet. Export needs neither -- only the module
    tree has to match the saved state_dict. `vision_learner` builds exactly
    `nn.Sequential(create_body(arch), custom_head)`, so building that directly is both
    equivalent and independent of the dataset being present, which matters because the
    artifact bundle should be reproducible from the .pt on any machine.
    """
    vocabs = checkpoint["vocabs"]
    cls2idx = checkpoint["cls2idx"]
    hierarchy_df = pd.DataFrame(checkpoint["hierarchy"])
    n_classes = [len(vocabs[level]) for level in HIERARCHY_LEVELS]

    sparse_masks = mod032.class_spec_from_hierarchy(hierarchy_df, vocabs, cls2idx)

    arch = getattr(importlib.import_module("fastai.vision.all"), checkpoint["model_arch_name"])
    nf = v4.body_out_features(arch)
    head = mod030.build_head(
        checkpoint["head"], nf, n_classes, cls2idx, sparse_masks,
        decoder_kwargs={"num_layers": 4, "nhead": 1},
    )
    body = create_body(arch(weights=None), n_in=3, pretrained=False)
    model = nn.Sequential(body, head)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocabs, n_classes


class ExportWrapper(nn.Module):
    """Normalization + backbone + head, returning one logit tensor per level.

    Input:  float32 [N, 3, H, W], RGB, already resized, values in [0, 1].
    Output: (species_logits, genus_logits, family_logits), raw.

    The `mean`/`std` buffers are registered rather than closed over so they are exported as
    initializers -- visible in the graph, and impossible for the frontend to disagree about.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        mean, std = imagenet_stats
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        out = self.model(x)  # list of per-level logits, finest -> coarsest
        return out[0], out[1], out[2]


# ---------------------------------------------------------------------------
# Taxonomy sidecar
# ---------------------------------------------------------------------------

def build_taxonomy(checkpoint, vocabs):
    """Vocabs in head-index order + parent index arrays, as a JSON-serializable dict.

    `parents["species_to_genus"][i]` is the genus-head index of species-head index `i`. Two
    consumers: the app (label lookup, GBIF deep links) and dev/042 (marginalization). They
    must agree, so there is exactly one place that derives them -- here.

    Built from the checkpoint's own `hierarchy` table, restricted to species the head can
    actually predict, for the same reason dev/032 restricts its masks: a mapping for a class
    with no output unit is meaningless, and `vocabs` is authoritative about what exists.
    """
    hierarchy_df = pd.DataFrame(checkpoint["hierarchy"])
    idx = {level: {str(k): i for i, k in enumerate(vocabs[level])} for level in HIERARCHY_LEVELS}
    sp_idx, gn_idx, fm_idx = (idx[level] for level in HIERARCHY_LEVELS)

    species_to_genus = np.full(len(vocabs[HIERARCHY_LEVELS[0]]), -1, dtype=np.int64)
    genus_to_family = np.full(len(vocabs[HIERARCHY_LEVELS[1]]), -1, dtype=np.int64)
    for row in hierarchy_df.itertuples(index=False):
        s, g, f = (str(getattr(row, level)) for level in HIERARCHY_LEVELS)
        if s in sp_idx and g in gn_idx:
            species_to_genus[sp_idx[s]] = gn_idx[g]
        if g in gn_idx and f in fm_idx:
            genus_to_family[gn_idx[g]] = fm_idx[f]

    for name, arr in [("species_to_genus", species_to_genus), ("genus_to_family", genus_to_family)]:
        missing = int((arr < 0).sum())
        if missing:
            # Not fatal for export (the ONNX graph never reads these), but it *is* fatal for
            # marginalization, so it must be loud rather than a -1 quietly indexing the last row.
            print(f"WARNING: {missing} entries of {name} have no parent in the saved hierarchy.")

    return {
        "levels": LEVEL_NAMES,
        "hierarchy_levels": HIERARCHY_LEVELS,
        "vocabs": {name: [str(v) for v in vocabs[level]]
                   for name, level in zip(LEVEL_NAMES, HIERARCHY_LEVELS)},
        "parents": {
            "species_to_genus": species_to_genus.tolist(),
            "genus_to_family": genus_to_family.tolist(),
        },
        "note": "vocab entries are GBIF taxon keys, in head-index order; "
                "GBIF page = https://www.gbif.org/species/<key>",
    }


# ---------------------------------------------------------------------------
# Export + verification
# ---------------------------------------------------------------------------

def total_size(onnx_path):
    """Bytes of the model *including* any external-data sidecar.

    `Path.stat().st_size` on a `.onnx` with external data reports the graph only -- 1.6 MB for
    a 171 MB model. That number is worse than useless in a project whose whole metric is file
    size, so size is always measured over the graph plus its `.data` siblings.
    """
    onnx_path = Path(onnx_path)
    total = onnx_path.stat().st_size
    for sidecar in onnx_path.parent.glob(onnx_path.name + ".data*"):
        total += sidecar.stat().st_size
    return total


def read_opset(onnx_path):
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=False)
    return max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=None)


def export(checkpoint_path, out_dir, img_size=256, opset=18, check=True, batch=1, single_file=True):
    checkpoint_path = mod032.resolve_model_path(str(checkpoint_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, vocabs, n_classes = build_model(checkpoint)
    print(f"Model: head={checkpoint['head']} arch={checkpoint['model_arch_name']} "
          f"classes={n_classes}")

    n_params = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in model[1].parameters())
    print(f"Params: {n_params/1e6:.2f} M total, {n_head/1e6:.2f} M head "
          f"({100*n_head/n_params:.0f}%), {n_params*4/1e6:.1f} MB fp32")

    wrapper = ExportWrapper(model).eval()
    dummy = torch.rand(batch, 3, img_size, img_size)

    # Warm the head's lazy caches in eager mode *before* tracing. `HierarchicalClassifier.masks`
    # and `ConditionalClassifier._weight_bias` are memoized on first access, and building them
    # runs `int(mask.max().item() + 1)` -- a data-dependent shape that torch.export cannot
    # specialize ("Could not extract specialized integer from data-dependent expression u0+1").
    # After one eager forward the caches are populated and both are plain buffer reads, so the
    # traced graph sees constants. The state_dict load is what dirties them, which is why this
    # cannot simply be left to the head's own constructor warm-up.
    with torch.no_grad():
        wrapper(dummy)

    onnx_path = out_dir / "model.onnx"
    print(f"Exporting to {onnx_path} (opset {opset}, {img_size}x{img_size})...")
    torch.onnx.export(
        wrapper, (dummy,), str(onnx_path),
        input_names=["image"],
        output_names=[f"logits_{n}" for n in LEVEL_NAMES],
        # Batch stays dynamic: the app runs batch 1, but the parity harness (dev/041) and any
        # offline re-scoring want to push many images through one session.
        dynamic_axes={"image": {0: "batch"}, **{f"logits_{n}": {0: "batch"} for n in LEVEL_NAMES}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=True,
        # Weights inside the .onnx rather than a sibling .onnx.data. The app fetches and caches
        # one file, and a two-file model is a way to ship a graph whose weights never arrived.
        # Safe while the artifact is far under protobuf's 2 GB ceiling, which it always is --
        # the whole point of the exercise is to end up at single-digit MB.
        external_data=not single_file,
    )
    size_bytes = total_size(onnx_path)
    print(f"Wrote {onnx_path.name}: {size_bytes/1e6:.1f} MB")

    # Read the opset back rather than trusting the request. torch.onnx down-converts as a
    # post-pass, and that pass can fail non-fatally (onnx's version_converter raises on some
    # graphs, is caught, and the un-converted model is written) -- so an export that *asked*
    # for 17 can silently be an 18 graph. ORT Web's minimum supported opset is the one thing
    # the app cannot negotiate at runtime, so the manifest must state what is really there.
    actual_opset = read_opset(onnx_path)
    if actual_opset != opset:
        print(f"NOTE: requested opset {opset}, graph is actually opset {actual_opset} "
              f"(down-conversion did not apply). Recording the real value.")

    tax = build_taxonomy(checkpoint, vocabs)
    tax_path = out_dir / "taxonomy.json"
    tax_path.write_text(json.dumps(tax))
    print(f"Wrote {tax_path.name}: {tax_path.stat().st_size/1e6:.1f} MB "
          f"({len(tax['vocabs']['species'])} species / {len(tax['vocabs']['genus'])} genera / "
          f"{len(tax['vocabs']['family'])} families)")

    manifest = {
        "source_checkpoint": str(checkpoint_path),
        "model_name": Path(checkpoint_path).stem,
        "head": checkpoint["head"],
        "arch": checkpoint["model_arch_name"],
        "n_classes": {n: c for n, c in zip(LEVEL_NAMES, n_classes)},
        "params_total": int(n_params),
        "params_head": int(n_head),
        "onnx_opset": actual_opset,
        "onnx_size_bytes": size_bytes,
        "preprocessing": {
            "input": "float32 NCHW, RGB, [0,1]",
            "img_size": img_size,
            # Recorded even though it is baked in: the app still has to reproduce the *resize*,
            # and a future artifact could bake different constants. The manifest is the only
            # place the two repos share a spec.
            "normalization": "baked into graph (imagenet_stats)",
            "imagenet_mean": list(imagenet_stats[0]),
            "imagenet_std": list(imagenet_stats[1]),
            "train_resize": "fastai Resize(aug_img_size=460) then aug_transforms(size=256)",
        },
        "outputs": [f"logits_{n}" for n in LEVEL_NAMES],
        "note": "raw logits; calibration + thresholds ship separately (dev/044)",
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    if check:
        verify(wrapper, onnx_path, img_size, batch=2)
    return onnx_path


@torch.no_grad()
def verify(wrapper, onnx_path, img_size, batch=2, tol=2e-3):
    """PyTorch vs ONNX Runtime on identical random input.

    This is *graph* parity only -- same tensor in, same numbers out. It does not test the
    preprocessing that produces that tensor in a browser, which is a different and more
    dangerous question; dev/041 covers it. Keeping them separate means a failure here points
    at the export and a failure there points at the resize, instead of one number that could
    mean either.
    """
    import onnxruntime as ort

    print("\nVerifying PyTorch vs ONNX Runtime...")
    x = torch.rand(batch, 3, img_size, img_size)
    torch_out = [o.numpy() for o in wrapper(x)]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": x.numpy()})

    ok = True
    for name, t, o in zip(LEVEL_NAMES, torch_out, ort_out):
        diff = float(np.abs(t - o).max())
        agree = float((t.argmax(1) == o.argmax(1)).mean())
        flag = "OK " if diff < tol else "FAIL"
        if diff >= tol:
            ok = False
        print(f"  [{flag}] {name:8s} max|Δlogit| = {diff:.3e}  top-1 agreement = {agree:.0%}")
    print("Graph parity: PASS" if ok else "Graph parity: FAIL")
    return ok


def cli():
    p = argparse.ArgumentParser(description="Export a dev/030 checkpoint to ONNX (Phase A1).")
    p.add_argument("-c", "--checkpoint", required=True, help="Path to a dev/030 .pt (glob allowed).")
    p.add_argument("-o", "--out-dir", required=True, help="Output directory for the artifact bundle.")
    p.add_argument("--img-size", type=int, default=256, help="Square input size (default: 256, the training size).")
    p.add_argument("--opset", type=int, default=18, help="ONNX opset (default 18: what the dynamo exporter emits).")
    p.add_argument("--external-data", action="store_true",
                   help="Write weights to a sibling model.onnx.data instead of embedding them.")
    p.add_argument("--no-check", action="store_true", help="Skip the ORT parity check.")
    a = p.parse_args()
    export(a.checkpoint, a.out_dir, img_size=a.img_size, opset=a.opset, check=not a.no_check,
           single_file=not a.external_data)


if __name__ == "__main__":
    cli()
