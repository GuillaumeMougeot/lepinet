"""ONNX export + taxonomy sidecar + (optional) hierarchy marginalization.

Exports the trained model to a browser-ready ONNX graph. Two deliberate choices carried over from
``dev/040``: normalization is **baked into the graph** (input is RGB ``[0,1]`` NCHW, exactly what a
canvas yields) so the frontend cannot get it wrong, and the graph emits **raw logits** (calibration
and thresholds ship separately).

Unlike ``dev/040``, this uses **``dynamo=False``** (the legacy TorchScript exporter) and needs **no
lazy-cache warm-up**: the clean :class:`~lepinet.heads.IndependentHead` has no data-dependent
control flow (no ``masks`` / ``_weight_bias`` on the forward path), so it traces directly. That is a
concrete payoff of the simplification (``journal/2026-07-src-lepinet-baseline-port.md``, D4).

:func:`marginalize` computes coarser-level probabilities from the finest level
(``P(genus) = Σ P(species∈genus)``) — export-only, used when shipping a species-only head.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .infer import IMAGENET_MEAN, IMAGENET_STD
from .test import load_model, resolve_checkpoint_path


class ExportWrapper(nn.Module):
    """Normalization + backbone + head, returning one raw-logit tensor per level.

    Input: float32 ``[N, 3, H, W]``, RGB, resized, values in ``[0, 1]``. The ``mean``/``std`` are
    registered buffers so they export as graph initializers — visible and impossible for the
    frontend to disagree about.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor):
        out = self.model((image - self.mean) / self.std)  # list of per-level logits
        return tuple(out)


def build_taxonomy(checkpoint: dict, meta: dict) -> dict:
    """Per-level vocabs in head-index order + parent index arrays (for labels, GBIF links, marginalization)."""
    levels = meta["levels"]
    vocabs = meta["vocabs"]
    hierarchy_df = meta["hierarchy"]
    idx = {level: {str(k): i for i, k in enumerate(vocabs[level])} for level in levels}

    parents = {}
    for child, parent in zip(levels[:-1], levels[1:]):
        arr = np.full(len(vocabs[child]), -1, dtype=np.int64)
        for row in hierarchy_df.itertuples(index=False):
            c, p = str(getattr(row, child)), str(getattr(row, parent))
            if c in idx[child] and p in idx[parent]:
                arr[idx[child][c]] = idx[parent][p]
        missing = int((arr < 0).sum())
        if missing:
            print(f"WARNING: {missing} entries of {child}->{parent} have no parent in the hierarchy.")
        parents[f"{child}_to_{parent}"] = arr.tolist()

    return {
        "levels": levels,
        "vocabs": {level: [str(v) for v in vocabs[level]] for level in levels},
        "parents": parents,
        "note": "vocab entries are GBIF taxon keys in head-index order; "
                "GBIF page = https://www.gbif.org/species/<key>",
    }


def _read_opset(onnx_path: Path):
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=None)


def export_onnx(
    checkpoint_path: str,
    out_dir: str,
    img_size: int = 256,
    opset: int = 17,
    check: bool = True,
    single_file: bool = True,
    dynamo: bool = False,
) -> Path:
    """Export a checkpoint to ``<out_dir>/model.onnx`` + ``taxonomy.json`` + ``MANIFEST.json``.

    ``dynamo=False`` (default) uses the legacy exporter — reliable for this graph and what the app
    pipeline is verified against. ``check`` runs a PyTorch-vs-ONNX-Runtime parity assertion.
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, meta = load_model(checkpoint, img_size=img_size)
    levels = meta["levels"]
    n_classes = [len(meta["vocabs"][level]) for level in levels]

    n_params = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in model[1].parameters())
    print(f"Model: head={checkpoint['head']} arch={checkpoint['model_arch_name']} classes={n_classes}")
    print(f"Params: {n_params/1e6:.2f} M total, {n_head/1e6:.2f} M head ({100*n_head/n_params:.0f}%)")

    wrapper = ExportWrapper(model).eval()
    dummy = torch.rand(1, 3, img_size, img_size)
    onnx_path = out_dir / "model.onnx"
    print(f"Exporting to {onnx_path} (opset {opset}, {img_size}x{img_size}, dynamo={dynamo})...")
    output_names = [f"logits_{level}" for level in levels]
    torch.onnx.export(
        wrapper, (dummy,), str(onnx_path),
        input_names=["image"],
        output_names=output_names,
        dynamic_axes={"image": {0: "batch"}, **{n: {0: "batch"} for n in output_names}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=dynamo,
    )
    actual_opset = _read_opset(onnx_path)

    tax = build_taxonomy(checkpoint, meta)
    (out_dir / "taxonomy.json").write_text(json.dumps(tax))
    manifest = {
        "source_checkpoint": str(checkpoint_path),
        "model_name": checkpoint_path.stem,
        "head": checkpoint["head"],
        "arch": checkpoint["model_arch_name"],
        "levels": levels,
        "n_classes": dict(zip(levels, n_classes)),
        "params_total": int(n_params),
        "params_head": int(n_head),
        "onnx_opset": actual_opset,
        "preprocessing": {
            "input": "float32 NCHW, RGB, [0,1]",
            "img_size": img_size,
            "normalization": "baked into graph (imagenet_stats)",
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
        },
        "outputs": output_names,
        "note": "raw logits; calibration + thresholds ship separately",
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote taxonomy.json ({len(tax['vocabs'][levels[0]])} {levels[0]}) and MANIFEST.json")

    if check:
        verify_onnx(wrapper, onnx_path, img_size, output_names)
    return onnx_path


@torch.no_grad()
def verify_onnx(wrapper, onnx_path, img_size, output_names, batch=2, tol=2e-3) -> bool:
    """Assert PyTorch and ONNX Runtime agree on identical random input (graph parity)."""
    import onnxruntime as ort

    x = torch.rand(batch, 3, img_size, img_size)
    torch_out = [o.numpy() for o in wrapper(x)]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": x.numpy()})
    ok = True
    print("\nPyTorch vs ONNX Runtime:")
    for name, t, o in zip(output_names, torch_out, ort_out):
        diff = float(np.abs(t - o).max())
        agree = float((t.argmax(1) == o.argmax(1)).mean())
        ok = ok and diff < tol
        print(f"  [{'OK ' if diff < tol else 'FAIL'}] {name:20s} max|Δlogit| {diff:.3e}  top-1 agree {agree:.0%}")
    print("Graph parity: PASS" if ok else "Graph parity: FAIL")
    return ok


# ---------------------------------------------------------------------------
# Marginalization (export-only): coarser levels from the finest head
# ---------------------------------------------------------------------------

def scatter_logsumexp(log_probs: torch.Tensor, parent_idx: torch.Tensor, n_parents: int) -> torch.Tensor:
    """``log P(parent) = logsumexp`` over its children (numerically stable, batched)."""
    n = log_probs.shape[0]
    idx = parent_idx.unsqueeze(0).expand(n, -1)
    maxes = torch.full((n, n_parents), float("-inf"), device=log_probs.device, dtype=log_probs.dtype)
    maxes = maxes.scatter_reduce(1, idx, log_probs, reduce="amax", include_self=True)
    shifted = (log_probs - maxes.gather(1, idx)).exp()
    sums = torch.zeros((n, n_parents), device=log_probs.device, dtype=log_probs.dtype)
    sums = sums.scatter_add(1, idx, shifted)
    return sums.log() + maxes


def marginalize(species_logits: torch.Tensor, taxonomy: dict) -> list[torch.Tensor]:
    """Coarser-level log-probabilities marginalized from species logits, using the parent arrays.

    Returns per-level log-probabilities (fine→coarse), each guaranteed consistent with the level
    below by construction. Lets a species-only head serve all levels (drops the coarse heads from
    the shipped artifact); verify it matches or beats the trained coarse heads before relying on it.
    """
    levels = taxonomy["levels"]
    log_sp = torch.log_softmax(species_logits.float(), dim=1)
    out = [log_sp]
    cur = log_sp
    for child, parent in zip(levels[:-1], levels[1:]):
        parent_idx = torch.as_tensor(taxonomy["parents"][f"{child}_to_{parent}"], dtype=torch.long,
                                     device=species_logits.device)
        cur = scatter_logsumexp(cur, parent_idx, len(taxonomy["vocabs"][parent]))
        out.append(cur)
    return out
