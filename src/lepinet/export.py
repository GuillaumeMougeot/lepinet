"""ONNX export + taxonomy sidecar + (optional) hierarchy marginalization.

Exports the trained model to a browser-ready ONNX graph. Two deliberate choices carried over from
``dev/040``: normalization is **baked into the graph** (input is RGB ``[0,1]`` NCHW, exactly what a
canvas yields) so the frontend cannot get it wrong, and the graph emits **raw logits** (calibration
and thresholds ship separately).

Unlike ``dev/040``, this uses **``dynamo=False``** (the legacy TorchScript exporter) and needs **no
lazy-cache warm-up**: the clean :class:`~lepinet.heads.IndependentHead` has no data-dependent
control flow (no ``masks`` / ``_weight_bias`` on the forward path), so it traces directly. That is a
concrete payoff of the simplification (``journal/2026-07-24-src-lepinet-baseline-port.md``, D4).

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


def friendly_level_names(levels) -> list[str]:
    """App-facing level names from the internal GBIF-column names: ``speciesKey`` → ``species``.

    The training/config side names levels by their parquet columns (``speciesKey``, ``genusKey``,
    ``familyKey``); the app (and any human) wants ``species`` / ``genus`` / ``family``. Stripping a
    trailing ``Key`` is the whole mapping for the Lepidoptera hierarchy, and it degrades to identity
    for any other level name — so a custom hierarchy just ships whatever names it uses.
    """
    return [lvl[:-3] if lvl.endswith("Key") else lvl for lvl in levels]


class Fp16ExportWrapper(nn.Module):
    """Half-precision **backbone** + fp32 **head**, exported from source (not converted after).

    Post-hoc fp16 conversion of our ONNX graph does not work: the legacy TorchScript exporter emits
    explicit ``Cast`` nodes that ``onnxconverter_common`` rewrites inconsistently (invalid graph), and
    blocking ``Cast`` converts nothing. Tracing a genuinely half-precision module instead produces a
    valid graph with fp16 initializers — roughly half the file, and (unlike int8) **no
    ``ConvInteger``/``MatMulInteger``**, which is what ORT-Web cannot execute.

    The cosine head stays **fp32**: it L2-normalizes and takes ``acos`` of values near ±1, which is
    exactly where fp16 loses the precision that matters (the same reason
    :class:`~lepinet.heads.PooledHead` forces fp32 under autocast). Input/output stay fp32 so the app
    feeds and reads the graph unchanged.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.body = model[0].half()
        self.head = model[1].float()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor):
        x = (image - self.mean) / self.std      # fp32 normalization
        x = self.body(x.half())                 # fp16 backbone (the bulk of the weights)
        out = self.head(x.float())              # fp32 cosine head
        return tuple(out)


def build_taxonomy(checkpoint: dict, meta: dict, level_names: list[str] | None = None) -> dict:
    """Per-level vocabs in head-index order + parent index arrays (for labels, GBIF links, marginalization).

    ``level_names`` relabels the emitted keys (``levels`` / ``vocabs`` / ``parents``) to the
    app-facing names; the internal ``meta['levels']`` still drives the lookup, so nothing about the
    trained model changes — only the JSON keys. Defaults to the internal names (no relabel).
    """
    levels = meta["levels"]
    names = list(level_names) if level_names is not None else list(levels)
    vocabs = meta["vocabs"]
    hierarchy_df = meta["hierarchy"]
    idx = {level: {str(k): i for i, k in enumerate(vocabs[level])} for level in levels}

    parents = {}
    for (child, parent), (cn, pn) in zip(zip(levels[:-1], levels[1:]), zip(names[:-1], names[1:])):
        arr = np.full(len(vocabs[child]), -1, dtype=np.int64)
        for row in hierarchy_df.itertuples(index=False):
            c, p = str(getattr(row, child)), str(getattr(row, parent))
            if c in idx[child] and p in idx[parent]:
                arr[idx[child][c]] = idx[parent][p]
        missing = int((arr < 0).sum())
        if missing:
            print(f"WARNING: {missing} entries of {cn}->{pn} have no parent in the hierarchy.")
        parents[f"{cn}_to_{pn}"] = arr.tolist()

    return {
        "levels": names,
        "vocabs": {name: [str(v) for v in vocabs[level]] for level, name in zip(levels, names)},
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
    level_names: list[str] | None = None,
    write_config: bool = True,
    bundle_name: str | None = None,
    precision: str = "fp32",
) -> Path:
    """Export a checkpoint to an **app-ready bundle**: ``model.onnx`` + ``taxonomy.json`` +
    ``config.json`` + ``MANIFEST.json`` in ``out_dir``.

    The graph output names and ``taxonomy.json`` keys use **app-facing level names**
    (:func:`friendly_level_names`, e.g. ``species``), and ``config.json`` is the bundle descriptor
    the ``lepinet-app`` loader reads — so the folder drops straight into the app (``names.json`` /
    ``calibration.json`` / ``thresholds.json`` are referenced by convention and the app degrades
    gracefully if they are absent; add them with ``dev/`` calibration + names scripts).

    ``dynamo=False`` (default) uses the legacy exporter — reliable for this graph and what the app
    pipeline is verified against. ``check`` runs a PyTorch-vs-ONNX-Runtime parity assertion.
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, meta = load_model(checkpoint, img_size=img_size)
    levels = meta["levels"]
    names = list(level_names) if level_names is not None else friendly_level_names(levels)
    n_classes = [len(meta["vocabs"][level]) for level in levels]

    n_params = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in model[1].parameters())
    print(f"Model: head={checkpoint['head']} arch={checkpoint['model_arch_name']} classes={n_classes}")
    print(f"Params: {n_params/1e6:.2f} M total, {n_head/1e6:.2f} M head ({100*n_head/n_params:.0f}%)")

    wrapper = (Fp16ExportWrapper(model) if precision == "fp16" else ExportWrapper(model)).eval()
    dummy = torch.rand(1, 3, img_size, img_size)
    onnx_path = out_dir / ("model.fp16.onnx" if precision == "fp16" else "model.onnx")
    print(f"Exporting to {onnx_path} (opset {opset}, {img_size}x{img_size}, dynamo={dynamo})...")
    output_names = [f"logits_{name}" for name in names]
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

    tax = build_taxonomy(checkpoint, meta, level_names=names)
    (out_dir / "taxonomy.json").write_text(json.dumps(tax))

    if write_config:
        config = {
            "name": bundle_name or f"{checkpoint['arch'] if 'arch' in checkpoint else checkpoint['model_arch_name']} · lepinet",
            "model": "model.onnx",
            "fallback": None,
            "taxonomy": "taxonomy.json",
            "names": "names.json",
            "calibration": "calibration.json",
            "thresholds": "thresholds.json",
            "imageSize": img_size,
            "inputName": "image",
            "outputs": dict(zip(names, output_names)),
            "gbifBase": "https://www.gbif.org/species/",
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        print(f"Wrote config.json (app bundle descriptor; outputs {config['outputs']})")

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
    print(f"Wrote taxonomy.json ({len(tax['vocabs'][names[0]])} {names[0]}) and MANIFEST.json")

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


# ---------------------------------------------------------------------------
# Quantization + one-command bundle (Phase 3: the teacher/student -> app bridge)
# ---------------------------------------------------------------------------

def quantize_dynamic_int8(onnx_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Dynamic int8 (weights-only) quantization of an ONNX graph via onnxruntime.

    ~3.9x smaller for ~-0.6 pp species macro-F1 on the cosine head (measured, ``dev/043`` /
    [[2026-07-20-lepi-app-compression]]) — the unit-norm prototypes share one dynamic range, so int8 is
    nearly free. Emits ``MatMulInteger``/``ConvInteger`` ops: fine for size + native ORT/CPU, but
    **not runnable in ORT-Web** (that needs static-QDQ, itself still unresolved in-browser — the app
    ships fp32 for now). So this is the size-reduced release variant, not yet the browser format.

    ORT's quantizer round-trips through shape-inference which trips on the exporter's ``value_info``;
    stripping it (derived data, lossless) is the known workaround.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(onnx_path)
    out_path = Path(out_path) if out_path else onnx_path.with_suffix(".int8.onnx")
    model = onnx.load(str(onnx_path))
    del model.graph.value_info[:]
    tmp = onnx_path.with_suffix(".stripped.onnx")
    onnx.save(model, str(tmp))
    try:
        quantize_dynamic(str(tmp), str(out_path), weight_type=QuantType.QInt8)
    finally:
        tmp.unlink(missing_ok=True)
    fp32_mb, int8_mb = onnx_path.stat().st_size / 1e6, out_path.stat().st_size / 1e6
    print(f"int8: {fp32_mb:.1f} MB -> {int8_mb:.1f} MB ({fp32_mb / max(int8_mb, 1e-9):.2f}x) -> {out_path.name}")
    return out_path


def make_bundle(
    checkpoint_path: str,
    out_dir: str,
    img_size: int = 256,
    quantize: bool = True,
    bundle_name: str | None = None,
    fp16: bool = True,
    publish_hf: str | None = None,
    hf_path: str | None = None,
) -> Path:
    """One command: checkpoint -> a deployable app bundle folder (Phase 3's "one button").

    Composes :func:`export_onnx` (fp32 ``model.onnx`` + ``taxonomy.json`` + ``config.json`` +
    ``MANIFEST.json``, the app-ready fp32 bundle) and, when ``quantize`` is set, adds a
    ``model.int8.onnx`` size-reduced variant. ``names.json`` / ``calibration.json`` /
    ``thresholds.json`` are data-dependent (``dev/044`` / ``dev/047``) and dropped in beside these
    when available; the app's ``config.json`` already references them and degrades if absent.
    """
    out_dir = Path(out_dir)
    onnx_path = export_onnx(checkpoint_path, str(out_dir), img_size=img_size, bundle_name=bundle_name)
    if fp16:
        # Half-precision backbone: ~1.4x smaller, ONNX-valid, and free of the integer ops ORT-Web
        # cannot run -- the leading browser-deployable format.
        export_onnx(checkpoint_path, str(out_dir), img_size=img_size, precision="fp16",
                    write_config=False, check=False)
    if quantize:
        quantize_dynamic_int8(onnx_path, out_dir / "model.int8.onnx")
    print(f"Bundle ready: {out_dir}  (files: {sorted(p.name for p in out_dir.iterdir())})")
    if publish_hf:
        publish_to_hf(out_dir, publish_hf, path_in_repo=hf_path, commit_message=bundle_name)
    return out_dir


def to_fp16_onnx(onnx_path: str | Path, out_path: str | Path | None = None,
                 keep_fp32_ops: tuple[str, ...] = ("Acos",)) -> Path:
    """Convert an fp32 ONNX graph to fp16 — the leading ORT-Web small-format candidate.

    Halves the file (~2x) with no ``ConvInteger``/``MatMulInteger`` (which ORT-Web can't run — the
    reason int8 QDQ failed in-browser, [[2026-07-20-lepi-app-compression]]). ``keep_io_types=True`` keeps
    the graph's inputs/outputs fp32 (the app feeds fp32, unchanged); internals run fp16. The cosine
    head's ``Acos`` is kept fp32 (``keep_fp32_ops``) — its domain-clamped ``acos`` is fp16-fragile
    (the same reason ``PooledHead`` runs the head in fp32). Verify top-1 parity, then browser-test.
    """
    import onnx
    from onnxconverter_common import float16

    onnx_path = Path(onnx_path)
    out_path = Path(out_path) if out_path else onnx_path.with_suffix(".fp16.onnx")
    model = onnx.load(str(onnx_path))
    block = sorted(set(keep_fp32_ops) | set(getattr(float16, "DEFAULT_OP_BLOCK_LIST", [])))
    model16 = float16.convert_float_to_float16(model, keep_io_types=True, op_block_list=block)
    onnx.save(model16, str(out_path))
    fp32_mb, fp16_mb = onnx_path.stat().st_size / 1e6, out_path.stat().st_size / 1e6
    print(f"fp16: {fp32_mb:.1f} MB -> {fp16_mb:.1f} MB ({fp32_mb / max(fp16_mb, 1e-9):.2f}x) -> {out_path.name}")
    return out_path


def publish_to_hf(bundle_dir: str | Path, repo_id: str, path_in_repo: str | None = None,
                  private: bool = False, commit_message: str | None = None) -> str:
    """Upload a bundle folder to the Hugging Face Hub and return its public base URL.

    **Why the Hub and not a GitHub release:** the app fetches the model *from a web page*, and
    release assets redirect to ``release-assets.githubusercontent.com``, which sends **no
    ``Access-Control-Allow-Origin``** — the browser blocks it (``curl`` succeeds, which is why this
    is easy to get wrong). The Hub sends CORS, versions every file, and is CDN-backed. GitHub
    releases remain useful as a human/script-facing *archive*, not as a runtime source.

    The returned URL is exactly the ``base`` the app's ``models.json`` expects: a folder whose
    ``config.json`` names the model file and its sidecars.
    """
    from huggingface_hub import HfApi

    bundle_dir = Path(bundle_dir)
    path_in_repo = path_in_repo or bundle_dir.name
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(folder_path=str(bundle_dir), path_in_repo=path_in_repo, repo_id=repo_id,
                      repo_type="model",
                      commit_message=commit_message or f"lepinet bundle: {path_in_repo}")
    base = f"https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}/"
    print(f"Published -> {base}\n"
          f'Add to the app\'s models.json:  {{"id": "{path_in_repo}", "base": "{base}", ...}}')
    return base
