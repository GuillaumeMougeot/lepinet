"""Phase B2 of the lepi-app plan: int8 post-training quantization, and what it costs.

The highest-yield, lowest-risk lever on this model, and the reason it is expected to be cheap
is structural rather than empirical: the head is a **cosine** classifier
(`mini_trainer.modeling.classifier.Classifier`, `normalized=True`), so every one of its 12,041
prototype rows is a unit vector. All rows therefore share an identical dynamic range, which is
the best case that exists for a per-tensor int8 scale -- unlike a normal classifier, where one
outlier row forces a coarse scale on everything else.

This script quantizes and then *measures*, at all three levels, both directly and through the
marginal path proven in dev/042, against the same held-out fold. A size win that quietly costs
species macro-F1 is not a win; the accuracy floor for the shipped app is 0.87 test species
macro-F1 (journal/2026-07-lepi-app-claude.md, §7).

Two practical notes, both learned the hard way here:

* **`value_info` must be stripped before quantizing.** ORT's quantizer round-trips the model
  through `save_and_reload_model_with_shape_infer`, which trips over the shape annotations the
  dynamo exporter leaves on the graph ("Inferred shape and existing shape differ in dimension
  0: (1280) vs (12041)") -- even though the same annotations pass `onnx.shape_inference` on
  their own, in both strict and non-strict mode. Dropping `graph.value_info` loses nothing:
  it is derived data, and inference recomputes it.
* **Dynamic, not static.** Static quantization would also quantize activations and shrink
  inference time further, but needs a calibration pass and produces a QDQ graph whose browser
  support is patchier. Dynamic quantizes weights only -- which is where all 43 M parameters
  are -- and keeps the graph shape ORT Web is happiest with. Static is a later experiment, and
  only worth it if latency rather than size turns out to bind.

Usage:
    python dev/043_quantize.py --onnx <bundle>/model.onnx --out <bundle>/model.int8.onnx -n 5000
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing

torch.multiprocessing.set_start_method("fork", force=True)  # see dev/030 for why

from sklearn.metrics import f1_score
from fastai.vision.all import imagenet_stats

v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod032 = importlib.import_module("032_hierarchical_heads_test")
mod040 = importlib.import_module("040_onnx_export")
mod042 = importlib.import_module("042_marginalize")

HIERARCHY_LEVELS = v4.HIERARCHY_LEVELS
LEVEL_NAMES = ["species", "genus", "family"]


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize(src, dst, weight_type="int8"):
    """Dynamic int8 quantization. WARNING: emits ConvInteger/MatMulInteger, which ONNX Runtime
    *Web* implements on no backend -- a model quantized this way loads fine in Python but fails
    in the browser with "Could not find an implementation for ConvInteger". Use for server-side
    size measurement only; for the app bundle use `quantize_static_qdq` below."""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src, dst = Path(src), Path(dst)
    stripped = dst.with_suffix(".stripped.onnx")
    model = onnx.load(str(src))
    del model.graph.value_info[:]  # see module docstring
    onnx.save(model, str(stripped))

    qtype = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}[weight_type]
    quantize_dynamic(str(stripped), str(dst), weight_type=qtype)
    stripped.unlink()

    before, after = mod040.total_size(src), mod040.total_size(dst)
    print(f"Quantized ({weight_type}): {before/1e6:.1f} MB -> {after/1e6:.1f} MB "
          f"({before/after:.2f}x smaller)")
    return dst, before, after


class _ImageCalibrationReader:
    """Feeds `quantize_static` a stream of preprocessed images to measure activation ranges.

    Same preprocessing as inference (dev/041: shorter-side resize + center crop, [0,1] RGB CHW),
    so the calibrated activation scales match what the model actually sees at run time.
    """

    def __init__(self, paths, img_size=256):
        self.paths = list(paths)
        self.img_size = img_size
        self._it = iter(self.paths)

    def _load(self, path):
        from PIL import Image
        S = self.img_size
        im = Image.open(path).convert("RGB")
        w, h = im.size
        sc = S / min(w, h)
        nw, nh = max(S, round(w * sc)), max(S, round(h * sc))
        im = im.resize((nw, nh), Image.BILINEAR)
        l, t = (nw - S) // 2, (nh - S) // 2
        im = im.crop((l, t, l + S, t + S))
        return (np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)[None]

    def get_next(self):
        path = next(self._it, None)
        if path is None:
            return None
        return {"image": self._load(path)}

    def rewind(self):
        self._it = iter(self.paths)


def quantize_static_qdq(src, dst, calib_paths, img_size=256):
    """Static int8 quantization in **QDQ** format -- the browser-compatible path.

    Static (activation scales precomputed from a calibration set) + QDQ format produces
    QuantizeLinear/DequantizeLinear nodes around ordinary Conv/MatMul, all of which ONNX Runtime
    Web supports -- unlike the ConvInteger that dynamic quantization emits. Numerically this is
    the same int8 weight compression measured in [[journal 2026-07-lepi-app-compression]]
    (-0.59 pp species macro-F1); only the op encoding differs, which is what makes it run in the
    browser.
    """
    import onnx
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod

    src, dst = Path(src), Path(dst)
    stripped = dst.with_suffix(".stripped.onnx")
    model = onnx.load(str(src))
    del model.graph.value_info[:]  # see module docstring
    onnx.save(model, str(stripped))

    reader = _ImageCalibrationReader(calib_paths, img_size)
    quantize_static(
        str(stripped), str(dst), reader,
        quant_format=QuantFormat.QDQ,
        # per-channel weights: the cosine prototype rows share a dynamic range, so this is
        # nearly free (see the int8 result in the journal).
        per_channel=True,
        weight_type=QuantType.QInt8,
        # Activations as uint8: ORT Web's QDQ kernels are happiest with u8 activations / s8 weights.
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    stripped.unlink()

    # Guard: assert the graph has no ConvInteger/MatMulInteger (the ops ORT Web can't run).
    qm = onnx.load(str(dst))
    bad = sorted({n.op_type for n in qm.graph.node if n.op_type in ("ConvInteger", "MatMulInteger")})
    if bad:
        raise RuntimeError(f"QDQ output still contains ORT-Web-unsupported ops: {bad}")

    before, after = mod040.total_size(src), mod040.total_size(dst)
    print(f"Static QDQ int8: {before/1e6:.1f} MB -> {after/1e6:.1f} MB ({before/after:.2f}x); "
          f"no ConvInteger/MatMulInteger (browser-safe).")
    return dst, before, after


# ---------------------------------------------------------------------------
# Streaming evaluation through the fastai validation pipeline
# ---------------------------------------------------------------------------

def build_test_df(parquet_path, img_dir, n, test_set, min_img_per_spc, vocabs, seed):
    img_dir = Path(img_dir)
    df = pd.read_parquet(parquet_path)
    df = v4.filter_df(df, keep_in=[test_set], min_img_per_spc=min_img_per_spc, family_filter=[])
    df["image_path"] = df["speciesKey"].astype(str) + "/" + df["filename"]
    df = df[df["image_path"].map(lambda p: (img_dir / p).exists())]
    df = mod032.filter_known_species(df, vocabs)
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    df = df.reset_index(drop=True)
    for level in HIERARCHY_LEVELS:
        df[level] = df[level].astype(str)
    df["is_valid"] = np.arange(len(df)) % 5 == 0  # dummy split; test_dl ignores it
    return df[["image_path", "is_valid", *HIERARCHY_LEVELS]]


def predict_onnx(onnx_path, test_df, img_dir, vocabs, sp2gn, gn2fm, n_gn, n_fm,
                 batch_size, num_workers, img_size, aug_img_size, providers=None):
    """Stream the fastai validation batches through an ONNX session.

    Streaming rather than materializing: 5,000 images at 3x256x256 float32 is ~3.9 GB, and the
    only things needed downstream are top-1 indices. The un-normalization mirrors dev/041 --
    the graph normalizes internally, so it must be handed [0,1] pixels.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
    # `Path`, not `str`: `make_dls` passes this to `ColReader(pref=...)`, which only inserts a
    # path separator when the prefix is a Path -- a string prefix silently concatenates into
    # ".../images5771877/xxx.jpg" and every image is reported missing.
    img_dir = Path(img_dir)
    dls = v4.make_dls(test_df, vocabs, img_dir, aug_img_size, img_size, batch_size,
                      num_workers, lowmem=False)
    dl = dls.test_dl(test_df, num_workers=num_workers)
    mean = torch.tensor(imagenet_stats[0]).view(1, 3, 1, 1)
    std = torch.tensor(imagenet_stats[1]).view(1, 3, 1, 1)

    sp, gn_d, fm_d, gn_m, fm_m = [], [], [], [], []
    for batch in dl:
        x01 = (batch[0].float().cpu() * std + mean).numpy()
        out = sess.run(None, {"image": np.ascontiguousarray(x01, dtype=np.float32)})
        log_sp = torch.log_softmax(torch.from_numpy(out[0]).float(), dim=1)
        log_gn = mod042.scatter_logsumexp(log_sp, sp2gn, n_gn)
        log_fm = mod042.scatter_logsumexp(log_gn, gn2fm, n_fm)
        sp.append(log_sp.argmax(1))
        gn_d.append(torch.from_numpy(out[1]).argmax(1))
        fm_d.append(torch.from_numpy(out[2]).argmax(1))
        gn_m.append(log_gn.argmax(1))
        fm_m.append(log_fm.argmax(1))
    return {k: torch.cat(v).numpy() for k, v in
            zip(["sp", "gn_direct", "fm_direct", "gn_marg", "fm_marg"], [sp, gn_d, fm_d, gn_m, fm_m])}


def score(true, pred):
    return {"acc": float((pred == true).mean()),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0))}


def run(onnx_path, out_path, parquet_path, img_dir, n, test_set, min_img_per_spc,
        batch_size, num_workers, img_size, aug_img_size, seed, weight_type, out_json):
    onnx_path = Path(onnx_path)
    tax = json.loads((onnx_path.parent / "taxonomy.json").read_text())
    vocabs = {level: tax["vocabs"][name] for name, level in zip(LEVEL_NAMES, HIERARCHY_LEVELS)}
    sp2gn = torch.tensor(tax["parents"]["species_to_genus"])
    gn2fm = torch.tensor(tax["parents"]["genus_to_family"])
    n_gn, n_fm = len(vocabs[HIERARCHY_LEVELS[1]]), len(vocabs[HIERARCHY_LEVELS[2]])

    qpath, size_before, size_after = quantize(onnx_path, out_path, weight_type)

    test_df = build_test_df(parquet_path, img_dir, n, test_set, min_img_per_spc, vocabs, seed)
    print(f"Evaluating on {len(test_df)} images.")

    idx = {level: {str(k): i for i, k in enumerate(vocabs[level])} for level in HIERARCHY_LEVELS}
    y = {name: np.array([idx[level][k] for k in test_df[level]])
         for name, level in zip(LEVEL_NAMES, HIERARCHY_LEVELS)}

    results = {"n_images": int(len(test_df)), "weight_type": weight_type,
               "size_bytes": {"fp32": size_before, "quantized": size_after},
               "variants": {}}
    preds = {}
    for tag, path in [("fp32", onnx_path), (weight_type, qpath)]:
        print(f"\nRunning {tag}...")
        p = predict_onnx(path, test_df, img_dir, vocabs, sp2gn, gn2fm, n_gn, n_fm,
                         batch_size, num_workers, img_size, aug_img_size)
        preds[tag] = p
        results["variants"][tag] = {
            "species": score(y["species"], p["sp"]),
            "genus_direct": score(y["genus"], p["gn_direct"]),
            "genus_marginal": score(y["genus"], p["gn_marg"]),
            "family_direct": score(y["family"], p["fm_direct"]),
            "family_marginal": score(y["family"], p["fm_marg"]),
        }
    results["agreement_with_fp32"] = {
        k: float((preds[weight_type][k] == preds["fp32"][k]).mean())
        for k in ["sp", "gn_direct", "fm_direct", "gn_marg", "fm_marg"]
    }

    print("\n" + "=" * 78)
    print(f"int8 quantization: {size_before/1e6:.1f} MB -> {size_after/1e6:.1f} MB "
          f"({size_before/size_after:.2f}x)   n={results['n_images']}")
    print("=" * 78)
    rows = []
    for metric in ["species", "genus_direct", "genus_marginal", "family_direct", "family_marginal"]:
        a, b = results["variants"]["fp32"][metric], results["variants"][weight_type][metric]
        rows.append({
            "metric": metric,
            "fp32_acc": a["acc"], f"{weight_type}_acc": b["acc"], "d_acc": b["acc"] - a["acc"],
            "fp32_f1": a["macro_f1"], f"{weight_type}_f1": b["macro_f1"],
            "d_f1": b["macro_f1"] - a["macro_f1"],
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ntop-1 agreement with fp32: "
          + "  ".join(f"{k}={v:.4f}" for k, v in results["agreement_with_fp32"].items()))
    print("=" * 78)

    if out_json:
        Path(out_json).write_text(json.dumps(results, indent=2))
        print(f"Wrote {out_json}")
    return results


def cli():
    p = argparse.ArgumentParser(description="int8 dynamic quantization + accuracy cost (Phase B2).")
    p.add_argument("--onnx", required=True, help="fp32 model.onnx from dev/040 (taxonomy.json beside it).")
    p.add_argument("--out", required=True, help="Destination for the quantized model.")
    p.add_argument("--parquet", default="../data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet")
    p.add_argument("--img-dir", default="../data/global/images")
    p.add_argument("-n", type=int, default=5000)
    p.add_argument("--test-set", default="0")
    # 0, matching dev/032's test path -- the one that produced the project's 0.9148 headline.
    # A non-zero value here silently restricts the test fold to well-represented species (50
    # gives 3,696 species / 484k images instead of 12,632 / 633k) and inflates every number.
    p.add_argument("--min-img-per-spc", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--aug-img-size", type=int, default=460)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-type", default="int8", choices=["int8", "uint8"])
    p.add_argument("--out-json", default=None)
    a = p.parse_args()
    run(a.onnx, a.out, a.parquet, a.img_dir, a.n, a.test_set, a.min_img_per_spc,
        a.batch_size, a.num_workers, a.img_size, a.aug_img_size, a.seed, a.weight_type, a.out_json)


if __name__ == "__main__":
    cli()
