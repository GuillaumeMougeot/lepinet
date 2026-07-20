"""Phase A2 of the lepi-app plan: does browser preprocessing agree with fastai's?

The failure this exists to catch is silent. A model trained through fastai's
`Resize(460) -> RandomResizedCropGPU(256)` validation pipeline and then fed a differently
resampled 256x256 crop in a browser still returns confident predictions -- it just returns
slightly worse ones, with nothing in the UI or the logs indicating anything is wrong. The
only way to know is to measure it against the pipeline that produced the published number.

Scope, stated precisely, because it is easy to over-claim here:

* dev/040 already established **graph** parity -- same tensor in, same logits out of ORT and
  PyTorch (max|Δ| ~2e-5). That question is closed and is not re-asked here.
* This script asks the **preprocessing** question: given the same JPEG on disk, does a
  candidate browser-style resize produce the same tensor, and the same prediction, as
  fastai's validation pipeline?
* It cannot fully answer that, and does not pretend to. Canvas `drawImage` downscaling is
  implementation-defined (Chrome, Firefox and Safari differ, and all differ from PIL). What
  this script *can* do is (a) rank Python-side candidates against the fastai reference, which
  tells the app which algorithm to aim for, and (b) emit a fixture -- real images plus the
  logits they must produce -- so `lepinet-app` can assert the same property in a real browser.
  (b) is the part that actually closes the question; this is the half that can be done here.

The fastai validation pipeline being matched (from the winning run's config: aug_img_size 460,
img_size 256, min_scale left at its 1.0 default) reduces to:
    shorter side -> 460, center-crop 460x460, resize to 256x256, ImageNet-normalize.

Usage:
    python dev/041_ort_parity.py --onnx <bundle>/model.onnx --n 200
    python dev/041_ort_parity.py --onnx ... --emit-fixture <dir>   # for the app repo
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

from PIL import Image
from fastai.vision.all import imagenet_stats

v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod032 = importlib.import_module("032_hierarchical_heads_test")

HIERARCHY_LEVELS = v4.HIERARCHY_LEVELS
LEVEL_NAMES = ["species", "genus", "family"]
MEAN = np.array(imagenet_stats[0], dtype=np.float32).reshape(3, 1, 1)
STD = np.array(imagenet_stats[1], dtype=np.float32).reshape(3, 1, 1)


# ---------------------------------------------------------------------------
# Candidate preprocessings (what the frontend might plausibly implement)
# ---------------------------------------------------------------------------

def _to_chw01(img):
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def prep_squash(img, size=256):
    """Resize straight to size x size, ignoring aspect ratio.

    The one-liner a frontend reaches for first (`drawImage(img, 0, 0, 256, 256)`), and the
    cheapest. Included because it is what will happen by default if nobody checks.
    """
    return _to_chw01(img.convert("RGB").resize((size, size), Image.BILINEAR))


def prep_short_side_crop(img, size=256):
    """Shorter side -> size, then center-crop size x size. One resample step."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    return _to_chw01(img.crop((left, top, left + size, top + size)))


def prep_two_step(img, size=256, inter=460):
    """fastai's actual validation path: shorter side -> 460, center-crop, then resize to 256.

    Two resample steps, so it is *not* equivalent to the one-step version above -- the
    intermediate 460 crop passes through a different low-pass. Whether that difference
    matters at the model's output is exactly what this script measures.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = inter / min(w, h)
    nw, nh = max(inter, round(w * scale)), max(inter, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - inter) // 2, (nh - inter) // 2
    img = img.crop((left, top, left + inter, top + inter))
    return _to_chw01(img.resize((size, size), Image.BILINEAR))


def prep_two_step_lanczos(img, size=256, inter=460):
    """As `prep_two_step`, but with a higher-quality kernel on the downscale.

    Browsers' `drawImage` downscaling is closer to a good pyramid filter than to naive
    bilinear, so this brackets the candidate space from the other side: if bilinear and
    Lanczos give the same predictions, the browser's exact kernel does not matter, which is
    the outcome that would let the app stop worrying about it.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = inter / min(w, h)
    nw, nh = max(inter, round(w * scale)), max(inter, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - inter) // 2, (nh - inter) // 2
    img = img.crop((left, top, left + inter, top + inter))
    return _to_chw01(img.resize((size, size), Image.LANCZOS))


CANDIDATES = {
    "squash_256": prep_squash,
    "short_side_crop_256": prep_short_side_crop,
    "two_step_460_256": prep_two_step,
    "two_step_lanczos": prep_two_step_lanczos,
}


# ---------------------------------------------------------------------------
# The fastai reference tensor
# ---------------------------------------------------------------------------

def fastai_reference_batch(test_df, img_dir, aug_img_size, img_size, vocabs, batch_size):
    """The exact tensor dev/032 fed the model, un-normalized back to [0,1].

    Going through `make_dls`/`test_dl` rather than reimplementing the pipeline is the whole
    point: a reimplementation would be testing this script's understanding of fastai against
    itself. The un-normalization is exact (an affine map), and is needed because the ONNX
    graph normalizes internally -- both paths must be handed pixels in the same space.
    """
    dls = v4.make_dls(test_df, vocabs, img_dir, aug_img_size, img_size, batch_size,
                      num_workers=0, lowmem=False)
    dl = dls.test_dl(test_df, num_workers=0)
    xs = []
    for batch in dl:
        xs.append(batch[0].float().cpu())
    x = torch.cat(xs)[: len(test_df)]
    mean = torch.tensor(imagenet_stats[0]).view(1, 3, 1, 1)
    std = torch.tensor(imagenet_stats[1]).view(1, 3, 1, 1)
    return (x * std + mean).numpy()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def run_session(sess, x01, batch=32):
    outs = [[] for _ in LEVEL_NAMES]
    for i in range(0, len(x01), batch):
        chunk = np.ascontiguousarray(x01[i : i + batch], dtype=np.float32)
        res = sess.run(None, {"image": chunk})
        for j in range(len(LEVEL_NAMES)):
            outs[j].append(res[j])
    return [np.concatenate(o) for o in outs]


def compare(onnx_path, parquet_path, img_dir, n, aug_img_size, img_size, test_set,
            min_img_per_spc, seed, emit_fixture=None):
    import onnxruntime as ort

    img_dir = Path(img_dir)
    print(f"Loading test fold '{test_set}' from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = v4.filter_df(df, keep_in=[test_set], min_img_per_spc=min_img_per_spc, family_filter=[])
    df["image_path"] = df["speciesKey"].astype(str) + "/" + df["filename"]

    # Only rows whose JPEG is actually on this box: the global image set here is a partial
    # mirror (12,632 species directories present), and a missing file would otherwise surface
    # as a fastai decode error halfway through the reference pass.
    present = df["image_path"].map(lambda p: (img_dir / p).exists())
    df = df[present]
    if len(df) == 0:
        raise SystemExit(f"No test images found under {img_dir}.")
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    for level in HIERARCHY_LEVELS:
        df[level] = df[level].astype(str)
    df["is_valid"] = True
    test_df = df[["image_path", "is_valid", *HIERARCHY_LEVELS]]
    print(f"Comparing on {len(test_df)} images.")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tax = json.loads((Path(onnx_path).parent / "taxonomy.json").read_text())
    vocabs = {level: tax["vocabs"][name] for name, level in zip(LEVEL_NAMES, HIERARCHY_LEVELS)}

    print("Building the fastai reference batch...")
    ref01 = fastai_reference_batch(test_df, img_dir, aug_img_size, img_size, vocabs, batch_size=32)
    ref_logits = run_session(sess, ref01)
    ref_top1 = [l.argmax(1) for l in ref_logits]

    truth = {}
    for i, (name, level) in enumerate(zip(LEVEL_NAMES, HIERARCHY_LEVELS)):
        idx = {k: j for j, k in enumerate(vocabs[level])}
        truth[name] = np.array([idx.get(k, -1) for k in test_df[level]])
    ref_acc = {name: float((ref_top1[i] == truth[name]).mean()) for i, name in enumerate(LEVEL_NAMES)}
    print(f"Reference (fastai pipeline) top-1 accuracy: "
          + "  ".join(f"{n}={ref_acc[n]:.3f}" for n in LEVEL_NAMES))

    paths = [img_dir / p for p in test_df["image_path"]]
    rows = []
    for cand_name, fn in CANDIDATES.items():
        x01 = np.stack([fn(Image.open(p), size=img_size) for p in paths])
        logits = run_session(sess, x01)
        top1 = [l.argmax(1) for l in logits]
        row = {
            "candidate": cand_name,
            "max_abs_dpixel": float(np.abs(x01 - ref01).max()),
            "mean_abs_dpixel": float(np.abs(x01 - ref01).mean()),
        }
        for i, name in enumerate(LEVEL_NAMES):
            row[f"agree_{name}"] = float((top1[i] == ref_top1[i]).mean())
        row["acc_species"] = float((top1[0] == truth["species"]).mean())
        row["dacc_species"] = row["acc_species"] - ref_acc["species"]
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\nPreprocessing candidates vs the fastai reference:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if emit_fixture:
        write_fixture(emit_fixture, paths, ref01, ref_logits, test_df, onnx_path, ref_acc)
    return table


def write_fixture(out_dir, paths, ref01, ref_logits, test_df, onnx_path, ref_acc):
    """Emit the browser-side test fixture: the images, and the logits they must reproduce.

    This is the half of the parity question that cannot be answered in Python. `lepinet-app`
    loads these JPEGs through its real canvas path in a real browser, runs the same ONNX, and
    asserts against `expected_logits`. Only the top-32 species logits per image are stored --
    the full 12,041-wide matrix would be 100x larger and the tail carries no signal anyone
    would assert on.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    k = 32
    top_idx = np.argsort(-ref_logits[0], axis=1)[:, :k]
    top_val = np.take_along_axis(ref_logits[0], top_idx, axis=1)

    manifest = []
    for i, p in enumerate(paths):
        name = f"{i:04d}_{Path(p).parent.name}_{Path(p).name}"
        (out_dir / "images" / name).write_bytes(Path(p).read_bytes())
        manifest.append({
            "file": f"images/{name}",
            "true_species_key": test_df[HIERARCHY_LEVELS[0]].iloc[i],
            "expected_top_species_idx": top_idx[i].tolist(),
            "expected_top_species_logits": [round(float(v), 4) for v in top_val[i]],
        })
    (out_dir / "fixture.json").write_text(json.dumps({
        "note": "Reference logits produced by the fastai validation pipeline "
                "(shorter side->460, center-crop, resize 256) + the exported ONNX graph. "
                "A browser implementation should reproduce the top-1 index on ~all images; "
                "logit values will differ slightly with the canvas resampling kernel.",
        "source_onnx": str(onnx_path),
        "reference_accuracy": ref_acc,
        "topk": k,
        "images": manifest,
    }, indent=2))
    print(f"\nFixture written to {out_dir} ({len(manifest)} images).")


def cli():
    p = argparse.ArgumentParser(description="Preprocessing parity: fastai vs browser-style resizes (Phase A2).")
    p.add_argument("--onnx", required=True, help="Path to model.onnx from dev/040 (taxonomy.json must sit beside it).")
    p.add_argument("--parquet", default="../data/global/models/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.lepinet.parquet")
    p.add_argument("--img-dir", default="../data/global/images")
    p.add_argument("-n", type=int, default=200, help="Number of test images (default 200).")
    p.add_argument("--aug-img-size", type=int, default=460)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--test-set", default="0")
    p.add_argument("--min-img-per-spc", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--emit-fixture", default=None, help="Directory to write the browser test fixture into.")
    a = p.parse_args()
    compare(a.onnx, a.parquet, a.img_dir, a.n, a.aug_img_size, a.img_size, a.test_set,
            a.min_img_per_spc, a.seed, emit_fixture=a.emit_fixture)


if __name__ == "__main__":
    cli()
