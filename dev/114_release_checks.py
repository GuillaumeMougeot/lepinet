"""Checks on a ToL release built by mini_trainer's `mt_release` that the generic `verify` does not make (CPU).

Runs with any venv that has torch, timm, onnxruntime and safetensors, and mini_trainer (branch
feature/export-tol) on PYTHONPATH; the export agent's run used a scratch venv, see the journal entry.

    python dev/114_release_checks.py crosscheck --release R     # predict.py against mt_hpredict's GPU argmax (emb_known)
    python dev/114_release_checks.py head --release R --weights W     # numpy head vs mt's classifier; Lepidoptera rows vs class mask
    python dev/114_release_checks.py fp16head --release R       # the head's prototypes stored in fp16: agreement
    python dev/114_release_checks.py transformers --release R   # trunk.safetensors into transformers' DINOv3ConvNextModel
    python dev/114_release_checks.py tl4-images                 # 200 test-fold moths + 100 trap crops -> data/release_checks/tl4_images.txt

Results in journal/engineering/2026-11-19-can-the-tree-of-life-model-ship-as-a-trunk-and-heads-in-a-few-commands.md.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOL = ROOT / "data/population_probes/tol_e1"
OUT = ROOT / "data/release_checks"


def _sample(n=300):
    """The 300 known-species test images `mt_release export --images emb_known/paths.json` verified on."""
    from mini_trainer.release.cli import image_list

    return image_list(str(TOL / "emb_known/paths.json"), n)


def _features(release, paths, threads=4):
    """The release's trunk.onnx features for `paths` (cached in data/release_checks), and the release."""
    from mini_trainer.release import standalone

    r = standalone.Release(release, threads=threads)
    cache = OUT / f"tol_features_{len(paths)}.npy"
    if cache.exists():
        return r, np.load(cache)
    out = [r.embed(paths[i : i + 16]) for i in range(0, len(paths), 16)]
    features = np.concatenate([f for f, _ in out])
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(cache, features)
    return r, features


def crosscheck(a):
    """Does predict.py, on CPU from the release, give the classes mini_trainer's own predict gave on GPU (2026-11-17)?"""
    from mini_trainer.release import standalone

    paths = _sample()
    index = {p: i for i, p in enumerate(json.loads((TOL / "emb_known/paths.json").read_text()))}
    rows = [index[p] for p in paths]
    gpu = np.load(TOL / "emb_known/argmax.npy")[rows]
    gpu_emb = np.load(TOL / "emb_known/embeddings.npy")[rows].astype(np.float32)
    release, features = _features(a.release, paths)
    embedding, ranks = standalone.score(features, release.head)
    agree = {name: float((z.argmax(1) == gpu[:, k]).mean()) for k, (name, z) in enumerate(zip(release.head.ranks, ranks))}
    cos = (embedding * gpu_emb).sum(1) / np.linalg.norm(gpu_emb, axis=1)
    res = {"images": len(paths), "top1_agreement_with_mt_hpredict_gpu": agree,
           "embedding_cosine_vs_gpu_fp16": {"min": float(cos.min()), "mean": float(cos.mean())}}
    print(json.dumps(res, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tol_crosscheck.json").write_text(json.dumps(res, indent=1))


def head(a):
    """The numpy head against mini_trainer's classifier on the same trunk features: where the logit differences sit
    (the full head), then the head restricted to its Lepidoptera rows (a species list) against set_active_features."""
    import torch

    from mini_trainer.release import standalone
    from mini_trainer.release.build import load_checkpoint, split

    paths = _sample()
    release, features = _features(a.release, paths)
    features = features[: a.n]
    head = release.head
    model, _ = load_checkpoint(a.weights)
    _, classifier = split(model)
    del model
    with torch.inference_mode():
        want = classifier(torch.from_numpy(features))[0].numpy()
    _, ranks = standalone.score(features, head)
    diff = np.abs(ranks[0] - want)
    x = standalone.embed_features(features, head) @ head.direction.T * head.magnitude  # what the clamp sees: g * cos
    near = np.abs(x) > 1 - 1e-4
    full = {"max_abs_logit_diff": float(diff.max()), "g_cos_at_max_diff": float(x.flat[diff.argmax()]),
            "share_of_pairs_within_1e-4_of_the_clamp": float(near.mean()),
            "max_abs_logit_diff_elsewhere": float(diff[~near].max()),
            "max_abs_logit_diff_at_each_image_top_class": float(diff[np.arange(len(diff)), want.argmax(1)].max())}

    rows = [i for i, label in enumerate(head.labels[0]) if label.split("|")[3] == "Lepidoptera"]
    local = standalone.subset(head, rows)
    _, ranks = standalone.score(features, local)
    classifier.set_active_features(rows)
    with torch.inference_mode():
        want = [z.numpy() for z in classifier(torch.from_numpy(features))]
    mapping = classifier._preprocess_metadata()["cls2idx"]
    res = {"images": len(features), "full_head_species": full, "lepidoptera_rows": len(rows), "ranks": {}}
    for k, (name, got, ref) in enumerate(zip(local.ranks, ranks, want)):
        ref = ref[:, [mapping[str(k)][label] for label in local.labels[k]]]
        res["ranks"][name] = {"classes": got.shape[1], "max_abs_logit_diff": float(np.abs(got - ref).max()),
                              "top1_agreement": float((got.argmax(1) == ref.argmax(1)).mean())}
    print(json.dumps(res, indent=1))
    (OUT / "tol_head.json").write_text(json.dumps(res, indent=1))


def fp16head(a):
    """Would storing the 190,199 x 1536 prototypes in fp16 (584 MB instead of 1.17 GB) change any prediction?"""
    import dataclasses

    from mini_trainer.release import standalone

    paths = _sample()
    release, features = _features(a.release, paths)
    head = release.head
    half = dataclasses.replace(head, direction=head.direction.astype(np.float16).astype(np.float32))
    _, full = standalone.score(features, head)
    _, low = standalone.score(features, half)
    res = {name: {"top1_agreement": float((x.argmax(1) == y.argmax(1)).mean()), "max_abs_logit_diff": float(np.abs(x - y).max())}
           for name, x, y in zip(head.ranks, full, low)}
    print(json.dumps(res, indent=1))
    (OUT / "tol_fp16head.json").write_text(json.dumps(res, indent=1))


def transformers(a):
    """Can the trunk be rebuilt with transformers' DINOv3ConvNextModel? Rename timm's tensors in order, compare features."""
    import timm
    import torch
    from safetensors.torch import load_file
    from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

    state = load_file(str(Path(a.release) / "trunk.safetensors"))
    tm = timm.create_model("convnext_large.dinov3_lvd1689m", pretrained=False, num_classes=0).eval()
    tm.load_state_dict(state, strict=True)
    hf = DINOv3ConvNextModel(DINOv3ConvNextConfig(depths=[3, 3, 27, 3], hidden_sizes=[192, 384, 768, 1536])).eval()
    hf_keys = [k for k, _ in hf.named_parameters()]
    timm_keys = [k for k, _ in tm.named_parameters()]
    assert [tuple(state[k].shape) for k in timm_keys] == [tuple(p.shape) for p in hf.parameters()]
    missing, unexpected = hf.load_state_dict({h: state[t] for h, t in zip(hf_keys, timm_keys)}, strict=False)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        want = tm(x)
        out = hf(pixel_values=x)
    got = out.pooler_output
    res = {"missing": missing, "unexpected": unexpected, "pooler_shape": list(got.shape),
           "max_abs_diff_vs_timm_features": float((got - want).abs().max()), "timm_feature_scale": float(want.abs().max())}
    print(json.dumps(res, indent=1))
    (OUT / "tol_transformers.json").write_text(json.dumps(res, indent=1))


def tl4_images(_):
    """TL4's verification set: 200 test-fold moth photos and 100 trap crops (the small, up-sampled ones)."""
    test = json.loads((ROOT / "data/tol_eval/index_testfold_s3000.json").read_text())["path"]
    trap = json.loads((ROOT / "data/tol_eval/index_trap.json").read_text())["path"]
    rng = random.Random(0)
    paths = rng.sample(test, 200) + rng.sample(trap, 100)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tl4_images.txt").write_text("\n".join(paths) + "\n")
    print(f"{len(paths)} paths -> {OUT / 'tl4_images.txt'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["crosscheck", "head", "fp16head", "transformers", "tl4-images"])
    p.add_argument("--release", help="an mt_release folder of tol_full")
    p.add_argument("--weights", default=str(TOL / "tol_full_last.pt"))
    p.add_argument("--n", type=int, default=64, help="head: images")
    a = p.parse_args()
    {"crosscheck": crosscheck, "head": head, "fp16head": fp16head, "transformers": transformers, "tl4-images": tl4_images}[a.step](a)


if __name__ == "__main__":
    main()
