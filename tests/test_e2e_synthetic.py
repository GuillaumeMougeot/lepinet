"""Self-contained end-to-end test on a **synthetic** dataset (no data/, no GPU).

Generates random images + a matching parquet in a tmp dir, then runs the whole pipeline:
train -> checkpoint -> evaluate -> predict (TTA) -> ONNX export. This is what CI runs to prove the
package works end to end, not just unit-tested in pieces. Needs fastai (train uses vision_learner);
skipped if fastai is unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastai")
pytest.importorskip("onnxruntime")

# Force CPU + the fork start method regardless of the runner.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _make_dataset(root: Path, n_species=6, n_per=10):
    """Write random JPEGs under ``root/images/<species>/<file>.jpg`` + a parquet describing them."""
    import pandas as pd
    from PIL import Image

    rng = np.random.default_rng(0)
    img_dir = root / "images"
    rows = []
    for s in range(n_species):
        genus = s // 2          # 2 species per genus
        family = genus // 2     # 2 genera per family (a proper species->genus->family tree)
        sp_dir = img_dir / f"{1000 + s}"
        sp_dir.mkdir(parents=True, exist_ok=True)
        for k in range(n_per):
            # give each species a distinct colour bias so there is a signal to learn
            base = rng.integers(0, 255, size=3)
            arr = np.clip(base + rng.integers(-30, 30, size=(32, 32, 3)), 0, 255).astype(np.uint8)
            fname = f"{s}_{k}.jpg"
            Image.fromarray(arr).save(sp_dir / fname, quality=90)
            rows.append({
                "speciesKey": 1000 + s, "genusKey": 100 + genus, "familyKey": 10 + family,
                "filename": fname, "set": str((k % 3) + 1),  # sets '1','2','3'
            })
    df = pd.DataFrame(rows)
    parquet = root / "meta.parquet"
    df.to_parquet(parquet, index=False)
    return parquet, img_dir


def test_synthetic_train_eval_predict_export(tmp_path):
    import yaml

    from lepinet.export import export_onnx
    from lepinet.infer import predict
    from lepinet.test import evaluate
    from lepinet.train import train_from_config

    parquet, img_dir = _make_dataset(tmp_path)
    cfg = {
        "version": 1.0, "desc": "synthetic-e2e",
        "train": {
            "parquet_path": str(parquet), "img_dir": str(img_dir),
            "out_dir": str(tmp_path / "models"), "fold": "1", "min_img_per_spc": 0,
            "model_name": "syn", "model_arch_name": "resnet18", "head": "independent",
            "nb_epochs": 1, "base_lr": 1e-3, "batch_size": 16, "aug_img_size": 48, "img_size": 32,
            "optimizer": "muon", "fp16": False, "precision": "bf16", "schedule": "one_cycle",
            "warmup_epochs": 0.3, "grad_clip": 5.0, "oversample_power": 0.5, "num_workers": 0,
            "mixup": 0.4,  # exercise multi-target MixUp through the pipeline
            "aug_kwargs": {"max_warp": 0.0, "max_lighting": 0.0, "flip_vert": True},
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_dir = train_from_config(str(cfg_path))
    ckpt = Path(run_dir) / "syn.pt"
    assert ckpt.exists(), "training did not produce a checkpoint"

    out = evaluate(model_path=str(ckpt), parquet_path=str(parquet), img_dir=str(img_dir),
                   out_dir=str(tmp_path / "preds"), eval_name="e2e", test_set="2",
                   batch_size=16, img_size=32, aug_img_size=48, num_workers=0)
    assert (out / "predictions.csv").exists()
    assert (out / "metrics.json").exists()

    imgs = sorted(img_dir.rglob("*.jpg"))[:3]
    preds = predict(str(ckpt), [str(i) for i in imgs], img_size=32, tta=True, topk=3)
    assert len(preds) == 3
    assert preds[0].levels[0].labels  # non-empty top-k

    onnx_path = export_onnx(str(ckpt), str(tmp_path / "onnx"), img_size=32, opset=17, check=True, dynamo=False)
    assert onnx_path.exists()
    assert (onnx_path.parent / "taxonomy.json").exists()
    # the app-ready bundle: config.json exists and uses friendly level names
    import json
    cfg_json = json.loads((onnx_path.parent / "config.json").read_text())
    assert cfg_json["outputs"] == {"species": "logits_species", "genus": "logits_genus", "family": "logits_family"}
    tax = json.loads((onnx_path.parent / "taxonomy.json").read_text())
    assert set(tax["vocabs"]) == {"species", "genus", "family"}


def test_synthetic_distill(tmp_path):
    """End-to-end distillation: train a teacher, then distil a student from it (full wiring)."""
    import yaml

    from lepinet.train import train, train_from_config

    parquet, img_dir = _make_dataset(tmp_path)

    def cfg_dict(name, extra):
        return {"version": 1.0, "desc": name, "train": {
            "parquet_path": str(parquet), "img_dir": str(img_dir), "out_dir": str(tmp_path / "models"),
            "fold": "1", "min_img_per_spc": 0, "model_name": name, "model_arch_name": "resnet18",
            "head": "independent", "nb_epochs": 1, "base_lr": 1e-3, "batch_size": 16,
            "aug_img_size": 48, "img_size": 32, "optimizer": "muon", "fp16": False, "precision": "bf16",
            "schedule": "one_cycle", "warmup_epochs": 0.3, "grad_clip": 5.0, "oversample_power": 0.5,
            "num_workers": 0, "aug_kwargs": {"max_warp": 0.0, "max_lighting": 0.0, "flip_vert": True}, **extra}}

    teacher_cfg = tmp_path / "teacher.yaml"
    teacher_cfg.write_text(yaml.safe_dump(cfg_dict("teacher", {})))
    teacher_run = train_from_config(str(teacher_cfg))
    teacher_ckpt = Path(teacher_run) / "teacher.pt"
    assert teacher_ckpt.exists()

    from lepinet.config import prepare_run_dir
    student_cfg = tmp_path / "student.yaml"
    student_cfg.write_text(yaml.safe_dump(cfg_dict("student", {
        "distill_teacher": str(teacher_ckpt), "distill_alpha": 0.5, "distill_temperature": 4.0})))
    cfg, _ = prepare_run_dir(str(student_cfg))
    train(cfg)                                            # runs DistillCallback + DistillLoss
    student_ckpt = Path(cfg.out_dir) / "student.pt"
    assert student_ckpt.exists(), "distillation did not produce a student checkpoint"
