"""End-to-end CPU test: train (tiny) -> checkpoint -> evaluate -> predict.

Gated: needs the ``data/small`` images and is slow, so it only runs when ``LEPINET_RUN_SLOW=1``.
On CI (no data, no GPU) it is skipped. Locally::

    LEPINET_RUN_SLOW=1 CUDA_VISIBLE_DEVICES= pytest tests/test_e2e_cpu.py -v
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LEPINET_RUN_SLOW") != "1" or not Path("data/small/images").is_dir(),
    reason="set LEPINET_RUN_SLOW=1 and provide data/small/images to run the CPU e2e test",
)

SMALL_PARQUET = "data/small/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.raw_subset.parquet"


def _tiny_parquet(tmp_path):
    import pandas as pd

    df = pd.read_parquet(SMALL_PARQUET)
    vc = df["speciesKey"].value_counts()
    keep = vc[vc >= 60].index[:5]
    sub = df[df["speciesKey"].isin(keep)].groupby("speciesKey", group_keys=False).head(30).reset_index(drop=True)
    p = tmp_path / "tiny.parquet"
    sub.to_parquet(p, index=False)
    return p


def test_train_eval_predict_export(tmp_path):
    import yaml

    from lepinet.export import export_onnx
    from lepinet.infer import predict
    from lepinet.test import evaluate
    from lepinet.train import train_from_config

    parquet = _tiny_parquet(tmp_path)
    cfg = {
        "version": 1.0, "desc": "pytest-e2e",
        "train": {
            "parquet_path": str(parquet), "img_dir": "data/small/images",
            "out_dir": str(tmp_path / "models"), "fold": "1", "min_img_per_spc": 0,
            "model_name": "e2e", "model_arch_name": "resnet18", "head": "independent",
            "nb_epochs": 1, "base_lr": 1e-3, "batch_size": 16, "aug_img_size": 128, "img_size": 96,
            "optimizer": "muon", "fp16": False, "precision": "bf16", "schedule": "one_cycle",
            "warmup_epochs": 0.3, "grad_clip": 5.0, "oversample_power": 0.5, "num_workers": 2,
            "aug_kwargs": {"max_warp": 0.0, "max_lighting": 0.0, "flip_vert": True},
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    run_dir = train_from_config(str(cfg_path))
    ckpt = Path(run_dir) / "e2e.pt"
    assert ckpt.exists()

    out = evaluate(model_path=str(ckpt), parquet_path=str(parquet), img_dir="data/small/images",
                   out_dir=str(tmp_path / "preds"), eval_name="e2e", test_set="2",
                   batch_size=16, img_size=96, aug_img_size=128, num_workers=2)
    assert (out / "predictions.csv").exists() and (out / "metrics.json").exists()

    # predict on a couple of images from the training set
    imgs = sorted(Path("data/small/images").rglob("*.jpg"))[:2]
    preds = predict(str(ckpt), [str(i) for i in imgs], img_size=96, tta=True, topk=3)
    assert len(preds) == 2 and preds[0].levels[0].labels

    onnx_path = export_onnx(str(ckpt), str(tmp_path / "onnx"), img_size=96, opset=17, check=True, dynamo=False)
    assert onnx_path.exists()
