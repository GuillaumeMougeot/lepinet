"""Loss / metrics / config unit tests. Synthetic — no dataset or GPU."""
import numpy as np
import pytest
import torch

from lepinet.config import TrainConfig
from lepinet.loss import FastaiLossWrapper, MultiLevelCELoss
from lepinet.test import macro_f1


def test_multilevel_loss_scalar_and_positive():
    crit = MultiLevelCELoss([6, 3, 2])
    preds = [torch.randn(4, 6), torch.randn(4, 3), torch.randn(4, 2)]
    targets = torch.stack([torch.randint(0, 6, (4,)), torch.randint(0, 3, (4,)), torch.randint(0, 2, (4,))], dim=1)
    loss = crit(preds, targets)
    assert loss.ndim == 0 and loss.item() > 0
    assert len(crit.per_level(preds, targets)) == 3


def test_label_smoothing_decreases_with_depth():
    crit = MultiLevelCELoss([6, 3, 2], label_smoothing=0.1)
    ls = [fn.label_smoothing for fn in crit._loss_fns]
    assert ls[0] > ls[1] > ls[2]  # coarser levels smooth less


def test_fastai_loss_wrapper_stacks_targets():
    crit = MultiLevelCELoss([6, 3, 2])
    wrap = FastaiLossWrapper(crit)
    preds = [torch.randn(4, 6), torch.randn(4, 3), torch.randn(4, 2)]
    yb = (torch.randint(0, 6, (4,)), torch.randint(0, 3, (4,)), torch.randint(0, 2, (4,)))
    assert wrap(preds, *yb).ndim == 0


def test_macro_f1_perfect_and_zero():
    classes = ["a", "b", "c"]
    true = np.array(["a", "b", "c", "a"])
    assert macro_f1(true.copy(), true, classes) == pytest.approx(1.0)
    wrong = np.array(["b", "c", "a", "b"])  # never correct
    assert macro_f1(wrong, true, classes) == pytest.approx(0.0)


def test_macro_f1_only_present_classes_counted():
    # class 'c' never appears as truth -> excluded from the average
    classes = ["a", "b", "c"]
    true = np.array(["a", "a", "b", "b"])
    pred = np.array(["a", "a", "b", "b"])
    assert macro_f1(pred, true, classes) == pytest.approx(1.0)


def test_config_rejects_dead_interventions():
    base = dict(parquet_path="x", img_dir="x", out_dir="x", model_name="m", model_arch_name="resnet18")
    with pytest.raises(ValueError, match="logit"):
        TrainConfig.from_dict({**base, "logit_adjust_tau": 1.0})
    with pytest.raises(ValueError, match="class-distribution"):
        TrainConfig.from_dict({**base, "class_reg_strength": 0.001})
    # zero values are fine (existing configs carry them)
    cfg = TrainConfig.from_dict({**base, "logit_adjust_tau": 0.0, "class_reg_strength": 0.0})
    assert cfg.head == "independent"


def test_config_muon_requires_unfrozen_schedule():
    base = dict(parquet_path="x", img_dir="x", out_dir="x", model_name="m", model_arch_name="resnet18")
    with pytest.raises(ValueError, match="unfrozen"):
        TrainConfig.from_dict({**base, "optimizer": "muon", "schedule": "fine_tune"})


def test_config_defaults_bf16():
    cfg = TrainConfig(parquet_path="x", img_dir="x", out_dir="x", model_name="m", model_arch_name="resnet18")
    assert cfg.precision == "bf16"
    assert cfg.levels == ["speciesKey", "genusKey", "familyKey"]
