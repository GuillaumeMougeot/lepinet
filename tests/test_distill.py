"""Distillation loss + config-guard unit tests. Synthetic — no dataset or GPU needed.

The end-to-end distill wiring (teacher load, vocab-alignment check, callback feeding the loss) is
exercised by ``test_e2e_synthetic.py::test_synthetic_distill``.
"""
import pytest
import torch

from lepinet.config import TrainConfig
from lepinet.loss import DistillLoss, MultiLevelCELoss


def _loss(alpha=0.5, T=4.0):
    return DistillLoss(MultiLevelCELoss([10, 5, 2]), alpha=alpha, temperature=T)


def test_distill_reduces_to_hard_ce_without_teacher():
    lf = _loss(alpha=0.5)
    preds = [torch.randn(4, c) for c in (10, 5, 2)]
    yb = [torch.randint(0, c, (4,)) for c in (10, 5, 2)]
    hard = MultiLevelCELoss([10, 5, 2])(preds, torch.stack(yb, 1))
    assert lf.teacher_logits is None
    assert torch.allclose(lf(preds, *yb), hard)          # no teacher → pure hard CE


def test_distill_alpha_zero_ignores_teacher():
    lf = _loss(alpha=0.0)
    preds = [torch.randn(4, c) for c in (10, 5, 2)]
    yb = [torch.randint(0, c, (4,)) for c in (10, 5, 2)]
    lf.teacher_logits = [torch.randn(4, c) for c in (10, 5, 2)]
    hard = MultiLevelCELoss([10, 5, 2])(preds, torch.stack(yb, 1))
    assert torch.allclose(lf(preds, *yb), hard)          # alpha=0 → teacher term dropped


def test_distill_kd_zero_when_student_matches_teacher():
    """KL(student‖teacher)=0 when logits are identical → total collapses to (1-α)·hard."""
    lf = _loss(alpha=0.7)
    preds = [torch.randn(4, c) for c in (10, 5, 2)]
    yb = [torch.randint(0, c, (4,)) for c in (10, 5, 2)]
    lf.teacher_logits = [p.clone() for p in preds]        # teacher == student
    hard = MultiLevelCELoss([10, 5, 2])(preds, torch.stack(yb, 1))
    assert torch.allclose(lf(preds, *yb), (1 - 0.7) * hard, atol=1e-5)


def test_distill_kd_positive_when_teacher_differs():
    lf = _loss(alpha=1.0)                                 # pure KD
    preds = [torch.zeros(4, c, requires_grad=True) for c in (10, 5, 2)]
    yb = [torch.randint(0, c, (4,)) for c in (10, 5, 2)]
    lf.teacher_logits = [torch.randn(4, c) for c in (10, 5, 2)]
    loss = lf(preds, *yb)
    assert float(loss.detach()) > 0 and torch.isfinite(loss)
    loss.backward()                                       # KD term is differentiable into the student


def test_distill_config_rejects_mixup():
    with pytest.raises(ValueError, match="distill_teacher is incompatible with mixup"):
        TrainConfig(parquet_path="p", img_dir="i", out_dir="o", model_name="m",
                    model_arch_name="efficientnet_v2_s", distill_teacher="teacher.pt", mixup=0.2)
