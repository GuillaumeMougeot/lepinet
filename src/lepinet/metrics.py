"""Per-level training metrics, generic in the number of hierarchy levels.

``LevelMacroF1`` reproduces ``mini_metrics``' macro-F1 exactly (verified like-for-like in
``journal/2026-07-why-was-fastai-behind-mini-trainer.md``), which is why the package can drop the
``mini_metrics`` dependency and still report the metric the project is scored on. All three are
streaming (accumulate TP/FP/FN), so they cost one vector per class and run at global scale.
"""
from __future__ import annotations

import torch
from fastai.metrics import Metric
from torch import nn


def level_pred_targ(learn, level_idx: int):
    """``(pred, target)`` for one level, robust to fastai's single-vs-multi-target shapes.

    With several levels fastai hands the metric a *tuple* of per-level targets, so ``learn.y[i]``
    picks a level. With a **single** level it hands over a bare tensor, and ``learn.y[0]`` would
    silently index the first *row* instead — yielding a 0-dim tensor and an ``IndexError`` deep in
    the Recorder. Normalising here keeps the metrics genuinely N-level generic (N=1 included), which
    matters for the flat-vs-multi-head comparison.
    """
    y = learn.y
    targ = y[level_idx] if isinstance(y, (tuple, list)) else y
    pred = learn.pred[level_idx] if isinstance(learn.pred, (tuple, list)) else learn.pred
    return pred, targ


class LevelAccuracy(Metric):
    """Top-1 accuracy for a single head, so per-level difficulty stays visible."""

    def __init__(self, level_idx: int, name: str):
        self.level_idx, self._name = level_idx, name
        self.reset()

    def reset(self):
        self.correct = self.total = 0

    def accumulate(self, learn):
        preds, targs = level_pred_targ(learn, self.level_idx)
        self.correct += (preds.argmax(dim=1) == targs).sum().item()
        self.total += targs.shape[0]

    @property
    def value(self):
        return self.correct / self.total if self.total else None

    @property
    def name(self):
        return self._name


class LevelMacroF1(Metric):
    """Per-level macro-F1 matching ``mini_metrics``' ``MacroF1``.

    Per class: precision (over rows predicted as that class), recall (over rows whose true label
    is that class), F1 = harmonic mean (0 when either is 0, e.g. a never-predicted class), then an
    unweighted mean over the classes present as a ground-truth label. Argmax, no threshold
    (training has none). This is the species number the project targets.
    """

    def __init__(self, level_idx: int, name: str):
        self.level_idx, self._name = level_idx, name
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = None

    def accumulate(self, learn):
        p, y = level_pred_targ(learn, self.level_idx)
        n = p.shape[1]
        if self.tp is None:
            self.tp, self.fp, self.fn = (p.new_zeros(n) for _ in range(3))
        pred_oh = nn.functional.one_hot(p.argmax(dim=1), n).bool()
        true_oh = nn.functional.one_hot(y, n).bool()
        self.tp += (pred_oh & true_oh).sum(dim=0)
        self.fp += (pred_oh & ~true_oh).sum(dim=0)
        self.fn += (~pred_oh & true_oh).sum(dim=0)

    @property
    def value(self):
        if self.tp is None:
            return None
        precision = self.tp / (self.tp + self.fp)  # 0/0 -> nan for never-predicted classes
        recall = self.tp / (self.tp + self.fn)
        denom = precision + recall
        f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom))
        present = (self.tp + self.fn) > 0
        return f1[present].mean().item() if present.any() else None

    @property
    def name(self):
        return self._name


class StreamingF1MultiHead(Metric):
    """Streaming macro/micro F1 averaged across all heads (a single summary number)."""

    def __init__(self, average: str = "macro", name: str | None = None):
        assert average in {"macro", "micro"}
        self.average = average
        self._name = name or f"F1_{average}_multihead"
        self.reset()

    def reset(self):
        self.tp, self.fp, self.fn = {}, {}, {}

    def accumulate(self, learn):
        for h, (p, y) in enumerate(zip(learn.pred, learn.y)):
            n_classes = p.shape[1]
            if h not in self.tp:
                self.tp[h] = p.new_zeros(n_classes)
                self.fp[h] = p.new_zeros(n_classes)
                self.fn[h] = p.new_zeros(n_classes)
            pred_oh = nn.functional.one_hot(p.argmax(dim=1), n_classes).bool()
            true_oh = nn.functional.one_hot(y, n_classes).bool()
            self.tp[h] += (pred_oh & true_oh).sum(dim=0)
            self.fp[h] += (pred_oh & ~true_oh).sum(dim=0)
            self.fn[h] += (~pred_oh & true_oh).sum(dim=0)

    @property
    def value(self):
        eps = 1e-8
        if self.average == "macro":
            f1s = []
            for h in self.tp:
                precision = self.tp[h] / (self.tp[h] + self.fp[h] + eps)
                recall = self.tp[h] / (self.tp[h] + self.fn[h] + eps)
                f1s.append((2 * precision * recall / (precision + recall + eps)).mean())
            return sum(f1s).item() / len(f1s)
        tp, fp, fn = (sum(v.sum() for v in d.values()) for d in (self.tp, self.fp, self.fn))
        precision, recall = tp / (tp + fp + eps), tp / (tp + fn + eps)
        return (2 * precision * recall / (precision + recall + eps)).item()

    @property
    def name(self):
        return self._name


def default_metrics(levels):
    """The standard per-level metric set (accuracy + macro-F1) plus the two multi-head summaries."""
    return [
        *(LevelAccuracy(i, f"acc_{level}") for i, level in enumerate(levels)),
        *(LevelMacroF1(i, f"f1_{level}") for i, level in enumerate(levels)),
        StreamingF1MultiHead(average="macro", name="F1(macro)"),
        StreamingF1MultiHead(average="micro", name="F1(micro)"),
    ]
