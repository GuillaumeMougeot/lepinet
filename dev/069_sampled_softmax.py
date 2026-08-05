"""Sampled-softmax training: does the ArcFace margin survive when most negatives are missing?

The question that decides whether a 1 M-class head is trainable at all
(`journal/2026-08-05-scaling-the-head.md`). At 1 M classes the prototype matrix is 5.1 GB plus
10.2 GB of Adam state; the practical fix is to keep it in CPU RAM and gather only the sampled rows
each step (21 MB at 4096 negatives). That works *if* accuracy survives partial negatives — and the
ArcFace margin is precisely the part that might not, because a margin acts against the **hardest**
negatives and uniform sampling misses them.

Testable now, and the answer transfers **pessimistically**: 1024 negatives of 12,041 is 8.5 %
coverage, while 1024 of 1 M is 0.1 %. Failure here means failure there; success makes the 1 M case
worth building properly.

**The sampled set always contains the batch's own classes.** With batch 64 over 12,041 species those
are ~64 distinct labels that the model is actively confusing right now — the cheap half of
hard-negative mining, and the half that matters most. Uniform draws fill the rest.

Implemented as a `Callback` so the package's default recipe cannot drift: it masks the non-sampled
logits to a large negative *after* the forward and before the loss, which is equivalent to computing
the softmax over the sampled subset only, but keeps the head untouched and the eval path exact
(masking is training-only).

    python dev/069_sampled_softmax.py train configs/<cfg>.yaml --n-negatives 1024
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from fastai.vision.all import Callback

MASK = -1e4  # finite, so bf16 stays well-behaved; -inf can produce NaNs through the margin path


class SampledSoftmax(Callback):
    """Restrict each step's softmax to ``n_negatives`` sampled classes plus the batch's own.

    ``order`` is after MixUp-style callbacks and before the loss. Validation is untouched, so the
    reported metric is always the full 12,041-way one and the arms stay comparable.
    """

    order = 70

    def __init__(self, n_negatives: int, n_classes: int, seed: int = 0):
        self.n_negatives = int(n_negatives)
        self.n_classes = int(n_classes)
        self.g = torch.Generator().manual_seed(seed)
        self._logged = False

    def after_pred(self):
        if not self.training or self.n_negatives >= self.n_classes:
            return
        pred = self.learn.pred
        pred = list(pred) if isinstance(pred, (list, tuple)) else [pred]
        y = self.learn.yb
        y0 = (y[0] if isinstance(y, (list, tuple)) else y)
        y0 = y0[:, 0] if y0.ndim > 1 else y0

        keep = torch.zeros(self.n_classes, dtype=torch.bool, device=pred[0].device)
        idx = torch.randint(self.n_classes, (self.n_negatives,), generator=self.g)
        keep[idx.to(keep.device)] = True
        keep[y0] = True                     # the batch's own classes are always in play
        if not self._logged:
            print(f"SampledSoftmax ON: {self.n_negatives} sampled + {int(y0.unique().numel())} "
                  f"in-batch of {self.n_classes} classes "
                  f"({100 * keep.float().mean():.1f}% per step)")
            self._logged = True

        # Only level 0 is masked. Coarse levels, where they exist, have few classes and no cost.
        pred[0] = pred[0].masked_fill(~keep.unsqueeze(0), MASK)
        self.learn.pred = pred


def main(argv):
    if len(argv) < 2 or argv[0] != "train":
        raise SystemExit(__doc__)
    cfg_path = argv[1]
    n_neg = int(argv[argv.index("--n-negatives") + 1]) if "--n-negatives" in argv else 1024

    from lepinet.config import load_config
    from lepinet.data import gen_df
    from lepinet.train import train_from_config

    cfg, _ = load_config(cfg_path)
    levels = list(cfg.levels)
    hierarchy = (Path(cfg.hierarchy_path) if cfg.hierarchy_path
                 else Path(cfg.parquet_path).parent / "hierarchy.csv")
    df, _ = gen_df(cfg.parquet_path, Path(cfg.out_dir), cfg.min_img_per_spc, cfg.fold,
                   hierarchy, cfg.family_filter, levels=levels)
    n_classes = df[levels[0]].nunique()
    print(f"{n_classes} classes; sampling {n_neg} negatives per step "
          f"({100 * n_neg / n_classes:.1f}% coverage)")
    train_from_config(cfg_path, extra_cbs=[SampledSoftmax(n_neg, n_classes)])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(sys.argv[1:])
