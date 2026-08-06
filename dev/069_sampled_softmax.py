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


class TaxonomyNegatives:
    """Negative sampler that draws **congeners first**, then fills uniformly to the budget.

    H2 measured that uniform sampling degrades smoothly with no plateau, and attributed it to the
    margin acting mostly against classes the model would never confuse: with 12,041 species over
    4,333 genera, a uniform draw of 1024 contains a given congener ~8.5 % of the time.

    This tests that attribution directly and cheaply. If congener-first recovers most of the loss at
    the same budget, the problem is *which* negatives, and hard-negative mining (over an ANN index at
    1 M scale) is the design. If it does not, the problem is *how many*, and the whole sampling
    family is dead -- which is worth knowing before building an index.

    Built from the checkpoint's own taxonomy, so it needs no new data.
    """

    def __init__(self, parent_of: torch.Tensor, n_classes: int, seed: int = 0):
        self.n_classes = int(n_classes)
        self.parent_of = parent_of                        # [n_classes] -> parent index
        order = torch.argsort(parent_of)                  # group siblings contiguously
        self.sorted_children = order
        self.sorted_parents = parent_of[order]
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, y0: torch.Tensor, budget: int) -> torch.Tensor:
        """Indices to keep: every congener of the batch's classes, then a uniform fill."""
        parents = torch.unique(self.parent_of[y0.cpu()])
        sib_mask = torch.isin(self.sorted_parents, parents)
        congeners = self.sorted_children[sib_mask]
        if len(congeners) >= budget:                      # already over budget: subsample them
            sel = torch.randperm(len(congeners), generator=self.g)[:budget]
            return congeners[sel]
        fill = torch.randint(self.n_classes, (budget - len(congeners),), generator=self.g)
        return torch.cat([congeners, fill])


class SampledSoftmax(Callback):
    """Restrict each step's softmax to ``n_negatives`` sampled classes plus the batch's own.

    ``order`` is after MixUp-style callbacks and before the loss. Validation is untouched, so the
    reported metric is always the full 12,041-way one and the arms stay comparable.
    """

    order = 70

    def __init__(self, n_negatives: int, n_classes: int, seed: int = 0, sampler=None):
        self.n_negatives = int(n_negatives)
        self.n_classes = int(n_classes)
        self.g = torch.Generator().manual_seed(seed)
        self.sampler = sampler          # None -> uniform; else a TaxonomyNegatives
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
        idx = (self.sampler(y0, self.n_negatives) if self.sampler is not None
               else torch.randint(self.n_classes, (self.n_negatives,), generator=self.g))
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
    sampler = None
    if "--taxonomy-negatives" in argv:
        import torch as _t

        from lepinet.heads import build_class_spec
        vocabs = {lv: sorted(df[lv].unique().tolist()) for lv in levels}
        _c2i, masks = build_class_spec(df, vocabs, levels)
        sampler = TaxonomyNegatives(_t.as_tensor(masks[0]), n_classes)
        print("negatives: CONGENERS FIRST, then uniform fill")
    print(f"{n_classes} classes; sampling {n_neg} negatives per step "
          f"({100 * n_neg / n_classes:.1f}% coverage)")
    train_from_config(cfg_path, extra_cbs=[SampledSoftmax(n_neg, n_classes, sampler=sampler)])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(sys.argv[1:])
