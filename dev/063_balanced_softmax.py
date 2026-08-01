"""Balanced Softmax as a callback, for the long-tail 2x2 (`journal/2026-08-01-imbalance-methods-bench.md`).

**This is logit adjustment at tau = 1, and this project already ran it.** Balanced Softmax
(Ren et al. 2020) trains with ``-log[ n_y e^{z_y} / sum_j n_j e^{z_j} ]``; expanding,
``log n_j = log pi_j + log N`` and the constant cancels inside the softmax, so it is exactly Menon
et al.'s ``z_j + tau*log pi_j`` at ``tau = 1``. `dev/034` implemented that and it scored 0.9031
against oversampling's 0.9148.

It is worth re-running anyway, because the *reason* it lost has been removed. The diagnosis was one
shared tau across three distributions of very different size (12,041 / 4,333 / 102), wrong for two of
the three. The current architecture supervises **one** distribution and derives the coarse levels as
marginals, so there is no shared constant left to get wrong.

Implemented as a fastai `Callback` rather than a config flag on purpose: `lepinet.config` explicitly
*rejects* ``logit_adjust_tau`` and points at `dev/`, because the option lost once and the package
should not carry a knob that reproduces a known-worse recipe by accident. This reaches training
through ``train(..., extra_cbs=[...])``.

Adjustment happens in ``after_pred``, between the forward pass and the loss, and only while
training — inference sees plain logits, which is Menon's prescription and what makes the exported
graph unchanged.

    python dev/063_balanced_softmax.py train configs/<cfg>.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from fastai.vision.all import Callback


class BalancedSoftmax(Callback):
    """Add ``tau * log(class prior)`` to the finest level's logits during training only.

    Only level 0 is adjusted. The coarse levels are *marginals* of it, so adjusting them separately
    would both double-count and reintroduce the many-constants problem that sank this method the
    first time. For a head that emits log-probabilities (``marginal``), adjusting post-normalisation
    would also break the marginal identity — so this callback refuses to run on those heads.
    """

    order = 60  # after MixUp-style callbacks, before the loss

    def __init__(self, counts, tau: float = 1.0, log_prior=None):
        self.tau = float(tau)
        if log_prior is None:
            c = torch.as_tensor(counts, dtype=torch.float32).clamp_min(1.0)
            log_prior = (c / c.sum()).log()
        self.log_prior = log_prior
        self._warned = False

    def after_pred(self):
        if not self.training:
            return
        pred = self.learn.pred
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        adj = self.log_prior.to(pred[0].device, pred[0].dtype) * self.tau
        if adj.numel() != pred[0].shape[1]:
            raise RuntimeError(
                f"BalancedSoftmax prior has {adj.numel()} classes but level 0 has "
                f"{pred[0].shape[1]}. The prior must be built from the same filtered dataframe the "
                f"loader uses (min_img_per_spc / family_filter / fold all change it)."
            )
        self.learn.pred = [pred[0] + adj, *pred[1:]]


def class_counts_from_config(cfg) -> torch.Tensor:
    """Training-fold counts per class at the finest level, ordered by the model's vocabulary.

    Rebuilt from the same `prepare_df` the loader uses, so the ordering matches the head's outputs.
    Getting this wrong silently trains against a permuted prior, which would look like a mildly bad
    run rather than a bug -- hence the shape assertion in the callback.
    """
    from pathlib import Path

    from lepinet import data as data_mod

    # Deliberately the same `gen_df` call `lepinet.train.train` makes, and the vocabulary built the
    # same way (`sorted(unique)`), because the prior must be indexed exactly like the head's outputs.
    # A permuted prior would train against the wrong class frequencies and look like a mildly bad run
    # rather than a bug -- which is why the callback also asserts the length.
    levels = list(cfg.levels)
    hierarchy_path = (Path(cfg.hierarchy_path) if cfg.hierarchy_path
                      else Path(cfg.parquet_path).parent / "hierarchy.csv")
    df, _ = data_mod.gen_df(cfg.parquet_path, Path(cfg.out_dir), cfg.min_img_per_spc, cfg.fold,
                            hierarchy_path, cfg.family_filter, levels=levels)
    vocab = sorted(df[levels[0]].unique().tolist())
    counts = df[~df["is_valid"]][levels[0]].value_counts()
    return torch.tensor([float(counts.get(v, 0)) for v in vocab])


def main(argv):
    if len(argv) < 2 or argv[0] != "train":
        raise SystemExit(__doc__)
    from lepinet.config import load_config
    from lepinet.train import train_from_config

    cfg, _ = load_config(argv[1])
    if cfg.head in {"marginal", "marginal_arcface"}:
        raise SystemExit(
            f"head={cfg.head!r} emits log-probabilities and derives coarse levels inside forward; "
            "adding a prior afterwards would break the marginal identity. Use head=independent."
        )
    counts = class_counts_from_config(cfg)
    tau = float(getattr(cfg, "_bs_tau", 1.0))
    print(f"BalancedSoftmax ON (tau={tau}, {int((counts > 0).sum())} classes, "
          f"counts {int(counts.min())}..{int(counts.max())})")
    train_from_config(argv[1], extra_cbs=[BalancedSoftmax(counts, tau=tau)])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(sys.argv[1:])
