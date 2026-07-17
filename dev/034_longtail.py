"""Long-tail (class-imbalance) helpers for the hierarchical trainer (dev/030, dev/028).

macro-F1 weights every species equally, so it is dominated by the tail: on the global set
~28% of trained species have <100 images and ~17% have <75 (median is 200). Those tail
species both learn less (few gradient updates land on them) and, as the epoch budget grows,
are the ones most at risk of memorisation (a 75-image species at 10 epochs sees the same 75
images 10x). These two orthogonal, individually-disableable interventions target that tail;
both are standard long-tail methods and both are opt-in (a zero value = off), so the default
behaviour of dev/030 is unchanged.

  1. Class-balanced resampling (`sample_weights` -> fastai `WeightedDL`): oversample rare
     species during training so the model sees them more often. fastai's WeightedDL samples
     n_train items *with replacement* each epoch, so epoch length -- and therefore LR-schedule
     timing -- is preserved (unlike physically duplicating rows). `power` interpolates the
     per-CLASS sampling probability: a class with n images gets total weight n**(1-power), so
       power=0   -> natural distribution (off),
       power=0.5 -> square-root sampling (Mahajan et al. 2018; the usual safe sweet spot),
       power=1   -> fully class-balanced (each species equally likely; can over-fit tiny ones).

  2. Logit adjustment (`LogitAdjustment`, Menon et al. 2021): add `tau * log(prior)` to the
     logits *during training only* (never at inference), per level. This shifts decision
     boundaries toward rare classes and is Fisher-consistent for the balanced (macro) error at
     tau=1. Orthogonal to (1): one reweights which data is seen, the other reshapes the loss.

Both compute their statistics from the TRAINING split only (rows with is_valid == False), so
no validation/test information leaks into training.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Class-balanced resampling
# ---------------------------------------------------------------------------

def _train_counts(df, level, train_mask):
    """Per-value image counts over the training split only."""
    return df.loc[train_mask, level].map(df.loc[train_mask, level].value_counts()).to_numpy()


def sample_weights(df, level="speciesKey", power=0.5):
    """Per-training-row sampling weights for a fastai `WeightedDL`, aligned to the training
    subset in df order (the order fastai's `ColSplitter` yields, i.e. the is_valid==False rows
    top-to-bottom), so it can be passed straight through as `dl_kwargs=({'wgts': w}, {})`.

    Per-row weight = count(row's class)**(-power); a class with n rows then carries total
    weight n**(1-power). Returns None when power == 0 (caller then skips weighting entirely).
    """
    if not power:
        return None
    train_mask = ~df["is_valid"].to_numpy()
    counts = _train_counts(df, level, train_mask).astype(np.float64)
    return counts ** (-float(power))


# ---------------------------------------------------------------------------
# 2. Logit adjustment
# ---------------------------------------------------------------------------

def logit_adjustments(df, vocabs, levels, tau=1.0, device="cpu"):
    """Per-level additive logit-adjustment tensors `tau * log(prior)`, index-aligned to each
    level's `vocab` (so `logits[i] + adj[i]` lines up with the head's class order).

    prior_c = count_c / sum(counts), computed on the training split only; classes with zero
    training rows (shouldn't happen after min_img_per_spc filtering, but guarded) get a floor
    of 1 count so log is finite. Returns None when tau == 0.
    """
    if not tau:
        return None
    train_mask = ~df["is_valid"].to_numpy()
    adjustments = []
    for level in levels:
        vc = df.loc[train_mask, level].value_counts()
        counts = np.array([max(int(vc.get(v, 0)), 1) for v in vocabs[level]], dtype=np.float64)
        log_prior = np.log(counts / counts.sum())
        adjustments.append(torch.tensor(float(tau) * log_prior, dtype=torch.float32, device=device))
    return adjustments


class LogitAdjustment:
    """Wraps a multi-level criterion to add per-level `tau*log(prior)` to the logits at train
    time only. Deliberately not an `nn.Module` (same reason as dev/030's `MultiLevelLossWrapper`:
    the underlying criterion isn't a well-formed module). `training` is toggled by the callback
    below so inference uses raw, unadjusted logits (Menon et al.: adjust while training, predict
    plain)."""

    def __init__(self, adjustments):
        self.adjustments = adjustments  # list of per-level tensors
        self.training = True

    def __call__(self, preds):
        if not self.training:
            return preds
        return [p + adj.to(p.device) for p, adj in zip(preds, self.adjustments)]
