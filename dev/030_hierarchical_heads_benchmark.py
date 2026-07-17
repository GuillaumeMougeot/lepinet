"""Benchmark mini_trainer's hierarchical classification heads inside the fastai framework.

mini_trainer (../mini_trainer, installed editable into this venv) already benchmarks
"flat" / "hierarchical" / "conditional" / "independent" / "autoregressive" heads for
Lepidoptera hierarchical classification using its own from-scratch training loop. This
script reuses mini_trainer's actual head and loss modules verbatim -- not a fastai
reimplementation of the same idea -- and trains them through fastai's `vision_learner`/
`Learner` instead. The training loop is the only thing that differs between this script
and mini_trainer's own `mt_htrain`, which is what makes "improvement isn't framework
dependent" a claim this script can actually support.

Head architecture (see mini_trainer/hierarchical/model.py):
  - hierarchical: one linear layer (species), coarser levels obtained by summing
    (logsumexp) child probabilities up the taxonomy via sparse parent-index masks.
  - independent: one linear layer per level, trained with no interaction between levels.
  - autoregressive: one linear layer per level (like independent) plus a transformer
    decoder (mini_trainer's XADecoder) that generates family -> genus -> species
    sequentially, conditioning each level on the previously decoded ones.

Known upstream issue worked around here: `AutoregressiveClassifier(IndependentClassifier,
AutoregressiveMixin)`'s MRO resolves `.forward` to `IndependentClassifier.forward`
(bypassing the decoder entirely) because `IndependentClassifier` precedes
`AutoregressiveMixin` in the base class list, and neither overrides `forward` in a way
that fixes this. `AutoregressiveMixin.forward` is never actually invoked anywhere else in
mini_trainer either (only `hierarchical`/`conditional`/`independent` are wired into
`mt_htrain`'s CLI), so this looks like unexercised code rather than an intentional
design. Worked around locally in `MTHeadAdapter` below by calling
`AutoregressiveMixin.forward(head, x, ...)` explicitly instead of `head(x)`, without
touching mini_trainer's source.

Usage
-----
uv run python dev/030_hierarchical_heads_benchmark.py --config <config.yaml>
"""

import argparse
import importlib
from collections import OrderedDict
from os.path import exists, join
from pathlib import Path
from shutil import copyfile

import torch
import torch.multiprocessing

# fastai forces 'fork' over Python's default multiprocessing start method, but only
# on Darwin (see fastai/torch_basics.py) -- Python 3.14 switched the Linux default to
# 'forkserver', which (unlike 'fork') must fully pickle the DataLoader worker state to
# hand it to the new process. That state ends up holding a GPU-resident tensor cached
# inside aug_transforms' Flip/Warp random state (populated by vision_learner's internal
# warm-up batch, which already runs on cuda), and CUDA storage can't be pickled --
# crashing every run with num_workers > 0 as soon as training starts. Must happen before
# any CUDA context/DataLoader is created.
torch.multiprocessing.set_start_method("fork", force=True)

import yaml
from torch import nn

from fastai.callback.core import Callback, CancelBatchException, CancelFitException
from fastai.callback.schedule import ParamScheduler, SchedCos, SchedLin, SchedNo, combine_scheds
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.optimizer import OptimWrapper
from fastai.vision.all import CSVLogger, GradientClip, L, SaveModelCallback, vision_learner

from mini_trainer.hierarchical.autoregressive import AutoregressiveMixin
from mini_trainer.hierarchical.integration import sparse_masks_from_labels
from mini_trainer.hierarchical.loss import MultiLevelWeightedCrossEntropyLoss
from mini_trainer.hierarchical.model import HierarchicalClassifier, IndependentClassifier, AutoregressiveClassifier
from mini_trainer.hierarchical.transformer import XADecoder
from mini_trainer.modeling import SupervisionContext
from mini_trainer.training.loss import class_weight_distribution_regularization
from mini_trainer.training.muon import MuonAuxAdamW

v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
longtail = importlib.import_module("034_longtail")

VALID_CONFIG_VERSIONS = [1.0]
HIERARCHY_LEVELS = v4.HIERARCHY_LEVELS  # ["speciesKey", "genusKey", "familyKey"], finest -> coarsest

HEAD_CLASSES = {
    "hierarchical": HierarchicalClassifier,
    "independent": IndependentClassifier,
    "autoregressive": AutoregressiveClassifier,
}


def muon_opt_func(param_groups, lr, wd=0.01, **kwargs):
    """fastai `opt_func` backed by mini_trainer's MuonAuxAdamW: Muon (Newton-Schulz
    orthogonalized momentum) on 2D backbone weights, AdamW on everything else.

    fastai hands us its splitter's parameter groups (body first, head last for
    `default_split`). We name the last group `head_nomuon` so MuonAuxAdamW routes the
    whole classification head to AdamW (matching mini_trainer, which never applies Muon
    to final classification layers), and Muon handles the 2D conv/linear weights of the
    backbone groups. Wrapped in fastai's `OptimWrapper` so the fastai training loop /
    schedulers drive it normally. Assumes an unfrozen model (see `schedule="flat_cos"`
    path in `train`): MuonAuxAdamW re-partitions param groups internally, which does not
    round-trip through fastai's freeze/`freeze_to`, so this is not used with `fine_tune`."""
    groups = list(param_groups)
    named = []
    for i, g in enumerate(groups):
        params = list(g) if isinstance(g, (list, tuple, L)) else [g]
        if not params:
            continue
        name = "head_nomuon" if i == len(groups) - 1 else f"backbone{i}"
        named.append({"params": params, "name": name, "lr": lr, "weight_decay": wd})
    return OptimWrapper(opt=MuonAuxAdamW(params=named, lr=lr, weight_decay=wd))


# ---------------------------------------------------------------------------
# Taxonomy -> mini_trainer class spec (cls2idx / labels / sparse_masks)
# ---------------------------------------------------------------------------

def build_class_spec(df, vocabs):
    """Builds mini_trainer's `cls2idx`/`labels`/`sparse_masks` from the same
    per-level vocabularies used to build the fastai `CategoryBlock`s, so label
    indices agree between the DataBlock and the head's masks."""
    cls2idx = {str(i): {v: idx for idx, v in enumerate(vocabs[level])} for i, level in enumerate(HIERARCHY_LEVELS)}

    unique = df.drop_duplicates("speciesKey")
    labels = OrderedDict(
        (row.speciesKey, tuple(getattr(row, level) for level in HIERARCHY_LEVELS)) for row in unique.itertuples(index=False)
    )
    sparse_masks = sparse_masks_from_labels(labels, cls2idx)
    return cls2idx, sparse_masks


# ---------------------------------------------------------------------------
# Head adapter: pools the fastai body's feature map, then hands it to a
# mini_trainer Classifier-family head.
# ---------------------------------------------------------------------------

class MTHeadAdapter(nn.Module):
    def __init__(self, head: nn.Module, autoregressive: bool = False, eval_method: str = "beam"):
        super().__init__()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = head
        self.autoregressive = autoregressive
        self.eval_method = eval_method

    def forward(self, x):
        x = self.pool(x)
        # Run the mini_trainer cosine head in fp32 even under fp16 AMP. Its forward does
        # F.normalize(hidden(x)); as weights grow during training the hidden linear layer's
        # output overflows fp16's ~65504 range -> inf -> normalize(inf) = NaN. This is the
        # classic ArcFace / cosine-margin instability, and it's why fp16 training NaN'd at
        # scale (the plain softmax head, being BatchNorm-bounded + log_softmax, doesn't).
        # fp32 has the range to avoid the overflow; the head is tiny vs the backbone so
        # keeping the backbone in fp16 preserves almost all of the speedup. When training is
        # already fp32 this block is a no-op.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if self.autoregressive:
                # See module docstring: works around mini_trainer's AutoregressiveClassifier
                # MRO bug by calling AutoregressiveMixin.forward explicitly. It already
                # branches on `self.head.training` internally (teacher-forcing/soft mix
                # while training, `self.eval_method` search while evaluating).
                return AutoregressiveMixin.forward(self.head, x, method=self.eval_method)
            return self.head(x)


def build_head(head_name, nf, n_classes, cls2idx, sparse_masks, decoder_kwargs=None):
    cls = HEAD_CLASSES[head_name]
    kwargs = dict(in_features=nf, out_features=n_classes[0], sparse_masks=sparse_masks, cls2idx=cls2idx)
    if head_name == "autoregressive":
        kwargs.update(decoder_cls=XADecoder, decoder_kwargs=decoder_kwargs or {})
    head = cls(**kwargs)
    return MTHeadAdapter(head, autoregressive=head_name == "autoregressive")


# ---------------------------------------------------------------------------
# Loss / supervision plumbing
# ---------------------------------------------------------------------------

class MultiLevelLossWrapper:
    """`MultiLevelWeightedCrossEntropyLoss` returns a list of per-level losses;
    fastai's `Learner` expects a single scalar from `loss_func`.

    Deliberately *not* an `nn.Module`: `MultiLevelWeightedCrossEntropyLoss` extends
    `nn.Module` but never calls `super().__init__()` (it places its tensors on
    `device` itself in its own constructor instead), so its module bookkeeping
    (`_modules` etc.) is never initialized. Registering it as a submodule here
    would make fastai's `TrainEvalCallback.before_fit` crash when it calls
    `loss_func.to(device)` -- which is unnecessary anyway since the criterion is
    already on the right device.
    """

    def __init__(self, criterion, logit_adjustment=None):
        self.criterion = criterion
        # Optional dev/034.LogitAdjustment: shifts logits by tau*log(prior) at train time only.
        self.logit_adjustment = logit_adjustment

    def __call__(self, preds, *yb):
        if self.logit_adjustment is not None:
            preds = self.logit_adjustment(preds)
        targets = torch.stack(yb, dim=1)
        return sum(self.criterion(preds, targets))


class GCCallback(Callback):
    """Works around a GPU memory leak in mini_trainer's `Classifier._weight_bias()`
    (mini_trainer/modeling/classifier.py): every forward call during training
    re-assigns a *live, graph-attached* view of the weight_norm-parametrized weight
    into a persistent buffer on `self`. PyTorch's weight_norm parametrization keeps a
    back-reference to its parent module, so this forms a reference cycle rooted at the
    classifier module that traps that batch's entire backward graph (all backbone
    activations, not just the classifier weight) until Python's cyclic GC happens to
    run. Refcounting alone never frees it, and the default GC's object-count-based
    trigger doesn't fire often enough at real batch/image sizes to keep pace, so
    memory climbs every batch until CUDA OOMs. Forcing a collection after every batch
    breaks the cycle deterministically; confirmed empirically to keep GPU memory flat
    where the leak otherwise OOMs a 32GB GPU within ~60 batches. Not needed for the
    hierarchical/autoregressive heads' cache (same pattern, same fix applies).

    Deliberately `gc.collect(0)` (generation 0 only), not a full `gc.collect()`: the
    cycle is created and dies within a single batch, so it's always still in the
    youngest generation and gen-0-only collection catches it just as reliably --
    confirmed empirically (GPU memory stays flat either way). A full collection also
    scans generations 1/2, which accumulate a lot of long-lived tracked objects (model
    parameters, optimizer state, DataLoader/dataset objects) that have nothing to do
    with this leak; at this model's real class count (12041/4333/102) that made a full
    `gc.collect()` cost ~250ms per batch -- 4-5x more than the ~50ms of actual GPU
    compute per batch -- which is what caused the GPU to visibly pulse between 0% and
    100% (idling in the collector, not on the data pipeline). `gc.collect(0)` costs
    ~4ms, restoring normal utilization."""

    def after_batch(self):
        import gc

        gc.collect(0)


class HostMemoryGuard(Callback):
    """Log host RAM every `every` batches, and abort loudly before the OOM killer does.

    Written after the 2026-07-17 UCloud benchmark: three 10-epoch jobs held the full decode
    ceiling for ~36k batches, then collapsed 440x and vanished mid-batch with **no traceback,
    no error, nothing**. That silence is the point -- the kernel's OOM killer SIGKILLs the
    process, so Python never runs another line and the log simply stops. Nine GPU-hours bought
    zero information. (A CUDA OOM is the opposite: a loud `torch.cuda.OutOfMemoryError`. The
    absence of one is how you tell host RAM from GPU RAM.)

    The cause is not a bug so much as arithmetic: dev/030 forks its dataloader workers, and
    CPython's refcounting writes to object headers on read, so copy-on-write breaks page by
    page and every worker converges on a private copy of the dataframe (~1.1 GB/worker at
    global scale, measured by dev/037_dl_memory_probe.py). num_workers therefore has a hard
    memory ceiling: 512 x 1.1 GB is ~2x a 288 GB node, 256 fits with little to spare. Because
    the leak saturates gradually, this is invisible for the first half hour and then fatal.

    So: report the number while there is still a process alive to report it, and if available
    RAM crosses `abort_below_gb`, raise -- turning a silent 3-hour death into an immediate,
    explained failure that names the knob to turn.
    """

    order = -7  # before anything that might allocate

    def __init__(self, every: int = 25, abort_at_frac: float = 0.92):
        # every=25, not 500: the 2026-07-17 bench2 attempt was SIGKILLed at batch ~315, i.e.
        # before a 500-batch interval ever reported, leaving the failure as unexplained as the
        # one this guard exists to explain. Reading two cgroup files costs nothing next to a
        # batch of 64 images at 460x460; there is no reason to sample coarsely.
        self.every = every
        self.abort_at_frac = abort_at_frac

    @staticmethod
    def _cgroup_mem():
        """(used_bytes, limit_bytes) for *this container*, or None outside one.

        `psutil.virtual_memory()` reads /proc/meminfo, which reports the physical host and
        ignores the cgroup entirely: inside a UCloud job it cheerfully says 2434 GB total /
        2307 GB available while the cgroup caps the job at 288 GB and the kernel kills it
        there. Watching the host figure is watching the wrong number -- it can never trip.
        The limit that actually binds is the cgroup's, so read that; fall back to the host
        only when there is no cgroup (e.g. the local workstation).
        """
        # A container namespaces the cgroup mount, so its own limit is at the root of
        # /sys/fs/cgroup. On a plain systemd host the process instead sits in a nested slice
        # named by /proc/self/cgroup, and the root carries no limit -- try both, nearest first.
        candidates = [(Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
                      (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
                       Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))]
        try:
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                rel = line.split(":")[-1].lstrip("/")
                if rel:
                    base = Path("/sys/fs/cgroup") / rel
                    candidates.insert(0, (base / "memory.current", base / "memory.max"))
        except OSError:
            pass

        for cur, mx in candidates:
            try:
                limit = mx.read_text().strip()
                if limit == "max":
                    return None  # cgroup exists but is uncapped: host RAM is the real limit
                limit = int(limit)
                # v1 signals "no limit" with a sentinel near 2^63, not the string "max".
                if limit > (1 << 62):
                    return None
                return int(cur.read_text().strip()), limit
            except (OSError, ValueError):
                continue
        return None

    def _mem(self):
        """(used_gb, limit_gb, source) -- the cgroup's numbers when capped, else the host's."""
        cg = self._cgroup_mem()
        if cg:
            return cg[0] / 1e9, cg[1] / 1e9, "cgroup"
        import psutil
        vm = psutil.virtual_memory()
        return vm.used / 1e9, vm.total / 1e9, "host"

    def _workers(self):
        """The real worker count.

        fastai's DataLoader keeps `self.num_workers = 1` and hands the configured value to the
        torch loader it wraps (`fake_l`), so reading `dls.train.num_workers` reports 1 no
        matter what the config said.
        """
        dl = self.learn.dls.train
        return getattr(getattr(dl, "fake_l", None), "num_workers", None) or \
            getattr(dl, "num_workers", "?")

    def before_fit(self):
        used, limit, src = self._mem()
        n = self._workers()
        est = f"{n * 1.1:.0f} GB" if isinstance(n, int) else "?"
        print(f"[mem] limit {limit:.0f} GB ({src}), used {used:.0f} GB | num_workers={n} "
              f"-> forked workers converge on ~1.1 GB each at global scale, so expect ~{est} "
              f"+ parent (dev/037_dl_memory_probe.py).", flush=True)
        if isinstance(n, int) and n * 1.1 > limit * 0.9:
            print(f"[mem] WARNING: {n} workers x ~1.1 GB is close to or over the {limit:.0f} GB "
                  f"limit. This is how the 2026-07-17 benchmark died -- silently, 36 min in.",
                  flush=True)

    def after_batch(self):
        if not self.training or self.learn.train_iter % self.every:
            return
        used, limit, src = self._mem()
        gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[mem] batch {self.learn.train_iter}: {used:.0f}/{limit:.0f} GB "
              f"({used/limit*100:.0f}% of {src}) | GPU allocated {gpu:.1f} GB", flush=True)
        if used / limit > self.abort_at_frac:
            raise RuntimeError(
                f"Memory at {used:.0f}/{limit:.0f} GB ({used/limit*100:.0f}% of the {src} "
                f"limit). The OOM killer takes this process next, with no traceback -- aborting "
                f"first so the reason is on the record. num_workers={self._workers()} x ~1.1 "
                f"GB/worker is the usual cause (forked workers each copy the dataframe); lower "
                f"it. See dev/037_dl_memory_probe.py."
            )


class NaNGuard(Callback):
    """Skip any training batch whose loss is non-finite, instead of letting it poison the
    weights. This ports the safety net from mini_trainer's own loop (train_one_epoch skips
    up to a few NaN batches) into fastai's loop, which otherwise has none: a single bad batch
    -- e.g. a corrupt/degenerate image in the 6.3M-image set whose embedding normalizes to
    NaN -- backprops through one optimizer step and permanently NaNs the whole model. Since
    the NaN originates in the forward, gradient clipping can't catch it; only skipping the
    batch (leaving weights untouched) does. Runs before backward via CancelBatchException, so
    no gradient from the bad batch is ever applied. Aborts training if NaNs are persistent
    (not a transient bad batch but genuinely diverged), so it fails loudly rather than
    silently training on nothing."""

    order = -5  # before GradientClip / the optimizer step

    def __init__(self, max_consecutive: int = 10):
        self.max_consecutive = max_consecutive
        self.n_consecutive = 0
        self.n_skipped = 0

    def after_loss(self):
        if not self.training:
            return
        if not torch.isfinite(self.loss):
            self.n_consecutive += 1
            self.n_skipped += 1
            if self.n_consecutive >= self.max_consecutive:
                raise CancelFitException(
                    f"NaNGuard: {self.n_consecutive} consecutive non-finite losses -- training has diverged."
                )
            # Zero the loss before CancelBatchException: fastai's `after_batch` (smoothed-loss
            # recorder) still runs on cancel, and lerping a NaN loss permanently poisons the
            # *displayed/logged* train_loss even though the model is untouched. Zeroing keeps
            # logging finite so a NaN in the CSV unambiguously means real divergence, not a
            # single skipped bad batch. (Negligible bias since skips are rare.)
            self.learn.loss = self.learn.loss_grad = torch.zeros_like(self.loss)
            raise CancelBatchException  # skip backward + optimizer step for this batch
        self.n_consecutive = 0

    def after_fit(self):
        if getattr(self, "n_skipped", 0):
            print(f"NaNGuard: skipped {self.n_skipped} non-finite-loss batch(es) during training.")


class ClassDistributionRegularizer(Callback):
    """Ports mini_trainer's `class_weight_distribution_regularization` (training/loss.py):
    an *ongoing* loss term -- not just an init -- that keeps the species classifier's row
    vectors spread apart on the unit sphere throughout training (adds `2*mean(relu(cosine
    z-score))` over a sqrt(n_classes)-sized random subset of class pairs each step). Without
    it, gradient descent is free to let the spherical-repulsion init's separation erode as
    training reshapes the embedding; mini_trainer's own loop applies this every step by
    default (`build_regularizer`, strength 1e-3). dev/030 never enabled it -- flagged as a
    caveat in dev/031's framework report ("mini_trainer arm ran with its class-distribution
    regularizer disabled") -- so this closes that specific, previously-known gap.

    Only regularizes the *species*-level linear layer (`head.linear`), matching mini_trainer's
    own `last_layer_weights()`: for multi-layer heads (`IndependentClassifier` etc.) that
    helper resolves to `Classifier.linear` (the finest level) only, never the coarser
    `self.layers[1:]` -- so mini_trainer itself doesn't regularize genus/family either.

    Added into `loss_grad` in `after_loss` (mirrors `NaNGuard`'s pattern: fastai backprops
    whatever `self.learn.loss_grad` holds after that event, per `Learner._do_one_batch`).
    Ordered after `NaNGuard` so a NaN batch's `CancelBatchException` skips this too.
    """

    order = -4  # after NaNGuard (-5), before GradientClip

    def __init__(self, strength: float = 1e-3):
        self.strength = strength

    def after_loss(self):
        if not self.training or self.strength == 0:
            return
        head = self.learn.model[1].head  # MTHeadAdapter.head: the mini_trainer Classifier
        reg = self.strength * class_weight_distribution_regularization(head.linear.weight)
        self.learn.loss_grad = self.learn.loss_grad + reg
        self.learn.loss = self.learn.loss_grad.clone()


class LogitAdjustCallback(Callback):
    """Turns the loss's `LogitAdjustment` (dev/034) on during training and off during
    validation, so tau*log(prior) is added only while learning (Menon et al.: predict on raw
    logits). Metrics are unaffected either way -- they read `learn.pred` (the model's raw
    logits), never the loss-internal adjusted copy -- so this only keeps `valid_loss`
    comparable across configs."""

    order = -6  # before the loss is computed

    def before_batch(self):
        la = getattr(self.learn.loss_func, "logit_adjustment", None)
        if la is not None:
            la.training = self.training


class SupervisionContextCallback(Callback):
    """Exposes the batch's hierarchy labels via `SupervisionContext` for the
    duration of the forward+loss pass, as mini_trainer's own `train_one_epoch`
    does with `with SupervisionContext(target): ...`. Only the autoregressive
    head reads this (for teacher forcing); harmless no-op for the others."""

    def before_batch(self):
        SupervisionContext.set(torch.stack(self.yb, dim=1))

    def after_loss(self):
        SupervisionContext.clear()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def warmup_cos_schedule(n_epoch, lr, warmup_epochs, schedule, warmup_div=100.0, cos_pct=0.25):
    """Builds the combined lr-schedule function (a fn of `pos` in [0,1] over the *whole*
    n_epoch run) used by `fit_warmup_cos`, without calling `learn.fit`. Split out so
    `fit_resume` can rebuild the exact same curve and continue it after an interruption,
    instead of restarting the anneal from scratch."""
    warmup_pct = warmup_epochs / n_epoch
    if not 0 < warmup_pct < 1:
        raise ValueError(f"warmup_epochs ({warmup_epochs}) must be in (0, n_epoch={n_epoch}).")
    ramp = SchedLin(lr / warmup_div, lr)
    if schedule == "one_cycle":
        pcts, scheds = [warmup_pct, 1 - warmup_pct], [ramp, SchedCos(lr, 0)]
    else:  # flat_cos: ramp -> flat -> cosine (cosine over the final cos_pct of the whole run)
        flat_pct = 1 - warmup_pct - cos_pct
        if flat_pct <= 0:
            raise ValueError(f"warmup_epochs too large: no room for the flat phase before the "
                             f"final {cos_pct:.0%} cosine anneal.")
        pcts = [warmup_pct, flat_pct, cos_pct]
        scheds = [ramp, SchedNo(lr, lr), SchedCos(lr, 0)]
    return combine_scheds(pcts, scheds)


def fit_warmup_cos(learn, n_epoch, lr, warmup_epochs, schedule, warmup_div=100.0, cos_pct=0.25):
    """`fit_flat_cos`/`fit_one_cycle` with a short LR warmup ramp prepended.

    Why a *ramp* rather than mini_trainer's frozen-backbone warmup: mini_trainer freezes the
    backbone for its first ~2 epochs, but that path is closed to us under the Muon optimizer
    (`MuonAuxAdamW` re-partitions param groups in a way fastai's freeze bookkeeping can't
    survive -- see `muon_opt_func`), which is exactly the config of our best baseline. A
    linear LR ramp from `lr/warmup_div` -> `lr` over `warmup_epochs` gives most of the same
    protection (the pretrained backbone isn't hit with full-strength Muon/AdamW steps while
    the freshly-initialised head is still producing large, poorly-aimed gradients) without
    freezing anything, so it is optimizer-agnostic. It is also cheap: unlike a frozen warmup
    it does not "waste" epochs -- the backbone is still learning throughout, just gently at
    first -- so a fraction of an epoch is enough (2 epochs would be far too much at ~5M
    training images: that is ~156k steps of hobbled LR out of a 5-epoch budget).

    Schedule shape over the full run: [linear ramp for `warmup_epochs`] then, for the
    remainder, the requested base shape -- `flat_cos` = hold flat then cosine-anneal the last
    `cos_pct`; `one_cycle` here = cosine-anneal straight down (the ramp already replaced its
    warmup phase). Only `lr` is scheduled (momentum left at the optimiser default) so this
    works for the tuple-hyper Muon optimiser too.
    """
    sched = {"lr": warmup_cos_schedule(n_epoch, lr, warmup_epochs, schedule, warmup_div, cos_pct)}
    learn.fit(n_epoch, cbs=ParamScheduler(sched))


def front_loaded_schedule(n_epoch, lr, warmup_epochs, fast_decay_epochs, lr_mid_frac, warmup_div=100.0):
    """Builds the combined lr-schedule function used by `fit_front_loaded` (see `warmup_cos_schedule`
    for why this is split out of the `learn.fit` call)."""
    warmup_pct = warmup_epochs / n_epoch
    fast_end_pct = fast_decay_epochs / n_epoch
    fast_pct = fast_end_pct - warmup_pct
    slow_pct = 1 - fast_end_pct
    if not (warmup_pct > 0 and fast_pct > 0 and slow_pct > 0):
        raise ValueError(f"Need 0 < warmup_epochs ({warmup_epochs}) < fast_decay_epochs "
                         f"({fast_decay_epochs}) < n_epoch ({n_epoch}).")
    lr_mid = lr * lr_mid_frac
    pcts = [warmup_pct, fast_pct, slow_pct]
    scheds = [SchedLin(lr / warmup_div, lr), SchedCos(lr, lr_mid), SchedCos(lr_mid, 0)]
    return combine_scheds(pcts, scheds)


def fit_front_loaded(learn, n_epoch, lr, warmup_epochs, fast_decay_epochs, lr_mid_frac, warmup_div=100.0):
    """Front-loaded LR schedule: peak early, drop hard within the first epoch(s), then a long
    gentle tail to 0.

    Motivation (vs cosine/one_cycle): cosine holds the LR near its peak early and only anneals
    steeply in the middle, so the model keeps exploring at high LR and doesn't converge until
    late -- the "big jump only at the end" behaviour. This schedule instead ramps to `lr` in
    `warmup_epochs`, then cosine-decays *fast* from `lr` down to `lr*lr_mid_frac` by the end of
    `fast_decay_epochs` (so the model settles into a good basin almost immediately, given the
    pretrained backbone + spherical-repulsion head init are already a strong starting point),
    then cosine-decays gently from that mid LR to 0 over the rest of training -- pure low-LR
    refinement. The trade-off is exploration: committing this early risks a lower final plateau
    if one high-LR epoch finds a worse basin than several would. LR-only scheduling (Muon-safe).
    """
    sched = {"lr": front_loaded_schedule(n_epoch, lr, warmup_epochs, fast_decay_epochs, lr_mid_frac, warmup_div)}
    learn.fit(n_epoch, cbs=ParamScheduler(sched))


def fit_resume(learn, full_sched, n_epoch, epochs_done):
    """Continues an interrupted run whose LR followed `full_sched` (a fn of `pos` in [0,1] over
    the *original* `n_epoch`-epoch run, as returned by `warmup_cos_schedule`/`front_loaded_schedule`).

    fastai's own `ParamScheduler` position always restarts at 0 for a fresh `learn.fit` call, so
    a naive resume would replay the schedule's *start* (e.g. the warmup ramp) instead of picking
    up where the anneal left off. This remaps that restarted [0,1] range back onto
    `[epochs_done/n_epoch, 1]` of the original curve, so the LR trajectory across the interruption
    is identical to what an uninterrupted run would have produced -- only the wall-clock gap
    differs, not the optimisation path.
    """
    start_pos = epochs_done / n_epoch
    remaining = n_epoch - epochs_done
    if remaining <= 0:
        raise ValueError(f"epochs_done ({epochs_done}) >= n_epoch ({n_epoch}); nothing to resume.")
    resumed = lambda pos: full_sched(start_pos + pos * (1 - start_pos))
    learn.fit(remaining, cbs=ParamScheduler({"lr": resumed}))


def train(
    parquet_path: str,
    img_dir: str,
    out_dir: str,
    fold: str,
    model_name: str,
    model_arch_name: str,
    head: str,
    nb_epochs: int,
    batch_size: int,
    aug_img_size: int,
    img_size: int,
    min_img_per_spc: int = 0,
    family_filter: list = [],
    base_lr: float = 1e-3,
    freeze_epochs: int = 1,
    level_weights: list = None,
    label_smoothing: float = None,
    hierarchy_path: str = None,
    num_workers: int = None,
    decoder_num_layers: int = 4,
    decoder_nhead: int = 1,
    optimizer: str = "adam",
    fp16: bool = False,
    schedule: str = "fine_tune",
    precision: str = "bf16",
    grad_clip: float = 1.0,
    warmup_epochs: float = 0.0,
    aug_kwargs: dict = None,
    fast_decay_epochs: float = 1.0,
    lr_mid_frac: float = 0.1,
    class_reg_strength: float = 0.0,
    oversample_power: float = 0.0,
    logit_adjust_tau: float = 0.0,
    resume_checkpoint: str = None,
    resume_epochs_done: int = 0,
):
    if head not in HEAD_CLASSES:
        raise ValueError(f"Unknown head {head!r}; must be one of {list(HEAD_CLASSES)}.")
    if optimizer not in ("adam", "muon"):
        raise ValueError(f"Unknown optimizer {optimizer!r}; must be 'adam' or 'muon'.")
    if schedule not in ("fine_tune", "flat_cos", "one_cycle", "front_loaded"):
        raise ValueError(f"Unknown schedule {schedule!r}; must be 'fine_tune', 'flat_cos', 'one_cycle' or 'front_loaded'.")
    if optimizer == "muon" and schedule == "fine_tune":
        # MuonAuxAdamW re-partitions param groups, which doesn't survive fastai's
        # freeze/freeze_to bookkeeping that `fine_tune` relies on. See muon_opt_func.
        raise ValueError("optimizer='muon' requires an unfrozen schedule ('flat_cos', 'one_cycle' or 'front_loaded').")

    parquet_path, img_dir, out_dir = Path(parquet_path), Path(img_dir), Path(out_dir)
    hierarchy_path = Path(hierarchy_path) if hierarchy_path else parquet_path.parent / "hierarchy.csv"

    df, hierarchy = v4.gen_df(parquet_path, out_dir, min_img_per_spc, fold, hierarchy_path, family_filter)

    vocabs = {level: sorted(df[level].unique().tolist()) for level in HIERARCHY_LEVELS}
    n_classes = [len(vocabs[level]) for level in HIERARCHY_LEVELS]
    print(f"Classes per level -- species: {n_classes[0]}, genus: {n_classes[1]}, family: {n_classes[2]}")

    cls2idx, sparse_masks = build_class_spec(df, vocabs)

    # Long-tail resampling (dev/034): oversample rare species in the train loader. Weights are
    # over the finest level; computed on the training split only. power=0 -> off (uniform).
    sample_wgts = longtail.sample_weights(df, level=HIERARCHY_LEVELS[0], power=oversample_power)
    if sample_wgts is not None:
        print(f"Rare-class oversampling ON (power={oversample_power}, level={HIERARCHY_LEVELS[0]}).")
    dls = v4.make_dls(df, vocabs, img_dir, aug_img_size, img_size, batch_size, num_workers,
                      aug_kwargs=aug_kwargs, sample_wgts=sample_wgts)

    arch = getattr(importlib.import_module("fastai.vision.all"), model_arch_name)
    nf = v4.body_out_features(arch)

    if label_smoothing is None:
        label_smoothing = 1 / n_classes[0]  # mini_trainer's own HierarchicalBuilder.build_criterion default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = MultiLevelWeightedCrossEntropyLoss(
        num_classes=n_classes, device=device, dtype=torch.float32, weights=level_weights, label_smoothing=label_smoothing
    )

    # Long-tail logit adjustment (dev/034): add tau*log(prior) to logits at train time only.
    logit_adjust = None
    if logit_adjust_tau:
        adjustments = longtail.logit_adjustments(df, vocabs, HIERARCHY_LEVELS, tau=logit_adjust_tau, device=device)
        logit_adjust = longtail.LogitAdjustment(adjustments)
        print(f"Logit adjustment ON (tau={logit_adjust_tau}).")

    custom_head = build_head(
        head, nf, n_classes, cls2idx, sparse_masks, decoder_kwargs={"num_layers": decoder_num_layers, "nhead": decoder_nhead}
    )

    metrics = [
        *(v4.LevelAccuracy(i, f"acc_{level}") for i, level in enumerate(HIERARCHY_LEVELS)),
        # Per-level macro-F1 matching mini_metrics (species number is the mini_trainer target).
        *(v4.LevelMacroF1(i, f"f1_{level}") for i, level in enumerate(HIERARCHY_LEVELS)),
        v4.StreamingF1MultiHead(average="macro", name="F1(macro)"),
        v4.StreamingF1MultiHead(average="micro", name="F1(micro)"),
    ]

    learner_kwargs = {}
    if optimizer == "muon":
        learner_kwargs["opt_func"] = muon_opt_func  # Muon on 2D backbone weights + AdamW rest

    learn = vision_learner(
        dls, arch,
        n_out=1,  # unused: `custom_head` builds the real head
        custom_head=custom_head,
        # mini_trainer's heads initialize themselves (weight_norm + spherical-repulsion
        # init on the cosine classifier layers, RMSNorm's own ones-init in the
        # autoregressive decoder). Letting fastai's default `kaiming_normal_` sweep run
        # over the custom head would crash on RMSNorm's 1D weight and would otherwise
        # silently fight mini_trainer's own initialization for the other heads.
        init=None,
        loss_func=MultiLevelLossWrapper(criterion, logit_adjustment=logit_adjust),
        metrics=metrics,
        model_dir=out_dir / "models",
        cbs=[
            GCCallback(),
            HostMemoryGuard(),
            NaNGuard(),
            *([ClassDistributionRegularizer(class_reg_strength)] if class_reg_strength and class_reg_strength > 0 else []),
            *([LogitAdjustCallback()] if logit_adjust is not None else []),
            SupervisionContextCallback(),
            # Gradient clipping: mini_trainer's own loop clips at 5.0 and skips NaN batches;
            # fastai's loop has neither, so without this an unfrozen Muon+AMP run on the full
            # 12k-class problem hits a gradient spike, overflows, and NaNs out permanently
            # (observed: epoch 0 fine at 82% species, epoch 1 -> nan). Clipping bounds the spike.
            *([GradientClip(grad_clip)] if grad_clip and grad_clip > 0 else []),
            CSVLogger(out_dir / f"{model_name}.csv", append=True),
            SaveModelCallback(fname=model_name, every_epoch=True),
            # TensorBoardCallback logs per-batch optimizer hypers, and MuonAuxAdamW
            # exposes non-scalar hypers (betas tuple) that its `float(tensor)` chokes on.
            # CSVLogger already records every metric, so just skip TB under Muon.
            *([] if optimizer == "muon" else
              [TensorBoardCallback(log_dir=out_dir / "tensorboard", trace_model=False, log_preds=False)]),
        ],
        **learner_kwargs,
    )

    if fp16:
        # Mixed precision. Default to bf16, not fp16: bf16 has fp32's exponent range so it
        # can't overflow to inf, removing the main path to NaN divergence on this large-class
        # unfrozen problem. mini_trainer uses fp16 but survives via its own NaN-skip guard,
        # which fastai's loop lacks. Set precision: fp16 in the config to force fp16 anyway.
        learn = learn.to_bf16() if precision == "bf16" else learn.to_fp16()

    if resume_checkpoint:
        # Recover from an interruption (crash, reboot, preemption): load the last
        # SaveModelCallback checkpoint's weights and continue the *same* LR curve from where
        # it left off, rather than restarting the anneal. SaveModelCallback here runs with its
        # default with_opt=False, so only model weights are on disk -- MuonAuxAdamW's momentum
        # buffers restart cold, which just costs a few batches of re-warming momentum, not
        # correctness. `resume_epochs_done` is the number of epochs already completed (0-indexed
        # epoch N-1's checkpoint -> resume_epochs_done=N).
        state = torch.load(resume_checkpoint, map_location=device)
        learn.model.load_state_dict(state)
        print(f"Resumed model weights from {resume_checkpoint} ({resume_epochs_done}/{nb_epochs} epochs already done).")

    if schedule in ("flat_cos", "one_cycle", "front_loaded"):
        # Train everything from step 0 at a uniform base_lr (no frozen-backbone warmup, no
        # discriminative LR). fastai's fine_tune trains the backbone at base_lr/100, which is
        # ~100x too gentle for this far-from-ImageNet 12k-class problem. one_cycle keeps the
        # (excellent) one-cycle LR shape; flat_cos holds LR flat then cosine-decays;
        # front_loaded peaks early then drops hard in the first epoch(s) then decays gently.
        learn.unfreeze()
        if schedule == "front_loaded":
            wu = warmup_epochs if warmup_epochs and warmup_epochs > 0 else 0.15
            full_sched = front_loaded_schedule(nb_epochs, base_lr, wu, fast_decay_epochs, lr_mid_frac)
        elif warmup_epochs and warmup_epochs > 0:
            full_sched = warmup_cos_schedule(nb_epochs, base_lr, warmup_epochs, schedule)
        else:
            full_sched = None  # built-in fit_one_cycle/fit_flat_cos: no resumable schedule fn

        if resume_checkpoint:
            if full_sched is None:
                raise ValueError("resume_checkpoint needs warmup_epochs > 0 or schedule='front_loaded' -- "
                                  "the built-in fit_one_cycle/fit_flat_cos paths don't expose a schedule "
                                  "function to resume from.")
            fit_resume(learn, full_sched, nb_epochs, resume_epochs_done)
        elif full_sched is not None:
            learn.fit(nb_epochs, cbs=ParamScheduler({"lr": full_sched}))
        elif schedule == "one_cycle":
            learn.fit_one_cycle(nb_epochs, base_lr)
        else:
            learn.fit_flat_cos(nb_epochs, base_lr)
    else:
        # Same freeze-then-unfreeze transfer learning as dev/028: `vision_learner`
        # loads real pretrained weights and gives the body/head separate param
        # groups, so `fine_tune`'s freeze phase actually freezes the body.
        learn.fine_tune(nb_epochs, base_lr, freeze_epochs=freeze_epochs)

    # `learn.export()` (whole-object pickling) doesn't work here: mini_trainer's
    # heads use `weight_norm` parametrization, which PyTorch explicitly refuses
    # to pickle outside of `state_dict()`. Save state_dict + reconstruction
    # metadata instead -- the same convention mini_trainer's own trainer.py uses.
    model_path = out_dir / f"{model_name}.pt"
    torch.save(
        {
            "model_state_dict": learn.model.state_dict(),
            "head": head,
            "model_arch_name": model_arch_name,
            "vocabs": vocabs,
            "cls2idx": cls2idx,
            # Derived from `df`, NOT from the `hierarchy` gen_df read off disk. The masks in
            # `cls2idx`/`sparse_masks` come from df, so saving a file-sourced hierarchy lets a
            # checkpoint disagree with itself whenever hierarchy.csv doesn't match the data --
            # and dev/032 rebuilds its masks from this field, so the disagreement surfaces
            # there, after training has finished and the GPU-hours are spent.
            # That is not hypothetical: /work/global_lepi/hierarchy.csv on UCloud has 11,939
            # species while the dataset has 12,041, so every checkpoint trained there carried a
            # hierarchy missing 102 of its own classes, and mini_trainer refused to build masks
            # from it ("Unable to construct sparse masks"). Deriving from df makes the
            # checkpoint self-consistent by construction, whatever hierarchy.csv happens to say.
            "hierarchy": v4.build_hierarchy(df).to_dict(orient="list"),
        },
        model_path,
    )
    print(f"Model exported to {model_path}")


def create_out_dir(out_dir, desc):
    """Create the output directory named after datetime-desc."""
    from datetime import datetime

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dirname = join(out_dir, current_time + "-" + desc)
    Path(out_dirname).mkdir(parents=True, exist_ok=True)
    return out_dirname


def cli(config_path: str = None):
    if config_path is None:
        parser = argparse.ArgumentParser(description="Hierarchical head benchmark (mini_trainer heads x fastai training loop).")
        parser.add_argument("-c", "--config", type=str, help="Path to config file.")
        config_path = parser.parse_args().config

    if not exists(config_path):
        raise FileNotFoundError(f"Path to config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert float(config["version"]) in VALID_CONFIG_VERSIONS, (
        f"Wrong config version: {config['version']}. Must be in {VALID_CONFIG_VERSIONS}."
    )

    config["train"]["out_dir"] = create_out_dir(config["train"]["out_dir"], config["desc"])
    copyfile(config_path, join(config["train"]["out_dir"], "config.yaml"))

    train(**config["train"])


if __name__ == "__main__":
    cli()
