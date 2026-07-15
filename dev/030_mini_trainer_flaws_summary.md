# mini_trainer issues found while driving its hierarchical heads through fastai's `Learner`

Context: `dev/030_hierarchical_heads_benchmark.py` in lepinet reuses mini_trainer's actual
head/loss modules (not a reimplementation) but trains them through fastai instead of
mini_trainer's own loop, at real scale (12,041 species / 4,333 genus / 102 family,
effnetv2s, fp16 AMP). All items below were worked around without touching mini_trainer's
source; workarounds live in dev/030 for reference.

1. **`AutoregressiveClassifier(IndependentClassifier, AutoregressiveMixin)` MRO bug.**
   Python's MRO resolves `.forward` to `IndependentClassifier.forward`, silently bypassing
   the decoder entirely — `AutoregressiveMixin.forward` is dead code via the normal
   `head(x)` call path. Neither base class overrides `forward` to fix the ambiguity. Also
   never exercised by `mt_htrain`'s own CLI (only hierarchical/conditional/independent are
   wired up there), which is probably why this hasn't surfaced. Workaround: call
   `AutoregressiveMixin.forward(head, x, method=...)` explicitly instead of `head(x)`.

2. **`MultiLevelWeightedCrossEntropyLoss` never calls `super().__init__()`.** It subclasses
   `nn.Module` and places its tensors on `device` in its own `__init__`, but skips
   `nn.Module.__init__()`, so `_modules`/`_parameters` etc. are never initialized. Any
   framework that expects it to behave like a real `nn.Module` (e.g. registering it as a
   submodule and calling `.to(device)` on it) will crash. Not a problem inside
   mini_trainer's own loop since it never does that, but it's a footgun for anyone
   embedding it elsewhere.

3. **GPU memory leak in `Classifier._weight_bias()`** (mini_trainer/modeling/classifier.py).
   Every forward call in training mode re-assigns a *live, graph-attached* view of the
   `weight_norm`-parametrized weight into a persistent buffer on `self`. `weight_norm`'s
   parametrization keeps a back-reference to its parent module, forming a reference cycle
   rooted at the classifier that traps the *entire* batch's backward graph (all backbone
   activations, not just the classifier weight) until Python's cyclic GC happens to run.
   Refcounting never frees it, and the GC's default object-count trigger doesn't fire often
   enough at real batch/image sizes — memory climbs monotonically until CUDA OOMs (empirically
   ~60 batches to blow a 32GB GPU at full class count). If mini_trainer's own training loop
   doesn't hit this, it's likely because it forces periodic GC somewhere already, or trains
   at a scale/duration where it doesn't matter — worth checking whether `mt_htrain` has the
   same latent issue at larger scale. Workaround: `gc.collect(0)` every batch (generation-0
   only — a *full* `gc.collect()` cost ~250ms/batch at 12k classes, 4-5x the actual compute,
   and visibly stalled the GPU; gen-0-only costs ~4ms and is just as effective since the
   cycle is always created and dies within a single batch).

4. **The cosine/weight-norm classification head overflows fp16.** Its forward computes
   `F.normalize(hidden(x))`; as weights grow during training, `hidden(x)`'s output magnitude
   can exceed fp16's ~65504 range, producing `inf`, and `normalize(inf)` = `NaN`. This is the
   standard ArcFace/cosine-margin-under-fp16 failure mode. Once it happens the NaN is
   persistent (weights stay large), not a one-off. We only hit this because we drive the
   head through fastai's `Learner`, which has no batch-level NaN-skip of its own — but it's
   worth flagging because mini_trainer's own AMP path presumably has the same overflow
   *arithmetically*; if its loop tolerates it, it's likely via the loss-skip mechanism in #5
   below papering over frequent-enough NaN batches rather than the head being numerically
   safe. Workaround: run just the head (not the backbone) in fp32 under fp16 autocast
   (`torch.autocast(..., enabled=False)` + `.float()` before the head) — negligible cost
   since the head is tiny relative to the backbone, and this is the standard fix for
   ArcFace-style heads under mixed precision. Might be worth building into the head itself
   rather than leaving it to callers.

5. **No exposed/reusable NaN-batch guard.** mini_trainer's own training loop apparently
   tolerates/skips non-finite losses internally (inferred from it training stably at scale
   in-house despite #4), but that behavior isn't factored out as something reusable outside
   `mt_htrain`'s loop — anyone driving the model/loss through a different training loop (like
   we did) has no equivalent safety net and will silently poison every weight on the first
   NaN batch. Might be worth exposing as a small reusable utility (e.g. "does this loss
   value indicate a skippable bad batch, and if so what's a `torch.optim` `Optimizer.step()`
   no-op path") rather than something baked only into your own loop.

6. **`MuonAuxAdamW` re-partitions parameter groups internally**, which doesn't survive being
   handed to a framework that manages its own param-group/freeze bookkeeping (e.g. fastai's
   `freeze`/`freeze_to`/discriminative-LR groups). Fine inside mini_trainer's own loop where
   it owns the whole optimizer lifecycle; a rough edge for anyone trying to use
   `MuonAuxAdamW` as a drop-in optimizer inside another training framework. Not something we
   needed fixed — just worth knowing it's not really "just another `torch.optim.Optimizer`"
   from an integration standpoint.

None of these blocked us — all six are worked around in `dev/030_hierarchical_heads_benchmark.py`
without touching mini_trainer's source, and models are training successfully at full scale
(12k species) with them in place. Flagging in case any are worth fixing upstream, especially
#1 (real bug, dead code path) and #3 (could bite `mt_htrain` itself at larger scale/duration).
