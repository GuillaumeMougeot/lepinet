# Why the autoregressive head trained broken, and the fix

**Status:** RESOLVED (2026-07-18). The autoregressive head is fp16-unstable; running it in fp32
fixes it. It was never a wiring bug.

## Symptom

In the UCloud MT-head benchmark (fp16 + Muon), the autoregressive head trained broken: epoch 0
train_loss **24.84** (higher than the ~21 random-init value -- it diverged *up*), species-F1
**0.000003**, and a NaN in the log (NaNGuard aborted). The run then failed at the test phase on
the separately-fixed dev/032 lowmem bug. The other two heads (independent, hierarchical) trained
fine to ~0.90+.

## It is NOT a wiring bug

Diagnosed on a fixed batch in fp32 (dev/030's exact head/loss):
- preds shapes correct (species/genus/family), loss finite (~21 init),
- gradients reach the backbone and 52/60 head params,
- it **learns a fixed batch to ~0 loss in 80 steps** (both the soft/supervised mix and forced
  supervised).

So the head, loss, MRO workaround, and SupervisionContext teacher-forcing all work. Earlier
worry about the MRO / a double-wrapped adapter was a red herring (build_head already returns the
MTHeadAdapter; my first probe double-wrapped it).

## The cause: fp16

The backbone runs fp16 while MTHeadAdapter already forces the *head* to fp32. The autoregressive
head is the only one that **generates a sequence autoregressively** through the XADecoder --
feeding predictions back through many transformer steps. fp16 backbone features (occasionally
large) compounding through that generation overflow to inf/NaN, where the plain independent/
hierarchical heads (a single linear per level) do not. That is why only this head diverged.

## The fix: bf16 (the existing default) -- NOT fp32

**The fix already existed and I initially missed it.** dev/030 *defaults* to `precision: "bf16"`
(line 602), and its own comment says bf16 "has fp32's exponent range so it doesn't overflow like
fp16." bf16 is the intended-safe path; the benchmark configs *override* it with `precision: fp16`,
and that override is what broke the autoregressive head. So the regression is the config, not the
code.

Two confirmations:
- fp32 works (`arfix-fp32`, full fp32): epoch 0 train_loss 1.37, species-F1 0.70. Proves the
  diagnosis but is the wrong (heavy) fix -- ~2x memory, ~1:45 vs 1:20/epoch.
- **bf16 works** (local, family 9717, 2 epochs): train_loss 3.59 -> 1.30, species-F1 0.24 -> 0.69,
  no NaN. bf16 is the same speed/memory as fp16, so this is the right fix, not fp32.

Why bf16 suffices where the fp32-*head* block alone didn't: the head runs fp32 (MTHeadAdapter
disables autocast), but its *input* is the backbone's features. In fp16 those occasionally
overflow (~65504) to inf before they even reach the head, and the autoregressive decoder's
compounding generation turns that into persistent NaN (the plain heads tolerate it). bf16 keeps
the *backbone* in fp32's exponent range, so the features never overflow -- fixing it upstream of
the head.

**Action:** set `precision: bf16` on the autoregressive config (done). Independent/hierarchical
were tuned and run in fp16 and are fine there, but they would also work in bf16 -- for a fair
3-head benchmark, run all three in bf16.

Related: [[2026-07-ucloud-benchmark-oom]] (the run it surfaced in), the dev/030 MTHeadAdapter
docstring (the fp32-head-under-fp16 pattern this extends).
