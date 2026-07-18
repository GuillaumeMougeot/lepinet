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

## The fix

Run the autoregressive head in **fp32** (`precision: fp32`, `fp16: false`). Confirmed
(`arfix-fp32`, 2-epoch UCloud run): epoch 0 train_loss **1.37** (learning normally, down from
~21 init), species-F1 **0.70** -- right alongside independent/hierarchical at epoch 0 (~0.78),
no NaN. It trains.

Cost: fp32 is ~2x the memory and slower (this run ~1:45/epoch vs the fp16 heads' ~1:20), and the
autoregressive beam eval is slow on top. For a fair 3-head benchmark, either run all three in
fp32, or -- cheaper -- keep independent/hierarchical in fp16 and run only autoregressive in fp32,
noting the precision differs. A tighter fix (keep the backbone fp16 but clamp/fp32 just the
decoder's feedback path, or add a decoder-input LayerNorm/clamp) would let it match the others'
precision; not yet attempted.

Related: [[2026-07-ucloud-benchmark-oom]] (the run it surfaced in), the dev/030 MTHeadAdapter
docstring (the fp32-head-under-fp16 pattern this extends).
