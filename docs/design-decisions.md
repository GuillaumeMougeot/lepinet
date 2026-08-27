# Design decisions — why the recipe looks like this

The [user guide](user-guide.md) says *how* to run lepinet and the [developer guide](developer-guide.md)
says *how* the code is arranged. This page is the third thing: **why each choice is what it is, and
what it was worth in points.**

Nothing here is a preference. Every line below is the residue of a run that was paid for, and each
links to the journal entry that argues it. It is written for the person who is about to change one
of these settings and deserves to know what happened last time.

The scientific claims — the ones that generalise beyond this dataset — are in
[`paper/DRAFT.md`](https://github.com/GuillaumeMougeot/lepinet/blob/main/paper/DRAFT.md). This page
is the engineering complement: how a working baseline was actually constructed.

---

## 1. The ladder

The baseline was not designed; it was climbed. Species macro-F1 on the held-out fold, one change at
a time:

| run | what changed | test F1 | Δ |
|---|---|---|---|
| `20260712-072542` | Muon + `flat_cos`, heavy augmentation, `grad_clip=1.0` | 0.8297 | — |
| `20260713-164456` | + 0.5 ep warmup, **light** augmentation, `grad_clip=5.0` | 0.8769 | **+4.7** |
| `20260714-072404` | `flat_cos` → **`one_cycle`** | 0.8887 | **+1.2** |
| `20260716-105029` | 5 → 10 epochs | 0.8976 | +0.9 |
| `20260716-154156` | 5 ep + **√-oversampling** | **0.9148** | **+2.6** |
| `20260725-*` | reimplemented from scratch in `src/lepinet` | 0.9152 | +0.0 |

Two things are worth reading off this table before anything else.

**The optimiser was never the lever.** Muon was in place from the first row and the model was still
at 0.83. What moved it was the *schedule*, the *augmentation strength* and the *sampler* — the parts
that decide what the model sees and for how long, not the parts that decide how a gradient is
applied. The instinct to reach for a fancier optimiser first was wrong here, and would likely be
wrong again.

**The largest single jump (+4.7) is a bundle**, and that is a mistake preserved on purpose: three
changes shipped in one run, so their individual contributions are unrecoverable. The honest state of
knowledge is "warmup + lighter augmentation + a looser gradient clip are jointly worth 4.7 points",
and no amount of later analysis can split it. Change one thing per run.

---

## 2. The recipe, component by component

### Schedule: `one_cycle`, not `flat_cos` — worth +1.2 pt

`flat_cos` holds the learning rate flat and spends only its final 25 % annealing. The model was still
converging hard when the run ended, so it was being **graded mid-descent**. `one_cycle` anneals over
~90 % of the run. Nothing else changed.

The general form of this lesson: when a fixed epoch budget is part of the comparison, the schedule's
job is to be *finished* at the budget, not to make the best progress per step.
→ [journal: 2026-07-16-why-was-fastai-behind-mini-trainer](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-16-why-was-fastai-behind-mini-trainer.md)

### Epoch budget: check the baseline's, don't assume it

Part of the "gap" against `mini_trainer` was never a gap. Its real budget was **10 epochs, not 5**,
so the comparison had been unfair from the start; matching it bought 0.9 points of the apparent
deficit. At 10 epochs the model was *still improving* (valid loss 0.4472 → 0.4327 across the last
two epochs), so longer schedules remain the cheapest untested lever.

The default is 5 epochs only because that is what the published comparisons used.
→ [journal: 2026-07-16-why-was-fastai-behind-mini-trainer](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-16-why-was-fastai-behind-mini-trainer.md)

### Long tail: √-oversampling (`power=0.5`) — worth +1.9 pt in-distribution, and **−2.9 pt where it matters**

> **Read this before enabling it.** The paragraph below is the in-distribution case, which is real.
> But measured off the training distribution, oversampling *costs*: −1.52 pt at 20 M, −2.88 pt at
> 198 M, and **−2.90 pt on species a model has never seen** (B6 vs B7, identical otherwise). It is
> the one intervention whose cost *grows* with capacity, because it reshapes the data rather than
> constraining the objective. For a model that will meet someone else's images, leave it off.

### The in-distribution case: √-oversampling, not logit adjustment — worth +2.6 pt

Both were tested head-to-head at 5 epochs. Oversampling reached **0.9148**; logit adjustment reached
0.9031 *and damaged genus and family* to buy it.

The mechanism is the interesting part. Oversampling reshapes **which data is seen** without changing
what the loss means for any given example, so it degrades gracefully across all three levels at
once. Logit adjustment reshapes **the loss itself** through a single τ shared across three genuinely
different distributions — and one constant cannot be simultaneously right for 12,041 species, 4,333
genera and 102 families. It was wrong for two of the three.

The transferable rule: in a hierarchy, **prefer interventions that do not require one shared constant
to be correct at every level at once.** → [journal: 2026-07-17-does-longtail-help](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-17-does-longtail-help.md)

### Precision: bf16, always

fp16 overflows the cosine head. The 0.9148 run survived fp16 only because the head was forced to
fp32 inside an adapter; the autoregressive head, which has no such protection, trained visibly broken
until precision was changed. This was diagnosed as a wiring bug for some time before the real cause
surfaced.

bf16 is the package default and there is no good reason to leave it.
→ [journal: 2026-07-18-autoregressive-fp16-instability](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-18-autoregressive-fp16-instability.md)

### Augmentation: light beats heavy

Part of the +4.7 bundle, and the direction is unambiguous even though the magnitude is not: the
initial configuration was fastai's heavier defaults, and reducing it helped. Fine-grained species ID
depends on *local texture* — wing scale patterns — and aggressive photometric and geometric
augmentation destroys exactly the signal the task needs. This is task-specific reasoning, not a
general rule, and it is the reason domain-mimicking augmentation is an **opt-in** module
(`domain_aug`, `src/lepinet/augment.py`) rather than a change to the default pipeline.

### Optimiser: Muon + AdamW, with a caveat that costs time

Muon (`MuonAuxAdamW`) **re-partitions parameter groups**, which breaks fastai's freeze bookkeeping —
it only works with unfrozen schedules. It also takes tuple betas, so fastai's stock `fit_one_cycle`
chokes on it, because that schedules momentum as well as learning rate. Hence the hand-built
**LR-only** schedules. Anyone swapping the optimiser should expect to touch the schedule code too.

### Head: a cosine head, and the bottleneck width

Class prototypes are L2-normalised and scored by angle. Why that beats a plain linear layer — and
why a raw cosine is a poor logit in high dimension — is a scientific claim and lives in the paper
(§2.1–2.2), not here.

The width was measured: **128 → 0.8843, 256 → 0.9002, 512 → 0.9058** at 5 epochs. 256 is the shipped
default for the small models because it is the knee — 512 buys 0.56 pt for double the head, and the
head is ~51 % of a small model's parameters, which is the whole reason the phone app was feasible.
→ [journal: 2026-07-20-lepi-app-compression](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-20-lepi-app-compression.md)

### Coarse levels: marginalise, don't add heads

Genus and family come from summing the species posterior, not from their own classifiers. This is
smaller, probabilistically coherent (the coarse posterior *is* the sum of the species one — argmax agreement is *not* guaranteed, see the journal entry on that), and scores better at every level (0.9135/0.9606/0.9739 vs
0.9110/0.9587/0.9708 like-for-like). It is one of the paper's contributions; see §2.4 there.

---

## 3. Things that did not pay (do not buy them twice)

| tried | result |
|---|---|
| **Class-distribution regularisation** (`class_reg_strength=0.001`) | 0.8860 vs 0.8880 unregularised — inside noise. A wash. |
| **Logit adjustment** (τ=1.0) | 0.9031, and it broke genus/family. Superseded by oversampling. |
| **Parent-conditioned hierarchical head** | 0.8845 vs 0.9110. Loses. |
| **Autoregressive head** | 0.69–0.73 even after the fp16 fix. Loses badly. |
| **int8 quantization for the browser** | Runs in Python, cannot run in ORT-Web (no `ConvInteger` kernel). fp16 ships instead. |
| **Tuning the ArcFace margin cheaply** | Two principled failures — see the paper's negative results. Needs full runs. |

---

### A noise floor belongs to (metric x benchmark x training procedure)

Our floors were measured on 5-epoch end-to-end 20 M runs and then applied to 2-epoch frozen-trunk
stages at 198 M, where the true spread is **7.2x larger** on held-out species (0.0374 vs 0.0052).
Two identical runs of G3 differed by 3.74 pt on that benchmark while agreeing to 0.0010
in-distribution. Re-measure the floor whenever the procedure changes, and prefer stating a margin in
points with the spread beside it over "Nx floor", which hides which floor is being invoked.
[[2026-08-27-the-noise-floor-does-not-transfer-across-training-regimes]]

### Measure at two capacities before claiming a method property

*Change one factor per run* is not sufficient. Five interventions here have changed magnitude across
the 10x scale step and two have changed sign, so a single-scale measurement establishes a fact about
a configuration, not a property of a method. Two conclusions moved on consecutive days in 2026-08 for
exactly this reason: B9 (20 M) dissolved the staged-vs-end-to-end trade, and G3 (198 M) restored it.

The cheap version of the rule: interventions that **reshape the data** (resampling, balanced
replication of pseudo-labels) get *worse* with capacity; interventions that **constrain the
optimisation** (auxiliary losses, classifier rebalancing) merely weaken. Classify the intervention
before predicting its sign, and say which bucket it is in when writing the hypothesis.

## 4. Lessons about measurement (the most transferable part)

These cost the most and generalise the furthest. Each one produced a wrong belief that survived for
days.

**Before hunting a bug, check that both numbers mean the same thing.** A "0.92 val vs 0.83 test fold
bug" was chased at length. There was no bug: the training log's `F1(macro)` averaged **all three
taxonomic levels** (inflated by easy genus/family) while the evaluation reported **species only**.
Like-for-like, both were 0.83. → [journal: 2026-07-16-why-was-fastai-behind-mini-trainer](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-16-why-was-fastai-behind-mini-trainer.md)

**Audit the evaluation set before believing the metric.** A ported model scored 0.9455 against a
0.9148 baseline — a suspiciously large win. The eval had passed `--min-img-per-spc 50`, dropping the
long tail from a *macro* average. With the filter removed it scored 0.9152, i.e. exactly reproducing
the baseline. A metric that improves when you change how it is computed has not improved.
→ [journal: 2026-07-24-src-lepinet-baseline-port](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-24-src-lepinet-baseline-port.md)

**Do not re-normalise a weight matrix you did not train.** The cosine head is *supposed* to keep
every prototype row at unit norm, so re-normalising it before use looks like a no-op. On one
checkpoint it was not: the rows had drifted, `argmax(g_c * cos_c)` is not `argmax(cos_c)`, and a
known-0.9135 model scored 0.5589. Use the weight as the model uses it, and assert the row norms if
the code depends on them being 1.

**A `dev/`-registered head is invisible unless its module is imported.** `HEAD_REGISTRY` is the
extension seam, but the entries only exist once `dev/050` has been executed — so any script that
loads a checkpoint trained with one dies in `build_head` with "Unknown head". It has now bitten
three separate scripts (`dev/061`, `dev/059`'s caller, `dev/070`), always at the moment a *previously
unrelated* config inherited such a head. If a script can load an arbitrary checkpoint, it must
register the dev heads.

The silent version is worse than the crash: `marginal_arcface` applies its margin inside `forward`
using labels supplied by a callback, so a runner that registers the head but forgets the callback
gets a working model that quietly is not the one the config describes.

**A cloned job spec inherits the wrong mounts.** `[[resources]]` is easy to overlook when a TOML is
copied from a neighbouring job: the B3 combine step was cloned from a flemming-only job and died on a
missing `/work/global_lepi` after five hours of the chain behind it sitting `BLOCKED`. When copying a
spec, check the mounts against what the command actually reads.

**A job can report SUCCESS while the script inside it exits 1.** Read the logs, not the job status.

**Framework attributes can lie.** fastai hardcodes `DataLoader.num_workers` to `1`; the real value
lives on `fake_l`. Reading the wrong one silently ran evaluations single-threaded — about **1 img/s
instead of 898**, a ~900× slowdown that was first misdiagnosed as a hardware sizing problem. The
`dl_num_workers()` helper in `lepinet/test.py` exists solely to stop this recurring.

**An all-`True` `is_valid` gives an `IndexError` about pandas indexing.** fastai's `ColSplitter`
leaves the *train* split empty and `DataBlock.setup` dies far from the cause. It cost debugging time
twice — both times on inference-only loaders, where every row was marked valid because the labels
were placeholders. `make_dls` now raises a message naming the actual problem.

**`isinstance(True, int)` is `True`** in Python, which is how a boolean reached a `sqrt()` and
produced `sqrt(-1)`.

---

## 5. Why the code was rebuilt from scratch

The original pipeline worked and scored 0.9148. It was still reimplemented as a clean, fastai-only
package, because the working version depended on `mini_trainer` and `mini_metrics` — two external
repositories carrying a `GCCallback` that existed to break a reference cycle, a `_weight_bias` cache,
a broken `--optimal` two-pass metric workaround, and control flow that made ONNX export fragile.

The rebuild was gated on a hard criterion: **reproduce 0.9148 from scratch, on the byte-identical
evaluation set.** It scored **0.9152**. Only then were the old dependencies dropped.

Two design rules from that rebuild are still in force:

- **Prove correctness without a GPU first.** The clean head was made to load the old checkpoint
  bit-exactly, so export, prediction and evaluation were all validated before any retraining.
- **Every extension point is a registry.** `HEAD_REGISTRY`, `DOMAIN_AUG_REGISTRY` — a `dev/`
  experiment adds an entry rather than editing the package, so the default recipe cannot drift out
  from under a published number.

→ [journal: 2026-07-24-src-lepinet-baseline-port](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/2026-07-24-src-lepinet-baseline-port.md)

---

## 6. Where to go next

- The numbers for every run: [`RESULTS.md`](https://github.com/GuillaumeMougeot/lepinet/blob/main/RESULTS.md)
- The reasoning, in order: [`journal/README.md`](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/README.md)
- What is running right now: [`journal/PLAN.md`](https://github.com/GuillaumeMougeot/lepinet/blob/main/journal/PLAN.md)
- The scientific claims: [`paper/DRAFT.md`](https://github.com/GuillaumeMougeot/lepinet/blob/main/paper/DRAFT.md)
