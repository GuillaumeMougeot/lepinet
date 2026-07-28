# Landscape & plan — where lepinet is, and the ordered work ahead

**Status:** planning doc (2026-07-28). Written at the owner's request to step back, take stock, and
order the whole backlog *before* more coding. Consolidates open threads from
[[2026-07-src-lepinet-baseline-port]], [[2026-07-bigger-everything]],
[[2026-07-teacher-student-app-bridge]], [[2026-07-lepi-app-compression]]. Nothing here is executed
yet; this is the map and the argument for the order.

## Where we are (settled)

| result | number | status |
|---|---|---|
| Port reproduces baseline (effnetv2_s, 5 ep) | species macro-F1 **0.9152** (≈0.9148) | milestone baseline |
| Bigger everything (ConvNeXtV2-L @320, 6 ep) | **0.9316** (+1.68 pp) | best teacher |
| Distillation (mock teacher → b0, **T=1**) | **0.8786** (> 0.8692 from-scratch) | method works |
| ArcFace head + DINOv3/ViT backbone | plumbing built, smoke-validated | opt-in, default off |
| Export → app-ready bundle (config.json + friendly taxonomy) | parity 2e-5 | interop done |

All three headline questions (port, scale, distillation) are answered as wins. The rest is
consolidation, a diagnostic the owner asked for, forward experiments, and productionisation.

---

## Answers to the specific questions raised (2026-07-28)

### Q1. export vs bundle — is there duplication?

Yes, latent, and worth resolving now. **Current reality:** there is *no* `bundle` command; `export`
already emits `model.onnx` + `taxonomy.json` + `MANIFEST.json` **and** (since the interop work) a
`config.json`, i.e. it half-produces an app bundle. The planned `bundle` was to add
quantize + calibrate + names. Leaving both would blur "make an ONNX" and "make a shippable package."

**Proposed seam (clean, non-duplicative):**
- **`export`** = checkpoint → the *model artifact*: `model.onnx` (fp32) + `taxonomy.json` +
  `MANIFEST.json`. Reproducible, self-contained, no app assumptions. **Move `config.json` out of
  export.**
- **`bundle`** = checkpoint → the *deployable app folder*: calls `export`, then adds `int8/qdq`
  quantization, `calibration.json` + `thresholds.json`, `names.json`, and writes `config.json`.
  This is the "one button" the owner wants; it is also what `release` uploads.

So `bundle` composes `export`; `export` never mentions the app. One capability per command.

### Q2. bigger-everything journal + the scaling rationale

The journal **exists**: [[2026-07-bigger-everything]] (now RESOLVED, with the 0.9316 result). The
owner didn't recall it — noting here so it's findable via the index. The scaling *rationale* the
owner asked to see argued and stored:

**Why DINOv3-distilled ConvNeXt before DINOv3 ViT.** Four reasons, in priority order — dataset size
is a *contributing* one, not the main one:
1. **Zero new plumbing / lowest risk.** `convnext_*.dinov3_lvd1689m` emit 4-D maps → drop straight
   into the `PooledHead` path we *just* validated with ConvNeXtV2-L (ran clean, 0.9316). The ViT
   path (ViTBody/FlatHead/manual Learner) is built but less battle-tested. Same-family next rung.
2. **Deployment.** The end product is a browser model. Conv nets give clean ONNX + fast WASM/WebGPU,
   **no attention kernels in the hot path** ([[2026-07-lepi-app-compression]] §C2). A ViT teacher
   distilled into a conv student is fine, but a conv teacher keeps the whole chain conv-native.
3. **Distillation-family alignment.** Student is a small conv net; a conv teacher's feature
   geometry transfers more directly, and the export/quantize pipeline is conv-proven.
4. **Dataset size (the owner's hypothesis — partly).** With ~3 M labelled images the SSL-vs-
   supervised gap narrows, so a DINOv3-*distilled* conv captures most of DINOv3's quality without the
   ViT. True, but secondary to (1)–(3). The ViT is still worth a **single** "can a ViT teacher beat
   the best conv teacher" run once the conv ladder is exhausted.

**Why only 6 epochs / limited augmentation / not "full blast big" immediately.** This is a
deliberate methodology, not timidity:
- **6 epochs** was a one-day wall-clock budget (~3 h/epoch), an explicit first guess. The result
  showed it was **not** under-annealed (macro *and* micro rose, tail levels most). More epochs
  (10–15) is the obvious cheap next gain — it just wasn't needed to answer "does scale help."
- **Moderate (not maximal) augmentation.** Heavy distortion hurts *short* runs (too few passes to
  learn through it), and **fine-grained wing texture is fragile** — aggressive warp/color can destroy
  the discriminative signal. Aug strength must scale *with* epochs; we hadn't validated that tradeoff
  at scale, so we raised aug moderately (warp 0.1, lighting 0.2) over the baseline's light aug, not
  to the max.
- **Not full-blast (biggest model + max res + max epochs + max aug at once)** on purpose:
  - **One-variable discipline** — the project's own ladder lesson. Full-blast means a failure
    (divergence, OOM, a wash) tells you nothing about *which* lever. Each rung isolates the change.
  - **Cost/evidence order** — you earn the right to spend 40 h on ConvNeXtV2-Huge by first proving
    cheaply (+1.68 pp) that the direction pays. We now have that evidence.
  - **Diminishing teacher returns for the end goal.** The teacher only needs to be good *enough* to
    distill from; the *student's* capacity is the real ceiling on the shipped model. A 0.9316 teacher
    is already a big step; 0.94+ from a huge model may barely move the distilled student.
  - **Conclusion:** full-blast is now *justified as the next rung* (bigger backbone, more epochs,
    higher res), precisely because the controlled step proved the direction. It was sequencing, not
    an argument against scale.

### Q3. Distillation — happy? keep in src? modular?

- **Happy?** Cautiously yes. The T=1 result (0.8786 > 0.8692 from-scratch) proves the method works
  and the failure→diagnosis→fix arc (T=4 hurt because the z-score head is already flat) is
  understood and recorded ([[distill-kd-temperature]]). But it used a *mock* teacher (0.911); the
  real payoff is re-running with the 0.9316 teacher (wider headroom). So: happy with the machinery
  and hyperparameters, the shippable-student gains are still ahead.
- **Keep in `src/`?** Yes. Producing shippable students is a first-class *product* function (the
  two-model vision), not an experiment — it belongs in the package, not `dev/`.
- **Modular / removable?** Yes, fully opt-in. `distill_teacher=None` (default) → ordinary training;
  `DistillLoss`/`DistillCallback` are only constructed when a teacher is set (`train.py` branch). It
  adds nothing to the normal path and can be ignored entirely. Confirmed.

### Q4. test / predict / TTA status

- **`lepinet test`** (`evaluate`) exists → native per-level macro-F1 + micro-acc + a
  `mini_metrics`-format `predictions.csv`. **It does NOT use TTA** (one forward per image).
- **`lepinet predict`** (`infer.predict`) exists → single image / folder, **TTA on by default**
  (`--tta`, 4-flip).
- **The app (ONNX)** does a single forward — **no TTA**.
- **So predictions are not uniformly TTA'd:** `predict` uses it; `test` and the shipped app do not.
  That is actually the right default (test metrics should match the deployed single-pass behaviour),
  but it's worth a **`--tta` flag on `test`** for a fair "best-case metric" number. Small addition.

### Q5. ArcFace for open-set / OOD species — yes, strong direction

The owner's intuition is correct and important. ArcFace was built for face *verification*: it shapes
an embedding space where each identity occupies a tight, well-separated angular region, and — the key
property — **distances stay meaningful for identities never seen in training**. That is exactly
open-set recognition, which species ID needs (new/rare/regional species arrive constantly).

**How an ArcFace model is used (both modes are valid, and we can support both):**
1. **Classifier mode** (what our `ArcFaceHead` already is): the learned prototypes *are* the class
   weights; softmax over them for known species. **OOD score = max cosine similarity to any
   prototype** — low max-cosine ⇒ "unknown species." No templates needed.
2. **Embedding/template (gallery) mode** (the face-verification pattern): ignore the classifier,
   store reference embeddings per species (templates), classify a new image by nearest template in
   cosine space. **Adds new species with no retraining** — just enroll their template embeddings.
   Best for open-set and for the long tail / regional additions.
- **Hierarchy extends it gracefully:** an OOD *species* may still be a known *genus*/*family* — score
  against genus/family prototypes (or marginalize) to say "unknown species, but Noctuidae." Exactly
  the graceful degradation a field app wants.
- **This connects to the flemming dataset:** its `referenced` vs `unreferenced` split looks like an
  open-set scenario (known vs novel). Natural OOD benchmark.
- **Plan:** train with ArcFace (species margin), then evaluate (i) known-species F1 *and* (ii) OOD
  detection (AUROC of max-cosine on held-out/unreferenced species), and prototype the template
  mode. Store design + results in a dedicated journal entry.

### Q6. Diataxis docs + remove emoji

Restructure `docs/` into the four Diataxis modes (like the ucloud-api docs): **Tutorials**
(learning), **How-to** (tasks), **Reference** (API/CLI/config), **Explanation** (the method + the
long-tail/fine-grained reasoning). Rewrite `README.md` to lead with the task and point into that
structure; **remove all emoji**. Folds together with the existing "reframe docs around the task, not
the port history" task.

---

## UCloud / disk state — inventory & cleanup plan

**`/12347837/repos/` leftovers** (keep `lepinet` only): `compat-check`, `demo-trainer`,
`extend-test`, `ioprobe` (ucloud-api test scaffolds), `mini_metrics`, `mini_trainer` (old-pipeline
deps; the lepinet package is standalone). → **remove** (all re-cloneable/regenerable).

**`repos/lepinet/data/ucloud_models/`** (34 dirs). Classification:
- **Keep + copy local** (important, not reproducible cheaply): the 5ep effnetv2_s baseline
  (`20260724-181230`, 0.9152 — already local at `data/local_models/5ep_baseline`), the ConvNeXtV2-L
  teacher (`20260724-202442`, 0.9316), the distilled-b0-T1 student (`20260725-232410`, 0.8786), the
  b0 from-scratch control (`20260725-164507`, 0.8692).
- **Delete** (throwaway smokes): `*-smoke-9717`, `*-convnextv2l-smoke*` (x2), `*-arcface-smoke*`
  (x2), `*-dinov3-vitb-smoke`, and the distill **T=4** run (`20260725-164500`, 0.8546 — negative
  result, fully documented, checkpoint not worth keeping).
- **Delete** (old dev/030 pipeline, numbers already in RESULTS.md): the `20260716–18` `bench-*`,
  `diag512`, `staged`, `arfix`, `heads-global-*` checkpoints (~20 dirs).

**`repos/lepinet/data/` prediction folders** — 8 of them (`ucloud_preds`, `_allspc`, `_cnxv2l`,
`_control`, `_distill`, `_distill_T1`, `ucloud_smoke`, `ucloud_smoke_preds`). Predictions.csv are
~130 MB each. → **Consolidate to a single `ucloud_preds/<run-name>/`** (owner's suggestion, agreed),
keep only the *useful* ones (baseline-allspc, cnxv2l teacher, distill-T1, control), **delete smoke
preds** and the superseded ones.

**Stray files** in `repos/lepinet/` root: `colo_1.log`, `colo_2.log`, `colo_3.log` (old
gpu-decode-colocate experiment). → **remove.**

**RESULTS.md** is auto-generated by `dev/036_ledger.py` from the *local* `data/global` (old
pipeline) and does **not** contain the UCloud lepinet-package runs. Plan: add a **hand-maintained
"lepinet package (UCloud) runs" table** with a **location column** (local / ucloud), clearly marked
as manual (the ledger can't see UCloud). Longer term, teach the ledger to ingest UCloud results.

> All deletions above are **proposals pending owner confirmation** — destructive UCloud ops will not
> run until confirmed. Copy-to-local of the four keep models is safe and can proceed first.

---

## The ordered plan

Ordered by: solid ground first (know & tidy what we have), then the requested diagnostic, then
forward experiments (cheap→expensive), then productionisation.

**Phase 0 — consolidate & document the ground (mostly non-code)**
1. Copy the 4 keep-models local; **confirm then** delete smokes/old-bench checkpoints, consolidate
   `ucloud_preds/`, remove repo leftovers + stray logs.
2. Complete RESULTS.md with the UCloud package runs + location column.
3. Expand [[2026-07-bigger-everything]] with the scaling rationale above (done here; cross-link).
4. Decide + document the export/bundle seam (above); no code yet.

**Phase 1 — the diagnostic the owner asked for**
5. **flemming_helsing `referenced` test** with the best model (ConvNeXtV2-L): build a labels parquet
   from the folder (species = dir name, genus/family via the taxonomy parents), run eval on UCloud,
   produce the **`mini_metrics` table at threshold 0,0,0** on all three levels. Handle OOD species
   (folder species not in the 12,041 vocab) explicitly — report in-vocab metrics + an OOD count.

**Phase 2 — forward model work (`dev/` experiments + `src/` features)**
6. **Real distillation**: ConvNeXtV2-L (0.9316) → small student, T=1 — the shippable student.
7. **Hierarchical + autoregressive heads** reimplemented fastai-only in `dev/` (import lepinet,
   register into `HEAD_REGISTRY`); benchmark independent vs hierarchical vs autoregressive on
   effnetv2_s — continues the open head-comparison question, all tries kept in `dev/`.
8. **ArcFace / OOD** (Q5): train with margin, evaluate known-species F1 + OOD AUROC (using
   flemming referenced/unreferenced), prototype template mode.
9. **Bigger teacher** ("full blast" next rung): DINOv3-distilled ConvNeXt-L / ConvNeXtV2-Huge, more
   epochs, higher res — now justified.

**Phase 3 — productionisation**
10. **`lepinet bundle`** (export + quantize + calibrate + names + config.json) + **`release`**.
11. **ORT-Web small-format fix** (per-tensor QDQ / fp16) — needs an in-browser test (owner opens a
    hosted URL; no remote desktop needed).
12. **Diataxis docs** + remove emoji + README rewrite.

Phases 0–1 are the immediate next block. 2 can run on UCloud in parallel with 0/3. Nothing is
started until the order + the cleanup deletions are confirmed.

## Execution log (2026-07-28, owner-approved)

**Phase 0 (done):** copied the 4 keep-models local (`data/local_models/{5ep_baseline,
convnextv2l_teacher,distill_b0_T1,b0_control}`); the 0.9148 run stays local (`data/global/models/
20260716-154156`, never on UCloud). Trashed (recoverable): repo leftovers `compat-check /
demo-trainer / extend-test / ioprobe` (**kept `mini_metrics` + `mini_trainer`** per owner — useful
for phase 1); 6 smoke checkpoints; `ucloud_smoke`, `ucloud_smoke_preds`, `ucloud_preds_distill` (T4).
RESULTS.md gained a hand-maintained "lepinet package (UCloud) runs" table with a location column.
*Deferred:* trimming the ~20 old dev/030 bench/heads checkpoints and consolidating the flat
`ucloud_preds_*` siblings into `ucloud_preds/<run>/` (new runs already write there) — a careful
follow-up, not blocking.

**Phase 1 (launched):** flemming open-set test running (ConvNeXtV2-L, `--no-drop-unknown-species`,
`mini_metrics -t 0 0 0` next). Parquet built from `example_pred.csv` (`dev/048`); note the owner's
flag that some "OOD" species may be GBIF-id renames — to double-check later, and to optionally accept
full species names instead of GBIF keys (deferred). **Correction:** the `unreferenced` folder is
*not* an OOD set (owner) — dropped that idea from the ArcFace/OOD plan.

**Phase 2 (launched, parallel B200):** #6 real distillation (ConvNeXtV2-L→b0, T=1) + #9
convnext_large.dinov3 teacher, each with an afterok eval. One queue daemon advances the chains +
auto-extend. #7 (hierarchical/autoregressive heads) and #8 (ArcFace/OOD) need code first — next.
