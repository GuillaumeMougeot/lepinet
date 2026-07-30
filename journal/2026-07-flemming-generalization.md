# How well does the best model generalise to an external dataset (flemming_helsing)?

**Status:** RESOLVED (first pass, 2026-07-28). The first test of a lepinet model on a **truly
external** dataset — not a held-out fold of the training distribution. Answers "does the 0.9316
in-distribution number survive a different data source?" Short answer: **no, it drops hard — that is
the honest generalisation gap, and it's the case for open-set/OOD work.**

## Setup

- **Model:** the ConvNeXtV2-L teacher (in-distribution fold-0 species macro-F1 **0.9316**).
- **Data:** `flemming_helsing/resized/valid/referenced` — **58,640 images, 522 species** (a different
  capture source from the GBIF training images). Labels parquet built from the dataset's own
  `example_pred.csv` (already mini_metrics long format), `dev/048_flemming_parquet.py`.
- **Open-set:** 17 of 522 species (**13.7 % of images**) are OOD — not in the model's 12,041-species
  vocab. Evaluated with `--no-drop-unknown-species` so OOD rows are kept (`known_label=False`).
- Metric: `mini_metrics -t 0 0 0` (threshold 0 on all three levels), plus known-only (`-K`).

## Result

| level | macro-F1 (all, incl OOD) | macro-F1 (known-only) | micro-acc | vocab-coverage |
|---|---|---|---|---|
| **species** | **67.2 %** | 69.9 % | 65.5 % | 86.3 % |
| genus | 76.4 % | 76.4 % | 78.0 % | 100 % |
| family | 80.9 % | 80.9 % | 94.9 % | 100 % |

(The package's native `macro_f1` reports 0.695 for species — it averages only over in-vocab species
present in truth, i.e. it *excludes* OOD species, matching mini_metrics' `-K` view. mini_metrics'
default includes OOD species as label-classes with F1=0, hence 67.2 % < 69.9 %. Both are correct,
different questions.)

## Interpretation

- **~23 pp generalisation gap** (in-distribution species 0.93 → external 0.67–0.70). This is the real
  story: strong on the training distribution, much weaker on a new data source. Expected — different
  cameras/backgrounds/lighting/geography — but the *size* is the useful, sobering number.
- **Coarser levels are far more robust.** Family micro-acc is **94.9 %** even though species micro-acc
  is 65.5 %: the model usually gets the broad group right when it misses the species. `theilU` ≈ 95 %
  at species means the errors are *structured* (predictions still correlate with truth), not random —
  consistent with confusing near-relatives, which the hierarchy absorbs.
- **OOD costs ~2.7 pp** on all-images species macro-F1 (67.2 vs 69.9 known-only). Modest, and some of
  those 17 "OOD" species are likely **GBIF-id renames** (owner's flag: GBIF's key convention changed),
  not truly novel taxa — reconciling ids would recover part of it. To do: (a) map flemming GBIF keys
  through GBIF's synonym/id history before calling them OOD; (b) optionally accept full species names
  instead of keys (owner has such a dataset) — deferred.

## What this argues for (feeds the plan)

1. **Open-set / OOD handling is not optional** for real deployment — 13.7 % of a real dataset was
   out-of-vocab. This is the concrete motivation for the **ArcFace/OOD** direction (#8): an embedding
   that flags "unknown species" (low max-cosine) and degrades gracefully to genus/family.
2. **Domain robustness is the real frontier**, not another point of in-distribution macro-F1. Levers:
   train on more diverse sources, stronger/**domain-style** augmentation, and — since the *teacher*
   generalises this way — the *distilled student* can't exceed it, so improving the teacher's
   robustness matters more than shrinking the student.
3. **The eval pipeline works end-to-end on an external, folder-structured, partly-OOD dataset** — the
   `--no-drop-unknown-species` path + the parquet-from-`example_pred` recipe are reusable for the
   next external set.

Predictions + tables: `data/ucloud_preds/flemming-cnxv2l/`. Companion: [[2026-07-bigger-everything]]
(the teacher), [[2026-07-landscape-and-plan]] (the plan this feeds).

## Follow-ups launched (2026-07-28)

- **The "names" flemming dataset** (`data/flemming`, 47,905 images / 486 species, folders named by
  species *name* with `name2id.csv` GBIF-API-reconciled keys). **OOD is only 6 species / 234 images
  (0.5%)** vs flemming_helsing's 13.7 % — **confirms the owner's hypothesis** that most of
  flemming_helsing's "OOD" was GBIF-id mismatch, not novel taxa. Better id reconciliation ⇒ a fairer
  test. Eval launched with the ConvNeXtV2-L teacher + **TTA**. (`dev/049` builds the parquet with an
  explicit `image_path=<name>/<file>` column, since folders aren't speciesKey-named; `test.evaluate`
  now uses a supplied `image_path` column when present.) Result pending.
- **TTA on `test`**: added `--tta` (4-flip softmax average, the same TTA `predict` uses) — `test` was
  single-pass before. Re-ran flemming_helsing with TTA on to quantify the lift. Result pending.
- **UCloud upload 502**: `ucloud files upload` *and* `ucloud sync push` both intermittently 502 on
  bulk upload (server gateway timeout) — it even fails the job-`[sync]` step sometimes. Workaround:
  `sync push` is **incremental** (skips uploaded files by size+mtime), so a retry loop converges;
  job submits succeed on retry. Not fixed at source (owner flagged an open ucloud-api issue) — using
  the retry workaround for now; the ucloud-api fix (chunked upload + retry) is a separate task.
