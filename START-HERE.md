# Start here

**lepinet** identifies moths and butterflies from photographs: one model predicts species, genus
and family at once, over 12,041 species with a heavy long tail. It began as a comparison of
hierarchical prediction heads. That comparison was a null result. A model scoring 0.93 on its own
data scores ~0.70 on light-trap images, and in the field it constantly meets species it was never
trained on. So the subject became **reliable prediction under domain shift**: knowing when an image
is something new (open-set), backing off to genus or family when the species is uncertain
(abstention), and adapting cheaply to a new camera.

## 1. Where things stand (2026-10-02)

- **The experiments behind the paper are finished.** There are about 80 lettered experiments (A1
  to P5, plus K1 and W1-W3); every one the paper uses is closed. [`EXPERIMENTS.md`](EXPERIMENTS.md) lists
  them all with their question and result.
- **Two models are public** on
  [Hugging Face](https://huggingface.co/collections/gmougeot/lepinet-lepidoptera-identification-6abbc33d250430f8c67db428):
  P5 (fine-tuned BioCLIP-2, the recommended one) and B8 (our own 198 M model).
- **The paper** ([`paper/DRAFT.md`](paper/DRAFT.md)) contains every result, but reads as a
  complete record rather than a paper. It still needs a scope decision, an introduction, figures and
  verified citations. [`journal/PLAN.md`](journal/PLAN.md) has the road to submission.
- **Running:** the TreeOfLife-200M download (W1), for a follow-up study (W3). The paper does not
  depend on it.

## 2. What to read, in what order

**20 minutes: what the project is.**
[`README.md`](README.md) (the problem and the method), then this page, then the "ten rows that
carry the paper" at the top of [`EXPERIMENTS.md`](EXPERIMENTS.md).

**2 hours: to work on the paper.**
In [`paper/DRAFT.md`](paper/DRAFT.md): the abstract, §1 (contributions), §4.0 (the models), §4.15
(the synthesis), §5 and §6. Then [`journal/PLAN.md`](journal/PLAN.md). Then the journal entries
linked from the ten rows.

**To run or change code.** [`docs/user-guide.md`](docs/user-guide.md) →
[`docs/developer-guide.md`](docs/developer-guide.md) → [`src/lepinet/README.md`](src/lepinet/README.md)
→ [`docs/design-decisions.md`](docs/design-decisions.md) (why every default is what it is) →
[`dev/README.md`](dev/README.md) and [`ucloud/README.md`](ucloud/README.md).

**Look up when needed.** [`docs/concepts.md`](docs/concepts.md) explains the vocabulary (cosine
head, margin, AUROC, marginalisation). [`journal/README.md`](journal/README.md) indexes every
question ever asked. [`RESULTS.md`](RESULTS.md) holds the July local runs.

**Safe to skip.** [the July reframe](journal/2026-07-29-the-reframe-directions.md) (the strategy that led to the paper, now
superseded by it); the app entries of 19-25 July; infrastructure and incident entries
unless that thing breaks again. [`CLAUDE.md`](CLAUDE.md) is the AI agent's operating manual.

## 3. The map: four projects in one repository

| part | what it is | where | state |
|---|---|---|---|
| **package** | the `lepinet` library: train, test, predict, export, bundle, distill | `src/`, `tests/`, `docs/` | stable |
| **research** | the experiments behind the paper | `dev/` scripts, `configs/` (one YAML per run), `ucloud/` (one TOML per cluster job, ~5 per experiment), `journal/`, `paper/` | experiments closed; paper being written |
| **release** | public models and the phone-app bridge | `dev/083_*`, [release entry](journal/2026-09-29-public-hf-release.md) | done |
| **TreeOfLife** | the 70 M-image download for W1-W3 | `dev/082`, `dev/084`, `dev/085`, `ucloud/*tol*` | running; separate from the paper |

**History, not needed to work:** `archive/`, `bash/`, and `dev/000`-`dev/033` (scripts and notebooks
from before the package existed).

## 4. The models to compare against

| model | what it is | in-dist | probe | open-set AUROC |
|---|---|---|---|---|
| **cheap reference** | effnetv2_s, one species head + marginalisation, √-oversampling (`singlehead`) | 0.9135 | 0.6270 | 0.8990 |
| **B8** | 198 M, self-training on 2 % trap images, no oversampling | 0.9060 | 0.7798 | 0.9153 |
| **P5** (recommended) | BioCLIP-2 fine-tuned + unfrozen adaptation | 0.9113 | 0.7757 | 0.9161 |
| best in-distribution | ConvNeXtV2-L, multi-head | **0.9316** | — | — |

*in-dist* = species macro-F1 on our held-out fold; *probe* = macro-F1 on held-out light-trap nights.
Compare only within a column, and check the noise floor first (in-dist ~0.000; probe ~0.004
end-to-end, ~0.012 for frozen-trunk stages). Config of the reference:
[`configs/20260729_ucloud_singlehead_species_effnetv2s.yaml`](configs/20260729_ucloud_singlehead_species_effnetv2s.yaml).

## 5. What is established

One line each, with the paper section and the experiments that carry it. Negative results count.

### 5a. Science

| # | finding | paper | IDs |
|---|---|---|---|
| 1 | Hierarchy-aware heads do not help; one species head plus marginalisation matches or beats them. Coarse *supervision* (a loss on the marginals) still buys ~1.4 pt under shift. | §4.1 | singlehead, marginal, A4 |
| 2 | In-distribution accuracy saturates near 0.93 but falls ~23 pt on trap images, and the three axes rank models differently. | §4.2, §4.10 | flemming, B4 |
| 3 | **The spine:** interventions belong in the classifier, not the representation: rebalancing (cRT), domain adaptation (83 % from the classifier alone), the prototype matrix (centroids for 0.29 pt). | §4.15 | L4, T2, H1 |
| 4 | Unlabelled trap images are the largest lever: self-training gives +7.94 probe at a 2 % share and beats 12,230 real labels; above 2 % adaptation becomes memorisation. | §4.11 | B3rep5x, T1 |
| 5 | Long-tail reweighting trades shift robustness for accuracy, monotonically; cRT removes the trade; capping training at 1,000 images per species helps shift. | §4.13 | L0-L4, L7 |
| 6 | The best open-set scoring rule changes with model scale (6-7.6 pt); one rule for all models produced a false ranking. | §4.9 | E2 |
| 7 | The angular margin relocates open-set signal rather than creating it: +0.78 AUROC best-vs-best (an earlier "31 pt" compared rules, and is retracted). | §4.3 | A1, C3r |
| 8 | Novelty detection improves with taxonomic distance (near < mid < far), and not because unseen taxa are rare. | §4.4 | C3, C3b |
| 9 | Abstention under shift is expensive, and two models with equal accuracy can differ by 17 pt in useful answers: the difference is calibration. | §4.6a | O1 |
| 10 | Two thirds of our test fold is inside BioCLIP-2's training data by GBIF occurrence id; a frozen probe understates that model by 7 pt; fine-tuned it is the better trunk, and our recipe closes the gap. | §4.14 | P1-P5 |
| 11 | A margin damages marginalisation more than classification, through calibration; replicated at 10x scale. | §4.7 | A1, A2, A4 |
| 12 | Augmentation that mimics trap conditions closes only 17 % of the gap. | §4.8 | B1 |
| 13 | Measurement: macro-F1 does not decompose over subsets; noise floors depend on the training regime (identical runs 3.74 pt apart); give every arm its own best configuration. | §3.2, §5 | A5, G3b, R5b |
| 14 | A regional checklist helps the average observation (+5 pt accuracy) and hurts per-species macro-F1 when the scored set is most of the checklist. *(Not in the paper.)* | — | K1 |

### 5b. Engineering (argued in [`docs/design-decisions.md`](docs/design-decisions.md))

| # | finding |
|---|---|
| 15 | The baseline climbed 0.8297 → 0.9152 through the schedule, the sampler and lighter augmentation, not the optimiser. |
| 16 | Margin heads need bf16 (fp16 overflows); fp16 is fine elsewhere. |
| 17 | Audit the evaluation set before believing a metric: filtering rare species out of the test fold inflated a macro average by 3 pt. |
| 18 | Framework attributes can lie: fastai's reported `num_workers` caused a ~900x slowdown first blamed on hardware. |
| 19 | Deployment: int8 cannot run in ORT-Web, fp16 ships; the classifier head is half a small model, so its width is the size knob. |
| 20 | 9.3 % of TreeOfLife "species" keys are not species; audit a dataset's key composition before quoting a class count. |
| 21 | Derive stored image size from the training transform: the input crops the short side, so store the short side. |
| 22 | UCloud jobs see the host's 256 cores; size thread pools from the cgroup quota, and cost CPU jobs before submitting them. |

## 6. Conventions

- **IDs.** Experiments are cited by letter-number (see [`EXPERIMENTS.md`](EXPERIMENTS.md)), July
  local runs by timestamp (`20260716-154156`). Letters have been reused; the registry says which is
  which.
- **The journal** is one file per question, dated by when the question was opened, with the
  prediction written before the result. `UPPERCASE.md` files (`PLAN`, `README`) are
  living documents.
- **The metric** is species macro-F1 on the held-out fold (`set == '0'`) over **all** species, so
  the long tail counts. Never filter the test fold.
- **`data/`** is machine-local and gitignored; a fresh clone has no runs.
