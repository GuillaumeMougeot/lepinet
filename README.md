# lepinet

**lepinet** identifies moths and butterflies from photographs: one model predicts species, genus
and family at once, over 12,041 species with a heavy long tail. It began as a comparison of
hierarchical prediction heads. That comparison was a null result. A model scoring 0.93 on its own
data scores ~0.70 on light-trap images, and in the field it constantly meets species it was never
trained on. So the subject became **reliable prediction under domain shift**: knowing when an image
is something new (open-set), backing off to genus or family when the species is uncertain
(abstention), and adapting cheaply to a new camera.

**Just want to identify moths?** The models are public and run with `onnxruntime` alone:
[`lepinet-bioclip2-vitl14`](https://huggingface.co/gmougeot/lepinet-bioclip2-vitl14) (P5,
answers most often), [`lepinet-dinov3-convnextl`](https://huggingface.co/gmougeot/lepinet-dinov3-convnextl)
(B8, names the species most often) and [`lepinet-effnetv2s`](https://huggingface.co/gmougeot/lepinet-effnetv2s) (small, fast).

## 1. Where things stand

- **Experiments behind the paper: finished.** About 80 lettered experiments, all resolved in
  [`RESULTS.md`](RESULTS.md).
- **Paper** ([`paper/DRAFT.md`](paper/DRAFT.md)): every result is in; it needs a scope decision, an
  introduction and verified citations. [`PLAN.md`](PLAN.md) is the road to submission.
- **Running:** the TreeOfLife-200M download (W1) for a follow-up study. The paper does not depend on it.

## 2. What to read

| time | read |
|---|---|
| **20 minutes** | this page, then [`RESULTS.md`](RESULTS.md) section 1 and "the ten rows that carry the paper" |
| **2 hours**, for the paper | [`paper/DRAFT.md`](paper/DRAFT.md): abstract, §1, §4.0, §4.15, §5, §6; then [`PLAN.md`](PLAN.md) |
| **to run or change code** | [`docs/user-guide.md`](docs/user-guide.md), [`docs/developer-guide.md`](docs/developer-guide.md), [`docs/design-decisions.md`](docs/design-decisions.md) |
| **look up** | [`docs/concepts.md`](docs/concepts.md) (vocabulary), [`journal/README.md`](journal/README.md) (every question) |

[`CLAUDE.md`](CLAUDE.md) is the AI agent's operating manual; humans can skip it.

## 3. The map

| part | what | where | state |
|---|---|---|---|
| **package** | the `lepinet` library: train, test, predict, export, bundle, distill | `src/`, `tests/`, `docs/` | stable |
| **research** | the experiments behind the paper | `dev/` (one script per experiment), `configs/`, `ucloud/` (cluster jobs), `journal/`, `paper/` | closed; paper being written |
| **release** | public models and the phone-app bridge | `dev/083_*` | done |
| **TreeOfLife** | the 70 M-image download for W1-W3 | `dev/082`, `dev/084`, `dev/085`, `dev/087` | running |

`archive/` holds what is no longer used; `journal/engineering/` holds side projects, cluster work and
incidents.

## 4. The models to compare against

| model | what it is | in-dist | probe | open-set AUROC |
|---|---|---|---|---|
| **cheap reference** | effnetv2_s, one species head + marginalisation, √-oversampling | 0.9135 | 0.6270 | 0.8990 |
| **B8** | 198 M, self-training on 2.6 % trap images, no oversampling | 0.9060 | 0.7798 | 0.9153 |
| **P5** | BioCLIP-2 fine-tuned + unfrozen adaptation | 0.9113 | 0.7757 | 0.9161 |
| best in-distribution | ConvNeXtV2-L, multi-head | **0.9316** | — | — |

*in-dist* = species macro-F1 on our held-out fold, over all species; *probe* = macro-F1 on
held-out light-trap nights. Compare only within a column, and check the noise floor (in-dist
~0.000; probe ~0.004, or ~0.012 for frozen-trunk stages). Reference config:
[`configs/20260729_ucloud_singlehead_species_effnetv2s.yaml`](configs/20260729_ucloud_singlehead_species_effnetv2s.yaml).

## 5. The method, briefly

A backbone feeds a **cosine classification head**; genus and family come from **marginalising** the
species posterior; rare species are **square-root oversampled** (better applied to the classifier
only under shift); **self-training** on unlabelled trap images is the largest robustness lever, and
adapting only the classifier recovers most of it. Why each default is what it is:
[`docs/design-decisions.md`](docs/design-decisions.md).

## 6. Install and run

```bash
uv pip install -e .              # library + CLI   (".[export]" for ONNX, ".[timm]" for timm backbones)
lepinet train   --config configs/20260729_ucloud_singlehead_species_effnetv2s.yaml
lepinet test    --model 'data/global/models/<run>/*.pt' --parquet <meta>.parquet \
                --img-dir data/global/images --out-dir data/global/preds --test-set 0
lepinet predict --model model.pt photo.jpg --topk 5
lepinet export  --model model.pt --out-dir artifact/ --img-size 256
pytest -q                        # CPU tests, no data needed
```

The training venv on the workstation is hand-managed: **never run `uv sync` on it**. Full CLI and
config reference: [`docs/user-guide.md`](docs/user-guide.md).

## 7. Conventions

- **IDs.** Experiments by letter-number ([`RESULTS.md`](RESULTS.md) section 2), July local runs by
  timestamp (`20260716-154156`).
- **The journal**: one file per question, dated when opened, prediction written before the result.
- **The metric** is species macro-F1 on the held-out fold (`set == '0'`) over **all** species, so
  the tail counts. Never filter the test fold.
- **`data/`** is machine-local and gitignored; a fresh clone has no runs.

GPL. See [`LICENSE`](LICENSE).
