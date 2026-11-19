# dev/ — experiment scripts

One script per experiment, numbered in the order the ideas came. They import the `lepinet` package
([`../src/lepinet/`](../src/lepinet/)) and register anything new through its registries
(`HEAD_REGISTRY`, `DOMAIN_AUG_REGISTRY`) instead of editing it. Each script's docstring says what it
measures and links its journal entry; the experiment IDs are resolved in
[`../RESULTS.md`](../RESULTS.md) section 2.

Scripts from before the package existed (`000`-`044`: the multilabel era, the `028`/`030`/`032`
trainer and its run ledger, the app-compression tools) are in [`../archive/dev/`](../archive/dev/),
kept as a record. Long-term tooling does not live here: the documentation check is
`tests/test_doc_health.py`, the CPU budget guard is `ucloud/budget_check.py`.

| script | what it is | IDs |
|---|---|---|
| `047_build_names.py` | the app's `names.json` from a bundle's taxonomy | release |
| `048`, `049` | eval parquets for the two trap datasets (`flemming_helsing`, `flemming`) | |
| `050_hierarchical_heads.py` | hierarchical, autoregressive and marginal-ArcFace heads, registered into `HEAD_REGISTRY` | A4, heads |
| `051_benchmark_run.py` | runs `lepinet train/test` with the dev heads registered | |
| `052_ood_score.py` | open-set scoring: AUROC of known vs novel species | A1, C2 |
| `053`, `054` | embedding geometry; 1-D score distributions for open-set | A1 |
| `055`-`057` | ArcFace margin search: range test and short grid, **both invalid** (journalled) | |
| `058_rank_abstention.py` | back off to genus/family: coverage and precision per rank | C1 |
| `059_hierarchical_ood.py` | open-set split by taxonomic distance (near/mid/far) | C3 |
| `061_ood_scoring_rules.py` | five open-set scoring rules in one pass | E2, O1 |
| `063_balanced_softmax.py` | balanced softmax as a callback | L1, L2 |
| `064_flemming_groups.py` | leak-free (trap, night) splits: `adapt`, `probe`, `probe-HO` | B3 |
| `065`, `066` | pseudo-label the trap images; merge them into training | B3, R |
| `067_label_budget.py` | real-labelled trap subsets | T1 |
| `068_centroid_retrieval.py` | class centroids vs the trained matrix; its rank | H1 |
| `069_sampled_softmax.py` | sampled softmax as a callback | H2, H3 |
| `070_two_stage.py` | cRT and classifier-only adaptation from a checkpoint | L4, T2, F, G |
| `071_clamp_rate.py` | how often the z-score transform clamps | |
| `072_holdout_common.py` | withhold common taxa for the novelty test | C3b |
| `073_proxy_free.py` | head with EMA centroids and no matrix | H4 |
| `074_figures.py` | paper figures | paper |
| `075_pretrained_trunk.py` | BioCLIP-2 as the trunk (frozen or fine-tuned) | P |
| `076`, `077`, `079` | TreeOfLife overlap with our data; BioCLIP-2 feature probe; ToL size under our policy | P, W1 |
| `078_head_cap_sweep.py` | cap training images per species | L7 |
| `081_restricted_label_set.py` | restrict the label space to a regional checklist | K1 |
| `082_tol_crawler.py` | the TreeOfLife-200M downloader | W1 |
| `083_hf_release.py` (+ `083_hf_release_files/`) | public Hugging Face release | release |
| `084_tol_globallepi_fill.py` | complete ToL Lepidoptera from our own download | W1 |
| `085_tol_audit.py` | duplicates and test-fold leakage in the ToL subset | W1 |
| `086_deployment_gap.py` | B8 / P5 / B3rep5x under the 95 %-precision back-off policy | O3 |
| `087_tol_upload.py` | pack the ToL images into 2 GB tar shards and upload them to UCloud | W1 |
| `088_trap_split_twins.py` | do individual moths leak across the adapt/probe split? pixel-level twin scan | B3, T1 |
| `089_trap_holdout.py` | self-training with a trap or site held out: fold data, job specs, pooled scoring | X1, X2 |
| `090_tol_share.py` | publish the ToL crawl as `/data/datasets/tol256` (one metadata parquet, exact species folders) | W1 |
| `091_paper_reader.py` (+ `.html`) | Paper Desk: cuts `paper/DRAFT.md` into reading items by priority, one HTML page; reviews come back through the artifact database | paper |
| `092_metric_conventions.py` | one predictions file scored under each macro-F1 convention (ours, mini_metrics with and without threshold) | paper |
| `093_rescore_mini_metrics.py` | re-score prediction folders in mini_metrics' current convention (zero and calibrated) next to ours; checkpoints per run in `data/rescore/`, progress in `data/rescore/progress.md` | paper |
| `094_merge_desk.py` (+ `.html`) | Merge Desk: the LaTeX pieces of `paper/merge/` as ordered insertion steps into the Overleaf draft; mirrors pasted pieces into the local copy | paper |
| `095_trap_quality_backoff.py` | the release back-off cascade broken down by crop sharpness and size, on held-out trap nights | N1 |
| `096_trap_openset.py` | open-set detection on all trap crops by stratum (near, mid, out-of-vocabulary), per species, and what the back-off policy answers on unseen species | N2, N3 |
| `097_backoff_policies.py` | back-off policies compared (greedy per-rank, lower genus target, expected utility, cost tuned to 95 %), any number of ranks; Pl@ntNet drops in with `--plantnet` | Q1 |
| `098_nemo_on_traps.py` | Nemo (mini_trainer's MAMBO-v3 release) on our reconciled trap export, full vocabulary and north_europe list, with and without TTA, in our long format for `dev/093` | VN |
| `099_openset_c3b.py` | open-set AUROC on C3b's withheld taxa (near, mid, far) from mini_trainer's per-image summaries (`dev/lepinet/openset.py`), dev/061's five rules at every rank, degenerate rules flagged | campaign open-set block |
| `100_campaign_autoscore.py` | cron scorer for the campaign: every finished output in mini_trainer's `tmp-runs/eval` (test fold, traps, open-set) scored once into `data/campaign/scores.csv` and `summary.md`, retried three times on failure | campaign |
| `101_colxr_relabel.py` | one command (`apply --upload`) that moves the Lepidoptera labels to Catalogue of Life XR from frozen tables in `dev/colxr/`: a new parquet (every row kept), trap labels, regional lists, report; writes new files only. `prepare` rebuilt the tables. Plan row TAX, [the labels](../journal/2026-11-03-the-training-labels-inherit-gbif-backbone-errors.md) |
| `102_tol_fix_ranks.py` | undo the ToL catalog's rank shift in `/data/datasets/tol256` (bare-epithet folders -> `Genus_epithet`, ranks realigned, `taxon_rank`), re-fetch what the corrected keys select (`delta`, `add`), journalled hard-link apply, `colxr` moves the Lepidoptera names to COL XR on dev/101's tables | TF |
| `103_tol_freeze.py` | the ToL freeze: COL XR lookups for names outside dev/101 (`names`), what the crawl misses by host and reason (`shortfall`), full decode (`decode`), benchmark photographs by difference hash (`bench`, a registry), and the single dry run (`plan`: names, rank fills, junk kingdoms, floor, held-out taxa, benchmark overlap, `split` by observation) that `dev/102 apply` applies; `sizes` (width/height of rows recovered from disk), `train` (`train.parquet` with `image_path`, `class_spec.json`) and `shards` (mini_trainer shards v1, `--pilot` 50/5/5 per species) | TF, W3 |
| `104_tol_shard_upload.py` | uploads a shard set to UCloud as `dev/103 shards` writes it: size-verified batches, a ledger, nothing deleted remotely and only its own partial files (a dropped upload) overwritten, `--free` drops each verified local `.tar`, manifest last | TF, W3 |
| `105_onestop_plan.py` | OneStop test set: GBIF images of the OneStop invasive insects observed after 2025-04-26 (after Nemo's and MILA's training data), queried through COL XR, up to 2,000 per species, hosts that refuse crawlers skipped; writes a dev/082 manifest | [OneStop](../journal/engineering/2026-11-11-onestop-nemo-against-mila-on-invasive-species.md) |
| `106_onestop_eval.py` | Nemo, MILA's UK & Denmark classifier and Otto on that set (`predict`), scored per Living Lab list with pinned mini_metrics (`score`: macro accuracy, calibrated macro-F1) and the OneStop figure (`figure`) | same |
| `107_population_probes.py` | CPU checks on saved predictions and logits: the test fold split by observer (`observer`), which trap errors tail weighting adds (`prior`), novelty read per group of crops (`ghosts`), a label-free presence filter (`presence`), a site prior estimated by EM (`siteprior`; `siteprior2` and `siteprior3` add the regional list, TTA views, temperature, per-site estimates and any folder of saved logits), how taxonomic the prototypes are (`geometry`; `congeners` and `severity` for the campaign's heads), the observer split on every run (`observers`); and for self-training against a prior: the campaign's calibrated macro-F1 under each prior rule (`calibrated`), pairwise discrimination that no prior can move (`discrimination`), recall on the test fold against the probe (`invariance`), crop-by-crop overlap of the repairs (`overlap`), each site as a new site (`newsite`); and for the first ToL model, its test-fold gap to VD by group of species (`tolgap`) and frozen encoders on moths by nearest class mean (`frozen_index`, `frozen`), how well confidence ranks right above wrong trap answers (`confidence`), accuracy and calibrated macro-F1 by crop-size quintile (`cropsize`) | Z, [routes](../journal/2026-11-13-beyond-scaling-what-would-make-a-better-biodiversity-backbone.md) |
| `108_lr_finder.py` | learning-rate range test (fastai `lr_find`) on mini_trainer's own training step, local GPU, gradient accumulation to the recipe's batch; recipes for Global Lepidoptera, Pl@ntNet-300K (checked against measured rates) and ToL (sampled rows, full 190 k-species head); `data/lr_finder/` (RESULTS LF) |
| `109_tol_eval.py` | evaluation of a ToL checkpoint (tol256-v1): ToL species to lepinet's legacy keys through dev/101's tables (`map`, `classlist`, `relabel`, so `mt_hpredict` output goes through `score.py` and `flemming.py score`), data indexes (`index`: test fold, trap crops, held-out taxa, Rare Species), embeddings and per-rank argmax through mini_trainer's predict (`embed`, GPU), held-out placement and few-shot (`placement`), Rare Species with BioCLIP-2's SimpleShot (`rare`), Z7 on the species prototypes (`congeners`), frozen BioCLIP-2 and raw DINOv3 baselines (`embed --model`), open set on the held-out taxa against test-split images (`index --set known`, `openset`), an E1-E2 table (`report`) | W3 |
| `110_species_floor.py` | what lifting the 50-image species floor would add to Global Lepidoptera under a genus or family rule, and images per family with no floor (journal 2026-11-17) |
| `111_tol_species_floor.py` | the same question for tol256-v1: below-floor species a genus or family rule would add, on disk (with the screens the freeze skipped for them) and in the TreeOfLife-200M catalog, and images per family with no floor; result in the docstring (journal 2026-11-17) |
| `112_embedding_structure.py` | is the ToL embedding taxonomically structured? Species means of the saved TL1/Z21 embeddings (held-out taxa, Rare Species) for ToL, BioCLIP-2, raw DINOv3, TL4, VD: nearest-species genus/family rate, cosine by shared rank, rank correlation with tree distance (RESULTS TL7, journal 2026-11-19) |
| `113_clamp_floor.py` | the cosine head's clamp floor: floor counts on a checkpoint (`stats`), the gradient a floored target gets with the current and fixed head (`grad`), and the explanatory figure (`figure`, journal/figures) (journal 2026-11-19, RESULTS CF) |

**Conventions.** Never `uv sync` or `uv run` (the venv is hand-managed); use `.venv/bin/python`
from the repo root.
