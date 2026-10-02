# dev/ — experiment scripts

One script per experiment, numbered in the order the ideas came. They import the `lepinet` package
([`../src/lepinet/`](../src/lepinet/)) and register anything new through its registries
(`HEAD_REGISTRY`, `DOMAIN_AUG_REGISTRY`) instead of editing it. Each script's docstring says what it
measures and links its journal entry; the experiment IDs are resolved in
[`../RESULTS.md`](../RESULTS.md) section 2.

Scripts from before the package existed (`000`-`044`: the multilabel era, the `028`/`030`/`032`
trainer, the app-compression tools) are in [`../archive/dev/`](../archive/dev/), kept as a record.
`036_ledger.py` stayed: it still generates `RESULTS.md`.

| script | what it is | IDs |
|---|---|---|
| `036_ledger.py` | prints every local run's config delta and score; `--snapshot` writes `RESULTS.md` | phase 0 |
| `047_build_names.py` | the app's `names.json` from a bundle's taxonomy | release |
| `048`, `049` | eval parquets for the two trap datasets (`flemming_helsing`, `flemming`) | |
| `050_hierarchical_heads.py` | hierarchical, autoregressive and marginal-ArcFace heads, registered into `HEAD_REGISTRY` | A4, heads |
| `051_benchmark_run.py` | runs `lepinet train/test` with the dev heads registered | |
| `052_ood_score.py` | open-set scoring: AUROC of known vs novel species | A1, C2 |
| `053`, `054` | embedding geometry; 1-D score distributions for open-set | A1 |
| `055`-`057` | ArcFace margin search: range test and short grid, **both invalid** (journalled) | |
| `058_rank_abstention.py` | back off to genus/family: coverage and precision per rank | C1 |
| `059_hierarchical_ood.py` | open-set split by taxonomic distance (near/mid/far) | C3 |
| `060_doc_health.py` | **not an experiment**: documentation checks, also run in CI | |
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

**Conventions.** Never `uv sync` or `uv run` (the venv is hand-managed); use `.venv/bin/python`
from the repo root.
