# lepinet

**Hierarchical fine-grained image classification.** Given a photo, predict a label at every level
of a taxonomy at once — for the reference dataset, the *species*, its *genus*, and its *family* of
moth or butterfly — over ~12,000 species with a heavy long tail (half the species have fewer than
200 images). The package is generic in the number and names of the levels: any fine→coarse label
hierarchy works.

Two things make this hard, and shape the whole design:

- **Fine-grained.** Neighbouring species differ by subtle wing-pattern detail, so the model needs
  high-resolution local texture, not just global shape.
- **Long-tailed.** Most images belong to a few common species; thousands of rare species have a
  handful each. The headline metric is therefore **macro-F1** (every species weighted equally, so
  the tail counts), and the training rebalances toward rare classes.

The method: a shared image backbone feeds a **per-level cosine classification head** (one set of
L2-normalised class prototypes per taxonomic level), trained with square-root class oversampling.
On the reference Lepidoptera dataset this reaches **test species macro-F1 0.9148**. See
[The method](#the-method) for why each piece is there.

📖 **Docs:** [user guide](docs/user-guide.md) · [developer guide](docs/developer-guide.md) ·
[design journal](journal/2026-07-src-lepinet-baseline-port.md)

## Install

```bash
uv pip install -e .              # library + CLI
uv pip install -e ".[export]"    # + ONNX export (onnx, onnxruntime)
uv pip install -e ".[timm]"      # + timm backbones
uv sync --group dev              # + dev tooling (pytest, ruff, mkdocs)
```

> The training venv is hand-managed and reproducible from `pyproject.toml` + `uv.lock` (the
> `torch==2.12.1+cu130` build for the Blackwell GPU comes from the pinned PyTorch index). See the
> developer guide for how the dev environment is defined and how `dev/` scripts pull their extra
> dependencies.

## Quickstart

Config-driven (the source of truth) or programmatic — both do the same thing.

```bash
lepinet train   --config configs/20260716_heads_global_independent_muon_5ep_oversample.yaml
lepinet test    --model 'data/global/models/*oversample*/*.pt' \
                --parquet data/global/<meta>.parquet --img-dir data/global/images \
                --out-dir data/global/preds --test-set 0
lepinet predict --model model.pt photo.jpg --topk 5
lepinet export  --model model.pt --out-dir artifact/ --img-size 256
```

```python
from lepinet import TrainConfig, train, evaluate, predict, export_onnx
train(TrainConfig(parquet_path="...", img_dir="...", out_dir="...",
                  model_name="run1", model_arch_name="efficientnet_v2_s", oversample_power=0.5))
```

## What it does

| stage | what you get |
|---|---|
| **train** | the winning recipe: efficientnet_v2_s, cosine independent head, Muon + one-cycle, warmup, grad-clip, square-root oversampling, bf16 |
| **test** | native per-level macro-F1 / micro-accuracy report (+ `predictions.csv` in `mini_metrics` format for interop) |
| **predict** | single-image / folder inference with test-time augmentation |
| **export** | browser-ready ONNX (normalization baked in, raw logits) + `taxonomy.json` for the companion PWA |

## The method

Each design choice answers one of the two difficulties above.

- **Per-level cosine head.** The backbone produces one embedding; a small bottleneck projects it,
  then each taxonomic level has its own layer of L2-normalised class prototypes and classifies by
  *cosine similarity* (angle to the nearest prototype), not an unbounded dot product. Normalising
  both sides tightens intra-class / widens inter-class angles — the property that helps most on
  fine-grained classes and on the tail, where a plain linear layer over-fits the few examples it
  sees. The levels are independent heads on the shared embedding, so adding or renaming levels is
  just a longer list.
- **Square-root oversampling** (`oversample_power 0.5`). Sampling rare classes more often — but at
  the square root of the inverse frequency, not the full inverse — lifts tail recall without
  drowning the common classes. This is the single biggest lever on macro-F1 in the experiment
  ladder (+1.7 pt over no oversampling).
- **Muon (backbone) + AdamW (head), one-cycle** with a short warmup and gradient clipping — the
  optimiser/schedule combination that trains this head stably in a few epochs.
- **bf16** throughout: enough exponent range for the cosine head (and for margin losses like
  ArcFace, a planned extension) without the fp16 overflow that NaNs them.

Reference recipe: efficientnet_v2_s · 460→256 px · batch 64 · 5 epochs · light aug. The full ladder
of experiments that led to each choice — and the negative results — is in [`journal/`](journal/)
and [`RESULTS.md`](RESULTS.md).

> **Provenance.** This is a from-scratch **fastai-only** reimplementation of an earlier
> `mini_trainer`/`mini_metrics`-based pipeline; it reproduces that pipeline's best result with no
> dependency on it (a `mini_metrics`-format `predictions.csv` is still emitted for interop). That
> lineage explains some naming in the design journal but is not something a user needs to care
> about — the package stands alone.

## Repository layout

```
src/lepinet/     the package (see docs/developer-guide.md for the module map)
configs/         training / evaluation YAML configs
tests/           unit + end-to-end tests
docs/            user & developer guides (published to GitHub Pages)
dev/             the numbered lab-notebook — frozen experiment record; new work imports lepinet
journal/         the reasoning behind every experiment (READ FIRST: journal/README.md)
RESULTS.md       the results table (generated by dev/036_ledger.py)
```

`dev/` stays as the historical record; it is not packaged. New experiments `import lepinet`
instead of the numbered scripts.

## Development

```bash
pytest -q                        # unit + synthetic end-to-end tests (CPU, no data needed)
ruff check src/lepinet tests     # lint
mkdocs serve                     # preview the docs site
```

CI (GitHub Actions) runs lint + unit + a self-contained end-to-end test (train a dummy model on a
generated dataset, then eval / predict / export). Docs deploy to GitHub Pages on push to `main`.

## License

GPL. See [`LICENSE`](LICENSE).
