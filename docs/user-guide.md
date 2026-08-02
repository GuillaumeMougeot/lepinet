# lepinet — user guide

`lepinet` trains, evaluates, and deploys hierarchical image classifiers (species / genus /
family, or any fine→coarse label hierarchy). It reproduces the project-best Lepidoptera baseline —
**test species macro-F1 0.9148** — from a clean, fastai-only package with no `mini_trainer`
dependency.

## Install

```bash
# into the project's hand-managed venv (never `uv sync` — see the developer guide)
uv pip install -e .

# optional extras
uv pip install -e ".[export]"   # ONNX export (onnx + onnxruntime)
uv pip install -e ".[timm]"     # timm backbones (fastvit / repvit / mobilenetv4 / ...)
uv pip install -e ".[dev]"      # pytest + ruff
```

Everything works from a config file **or** from Python **or** from the `lepinet` CLI.

## Quickstart (CLI)

```bash
# train (writes <out_dir>/<timestamp>-<desc>/<model_name>.pt)
lepinet train -c configs/20260716_heads_global_independent_muon_5ep_oversample.yaml

# evaluate on the held-out fold (set '0' is the global test fold)
lepinet test -m 'data/global/models/*-oversample-effnetv2s/*.pt' \
             -p data/global/<parquet>.parquet -i data/global/images \
             -o data/global/preds --test-set 0

# predict on an image or a folder (TTA on by default)
lepinet predict -m model.pt path/to/moth.jpg --topk 5

# export to ONNX + taxonomy.json (dynamo-free, with an ORT parity check)
lepinet export -m model.pt -o artifact/ --img-size 256
```

## Quickstart (Python)

```python
from lepinet import TrainConfig, train, evaluate, predict, export_onnx, load_config

# train from a config...
cfg, _raw = load_config("configs/....yaml")
train(cfg)

# ...or build the config in code
train(TrainConfig(parquet_path="...", img_dir="...", out_dir="...",
                  model_name="run1", model_arch_name="efficientnet_v2_s",
                  oversample_power=0.5))

# evaluate
evaluate(model_path="run1.pt", parquet_path="...", img_dir="...", out_dir="preds", test_set="0")

# predict (returns ImagePrediction objects; .as_dict() for JSON)
preds = predict("run1.pt", "photo.jpg", topk=5, tta=True)
print(preds[0].levels[0].top)           # (top species key, confidence)

# export
export_onnx("run1.pt", "artifact/", img_size=256)
```

## The baseline recipe (what to compare against)

The winning configuration (held fixed unless you are experimenting):

| knob | value |
|---|---|
| backbone | `efficientnet_v2_s` |
| head | `independent` (cosine, per-level prototypes) |
| optimizer | `muon` (backbone) + AdamW (head) |
| schedule | `one_cycle`, `warmup_epochs: 0.5` |
| grad clip | `5.0` |
| oversampling | `oversample_power: 0.5` (square-root) |
| precision | `bf16` (default; fp16 also works) |
| images | `aug_img_size: 460` → `img_size: 256`, `batch_size: 64` |
| epochs | 5 |
| aug | light: no warp, no lighting, `flip_vert: true`, rotate 15°, zoom 1.1 |

## Beyond training: the rest of the CLI

```bash
lepinet distill -c student.yaml --teacher teacher.pt   # KD from a teacher (use T=1, see below)
lepinet test  -m model.pt ... --marginal               # derive genus/family from the species head
lepinet test  -m model.pt ... --tta --limit 20000      # 4-flip TTA; --limit for a quick probe
lepinet bundle -m model.pt -o bundles/x --publish-hf user/repo   # deployable ONNX bundle + upload
```

Flags worth knowing, and why they exist:

| flag | when you need it |
|---|---|
| `--marginal` / `--eval-levels` | Score coarser ranks from the species posterior. **Required** for a single-head model (it has no genus/family head); optional otherwise, to compare marginals against trained coarse heads. |
| `--skip-missing` (default on) | The parquet is a catalogue; image mirrors are often incomplete. Without this one absent file kills a multi-hour eval. |
| `--num-workers` | Set it explicitly. On a network mount this is the single biggest throughput lever (measured 1 → 898 img/s). |
| `--tta` | 4-flip average. Worth ~+0.3 pp for 4× the cost — usually not worth it. |
| `--no-drop-unknown-species` | Open-set evaluation: keep out-of-vocabulary species instead of dropping them. |

### Distillation

`distill_temperature: 1.0` is the default **on purpose**: the textbook T=4 *hurt* this head
(0.8546 vs 0.8692 from-scratch) because the cosine z-score logits are already near unit scale, so
dividing by 4 flattens the target toward uniform. At T=1 the student beats from-scratch (0.8786).

## Config reference (the `train:` block)

Field names match `lepinet.config.TrainConfig`. Common ones:

- `parquet_path`, `img_dir`, `out_dir`, `model_name`, `model_arch_name`
- `fold` (validation set id), `min_img_per_spc`, `family_filter`
- `levels` — the hierarchy, fine→coarse (default `[speciesKey, genusKey, familyKey]`)
- `head` (`independent`), `hidden` (bottleneck width; `true` = backbone width, or an int)
- `nb_epochs`, `batch_size`, `base_lr`, `optimizer`, `schedule`, `warmup_epochs`, `grad_clip`
- `precision` (`bf16` / `fp16`), `oversample_power`, `aug_kwargs`

Two options are **rejected on purpose** (they were measured to lose): `logit_adjust_tau` and
`class_reg_strength`. See `journal/2026-07-17-does-longtail-help.md`.

## Outputs

- **Training** → `<out_dir>/<timestamp>-<desc>/`: `<model_name>.pt` (self-contained checkpoint:
  weights + vocabs + hierarchy), per-epoch `.csv`, per-epoch `models/*.pth`.
- **Evaluation** → `<out_dir>/<model_name>/<eval_name>/`: `predictions.csv` (in `mini_metrics`
  long input format, for interoperability), `metrics.json` (native per-level macro-F1 +
  micro-accuracy), `combinations.csv` (the hierarchy used).
- **Export** → `<out_dir>/`: `model.onnx` (normalization baked in, raw logits out),
  `taxonomy.json` (per-level vocabs in head-index order + parent arrays), `MANIFEST.json`.

## Notes

- **Local GPU workaround:** if the box shows an `nvmlInit`/driver-version-mismatch error during
  training, the NVIDIA driver needs a reload/reboot. `torch.cuda` basic ops can still pass while
  training's allocator fails — that is the same problem.
- **ONNX exporter:** `lepinet` uses the legacy TorchScript exporter (`dynamo=False`) by default;
  it is reliable for this graph. It is deprecated in newer torch, so `--dynamo` is available if
  you want the newer exporter.
- **GBIF labels:** vocab entries are GBIF taxon keys; a species page is
  `https://www.gbif.org/species/<key>`.

### A release bundle (with calibration)

`lepinet bundle` alone emits the model and taxonomy. A *release* should also carry display names and
calibrated thresholds, so the app can grey a name on a defensible claim instead of on `p > 0.5`:

```bash
lepinet bundle -m 'data/models/<run>/*.pt' -o bundles/v2 \
    --parquet data/global/metadata.parquet \
    --img-dir data/global/images \
    --calibrate --target-precision 0.95
```

- `--parquet` alone adds **`names.json`** (display names aligned to the vocab order). Cheap.
- `--calibrate` additionally fits a **temperature per level** on the validation fold and a
  **precision-targeted threshold**, then *verifies both on the held-out test fold* and writes the
  achieved precision alongside. Costs one inference pass per fold.

Temperature scaling changes no prediction — accuracy is untouched by construction — it only makes
the number attached to the prediction honest. See [concepts](concepts.md#12-evaluation-vocabulary).
