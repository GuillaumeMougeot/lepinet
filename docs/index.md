# lepinet

**Hierarchical fine-grained image classification.** Predict a label at every level of a taxonomy
from one photo — species, genus and family for the reference moth/butterfly dataset (~12,000
species, heavy long tail), or any fine→coarse label hierarchy of your own.

The task is hard for two reasons that shape the design:

- **Fine-grained** — species differ by subtle local detail, so features must capture texture, not
  just shape.
- **Long-tailed** — a few species dominate the images, thousands are rare. The headline metric is
  **macro-F1** (all species weighted equally), and training oversamples the tail.

The method is a shared backbone with a **per-level cosine classification head** (L2-normalised
class prototypes per level) trained with square-root class oversampling; on the reference
Lepidoptera dataset it reaches **test species macro-F1 0.9148**. The
[developer guide](developer-guide.md) explains *why* each piece is there.

!!! note "Provenance"
    `lepinet` is a from-scratch **fastai-only** reimplementation of an earlier
    `mini_trainer`/`mini_metrics` pipeline, reproducing its best result without depending on it
    (it still emits a `mini_metrics`-format `predictions.csv` for interop). That lineage is a
    project-history detail, not something you need to use the package.

## What you get

- **Train** the winning recipe (efficientnet_v2_s, cosine independent head, Muon + one-cycle,
  square-root oversampling) from a config or from Python.
- **Evaluate** on a held-out fold with a native per-level macro-F1 / micro-accuracy report
  (plus `predictions.csv` in `mini_metrics` format for interoperability).
- **Predict** on an image or folder with test-time augmentation.
- **Export** to a browser-ready ONNX graph (normalization baked in, raw logits out) plus a
  `taxonomy.json` sidecar — the artifact the companion PWA consumes.

## Install

```bash
uv pip install -e .
uv pip install -e ".[export]"   # + ONNX export
```

## 30-second tour

```bash
lepinet train   --config configs/20260716_heads_global_independent_muon_5ep_oversample.yaml
lepinet test    --model 'runs/*/*.pt' --parquet meta.parquet --img-dir images --out-dir preds --test-set 0
lepinet predict --model model.pt photo.jpg --topk 5
lepinet export  --model model.pt --out-dir artifact/
```

See the **[user guide](user-guide.md)** for the full API and config reference, and the
**[developer guide](developer-guide.md)** for the architecture and the lessons encoded in the code.

---

Design decisions and the full execution log live in the repository's
[`journal/`](https://github.com/GuillaumeMougeot/lepinet/tree/main/journal).
