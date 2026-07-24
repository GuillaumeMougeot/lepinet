# lepinet

**Hierarchical image classification** (species / genus / family, or any fine→coarse label
hierarchy) — a clean, **fastai-only, `mini_trainer`-free** Python package.

It reproduces the project-best Lepidoptera baseline — **test species macro-F1 0.9148** — and is
generic in the number of hierarchy levels.

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
