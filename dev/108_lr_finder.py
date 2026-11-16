"""Learning-rate range test (fastai's `lr_find`) for mini_trainer recipes, run on the local GPU.

Owner (2026-11-16): choose the ToL run's learning rate with an lr finder, after checking that it finds
values close to the ones measured on Global Lepidoptera (DINOv3 ConvNeXt-L, Muon 8e-4, VD) and on
Pl@ntNet-300K (same trunk, 2e-4 better than 8e-4 by 4-8 points, VD/VP); default to 8e-4 if it is not
credible. See journal/2026-10-01-does-seeing-the-whole-tree-teach-a-model-what-it-does-not-know.md.

Method (Smith 2015; fastai `lr_find`): from the recipe's initial state, raise the learning rate
exponentially over `--steps` optimizer steps from `--start` to `--end`, all parameter groups together (their
ratios kept), and record the training loss of every step. Stop when the smoothed loss exceeds 4x its minimum
or turns non-finite. Suggestions, as fastai: `minimum/10` (lr at the smoothed minimum, divided by 10),
`steep` (steepest descent of the smoothed loss against log lr) and `valley` (fastai 2.7's rule). The training
step is mini_trainer's own (`trainer.train_one_epoch`: augmentation, preprocess, autocast, per-level loss
summed, GradScaler, clip 5, optimizer), with gradient accumulation so that `--batch` can exceed what one
RTX 5090 holds; ConvNeXt has no batch norm, so accumulation is exact.

    PYTHONPATH=$MT .venv/bin/python dev/108_lr_finder.py --dataset plantnet --out data/lr_finder
    PYTHONPATH=$MT .venv/bin/python dev/108_lr_finder.py --dataset lepi --out data/lr_finder
    PYTHONPATH=$MT .venv/bin/python dev/108_lr_finder.py --dataset tol --out data/lr_finder
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from torch.amp import autocast

MT = Path(os.environ.get("MT", "/home/au761367/codes/mini_trainer/.worktrees/lepinet-integration"))
TRUNK = "timm:convnext_large.dinov3_lvd1689m"
GLOBAL = "data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"
TOL = Path("/data/datasets/tol256")

# The recipes as they were trained (campaign plans c-dinov3-grid, pp-dinov3 / pp-dinov3-lr) and as planned for ToL.
RECIPES = {
    "lepi": {"base": "lepinet-BU.yaml", "input": GLOBAL, "set": {
        "size": 320, "augmentation_builder_kwargs.size": 320, "model_builder_kwargs.read_size": 384,
        "dataloader_builder_kwargs.read_size": 384}, "batch": 96, "known": {"8e-4": "trained (VD, OT4)"}},
    "plantnet": {"base": "lepinet-BU.yaml", "input": "/data/au761367/plantnet300k/mt/plantnet300k.parquet", "set": {
        "size": 320, "augmentation_builder_kwargs.size": 320, "model_builder_kwargs.read_size": 384,
        "dataloader_builder_kwargs.read_size": 384, "spec_model_dataloader_kwargs.levels": 5,
        "criterion_builder_kwargs.weights": [1, 1, 1, 1, 1]}, "batch": 96,
        "known": {"2e-4": "best measured", "8e-4": "4-8 points worse"}},
    "tol": {"base": "lepinet-BU.yaml", "input": None, "set": {
        "size": 224, "augmentation_builder_kwargs.size": 224, "model_builder_kwargs.read_size": 256,
        "dataloader_builder_kwargs.read_size": 256, "spec_model_dataloader_kwargs.levels": 7,
        "spec_model_dataloader_kwargs.min_count": 0, "dataloader_builder_kwargs.class_sampling_power": 0.0,
        "criterion_builder_kwargs.weights": [1] * 7}, "batch": 768, "known": {}},
}
COMMON = {"model_builder_kwargs.model_type": TRUNK, "optimizer_builder_kwargs.optimizer_cls": "mini_trainer.training.MuonAuxAdamW",
          "optimizer_builder_kwargs.head_muon": False, "seed": 42,
          # the campaign's pinned settings (.agents/local/campaign/build.py PINNED): every species, no oversampling
          "spec_model_dataloader_kwargs.min_count": 0, "dataloader_builder_kwargs.class_sampling_power": 0.0}


def put(cfg: dict, dotted: str, value):
    d = cfg
    *path, last = dotted.split(".")
    for k in path:
        d = d.setdefault(k, {})
    d[last] = value


def tol_subset(out: Path, rows: int, seed: int = 0) -> tuple[str, str]:
    """A random sample of ToL training rows as a parquet mini_trainer can read (images/<speciesKey>/<filename>),
    with the full 190,199-species class specification (the head must have its real size)."""
    d = out / "tol_subset"
    pq_path = d / "train.parquet"
    if not pq_path.exists():
        d.mkdir(parents=True, exist_ok=True)
        import pyarrow as pa
        import pyarrow.compute as pc
        f = pq.ParquetFile(TOL / "train/train.parquet")  # streamed by row group: the full table is ~60.9 M rows
        cols = ["filename", "set", "speciesKey", "genusKey", "familyKey", "orderKey", "classKey", "phylumKey", "kingdomKey"]
        frac = rows / 54_818_719  # training rows (set 2), train_summary.json
        rng = np.random.default_rng(seed)
        parts = []
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=cols)
            t = t.filter(pc.equal(t.column("set"), "2"))
            keep = rng.random(t.num_rows) < frac
            parts.append(t.filter(pa.array(keep)))
        t = pa.concat_tables(parts)
        pq.write_table(t, pq_path)
        img = d / "images"
        img.mkdir(exist_ok=True)
        for key in set(t.column("speciesKey").to_pylist()):  # each class folder points at the image root
            link = img / key
            if not link.exists():
                link.symlink_to(TOL / "images")
    return str(pq_path), str(TOL / "train/class_spec.json")


def build(dataset: str, micro: int, workers: int, out: Path, rows: int):
    from mini_trainer.config import load_yaml_config

    r = RECIPES[dataset]
    cfg = load_yaml_config(str(MT / "dev/lepinet" / r["base"]))
    for k, v in {**COMMON, **r["set"]}.items():
        put(cfg, k, v)
    import importlib
    mod, cls = cfg["optimizer_builder_kwargs"]["optimizer_cls"].rsplit(".", 1) if isinstance(
        cfg["optimizer_builder_kwargs"]["optimizer_cls"], str) else (None, None)
    if mod:
        cfg["optimizer_builder_kwargs"]["optimizer_cls"] = getattr(importlib.import_module(mod), cls)
    cfg["dataloader_builder_kwargs"]["batch_size"] = micro
    cfg["dataloader_builder_kwargs"]["num_workers"] = workers
    builder = cfg["builder"]
    if isinstance(builder, str):
        m, c = builder.rsplit(".", 1)
        builder = getattr(importlib.import_module(m), c)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device, dtype = torch.device("cuda"), getattr(torch, cfg["dtype"])
    if dataset == "tol":
        inp, spec_path = tol_subset(out, rows)
    else:
        inp, spec_path = str(Path(r["input"]).resolve()), None
    spec = builder.build_class_spec(path=spec_path, dir=inp, **cfg.get("spec_model_dataloader_kwargs", {}))
    spec["resize_size"] = cfg["size"]
    model, preprocess = builder.build_model(device=device, dtype=torch.float32, **{**spec, **cfg["model_builder_kwargs"]})
    model = model.to(memory_format=torch.channels_last)
    labels, train_loader, _val = builder.build_dataloader(input_dir=inp, output_dir=None, device=device, dtype=dtype,
                                                          splits=("train", "val"), **{**spec, **cfg["dataloader_builder_kwargs"]})
    augmentation = builder.build_augmentation(dtype=dtype, **cfg["augmentation_builder_kwargs"])
    optimizer = builder.build_optimizer(model=model, **cfg["optimizer_builder_kwargs"])
    criterion = builder.build_criterion(labels=labels, num_classes=spec["num_classes"], device=device, dtype=dtype,
                                        **cfg["criterion_builder_kwargs"])
    scaler = builder.build_scaler(device=device, enabled=dtype != torch.float32, **cfg.get("scaler_builder_kwargs", {}))
    regularizer = builder.build_regularizer(**cfg.get("regularizer_builder_kwargs", {}))
    return dict(model=model, preprocess=preprocess, loader=train_loader, augmentation=augmentation, optimizer=optimizer,
                criterion=criterion, scaler=scaler, regularizer=regularizer, device=device, dtype=dtype,
                num_classes=spec["num_classes"])


def range_test(c: dict, batch: int, micro: int, steps: int, start: float, end: float, log=print):
    from mini_trainer.trainer import _optimizer_step
    import mini_trainer.trainer as T

    model, opt, scaler = c["model"], c["optimizer"], c["scaler"]
    accum = batch // micro
    assert accum * micro == batch, (batch, micro)
    base = [g["lr"] for g in opt.param_groups]
    ref = max(base)
    ratios = [b / ref for b in base]
    mult = (end / start) ** (1 / max(1, steps - 1))
    rec, best, avg = [], math.inf, 0.0
    it = iter(c["loader"])
    model.train()
    t0 = time.time()
    for s in range(steps):
        lr = start * mult ** s
        for g, rt in zip(opt.param_groups, ratios):
            g["lr"] = lr * rt
        opt.zero_grad()
        tot, levels = 0.0, None
        for _ in range(accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(c["loader"])
                x, y = next(it)
            x, y = x.to(c["device"], non_blocking=True), y.to(c["device"], non_blocking=True)
            with autocast(device_type="cuda", dtype=c["dtype"], enabled=c["dtype"] != torch.float32), \
                    T.SupervisionContext(y), T.EmbeddingContext():
                logits = model(c["preprocess"](c["augmentation"](x)))
                loss = c["criterion"](logits, y)
                reg = c["regularizer"](model)
            if isinstance(loss, torch.Tensor) and loss.numel() == 1:
                loss = [loss]
            ltot = sum(loss) + reg
            if not torch.isfinite(ltot):
                tot = math.nan
                break
            scaler.scale(ltot / accum).backward()
            tot += float(ltot.detach()) / accum
            lv = [float(l.detach()) / accum for l in loss]
            levels = lv if levels is None else [a + b for a, b in zip(levels, lv)]
        if not math.isfinite(tot):
            rec.append({"step": s, "lr": lr, "loss": math.nan})
            log(f"step {s} lr {lr:.2e}: non-finite loss, stop")
            break
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        stepped = _optimizer_step(opt, scaler)
        avg = 0.98 * avg + 0.02 * tot
        smooth = avg / (1 - 0.98 ** (s + 1))
        best = min(best, smooth)
        rec.append({"step": s, "lr": lr, "loss": tot, "smooth": smooth, "levels": levels, "stepped": bool(stepped)})
        if s % 10 == 0:
            log(f"step {s:4d} lr {lr:.2e} loss {tot:.3f} smooth {smooth:.3f} ({time.time() - t0:.0f} s)")
        if s > 10 and smooth > 4 * best:
            log(f"step {s} lr {lr:.2e}: smoothed loss above 4x its minimum, stop")
            break
    return rec


def suggest(rec: list[dict]) -> dict:
    r = [x for x in rec if math.isfinite(x.get("loss", math.nan)) and "smooth" in x]
    lrs = np.array([x["lr"] for x in r])
    sm = np.array([x["smooth"] for x in r])
    # fastai skips the first and last points; use the part before the blow-up
    i_min = int(np.argmin(sm))
    out = {"lr_at_min": float(lrs[i_min]), "minimum/10": float(lrs[i_min] / 10)}
    grad = np.gradient(sm, np.log(lrs))
    lo = max(1, int(0.05 * len(sm)))
    i_steep = lo + int(np.argmin(grad[lo:i_min + 1])) if i_min > lo else i_min
    out["steep"] = float(lrs[i_steep])
    # fastai 2.7 valley: the longest strictly-decreasing run of the (smoothed) loss; lr two thirds of the way down
    n = len(sm)
    run = [1] * n
    prev = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if sm[i] < sm[j] and run[i] < run[j] + 1:
                run[i], prev[i] = run[j] + 1, j
    end = int(np.argmax(run))
    path = [end]
    while prev[path[-1]] != -1:
        path.append(prev[path[-1]])
    s0, s1 = path[-1], path[0]
    out["valley"] = float(lrs[s0 + int(round((s1 - s0) * 2 / 3))])
    return out


def plot(rec, sug, known, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = [x for x in rec if "smooth" in x]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x["lr"] for x in r], [x["loss"] for x in r], color="#bbbbbb", lw=0.8, label="loss")
    ax.plot([x["lr"] for x in r], [x["smooth"] for x in r], color="#1f5f99", lw=2, label="smoothed")
    for k, col in (("minimum/10", "#d1495b"), ("steep", "#edae49"), ("valley", "#00798c")):
        ax.axvline(sug[k], color=col, ls="--", lw=1.2, label=f"{k} {sug[k]:.1e}")
    for k, v in known.items():
        ax.axvline(float(k), color="#333333", ls=":", lw=1.2)
        ax.text(float(k), ax.get_ylim()[1], f" {k}: {v}", rotation=90, va="top", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (peak of the one-cycle schedule)")
    ax.set_ylabel("training loss (sum over ranks)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(RECIPES), required=True)
    ap.add_argument("--out", default="data/lr_finder")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--start", type=float, default=1e-6)
    ap.add_argument("--end", type=float, default=1e-1)
    ap.add_argument("--micro", type=int, default=32, help="images per forward pass (memory)")
    ap.add_argument("--batch", type=int, default=None, help="images per optimizer step (default: the recipe's global batch)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--rows", type=int, default=200_000, help="ToL rows sampled for the test")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    batch = a.batch or RECIPES[a.dataset]["batch"]
    name = f"{a.dataset}_b{batch}{a.tag}"
    c = build(a.dataset, a.micro, a.workers, out, a.rows)
    print(f"{name}: classes {c['num_classes']}, batch {batch} = {batch // a.micro} x {a.micro}, {a.steps} steps "
          f"{a.start:.0e} -> {a.end:.0e}", flush=True)
    rec = range_test(c, batch, a.micro, a.steps, a.start, a.end, log=lambda m: print(m, flush=True))
    sug = suggest(rec)
    print(json.dumps(sug, indent=1))
    (out / f"{name}.json").write_text(json.dumps({"dataset": a.dataset, "batch": batch, "micro": a.micro, "steps": a.steps,
                                                   "start": a.start, "end": a.end, "suggest": sug, "record": rec}, indent=1))
    plot(rec, sug, RECIPES[a.dataset]["known"], f"{a.dataset}, DINOv3 ConvNeXt-L, BU, batch {batch}", out / f"{name}.png")
    print("wrote", out / f"{name}.png")


if __name__ == "__main__":
    main()
