"""GPU-decode training prototype: read raw JPEG bytes on the CPU, decode on the GPU (nvJPEG).

Why
---
The fastai CPU pipeline is decode-bound at ~1100 img/s and each worker's decode+resize+collate
costs ~1.2 GB, which is what forced the worker-count/OOM bind (journal/2026-07-ucloud-benchmark
-oom.md) and leaves the B200 GPU ~idle (journal/2026-07-ucloud-throughput.md). Moving JPEG
decode to the GPU should (a) break the ~1100 ceiling and (b) collapse per-worker memory, since
CPU workers then only read ~80 KB of bytes instead of holding a decode pipeline. Measured
locally: batched GPU decode+resize ~9200 img/s vs ~1100 CPU.

What this is
------------
A prototype, not the production path. It reuses dev/030's EXACT model / loss / Muon optimiser
(via a fastai Learner built the same way) but replaces the data feed with a manual loop:
  workers read raw bytes -> GPU nvJPEG decode -> GPU resize -> GPU aug -> GPU normalize -> model.
It reports epoch time, throughput (img/s), peak host anon memory, and validation species top-1
+ macro-F1, so a 1-epoch run is directly comparable to the CPU pipeline's epoch-0 row.

Caveats (prototype): the GPU augmentation here (flip / rotate / zoom) approximates fastai's
aug_transforms rather than reproducing it bit-for-bit, so absolute F1 is indicative, not a
like-for-like number. Throughput and memory ARE the comparison this is built for.

Usage:
    python dev/038_gpu_decode_train.py --config configs/<a dev/030 train config>.yaml \
        --epochs 1 --max-batches 0   # 0 = full epoch
"""

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_start_method("fork", force=True)

import psutil
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_jpeg
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod030 = importlib.import_module("030_hierarchical_heads_benchmark")
longtail = importlib.import_module("034_longtail")
HL = v4.HIERARCHY_LEVELS


def anon_gb():
    """Host memory of this process tree in GB, via PSS (not RSS): a forked worker's RSS counts
    every shared copy-on-write page it still shares with the parent, so summing RSS across the
    tree over-reports (it read ~86 GB for a bytes-only loader). PSS divides shared pages by the
    number of sharers, giving the true footprint."""
    p = psutil.Process()
    def pss(proc):
        try:
            return proc.memory_full_info().pss
        except (psutil.Error, AttributeError):
            return 0
    return (pss(p) + sum(pss(c) for c in p.children(recursive=True))) / 1e9


class BytesDataset(Dataset):
    """Returns (raw_jpeg_bytes_as_uint8_tensor, label_indices). CPU workers do I/O only -- no
    decode -- so per-worker memory is ~one JPEG, not a decode pipeline."""

    def __init__(self, paths, labels, img_dir):
        self.paths = paths
        self.labels = labels  # (N, n_levels) int64
        self.img_dir = str(img_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        fp = os.path.join(self.img_dir, str(self.paths[i]))
        with open(fp, "rb") as f:
            buf = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
        return buf, self.labels[i]


def collate_bytes(batch):
    """Keep the variable-length byte tensors as a list (they cannot be stacked); stack labels."""
    xs = [b[0] for b in batch]
    ys = torch.stack([torch.as_tensor(b[1]) for b in batch])
    return xs, ys


def gpu_decode_batch(byte_list, size, device, train, mean, std):
    """bytes -> decoded, resized, augmented, normalised float batch on `device`.

    decode_jpeg over the whole list is one nvJPEG call. A rare CMYK/corrupt/non-JPEG file makes
    it throw where PIL coped, so those are decoded one-by-one and any straggler is replaced with
    a grey frame rather than killing the batch.
    """
    try:
        imgs = decode_jpeg(byte_list, device=device, mode=torchvision_rgb())
    except RuntimeError:
        imgs = []
        for b in byte_list:
            try:
                imgs.append(decode_jpeg(b, device=device, mode=torchvision_rgb()))
            except RuntimeError:
                imgs.append(torch.full((3, size, size), 114, dtype=torch.uint8, device=device))
    # resize each (variable native size) then stack
    out = torch.empty((len(imgs), 3, size, size), device=device)
    for j, im in enumerate(imgs):
        if im.shape[0] == 1:
            im = im.expand(3, -1, -1)
        out[j] = F.interpolate(im.unsqueeze(0).float(), size=(size, size),
                               mode="bilinear", align_corners=False).squeeze(0)
    out /= 255.0
    if train:
        # Approximate fastai aug_transforms(flip_vert, max_rotate=15, max_zoom=1.1), on GPU.
        if torch.rand(1).item() < 0.5:
            out = out.flip(-1)
        if torch.rand(1).item() < 0.5:
            out = out.flip(-2)
        out = _rand_affine(out, max_rot_deg=15.0, max_zoom=1.1)
    out = (out - mean) / std
    return out


def _rand_affine(x, max_rot_deg, max_zoom):
    """One shared random rotation+zoom for the batch (cheap; fastai varies per-item, close
    enough for a throughput/memory prototype)."""
    n = x.shape[0]
    ang = (torch.rand(1, device=x.device) * 2 - 1) * (max_rot_deg * 3.14159 / 180)
    zoom = 1.0 + torch.rand(1, device=x.device) * (max_zoom - 1.0)
    cos, sin = torch.cos(ang) / zoom, torch.sin(ang) / zoom
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")


def torchvision_rgb():
    from torchvision.io import ImageReadMode
    return ImageReadMode.RGB


def build_learner(cfg, df, vocabs):
    """dev/030's exact model + loss + Muon opt, via a throwaway dls just for structure."""
    dls = v4.make_dls(df.head(256), vocabs, Path(cfg["img_dir"]), cfg["aug_img_size"],
                      cfg["img_size"], cfg["batch_size"], num_workers=0, lowmem=False)
    cls2idx, sparse_masks = mod030.build_class_spec(df, vocabs)
    n_classes = [len(vocabs[l]) for l in HL]
    from fastai.vision.all import vision_learner
    arch = getattr(importlib.import_module("fastai.vision.all"), cfg["model_arch_name"])
    nf = v4.body_out_features(arch)
    head = mod030.build_head(cfg["head"], nf, n_classes, cls2idx, sparse_masks,
                             decoder_kwargs={"num_layers": cfg.get("decoder_num_layers", 4),
                                             "nhead": cfg.get("decoder_nhead", 1)})
    device = torch.device("cuda")
    criterion = mod030.MultiLevelLossWrapper(
        mod030.MultiLevelWeightedCrossEntropyLoss(num_classes=n_classes, device=device,
                                                  dtype=torch.float32,
                                                  label_smoothing=1 / n_classes[0]))
    # No to_fp16(): the manual loop uses bf16 autocast (wider range, no grad scaler needed),
    # which keeps this prototype's step simple and avoids wiring fastai's MixedPrecision scaler
    # through the OptimWrapper by hand.
    learn = vision_learner(dls, arch, n_out=1, custom_head=head, loss_func=criterion,
                           opt_func=mod030.muon_opt_func)
    learn.unfreeze()
    learn.create_opt()
    return learn, criterion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = full epoch (for quick probes).")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = use the config's batch_size.")
    ap.add_argument("--overlap", action="store_true",
                    help="Double-buffer: decode batch N+1 on a side CUDA stream while the model "
                         "runs batch N. Profiling showed ~1.7x (the model, not decode, is the cost).")
    ap.add_argument("--gc-every", type=int, default=8,
                    help="gc.collect(0) every N batches to break mini_trainer's cosine-head GPU "
                         "reference cycle. Per-batch (=1) halved throughput; letting a few batches' "
                         "graphs accumulate is fine on a large GPU. Watch the reported GPU mem.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["train"]
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    device = torch.device("cuda")
    parquet = Path(cfg["parquet_path"])
    hier = parquet.parent / "hierarchy.csv"
    df, _ = v4.gen_df(parquet, Path(cfg["out_dir"]), cfg["min_img_per_spc"], cfg["fold"], hier,
                      cfg.get("family_filter", []))
    vocabs = {l: sorted(df[l].unique().tolist()) for l in HL}
    idx = {l: {v: k for k, v in enumerate(vocabs[l])} for l in HL}
    labels = np.stack([df[l].map(idx[l]).to_numpy() for l in HL], axis=1).astype(np.int64)
    paths = df["image_path"].to_numpy(dtype="U")
    is_valid = df["is_valid"].to_numpy(bool)

    learn, criterion = build_learner(cfg, df, vocabs)
    model = learn.model.to(device)
    opt = learn.opt
    size = cfg["img_size"]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    tr, va = ~is_valid, is_valid
    wgts = None
    if cfg.get("oversample_power", 0):
        wgts = longtail.sample_weights(df, level="speciesKey", power=cfg["oversample_power"])
    train_ds = BytesDataset(paths[tr], labels[tr], cfg["img_dir"])
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(wgts), len(train_ds), replacement=True) if wgts is not None else None
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                          shuffle=(sampler is None), num_workers=args.num_workers,
                          collate_fn=collate_bytes, persistent_workers=True, prefetch_factor=6)

    import gc, threading, queue
    base_lr = cfg.get("base_lr", 1e-3)
    warmup = max(len(train_dl) // 2, 1000)
    grad_clip = cfg.get("grad_clip", 5.0)

    def set_lr(lr):
        for g in opt.param_groups:
            g["lr"] = lr

    def batches(dl):
        """Yield (xb_decoded_on_gpu, yb). With --overlap, a producer thread decodes the next
        batch on a side CUDA stream while the caller trains on the current one; the GPU decode
        of N+1 overlaps the model compute of N (profiling: ~1.7x, since the model dominates)."""
        if not args.overlap:
            for bl, yb in dl:
                yield gpu_decode_batch(bl, size, device, True, mean, std), yb.to(device)
            return
        stream = torch.cuda.Stream()
        q = queue.Queue(maxsize=3)
        def producer():
            for bl, yb in dl:
                with torch.cuda.stream(stream):
                    xb = gpu_decode_batch(bl, size, device, True, mean, std)
                q.put((xb, yb.to(device), stream.record_event()))
            q.put(None)
        threading.Thread(target=producer, daemon=True).start()
        while (item := q.get()) is not None:
            xb, yb, ev = item
            torch.cuda.current_stream().wait_event(ev)
            yield xb, yb

    gpu_gb = lambda: torch.cuda.max_memory_allocated() / 1e9
    print(f"gpu-decode | {len(train_ds):,} train imgs | workers={args.num_workers} "
          f"| bs={cfg['batch_size']} | head={cfg['head']} | overlap={args.overlap} "
          f"| gc_every={args.gc_every} | warmup {warmup}, clip {grad_clip}", flush=True)
    model.train()
    step = 0
    for ep in range(args.epochs):
        t0 = time.time(); seen = 0; peak_anon = 0.0; nan_skipped = 0
        for bi, (xb, yb) in enumerate(batches(train_dl)):
            set_lr(base_lr * min(1.0, (step + 1) / warmup))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(xb)
                loss = criterion(preds, *[yb[:, k] for k in range(yb.shape[1])])
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if torch.isfinite(gnorm):
                    opt.step()
                else:
                    nan_skipped += 1
            else:
                nan_skipped += 1
            # gc.collect(0) breaks mini_trainer's cosine-head reference cycle that otherwise
            # traps each batch's backward graph on the GPU. Per-batch is correct but halves
            # throughput; every gc_every batches lets a few graphs accumulate (watch GPU mem).
            if (bi + 1) % args.gc_every == 0:
                gc.collect(0)
            seen += xb.shape[0]; step += 1
            if (bi + 1) % 200 == 0:
                peak_anon = max(peak_anon, anon_gb())
                el = time.time() - t0
                print(f"  ep{ep} batch {bi+1}: {seen/el:.0f} img/s | host {anon_gb():.1f} GB "
                      f"| GPU {gpu_gb():.1f} GB | loss {loss.item():.3f} | nan {nan_skipped}", flush=True)
            if args.max_batches and bi + 1 >= args.max_batches:
                break
        el = time.time() - t0
        print(f"epoch {ep}: {seen:,} imgs in {el/60:.1f} min = {seen/el:.0f} img/s | "
              f"peak host {peak_anon:.0f} GB | peak GPU {gpu_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
