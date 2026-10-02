"""Profile the GPU-decode pipeline: where does the ~650-860 img/s cap come from, and does a
bigger batch or overlap help? Reuses dev/038's pieces. Local, no config changes to dev/038.

Two measurements:
  1. Per-stage synced timing at a given bs: data-wait / decode / resize / aug / fwd / bwd / opt.
     Synced (cuda.synchronize between stages) over-counts vs the real async overlap, but it
     ranks the stages -- the bottleneck to attack.
  2. End-to-end throughput (unsynced, realistic) across batch sizes, with and without a simple
     double-buffer overlap (decode batch N+1 while the model runs batch N).

Usage: python dev/039_gpu_decode_profile.py -c <config> --bs 64,128,256 --warmup 30 --iters 60
"""
import argparse, importlib, sys, time, threading, queue
from pathlib import Path
import numpy as np, torch, yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
d38 = importlib.import_module("038_gpu_decode_train")
v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod030 = importlib.import_module("030_hierarchical_heads_benchmark")
longtail = importlib.import_module("034_longtail")
HL = v4.HIERARCHY_LEVELS


def stage_times(model, opt, crit, dl_iter, size, dev, mean, std, iters):
    """Synced per-stage timing, averaged over `iters` batches."""
    from torchvision.io import decode_jpeg, ImageReadMode
    acc = dict(wait=0, decode=0, resize=0, aug=0, fwd=0, bwd=0, opt=0)
    def sync(): torch.cuda.synchronize()
    for _ in range(iters):
        t = time.time(); byte_list, yb = next(dl_iter); acc["wait"] += time.time() - t
        yb = yb.to(dev)
        sync(); t = time.time()
        imgs = decode_jpeg(byte_list, device=dev, mode=ImageReadMode.RGB)
        sync(); acc["decode"] += time.time() - t; t = time.time()
        out = torch.empty((len(imgs), 3, size, size), device=dev)
        for j, im in enumerate(imgs):
            out[j] = F.interpolate(im.unsqueeze(0).float(), (size, size), mode="bilinear", align_corners=False).squeeze(0)
        out /= 255.0
        sync(); acc["resize"] += time.time() - t; t = time.time()
        out = d38._rand_affine(out, 15.0, 1.1); out = (out - mean) / std
        sync(); acc["aug"] += time.time() - t; t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(out); loss = crit(preds, *[yb[:, k] for k in range(yb.shape[1])])
        sync(); acc["fwd"] += time.time() - t; t = time.time()
        opt.zero_grad(); loss.backward()
        sync(); acc["bwd"] += time.time() - t; t = time.time()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)):
            opt.step()
        sync(); acc["opt"] += time.time() - t
    return {k: v / iters * 1000 for k, v in acc.items()}   # ms/batch


def throughput(model, opt, crit, dl, size, dev, mean, std, bs, warmup, iters, overlap):
    """Realistic (unsynced) img/s. overlap=True double-buffers: a background thread decodes the
    next batch on a side CUDA stream while the model runs the current one."""
    it = iter(dl)
    if not overlap:
        for i, (bl, yb) in enumerate(it):
            xb = d38.gpu_decode_batch(bl, size, dev, True, mean, std); yb = yb.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(xb), *[yb[:, k] for k in range(yb.shape[1])])
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if i == warmup: torch.cuda.synchronize(); t0 = time.time(); seen = 0
            elif i > warmup:
                seen += xb.shape[0]
                if i - warmup >= iters: break
        torch.cuda.synchronize(); return seen / (time.time() - t0)
    # overlap: prefetch-decode on a side stream
    stream = torch.cuda.Stream()
    q = queue.Queue(maxsize=3)
    def producer():
        for bl, yb in it:
            with torch.cuda.stream(stream):
                xb = d38.gpu_decode_batch(bl, size, dev, True, mean, std)
            q.put((xb, yb.to(dev), stream.record_event()))
        q.put(None)
    threading.Thread(target=producer, daemon=True).start()
    i = 0; seen = 0; t0 = None
    while True:
        item = q.get()
        if item is None: break
        xb, yb, ev = item; torch.cuda.current_stream().wait_event(ev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(xb), *[yb[:, k] for k in range(yb.shape[1])])
        opt.zero_grad()
        if torch.isfinite(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if i == warmup: torch.cuda.synchronize(); t0 = time.time(); seen = 0
        elif i > warmup:
            seen += xb.shape[0]
            if i - warmup >= iters: break
        i += 1
    torch.cuda.synchronize(); return seen / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--bs", default="64,128,256")
    ap.add_argument("--num-workers", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["train"]
    dev = torch.device("cuda")
    pq = Path(cfg["parquet_path"])
    df, _ = v4.gen_df(pq, Path(cfg["out_dir"]), cfg["min_img_per_spc"], cfg["fold"],
                      pq.parent / "hierarchy.csv", cfg.get("family_filter", []))
    vocabs = {l: sorted(df[l].unique().tolist()) for l in HL}
    idx = {l: {v: k for k, v in enumerate(vocabs[l])} for l in HL}
    labels = np.stack([df[l].map(idx[l]).to_numpy() for l in HL], 1).astype(np.int64)
    paths = df["image_path"].to_numpy(dtype="U"); tr = ~df["is_valid"].to_numpy(bool)
    learn, crit = d38.build_learner(cfg, df, vocabs)
    model = learn.model.to(dev).train(); opt = learn.opt
    size = cfg["img_size"]
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    ds = d38.BytesDataset(paths[tr], labels[tr], cfg["img_dir"])

    for bs in [int(b) for b in args.bs.split(",")]:
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                        collate_fn=d38.collate_bytes, persistent_workers=True, prefetch_factor=6)
        if bs == int(args.bs.split(",")[0]):
            st = stage_times(model, opt, crit, iter(dl), size, dev, mean, std, 40)
            tot = sum(st.values())
            print(f"\n=== per-stage ms/batch @ bs={bs} (synced; ranks the bottleneck) ===")
            for k, v in sorted(st.items(), key=lambda x: -x[1]):
                print(f"  {k:7} {v:6.1f} ms  ({v/tot*100:4.1f}%)")
            print(f"  {'TOTAL':7} {tot:6.1f} ms  -> serial ceiling {bs/tot*1000:.0f} img/s")
        base = throughput(model, opt, crit, dl, size, dev, mean, std, bs, args.warmup, args.iters, False)
        ovl = throughput(model, opt, crit, dl, size, dev, mean, std, bs, args.warmup, args.iters, True)
        print(f"bs={bs:4}: serial {base:5.0f} img/s | overlapped {ovl:5.0f} img/s  ({ovl/base:.2f}x)")


if __name__ == "__main__":
    main()
