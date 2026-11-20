"""C3 on a GPU (UCloud, one B200 MIG slice): the specialist retrained, the ToL trunk under test-time views, and a head
trained through the frozen trunk. Stages, each writing to --out; runs on CPU with --limit as a check.

    python dev/117_broad_taxa_gpu.py prep   --zip broadTaxonClassifier.zip --root /tmp/mcc24 --index index.csv
    python dev/117_broad_taxa_gpu.py kim    --root R --index I --out O            # BT2: Kim's recipe, 4 folds
    python dev/117_broad_taxa_gpu.py embed  --root R --index I --out O --weights last.pt --sha256 H   # BT3: views
    python dev/117_broad_taxa_gpu.py frozen --root R --index I --out O --weights last.pt --sha256 H   # BT4: head

Folds (from dev/115's index.csv): `kim` = his random 80/20 split (train on his training list, score his validation
list); `LV`, `OH`, `SS` = train on the other two sites (all crops), score the held-out site.

- kim: Kim Bjerge's broad-taxon classifier as his repository and paper give it (MCC24-trap `resnet50.py`: torchvision
  ResNet50 with ImageNet weights, fc removed, dropout 0.5, 2048-512 ReLU, 512-16; crops squashed to 128 x 128 with a
  bilinear resize without antialiasing, as his published logits match; no normalisation, as his inference code;
  Adam 1e-4, batch 256, 25 epochs, cross-entropy; augmentation "image scaling, horizontal and vertical flip, colour
  jitter for brightness, contrast and saturation", ranges ours: scale 0.8-1.2, factors 0.8-1.2). His training script
  is not public. bf16 autocast. Writes per-epoch scores on the scored part and its logits at the last epoch and at the
  epoch with the best accuracy there (his `dhc_best` protocol on his split; reported, not used, on the site folds).
- embed: tol_full's trunk (timm ConvNeXt-L, the checkpoint's own keys) and its head's hidden layer, so the output is
  ToL's normalised embedding, for views r0 (the evaluation read), rot-30bm_pad25 and rot30bm_pad25 (tta3bm's other two
  views: rotation on an expanded canvas, then 25 % padding, both filled with the border-ring mean; mini_trainer
  campaign `logits.py`) and padsq (padded to square with the border-ring mean instead of cutting the centre). Every
  view is then read as mini_trainer's evaluation reads: largest centred square, PIL bilinear to 256, resampled to 224
  in the model (identity grid, bilinear, reflection, align_corners), ImageNet normalisation. fp32 with TF32 matmuls.
  Writes emb_<view>.npy (fp16, rows in index order); the 190 k-class head is scored on CPU by dev/115.
- frozen: a new head on the frozen trunk, trained with augmentation, as TL8 (mini_trainer --freeze-backbone): dropout
  0.1, hidden 1536-1536 with leaky ReLU, normalised cosine head with learned magnitude and bias
  (sqrt(D-2) asin(clamp(g cos)) + b), AdamW 8e-4, weight decay 0.01, 0.5-epoch warmup from 1 %, cosine to 0, batch 96,
  10 epochs, label smoothing 1/16; per image TrivialAugmentWide on the 256 read, dihedral flips, rotation up to 15
  degrees and zoom 1-1.1 with probability 0.75, then the model's resample to 224. Trunk in eval mode, no gradient,
  bf16 autocast; the head fp32. Writes its logits on the scored part (plain read, no augmentation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

N_CLS = 16
READ, SIZE = 256, 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def cpus():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return max(1, int(q) // int(p)) if q != "max" else os.cpu_count()
    except OSError:
        return os.cpu_count()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def folds(idx, names):
    site = idx.trap.fillna("").str[:2].to_numpy()
    for f in names:
        if f == "kim":
            yield f, (idx.split == "train").to_numpy(), (idx.split == "val").to_numpy()
        else:
            yield f, (site != f) & (site != ""), site == f


def macro(y, p):
    f1 = []
    for c in range(N_CLS):
        tp, fp, fn = ((p == c) & (y == c)).sum(), ((p == c) & (y != c)).sum(), ((p != c) & (y == c)).sum()
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1)), float((y == p).mean())


def load_index(a):
    idx = pd.read_csv(a.index, keep_default_na=False)
    if a.limit:   # about a.limit crops, evenly over the classes
        step = (idx.groupby("label").label.transform("size") * N_CLS // a.limit).clip(lower=1)
        idx = idx[idx.groupby("label").cumcount() % step == 0].reset_index(drop=True)
    return idx


def path_of(a, idx, i):
    return os.path.join(a.root, idx.folder.iat[i], idx.file.iat[i])


# --------------------------------------------------------------------------------------------------- prep
def prep(a):
    os.makedirs(a.root, exist_ok=True)
    with zipfile.ZipFile(a.zip) as z:
        names = [n for n in z.namelist() if n.startswith("broadTaxonClassifier/") and ".." not in n]
        z.extractall(os.path.dirname(a.root.rstrip("/")), members=names)
    idx = pd.read_csv(a.index, keep_default_na=False)
    missing = [i for i in range(len(idx)) if not os.path.exists(path_of(a, idx, i))]
    log(f"prep: {len(idx):,} indexed crops, missing {len(missing)}")
    assert not missing


# ---------------------------------------------------------------------------------------------- kim (BT2)
class KimResNet50(nn.Module):
    """MCC24-trap resnet50.py: torchvision ResNet50, fc removed, dropout 0.5, 2048-512-ReLU-16."""

    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None   # pretrained=True in his code
        self.model_ft = torchvision.models.resnet50(weights=w)
        self.model_ft.fc = nn.Identity()
        self.drop = nn.Dropout(p=0.5)
        self.linear_lvl1 = nn.Linear(2048, 512)
        self.relu_lv1 = nn.ReLU(inplace=False)
        self.softmax_reg1 = nn.Linear(512, N_CLS)

    def forward(self, x):
        return self.softmax_reg1(self.relu_lv1(self.linear_lvl1(self.drop(self.model_ft(x)))))


def _one_thread():
    torch.set_num_threads(1)   # a pool worker: torch would otherwise size its pool from the host's 256 cores


def _squash128(path):
    with Image.open(path) as im:
        x = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float()[None]
    x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False, antialias=False)
    return x[0].round().clamp(0, 255).to(torch.uint8).numpy()


def kim_augment(x, g):
    """x float [N,3,H,W] in [0,1]; per-image scale, flips and colour jitter (brightness, contrast, saturation)."""
    n = len(x)
    s = torch.empty(n, device=x.device).uniform_(0.8, 1.2, generator=g)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = 1 / s
    theta[:, 1, 1] = 1 / s
    theta[:, 0, 0] *= torch.where(torch.rand(n, device=x.device, generator=g) < 0.5, -1.0, 1.0)
    theta[:, 1, 1] *= torch.where(torch.rand(n, device=x.device, generator=g) < 0.5, -1.0, 1.0)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    b, c, sa = (torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.8, 1.2, generator=g) for _ in range(3))
    x = x * b
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    gray = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * sa + gray
    return x.clamp(0, 1)


def kim(a):
    from multiprocessing import Pool
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = load_index(a)
    cache = Path(a.root).parent / f"kim128_{len(idx)}.npy"
    if not cache.exists():
        t = time.time()
        with Pool(cpus(), initializer=_one_thread) as pool:
            arr = np.stack(pool.map(_squash128, [path_of(a, idx, i) for i in range(len(idx))], chunksize=256))
        np.save(cache, arr)
        log(f"kim: {len(arr):,} crops squashed to 128 in {time.time() - t:.0f} s")
    X = torch.from_numpy(np.load(cache))   # host memory (6 GB at full size); batches go to the GPU
    X = X.pin_memory() if dev.type == "cuda" else X
    y_all = torch.from_numpy(idx.label.to_numpy())
    os.makedirs(a.out, exist_ok=True)
    for name, tr, te in folds(idx, a.folds.split(",")):
        dst = Path(a.out) / f"kim_{name}.npz"
        if dst.exists():
            log(f"kim {name}: done already")
            continue
        torch.manual_seed(42)
        g = torch.Generator(device=dev).manual_seed(42)
        gc = torch.Generator().manual_seed(42)
        model = KimResNet50(pretrained=not a.limit).to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        tri, tei = torch.from_numpy(np.flatnonzero(tr)), torch.from_numpy(np.flatnonzero(te))
        hist, best, t0 = [], (-1.0, None, -1), time.time()
        for ep in range(a.epochs):
            model.train()
            perm = tri[torch.randperm(len(tri), generator=gc)]
            for s in range(0, len(perm), 256):
                b = perm[s:s + 256]
                x = X[b].to(dev, non_blocking=True).float() / 255
                x = kim_augment(x, g).contiguous(memory_format=torch.channels_last)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    loss = F.cross_entropy(model(x), y_all[b].to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            model.eval()
            out = []
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                for s in range(0, len(tei), 1024):
                    b = tei[s:s + 1024]
                    out.append(model((X[b].to(dev).float() / 255).contiguous(memory_format=torch.channels_last)).float())
            logits = torch.cat(out).cpu().numpy()
            f1, acc = macro(idx.label.to_numpy()[te], logits.argmax(1))
            hist.append({"epoch": ep + 1, "macro_f1": f1, "accuracy": acc, "loss_last_batch": float(loss)})
            if acc > best[0]:
                best = (acc, logits, ep + 1)
            log(f"kim {name} epoch {ep + 1}: scored part macro F1 {f1:.4f} acc {acc:.4f} ({(time.time() - t0) / 60:.1f} min)")
        np.savez(dst, kim_idx=idx.kim_idx.to_numpy()[te], label=idx.label.to_numpy()[te], logits_last=logits,
                 logits_best=best[1], best_epoch=best[2], history=json.dumps(hist), n_train=int(tr.sum()))


# -------------------------------------------------------------------------------------- ToL trunk (BT3, BT4)
def check_sha(path, want):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    assert not want or h.hexdigest() == want, f"{path}: sha256 {h.hexdigest()} is not {want}"
    log(f"weights sha256 {h.hexdigest()}")


def tol_trunk(a, dev):
    import timm
    check_sha(a.weights, a.sha256)
    sd = torch.load(a.weights, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    model = timm.create_model("convnext_large.dinov3_lvd1689m", pretrained=False, num_classes=0)
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head.fc.")}, strict=True)
    hidden = nn.Linear(1536, 1536)
    hidden.load_state_dict({"weight": sd["head.fc.hidden.weight"], "bias": sd["head.fc.hidden.bias"]})
    return model.eval().to(dev), hidden.eval().to(dev)


def resample_normalise(x):
    """uint8 [N,3,256,256] -> the model's input: mini_trainer's ResampleNormalize (as the release graph)."""
    x = x.float().div(255)
    theta = torch.eye(2, 3, device=x.device).expand(len(x), 2, 3)
    grid = F.affine_grid(theta, [len(x), 3, SIZE, SIZE], align_corners=True)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    m, s = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1), torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def read_square_image(image, size=READ):
    """The release's read_square on an image already decoded: largest centred square, PIL bilinear to size."""
    width, height = image.size
    side = int((width / size if width / size < height / size else height / size) * size)
    left, top = int(0.5 * (width - side)), int(0.5 * (height - side))
    image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1))


def border_mean(img):
    ring = np.ones(img.shape[1:], bool)
    ring[2:-2, 2:-2] = False
    return tuple(int(round(float(c))) for c in img[:, ring].mean(1))


class BorderRotatePad:
    """mini_trainer campaign logits.py: rotation on an expanded canvas, then padding by `padding` of each axis, both
    filled with the mean of the image's outer 2-pixel ring."""

    def __init__(self, degrees, padding):
        self.degrees, self.padding = degrees, padding

    def __call__(self, image):
        fill = border_mean(image)
        rotated = np.asarray(Image.fromarray(image.transpose(1, 2, 0)).rotate(
            self.degrees, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=fill)).transpose(2, 0, 1)
        h, w = rotated.shape[1:]
        y, x = int(np.ceil(h * self.padding)), int(np.ceil(w * self.padding))
        return np.stack([np.pad(c, ((y, y), (x, x)), constant_values=f) for c, f in zip(rotated, fill)])


def pad_square(image):
    fill = border_mean(image)
    h, w = image.shape[1:]
    d = abs(h - w)
    pad = ((d // 2, d - d // 2), (0, 0)) if h < w else ((0, 0), (d // 2, d - d // 2))
    return np.stack([np.pad(c, pad, constant_values=f) for c, f in zip(image, fill)])


VIEWS = {"r0": lambda im: im, "rot-30bm_pad25": BorderRotatePad(-30, 0.25), "rot30bm_pad25": BorderRotatePad(30, 0.25),
         "padsq": pad_square}


def decode(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)


class ViewSet(torch.utils.data.Dataset):
    def __init__(self, a, idx, views):
        self.a, self.idx, self.views = a, idx, views

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = decode(path_of(self.a, self.idx, i))
        return i, torch.stack([torch.from_numpy(read_square_image(Image.fromarray(
            np.ascontiguousarray(VIEWS[v](img)).transpose(1, 2, 0)))) for v in self.views])


def embed(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    idx = load_index(a)
    views = a.views.split(",")
    model, hidden = tol_trunk(a, dev)
    os.makedirs(a.out, exist_ok=True)
    outs = {v: np.lib.format.open_memmap(Path(a.out) / f"emb_{v}.npy", mode="w+", dtype=np.float16,
                                         shape=(len(idx), 1536)) for v in views}
    loader = torch.utils.data.DataLoader(ViewSet(a, idx, views), batch_size=a.batch, num_workers=max(1, cpus() - 1),
                                         prefetch_factor=2)
    t0, done = time.time(), 0
    with torch.no_grad():
        for i, x in loader:
            for k, v in enumerate(views):
                f = model(resample_normalise(x[:, k].to(dev)))
                e = F.normalize(F.leaky_relu(hidden(f.float())), dim=1)
                outs[v][i.numpy()] = e.cpu().numpy().astype(np.float16)
            done += len(i)
            if done // 8192 != (done - len(i)) // 8192:
                log(f"embed {done:,}/{len(idx):,} crops, {done / (time.time() - t0):.0f} crops/s ({len(views)} views each)")
    for o in outs.values():
        o.flush()
    json.dump({"views": views, "n": len(idx), "kim_idx": idx.kim_idx.tolist(), "seconds": round(time.time() - t0)},
              open(Path(a.out) / "emb_meta.json", "w"))
    log(f"embed done: {len(idx):,} crops in {(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------------------------------- frozen head (BT4)
class CosineHead(nn.Module):
    """mini_trainer's normalised Classifier with a hidden layer, trainable magnitude and bias (TL8's head)."""

    def __init__(self, d=1536):
        super().__init__()
        self.drop, self.hidden = nn.Dropout(0.1), nn.Linear(d, d)
        self.direction = nn.Parameter(F.normalize(torch.randn(N_CLS, d), dim=1))
        self.g, self.b = nn.Parameter(torch.ones(N_CLS)), nn.Parameter(torch.zeros(N_CLS))
        self.scale = math.sqrt(d - 2)

    def forward(self, f):
        e = F.normalize(F.leaky_relu(self.hidden(self.drop(f))), dim=1)
        cos = e @ F.normalize(self.direction, dim=1).T
        return self.scale * torch.asin((self.g * cos).clamp(-1 + 1e-7, 1 - 1e-7)) + self.b


class AugSet(torch.utils.data.Dataset):
    def __init__(self, a, idx, rows, train):
        import torchvision.transforms as T
        self.a, self.idx, self.rows, self.train = a, idx, rows, train
        self.taw = T.TrivialAugmentWide()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        i = int(self.rows[j])
        with Image.open(path_of(self.a, self.idx, i)) as im:
            im = Image.fromarray(read_square_image(im.convert("RGB")).transpose(1, 2, 0))
        if self.train:
            im = self.taw(im)
            if np.random.rand() < 0.5:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            im = im.rotate(90 * np.random.randint(4))
            if np.random.rand() < 0.75:
                fill = border_mean(np.asarray(im).transpose(2, 0, 1))
                zoom = np.random.uniform(1.0, 1.1)
                im = im.rotate(np.random.uniform(-15, 15), resample=Image.Resampling.BILINEAR, fillcolor=fill)
                s = int(round(READ / zoom))
                o = (READ - s) // 2
                im = im.crop((o, o, o + s, o + s)).resize((READ, READ), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(im).transpose(2, 0, 1).copy()), int(self.idx.label.iat[i])


def frozen(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    idx = load_index(a)
    model, _ = tol_trunk(a, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(memory_format=torch.channels_last)
    os.makedirs(a.out, exist_ok=True)
    workers = max(1, cpus() - 1)
    for name, tr, te in folds(idx, a.folds.split(",")):
        dst = Path(a.out) / f"frozen_{name}.npz"
        if dst.exists():
            log(f"frozen {name}: done already")
            continue
        torch.manual_seed(42)
        np.random.seed(42)
        head = CosineHead().to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=8e-4, weight_decay=0.01)
        dl = torch.utils.data.DataLoader(AugSet(a, idx, np.flatnonzero(tr), True), batch_size=96, shuffle=True,
                                         num_workers=workers, drop_last=True, persistent_workers=True, prefetch_factor=4)
        total, warm = a.epochs * len(dl), max(1, len(dl) // 2)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 + 0.99 * s / warm if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        t0, step = time.time(), 0
        for ep in range(a.epochs):
            head.train()
            for x, y in dl:
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    f = model(resample_normalise(x.to(dev, non_blocking=True)).contiguous(memory_format=torch.channels_last))
                loss = F.cross_entropy(head(f.float()), y.to(dev), label_smoothing=1 / N_CLS)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                step += 1
            log(f"frozen {name} epoch {ep + 1}/{a.epochs}: loss {float(loss):.3f} "
                f"({step * 96 / (time.time() - t0):.0f} img/s, {(time.time() - t0) / 60:.1f} min)")
        head.eval()
        out = []
        dl = torch.utils.data.DataLoader(AugSet(a, idx, np.flatnonzero(te), False), batch_size=128, num_workers=workers)
        with torch.no_grad():
            for x, _ in dl:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    f = model(resample_normalise(x.to(dev)).contiguous(memory_format=torch.channels_last))
                out.append(head(f.float()).cpu())
        logits = torch.cat(out).numpy()
        f1, acc = macro(idx.label.to_numpy()[te], logits.argmax(1))
        log(f"frozen {name}: scored part macro F1 {f1:.4f} acc {acc:.4f}")
        np.savez(dst, kim_idx=idx.kim_idx.to_numpy()[te], label=idx.label.to_numpy()[te], logits=logits,
                 n_train=int(tr.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["prep", "kim", "embed", "frozen"])
    p.add_argument("--zip")
    p.add_argument("--root", required=True, help="the broadTaxonClassifier folder")
    p.add_argument("--index", required=True, help="dev/115's index.csv")
    p.add_argument("--out", default="out")
    p.add_argument("--weights", help="tol_full last.pt")
    p.add_argument("--sha256", default="", help="expected sha256 of --weights")
    p.add_argument("--folds", default="kim,LV,OH,SS")
    p.add_argument("--views", default="r0,rot-30bm_pad25,rot30bm_pad25,padsq")
    p.add_argument("--epochs", type=int, default=0, help="default: kim 25, frozen 10")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="a CPU check on about N crops")
    a = p.parse_args()
    torch.set_num_threads(cpus())   # inside a UCloud job torch sees the host's 256 cores; the cgroup quota is right
    a.epochs = a.epochs or {"kim": 25, "frozen": 10}.get(a.stage, 0)
    {"prep": prep, "kim": kim, "embed": embed, "frozen": frozen}[a.stage](a)


if __name__ == "__main__":
    main()
