"""Early-convergence race on the GLOBAL dataset: fastai-only vs mini_trainer-only.

Question: with everything held as identical as possible, does one framework's training
loop make validation accuracy climb *faster per gradient step* at the start of training?
We don't wait for a full epoch -- we probe a fixed validation subset every N steps for the
first ~1.5k steps and compare the climb.

Held identical across both arms (as close as possible):
  backbone (effnetv2s, same pretrained weights, same seed/init), batch size, image size,
  MINIMAL augmentation (horizontal flip only -- "little aug"), base LR, label smoothing,
  the model head+loss (mini_trainer IndependentClassifier + MultiLevelWeightedCE), the
  train/val split, the fixed validation probe subset, NO warmup, and UNFROZEN from step 0.

The only differences are each framework's native training core:
  - fastai       : fastai `Learner.fit` (constant LR, no warmup), torch Adam=AdamW
                   (fastai default, decoupled wd), fp32.
  - mini_trainer : mini_trainer's loop -- MuonAuxAdamW (Muon on 2D backbone weights +
                   AdamW rest), fp16 AMP + GradScaler, grad-clip 5. Constant LR, no warmup.

Usage: python dev/031_framework_climb_race.py [--arch effnetv2s] [--bs 32] [--steps 1500]
"""

import sys, importlib, time, argparse, gc
import torch, torch.multiprocessing
torch.multiprocessing.set_start_method("fork", force=True)
import pandas as pd
from pathlib import Path
from torch import nn

sys.path.insert(0, "dev")
v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod030 = importlib.import_module("030_hierarchical_heads_benchmark")

import fastai.vision.all as fv
from fastai.callback.core import Callback, CancelFitException
from mini_trainer.hierarchical.loss import MultiLevelWeightedCrossEntropyLoss
from mini_trainer.hierarchical.integration import HierarchicalBuilder
from mini_trainer.training.muon import MuonAuxAdamW
from mini_trainer.modeling import SupervisionContext, EmbeddingContext

PARQUET = "data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"
IMG_DIR = "data/global/images"
OUT_DIR = "data/global/models"
ARCH_MAP = {"effnetv2s": (fv.efficientnet_v2_s, fv.EfficientNet_V2_S_Weights.DEFAULT),
            "resnet18": (fv.resnet18, fv.ResNet18_Weights.DEFAULT),
            "resnet50": (fv.resnet50, fv.ResNet50_Weights.DEFAULT)}


def minimal_aug_dls(df, vocabs, img_dir, img_size, bs, num_workers):
    """Same as v4.make_dls but with 'little aug': horizontal flip only (no warp/rotate/
    zoom/lighting), plus imagenet normalization."""
    from fastai.vision.all import (DataBlock, ImageBlock, CategoryBlock, ColSplitter, ColReader,
                                   Resize, Flip, Normalize, imagenet_stats)
    dblock = DataBlock(
        blocks=(ImageBlock, *(CategoryBlock(vocab=vocabs[l]) for l in v4.HIERARCHY_LEVELS)),
        n_inp=1, splitter=ColSplitter(),
        get_x=ColReader("image_path", pref=img_dir),
        get_y=[ColReader(l) for l in v4.HIERARCHY_LEVELS],
        item_tfms=Resize(img_size),
        batch_tfms=[Flip(), Normalize.from_stats(*imagenet_stats)],
    )
    kw = {} if num_workers is None else {"num_workers": num_workers}
    return dblock.dataloaders(df, bs=bs, **kw)


@torch.no_grad()
def evaluate(model, probe_batches, device):
    was = model.training
    model.eval()
    corr = [0, 0, 0]; total = 0
    for xb, ys in probe_batches:
        xb = xb.to(device)
        logits = model(xb)
        for i in range(3):
            corr[i] += (logits[i].argmax(1) == ys[i].to(device)).sum().item()
        total += ys[0].shape[0]
    model.train(was)
    return [c / total for c in corr]


class Net(nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.body, self.head = body, head
        self._backbone_output_name = "head"
    def forward(self, x):
        return self.head(self.body(x))


def build_body_head(arch, weights, nf, n_classes, cls2idx, sparse_masks, device, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    body = fv.create_body(arch(weights=weights)).to(device)
    head = mod030.build_head("independent", nf, n_classes, cls2idx, sparse_masks).to(device)
    return body, head


class ClimbProbe(Callback):
    """fastai callback: probe fixed val subset every `every` steps, stop after `max_steps`."""
    order = 100
    def __init__(self, probe_batches, device, every, max_steps, hist, t0):
        self.pb, self.device, self.every, self.max_steps, self.hist, self.t0 = \
            probe_batches, device, every, max_steps, hist, t0
    def after_batch(self):
        if not self.training:
            return
        step = self.train_iter
        if step % self.every == 0:
            accs = evaluate(self.model, self.pb, self.device)
            self.hist.append((step, time.time() - self.t0, *accs))
            print(f"[fastai]      step {step:4d}  sp {accs[0]:.4f}  ge {accs[1]:.4f}  fa {accs[2]:.4f}  ({time.time()-self.t0:.0f}s)", flush=True)
        if step >= self.max_steps:
            raise CancelFitException()


def run_fastai(dls, probe_batches, arch, weights, nf, n_classes, cls2idx, sparse_masks,
               device, lr, every, max_steps, seed):
    body, head = build_body_head(arch, weights, nf, n_classes, cls2idx, sparse_masks, device, seed)
    model = nn.Sequential(body, head)
    crit = mod030.MultiLevelLossWrapper(MultiLevelWeightedCrossEntropyLoss(
        num_classes=n_classes, device=device, dtype=torch.float32, label_smoothing=1 / n_classes[0]))
    hist = []
    learn = fv.Learner(dls, model, loss_func=crit,
                       cbs=[mod030.GCCallback(), mod030.SupervisionContextCallback(),
                            ClimbProbe(probe_batches, device, every, max_steps, hist, time.time())])
    learn.unfreeze()
    try:
        learn.fit(1, lr)  # constant LR, no warmup; ClimbProbe stops it after max_steps
    except CancelFitException:
        pass
    return hist


def run_minitrainer(dls, probe_batches, arch, weights, nf, n_classes, cls2idx, sparse_masks,
                    device, lr, every, max_steps, seed):
    body, head = build_body_head(arch, weights, nf, n_classes, cls2idx, sparse_masks, device, seed)
    model = Net(body, head).to(device)
    opt = HierarchicalBuilder.build_optimizer(model, optimizer_cls=MuonAuxAdamW, lr=lr,
                                              weight_decay=0.01, backbone_lr=lr)  # backbone_lr=lr: match fastai (no discriminative)
    scaler = torch.amp.GradScaler(device.type)
    crit = MultiLevelWeightedCrossEntropyLoss(num_classes=n_classes, device=device,
                                             dtype=torch.float32, label_smoothing=1 / n_classes[0])
    hist = []; t0 = time.time(); step = 0
    model.train()
    while step < max_steps:
        for xb, *yb in dls.train:
            if step % every == 0:
                accs = evaluate(model, probe_batches, device)
                hist.append((step, time.time() - t0, *accs))
                print(f"[minitrainer] step {step:4d}  sp {accs[0]:.4f}  ge {accs[1]:.4f}  fa {accs[2]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            xb = xb.to(device).as_subclass(torch.Tensor)
            yb = [y.to(device).as_subclass(torch.Tensor).long() for y in yb]
            target = torch.stack(yb, 1)
            with torch.autocast(device.type, dtype=torch.float16), SupervisionContext(target), EmbeddingContext():
                loss = sum(crit(model(xb), target))
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); gc.collect(0)
            step += 1
            if step >= max_steps:
                break
    accs = evaluate(model, probe_batches, device)
    hist.append((step, time.time() - t0, *accs))
    print(f"[minitrainer] step {step:4d}  sp {accs[0]:.4f}  FINAL  ({time.time()-t0:.0f}s)", flush=True)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="effnetv2s")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--min_img_per_spc", type=int, default=50)
    ap.add_argument("--probe_batches", type=int, default=60)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="fastai,minitrainer")
    args = ap.parse_args()

    device = torch.device("cuda")
    arch, weights = ARCH_MAP[args.arch]

    df, hierarchy = v4.gen_df(Path(PARQUET), Path(OUT_DIR), args.min_img_per_spc, "1",
                              Path("data/global/hierarchy.csv"), [])
    vocabs = {l: sorted(df[l].unique().tolist()) for l in v4.HIERARCHY_LEVELS}
    n_classes = [len(vocabs[l]) for l in v4.HIERARCHY_LEVELS]
    print(f"n_classes={n_classes} arch={args.arch} bs={args.bs} img={args.img} lr={args.lr} "
          f"steps={args.steps}", flush=True)

    dls = minimal_aug_dls(df, vocabs, Path(IMG_DIR), args.img, args.bs, args.num_workers)
    nf = v4.body_out_features(arch)
    cls2idx, sparse_masks = mod030.build_class_spec(df, vocabs)

    # Fixed validation probe subset -- materialize once, reuse identically for both arms.
    print(f"Materializing fixed probe subset ({args.probe_batches} batches)...", flush=True)
    probe_batches = []
    for i, (xb, *yb) in enumerate(dls.valid):
        if i >= args.probe_batches:
            break
        probe_batches.append((xb.as_subclass(torch.Tensor).cpu(),
                              [y.as_subclass(torch.Tensor).long().cpu() for y in yb]))
    print(f"probe subset: {sum(b[0].shape[0] for b in probe_batches)} images", flush=True)

    R = {}
    for arm in args.arms.split(","):
        print(f"\n===== {arm} =====", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        fn = run_fastai if arm == "fastai" else run_minitrainer
        R[arm] = fn(dls, probe_batches, arch, weights, nf, n_classes, cls2idx, sparse_masks,
                    device, args.lr, args.every, args.steps, args.seed)

    print("\n\n===== CLIMB: val species top-1 vs step =====")
    print("step   " + "".join(f"{a:>14}" for a in R))
    steps = sorted({h[0] for hs in R.values() for h in hs})
    lut = {a: {h[0]: h[2] for h in R[a]} for a in R}
    for s in steps:
        print(f"{s:<6d} " + "".join(f"{lut[a].get(s, float('nan')):>14.4f}" for a in R))
    print("\n===== wall-clock to reach species-acc milestones =====")
    for a in R:
        line = f"{a:14s}"
        for milestone in (0.1, 0.2, 0.3, 0.4):
            hit = next((h for h in R[a] if h[2] >= milestone), None)
            line += f"  {int(milestone*100)}%@{hit[1]:.0f}s/step{hit[0]}" if hit else f"  {int(milestone*100)}%:--"
        print(line, flush=True)


if __name__ == "__main__":
    main()
