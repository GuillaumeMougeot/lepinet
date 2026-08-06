"""How often does the cosine head's z-score transform actually clamp?

`cosine_to_zscore` clamps its input to +-(1 - 1e-7) before `acos`. That is a no-op *if* the
prototype rows are unit-norm, because then the input is a genuine cosine. They are not
(`journal/2026-08-06-the-cosine-head-is-not-unit-norm.md`): measured row norms reach 1.71 on the
ArcFace checkpoint and 2.37 on the plain one, so scores can exceed 1 and saturate.

This measures the consequence, which bounds how much of the paper's calibration argument is
affected:

* **clamp rate** -- fraction of all logits that saturate, and fraction of *images* whose top-1
  saturates. The second matters more: a saturated top-1 has an uninformative confidence.
* **tie rate** -- fraction of images where two or more classes saturate, so the argmax is decided
  by index order rather than by the model. This is the case that would cost accuracy.
* **the raw score distribution**, to show how far past 1 it actually goes.

No training, no labels needed beyond the eval fold. Runs in minutes.

    python dev/071_clamp_rate.py --model '...*.pt' --parquet ... --img-dir ... --out clamp.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

LIMIT = 1.0 - 1e-7


@torch.no_grad()
def measure(model, dls, df, device, num_workers=32, n_max=40000):
    from lepinet.test import dl_num_workers

    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    dl = dls.test_dl(df, num_workers=nw)
    model.to(device).eval()
    body, head = model[0], model[1].head
    W = head.layers[0].weight.detach()          # as the model uses it -- do NOT normalise

    n_logits = n_clamped = n_img = n_top1_clamped = n_tied = 0
    mx, quant = [], []
    for batch in dl:
        feats = body(batch[0].to(device))
        pooled = (F.adaptive_avg_pool2d(feats, 1).flatten(1) if feats.ndim == 4 else feats).float()
        raw = head.preclassification(pooled) @ W.T.float()     # the pre-clamp "cosine"
        n_logits += raw.numel()
        n_clamped += int((raw.abs() >= LIMIT).sum())
        n_img += raw.shape[0]
        top = raw.max(dim=1).values
        n_top1_clamped += int((top >= LIMIT).sum())
        n_tied += int(((raw >= LIMIT).sum(dim=1) >= 2).sum())
        mx.append(top.cpu().numpy())
        quant.append(raw[:, ::97].flatten().cpu().numpy())     # thinned sample of all logits
        if n_img >= n_max:
            break
    mx = np.concatenate(mx)
    quant = np.concatenate(quant)
    return {
        "n_images": n_img,
        "clamp_rate_all_logits": n_clamped / max(n_logits, 1),
        "clamp_rate_top1": n_top1_clamped / max(n_img, 1),
        "tie_rate_two_or_more_clamped": n_tied / max(n_img, 1),
        "top1_raw_score": {"mean": float(mx.mean()), "p50": float(np.percentile(mx, 50)),
                           "p99": float(np.percentile(mx, 99)), "max": float(mx.max())},
        "all_logits_sample": {"p50": float(np.percentile(quant, 50)),
                              "p999": float(np.percentile(quant, 99.9)),
                              "max": float(quant.max())},
        "prototype_row_norms": {"mean": float(W.norm(dim=1).mean()),
                                "max": float(W.norm(dim=1).max())},
    }


def main(a):
    import pandas as pd

    from lepinet.data import DEFAULT_LEVELS, filter_df, make_dls
    from lepinet.test import load_model, resolve_checkpoint_path

    path = resolve_checkpoint_path(a.model)
    print(f"checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    levels, vocabs = ckpt.get("levels", DEFAULT_LEVELS), ckpt["vocabs"]
    df = filter_df(pd.read_parquet(a.parquet), keep_in=[a.test_set], levels=levels)
    df = df[df[levels[0]].astype(str).isin({str(v) for v in vocabs[levels[0]]})]
    if "image_path" not in df.columns:
        df["image_path"] = df[levels[0]].astype(str) + "/" + df["filename"]
    for lv in levels:
        df[lv] = df[lv].astype(str)
    df = df.reset_index(drop=True)
    df["is_valid"] = np.arange(len(df)) % 5 == 0
    dls = make_dls(df[["image_path", "is_valid", *levels]], vocabs, a.img_dir,
                   int(a.img_size * 460 / 256), a.img_size, 128, a.num_workers,
                   lowmem=False, levels=levels)
    model, _ = load_model(ckpt, img_size=a.img_size)
    res = measure(model, dls, df, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  a.num_workers, a.n_max)
    res["head"] = ckpt.get("head")
    print(json.dumps(res, indent=2))
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", default="clamp.json")
    ap.add_argument("--test-set", default="0")
    ap.add_argument("--n-max", type=int, default=40000)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=32)
    main(ap.parse_args())
