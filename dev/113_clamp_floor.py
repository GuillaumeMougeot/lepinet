"""The cosine head's clamp floor: where it comes from, how much it costs, and what the proposed fix changes
(owner, 2026-11-19). CPU only.

mini_trainer's normalised head (lepinet's recipe: learned per-class magnitude g and bias b) computes
    logit_c = sqrt(D-2) * asin(clamp(g_c * cos(h, v_c), -1, 1)) + b_c
The proposed fix applies g after the transform:
    logit_c = g_c * sqrt(D-2) * asin(clamp(cos(h, v_c), -1, 1)) + b_c

    .venv/bin/python dev/113_clamp_floor.py stats --weights CKPT --emb DIR   # floor counts on a checkpoint
    .venv/bin/python dev/113_clamp_floor.py grad                             # the gradient a floored target gets
    .venv/bin/python dev/113_clamp_floor.py figure                           # journal/figures/2026-11-19-clamp-floor.png

`--emb` is a dev/109 `embed --set known` folder (embeddings of test images of trained ToL species, paths.json gives
the species folder). Results in journal 2026-11-19 (the clamp entry).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

D = 1536
S = math.sqrt(D - 2)
EPS = 1e-7


def z_current(cos, g):
    return S * torch.asin((g * cos).clamp(-1 + EPS, 1 - EPS))


def z_fixed(cos, g):
    return g * S * torch.asin(cos.clamp(-1 + EPS, 1 - EPS))


def load(weights, emb, n=20000):
    sd = torch.load(weights, map_location="cpu", mmap=True, weights_only=False)
    sd = sd.get("model", sd)
    g = sd["head.fc.linear.parametrizations.weight.original0"].float().squeeze(1)
    b = sd["head.fc.linear.bias"].float()
    V = F.normalize(sd["head.fc.linear.parametrizations.weight.original1"].float(), dim=1)
    cls = json.load(open(f"{emb}/classes.json"))["0"]
    tail = Counter(k.split("|")[-1] for k in cls)
    by_folder = {k.split("|")[-1]: v for k, v in cls.items() if tail[k.split("|")[-1]] == 1}
    paths = json.load(open(f"{emb}/paths.json"))[:n]
    y = torch.tensor([by_folder.get(p.split("/")[-2], -1) for p in paths])
    e = F.normalize(torch.from_numpy(np.load(f"{emb}/embeddings.npy")[:n].astype(np.float32)), dim=1)
    ok = y >= 0
    return g, b, V, e[ok], y[ok]


def stats(a):
    g, b, V, e, y = load(a.weights, a.emb)
    cos = e @ V.T
    x = cos * g
    t = x[torch.arange(len(y)), y]
    wrong = x.argmax(1) != y
    print(f"images {len(y):,}; g median {g.median():.3f} (p1 {g.quantile(.01):.3f}, p99 {g.quantile(.99):.3f})")
    print(f"(image, class) pairs on the floor {(x <= -1).float().mean():.3f}; at the ceiling {(x >= 1).float().mean():.5f}")
    print(f"classes off the floor per image: median {(x > -1).sum(1).float().median():.0f}")
    print(f"true class on the floor {(t <= -1).float().mean():.4f} of images; top-1 {(~wrong).float().mean():.3f}; "
          f"share of errors with the true class on the floor {((t <= -1) & wrong).sum() / wrong.sum():.3f}")
    print(f"median cosine: true class {cos[torch.arange(len(y)), y].median():.3f}, all classes {cos.median():.3f}")


def grad(_):
    """One image, three classes: the target floored, two competitors alive. Gradient of the cross-entropy."""
    for name, fn in (("current", z_current), ("fixed", z_fixed)):
        cos = torch.tensor([-0.80, -0.45, -0.50], requires_grad=True)  # target first, at g*cos = -1.2
        g = torch.tensor([1.5, 1.5, 1.5])
        logits = fn(cos, g)
        loss = F.cross_entropy(logits[None], torch.tensor([0]))
        loss.backward()
        p = logits.softmax(0)
        print(f"{name:8s} logits {logits.detach().numpy().round(1)}  p(target) {p[0]:.2e}  "
              f"dloss/dcos target {cos.grad[0]:+.3f}, competitors {cos.grad[1]:+.3f} {cos.grad[2]:+.3f}")
    c = torch.linspace(-0.99, 0.99, 7)
    assert torch.allclose(z_current(c, torch.ones(7)), z_fixed(c, torch.ones(7)))
    print("g = 1: both forms give identical logits")


def figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g, b, V, e, y = load(a.weights, a.emb, n=5000)
    cos = e @ V.T
    gm = float(g.median())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    c = torch.linspace(-1, 1, 801)
    ax[0].plot(c, z_current(c, torch.full_like(c, gm)), color="#b4462e", lw=2, label=f"current: asin(clamp(g cos)), g = {gm:.2f}")
    ax[0].plot(c, z_fixed(c, torch.full_like(c, gm)), color="#2e6fb4", lw=2, label=f"fixed: g asin(cos), g = {gm:.2f}")
    ax[0].plot(c, z_current(c, torch.ones_like(c)), color="#888", lw=1, ls="--", label="g = 1 (both)")
    ax[0].axvspan(-1, -1 / gm, color="#b4462e", alpha=0.10)
    ax[0].text(-0.98, -20, "floor: flat,\nno gradient", color="#b4462e", fontsize=9)
    ax[0].set(xlabel="cosine between embedding and class prototype", ylabel="logit before bias",
              title="The transform, at the median learned magnitude")
    ax[0].legend(fontsize=8, loc="upper left")
    x = (cos * g)
    t = x[torch.arange(len(y)), y].numpy()
    allc = x[:, ::37].flatten().numpy()
    bins = np.linspace(-1.6, 0.6, 111)
    ax[1].hist(np.clip(allc, -1.6, 0.6), bins=bins, density=True, color="#999", alpha=0.6, label="every class (sample)")
    ax[1].hist(np.clip(t, -1.6, 0.6), bins=bins, density=True, color="#2e6fb4", alpha=0.7, label="the true class")
    ax[1].axvline(-1, color="#b4462e", lw=1.5)
    ax[1].text(-1.58, ax[1].get_ylim()[1] * 0.85, f"left of -1: on the floor\n{(x <= -1).float().mean():.0%} of all pairs,\n"
               f"{(t <= -1).mean():.1%} of true classes", color="#b4462e", fontsize=9)
    ax[1].set(xlabel="g x cosine (what the clamp sees)", ylabel="density",
              title="tol_full, 5,000 test images of trained species")
    ax[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = "journal/figures/2026-11-19-clamp-floor.png"
    fig.savefig(out, dpi=130)
    print(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["stats", "grad", "figure"])
    p.add_argument("--weights", default="data/population_probes/tol_e1/tol_full_last.pt")
    p.add_argument("--emb", default="data/population_probes/tol_e1/emb_known")
    a = p.parse_args()
    {"stats": stats, "grad": grad, "figure": figure}[a.step](a)


if __name__ == "__main__":
    main()
