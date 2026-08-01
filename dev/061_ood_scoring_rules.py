"""Is the open-set loss a property of the *embedding*, or of the *scoring rule*? (E2)

Every open-set number in this project uses one score: ``-max_logit`` (`dev/052`). Across the
capacity x augmentation factorial that score falls monotonically — 0.9068 -> 0.8132 — and we
concluded the larger, augmented models are worse at novelty detection. **That conclusion assumes the
score is a fair readout of the embedding**, and it has never been checked.

It is worth checking before spending ~36 GPU-hours on E1 (re-tuning the margin at scale), because if
a different rule reads the same embedding better, there is nothing to retune.

**Why not temperature scaling.** The obvious idea — T-scale the logits and re-score — is
mathematically vacuous here. AUROC is a rank statistic and ``max_logit / T`` is a strictly monotone
transform of ``max_logit``, so the AUROC is *identical* for every T. Anything that changes the
ranking must use the logit vector's shape, not just its maximum.

So this compares rules that genuinely differ:

* ``max``      — the incumbent: ``max_j z_j``.
* ``energy``   — ``logsumexp_j z_j``. Uses the whole vector; the standard OOD baseline
                 (Liu et al. 2020) and provably better-behaved than max-logit under a Gibbs model.
* ``msp``      — max softmax probability. Normalised, so it asks "is the top class dominant?"
                 rather than "is it close?" — a different question, and not monotone in ``max``.
* ``entropy``  — ``-H(softmax(z))``. Pure shape: a diffuse posterior means "unfamiliar" even if some
                 class scores high.
* ``margin``   — ``z_(1) - z_(2)``. Novel inputs often sit between two known prototypes; this is the
                 only rule here that is insensitive to overall magnitude, which matters because
                 domain shift is known to depress *all* logits ([[2026-07-28-flemming-generalization]]).

Each is computed in the same forward pass, so five hypotheses cost one inference.

    python dev/061_ood_scoring_rules.py --model '...*.pt' --parquet ... --img-dir ... --out s.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

#: name -> f(logits [N,C]) -> score, **higher meaning more novel** (so AUROC is directly comparable).
SCORE_RULES = {
    "max":     lambda z: -z.max(dim=1).values,
    "energy":  lambda z: -torch.logsumexp(z, dim=1),
    "msp":     lambda z: -F.softmax(z, dim=1).max(dim=1).values,
    "entropy": lambda z: -(F.softmax(z, dim=1) * F.log_softmax(z, dim=1)).sum(dim=1),
    "margin":  lambda z: -(z.topk(2, dim=1).values[:, 0] - z.topk(2, dim=1).values[:, 1]),
}


def degenerate_rules(z: torch.Tensor) -> set[str]:
    """Rules that carry no information for *this* head's output convention.

    The `marginal*` heads emit **log-probabilities**, not raw z-scores, and two rules collapse:

    * ``energy`` = ``logsumexp(log p)`` = ``log(sum p)`` = **0 for every image** -- a constant, so its
      AUROC is whatever the tie-breaking in the rank sort happens to produce (it printed 0.4399 for
      A4, which looks like a catastrophic result and is actually an empty one).
    * ``msp`` = ``softmax(log p)`` = ``p``, so it is a monotone function of ``max`` and must give the
      *identical* AUROC (A4 printed 0.8591 for both, exactly).

    Reporting those as findings would be worse than not reporting them, so they are flagged. Detected
    from the data rather than from the head name, because a future head could do the same thing.
    """
    out = set()
    lse = torch.logsumexp(z, dim=1)
    if (lse.abs() < 1e-3).all():          # rows already normalised: sum of exp == 1
        out |= {"energy", "msp"}
    return out


def auroc(novel: np.ndarray, known: np.ndarray) -> float:
    """P(score_novel > score_known), by Mann-Whitney. No sklearn dependency."""
    if len(novel) == 0 or len(known) == 0:
        return float("nan")
    allv = np.concatenate([known, novel])
    ranks = allv.argsort().argsort()
    r_novel = ranks[len(known):].sum()
    n_k, n_o = len(known), len(novel)
    return float((r_novel - n_o * (n_o - 1) / 2) / (n_k * n_o))


@torch.no_grad()
def all_scores(model, dls, df, device, num_workers=32):
    """One forward pass, every rule. Only the reductions are kept, never the [N, 12041] matrix."""
    from lepinet.test import dl_num_workers

    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    dl = dls.test_dl(df, num_workers=nw)
    print(f"  dataloader: num_workers={nw}, batches={len(dl)}")
    model.to(device).eval()
    out = {k: [] for k in SCORE_RULES}
    degenerate: set[str] = set()
    t0 = time.time()
    for batch in dl:
        logits = model(batch[0].to(device))
        z = (logits[0] if isinstance(logits, (list, tuple)) else logits).float()
        if not degenerate:
            degenerate = degenerate_rules(z)
        for name, fn in SCORE_RULES.items():
            out[name].append(fn(z).cpu().numpy())
    dt = time.time() - t0
    n = sum(len(a) for a in out["max"])
    print(f"  inference: {n} images in {dt:.1f}s = {n / max(dt, 1e-9):.1f} img/s")
    return {k: np.concatenate(v) for k, v in out.items()}, degenerate


def _register_dev_heads():
    """dev/-registered heads (marginal, marginal_arcface, hierarchical) are invisible to the package
    unless dev/050 is imported. Scoring a checkpoint trained with one otherwise dies in build_head."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "dev050_heads", Path(__file__).with_name("050_hierarchical_heads.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main(a):
    import pandas as pd

    _register_dev_heads()

    from lepinet.data import DEFAULT_LEVELS, filter_df, make_dls
    from lepinet.test import load_model, resolve_checkpoint_path

    # Deliberately mirrors dev/052's setup line for line. The AUROCs here are meant to be compared
    # against the numbers that script produced (0.9068 / 0.8132 / ...), so any divergence in which
    # images are scored would silently invalidate the comparison this experiment exists to make.
    ckpt = torch.load(resolve_checkpoint_path(a.model), map_location="cpu", weights_only=False)
    levels = ckpt.get("levels", DEFAULT_LEVELS)
    vocabs = ckpt["vocabs"]
    known = {str(v) for v in vocabs[levels[0]]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = filter_df(pd.read_parquet(a.parquet), keep_in=["0"], levels=levels)
    if "image_path" not in df.columns:
        df["image_path"] = df[levels[0]].astype(str) + "/" + df["filename"]
    for lv in levels:
        df[lv] = df[lv].astype(str)
    df["is_valid"] = np.arange(len(df)) % 5 == 0
    is_novel = ~df[levels[0]].isin(known)
    print(f"{len(df)} images | novel-species images: {int(is_novel.sum())} "
          f"({100 * is_novel.mean():.1f}%)")

    dls = make_dls(df[["image_path", "is_valid", *levels]].reset_index(drop=True),
                   vocabs, a.img_dir, int(a.img_size * 460 / 256), a.img_size, 128,
                   a.num_workers, lowmem=False, levels=levels)
    model, _ = load_model(ckpt, img_size=a.img_size)
    scores, degenerate = all_scores(model, dls, df, device, a.num_workers)

    nv = is_novel.to_numpy()
    res = {"n_known": int((~nv).sum()), "n_novel": int(nv.sum()),
           "head": ckpt.get("head"), "degenerate": sorted(degenerate), "auroc": {}}
    for name, s in scores.items():
        res["auroc"][name] = auroc(s[nv], s[~nv])

    print("\nAUROC by scoring rule (same embedding, same images):")
    live = {k: v for k, v in res["auroc"].items() if k not in degenerate}
    best = max(live, key=lambda k: live[k])
    for name, v in sorted(res["auroc"].items(), key=lambda kv: -kv[1]):
        if name in degenerate:
            why = ("constant (log-probs sum to 1)" if name == "energy"
                   else "identical to max by construction")
            print(f"  {name:8s} {v:.4f}   -- NOT MEANINGFUL for this head: {why}")
            continue
        mark = "  <- best" if name == best else ""
        print(f"  {name:8s} {v:.4f}   ({v - res['auroc']['max']:+.4f} vs max-logit){mark}")
    if degenerate:
        print(f"  (head emits log-probabilities; {sorted(degenerate)} excluded from the ranking)")
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", default="ood_scoring_rules.json")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=32)
    main(ap.parse_args())
