"""Deployment restriction: what is a regional species checklist worth?

The paper's stated subject is deployment, and every number in it evaluates a **12,041-class global
Lepidoptera head**. No real deployment looks like that. A Danish user running a moth trap will meet
the ~500 species that occur in Denmark, and that list is *known in advance* -- national checklists
are published, curated, and free. Constraining the classifier to it costs nothing at inference and
throws away 96 % of the ways the model can be wrong.

This measures that. It is inference-only: no retraining, one forward pass per configuration.

## What is restricted, and what is NOT

**We restrict the LABEL SPACE, not the evaluation set.** The eval parquet, the images, the species
present, and therefore the macro-F1 denominator are all byte-identical between arms. Only the set of
labels the model is *allowed to emit* changes.

This distinction is the one that has bitten this project hardest (`--min-img-per-spc` on a test fold
silently dropped the tail out of a macro average and inflated it by ~3 pt), so it is worth being
explicit: filtering the eval set changes the question; restricting the label space changes the model.
The assertion below enforces it -- the row count and species set of the scored frame must not move.

## Why the checklist must not be derived from the eval split

Restricting to exactly the species present in the probe fold (368) would leak the answer set and
report a meaningless number. The honest checklist is a **regional** artefact defined independently of
how we split: the 486 species of the full trap corpus, i.e. "moths recorded at these Danish traps".
Evaluating the 368-species probe under a 486-species checklist is legitimate -- the checklist is
strictly larger than the eval set and was not chosen by looking at it.

`--checklist-from` therefore takes a parquet whose species column defines the region, and it should
normally be the *full* trap corpus, never the fold being scored.

## The curve, not the point

A single restricted number is not very interesting -- of course it is higher. The question is the
**shape**: how does accuracy scale with checklist size, and how much slack can a checklist carry
before the benefit disappears? `--pad-to` grows the checklist with random extra species drawn from
the global vocabulary, which simulates a checklist that is over-inclusive (a national list covering
a whole country when the trap sees one habitat). That curve is what tells a practitioner whether to
bother curating.

## The cost, which must be reported alongside

A restricted head **cannot** predict a species outside the checklist. Every genuinely novel taxon is
now a guaranteed error rather than a possible one, so this trades open-set recall for closed-set
accuracy. That is exactly the tension the paper is about, so the script reports the restricted
model's score on the held-out-species fold too -- the arm where the trade should hurt most.

    python dev/081_restricted_label_set.py \
        --model '/work/.../B8/*.pt' \
        --parquet /work/.../flemming_probe.parquet \
        --checklist-from /work/.../flemming_probe.parquet /work/.../flemming_adapt.parquet \
        --img-dir /work/flemming/images \
        --out-dir /work/lepinet/data/ucloud_preds/D1-probe \
        --pad-to -1 0 1000 2000 4000   # -1 = unrestricted (12,041); 0 = checklist as-is
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class RestrictedHead(nn.Module):
    """Wrap a model so the species head can only emit labels in ``keep``.

    Masking happens on the **species logits, before the softmax**, which is the only correct place:
    the posterior then renormalises over the allowed set, and because every coarser rank is derived
    from it by log-sum-exp over children (`lepinet.test.predict_df`), the genus and family marginals
    automatically sum over retained children only. Masking after the softmax, or masking the coarse
    heads separately, would break the coherence that makes marginalisation worth using.

    ``-inf`` rather than a large negative constant: the z-score head's logits already reach ~35, so
    a "large" constant is a guess, and ``-inf`` is exact under softmax.
    """

    def __init__(self, model: nn.Module, keep_idx: np.ndarray, n_species: int):
        super().__init__()
        self.model = model
        mask = torch.zeros(n_species, dtype=torch.bool)
        mask[torch.as_tensor(keep_idx, dtype=torch.long)] = True
        self.register_buffer("keep", mask)

    def forward(self, x):
        out = self.model(x)
        out = list(out) if isinstance(out, (list, tuple)) else [out]
        sp = out[0]
        out[0] = sp.masked_fill(~self.keep.to(sp.device).unsqueeze(0), float("-inf"))
        return out


def build_checklist(checklist_parquets: list[str], species_col: str, vocab: list[str],
                    pad_to: int, seed: int) -> np.ndarray:
    """Indices into the model's species vocabulary that the classifier may emit.

    Takes the **union** over several parquets, because the region is defined by every trap and every
    night, not by whichever split we happen to be scoring. For the Danish traps that is
    `flemming_probe` union `flemming_adapt` = the full 486-species trap list.

    Species in the regional list that the model has never heard of are silently dropped -- it cannot
    predict them either way, and keeping them in the count would overstate the checklist's size.
    """
    regional: set[str] = set()
    for f in checklist_parquets:
        d = pd.read_parquet(f, columns=[species_col])
        regional |= {str(s) for s in d[species_col].dropna().unique()}
    v2i = {str(v): i for i, v in enumerate(vocab)}
    keep = sorted(v2i[s] for s in regional if s in v2i)
    dropped = len(regional) - len(keep)
    print(f"  checklist: {len(regional):,} regional species, {len(keep):,} in vocabulary "
          f"({dropped:,} unknown to the model)")

    if pad_to and pad_to > len(keep):
        rng = np.random.default_rng(seed)
        rest = np.setdiff1d(np.arange(len(vocab)), np.asarray(keep))
        extra = rng.choice(rest, size=min(pad_to - len(keep), len(rest)), replace=False)
        keep = np.sort(np.concatenate([np.asarray(keep), extra]))
        print(f"  padded to {len(keep):,} with random non-regional species (slack simulation)")
    return np.asarray(keep, dtype=np.int64)


def main(a):
    import lepinet.test as T

    # --- one shared guard, applied to every arm -------------------------------------------------
    # `evaluate` is what actually loads the checkpoint and builds the frame. We wrap the model it
    # builds rather than reimplementing the eval loop, so the metric path is bit-identical to every
    # other number in the project.
    original_predict_df = T.predict_df
    state: dict = {}

    def patched_predict_df(model, dls, test_df, vocabs, levels, device, **kw):
        n_rows, n_spc = len(test_df), test_df[a.species_col].nunique()
        if "denom" not in state:
            state["denom"] = (n_rows, n_spc)
            print(f"  scored frame: {n_rows:,} rows, {n_spc:,} species")
        assert state["denom"] == (n_rows, n_spc), (
            f"the evaluation frame changed between arms {state['denom']} -> {(n_rows, n_spc)}; "
            "this script restricts the LABEL SPACE, never the eval set -- refusing to continue")
        keep = state.get("keep")
        if keep is not None:
            model = RestrictedHead(model, keep, len(vocabs[levels[0]])).to(device).eval()
        return original_predict_df(model, dls, test_df, vocabs, levels, device, **kw)

    T.predict_df = patched_predict_df

    results = []
    for pad in a.pad_to:
        tag = "unrestricted" if pad == -1 else (f"checklist" if pad == 0 else f"checklist+pad{pad}")
        print(f"\n=== {tag} ===", flush=True)

        if pad == -1:
            state["keep"] = None
        else:
            # vocab comes from the checkpoint; read it once via a throwaway load
            if "vocab" not in state:
                ck = torch.load(_resolve(a.model), map_location="cpu", weights_only=False)
                vocabs = ck.get("vocabs") or ck["cfg"]["vocabs"]
                state["vocab"] = [str(v) for v in vocabs[a.level]]
                print(f"  model vocabulary: {len(state['vocab']):,} species")
            state["keep"] = build_checklist(a.checklist_from, a.species_col,
                                            state["vocab"], pad, a.seed)

        out = T.evaluate(
            model_path=_resolve(a.model), parquet_path=a.parquet, img_dir=a.img_dir,
            out_dir=a.out_dir, eval_name=tag, test_set=a.test_set,
            min_img_per_spc=0,               # NEVER filter the eval fold
            batch_size=a.batch_size, aug_img_size=a.aug_img_size, img_size=a.img_size,
            num_workers=a.num_workers, drop_unknown_species=False, marginal=True,
        )
        m = json.loads((Path(out) / "metrics.json").read_text())
        f1 = m.get("species_macro_f1") or m.get("macro_f1") or m
        n_allowed = len(state["vocab"]) if pad == -1 else len(state["keep"])
        results.append({"arm": tag, "n_allowed_labels": n_allowed, "species_macro_f1": f1})
        print(f"  -> {tag}: {f1}", flush=True)

    print("\n" + json.dumps(results, indent=2, default=str))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / "restriction_curve.json").write_text(json.dumps(results, indent=2, default=str))


def _resolve(pattern: str) -> str:
    from glob import glob
    hits = sorted(glob(pattern))
    if not hits:
        raise SystemExit(f"no checkpoint matches {pattern!r}")
    return hits[-1]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--parquet", required=True, help="the fold to score; never filtered")
    p.add_argument("--checklist-from", required=True, nargs="+",
                   help="parquet(s) defining the region; the union is taken. Must be a superset of, "
                        "and not derived from, the fold being scored.")
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--species-col", default="speciesKey")
    p.add_argument("--level", default="species")
    p.add_argument("--test-set", default="0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--aug-img-size", type=int, default=460)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pad-to", type=int, nargs="+", default=[-1, 0, 1000, 2000, 4000],
                   help="-1 = unrestricted baseline; 0 = the checklist as-is; N = pad to N species")
    main(p.parse_args())
