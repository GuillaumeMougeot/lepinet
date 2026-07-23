"""Phase B1 of the lepi-app plan: is marginalizing beats reading the genus/family heads?

The claim under test, from journal/2026-07-lepi-app.md's Review section: because the winning
model uses **independent** heads, its three predictions are not guaranteed consistent -- it
can name a species whose genus contradicts the genus head. Deriving genus and family from
the species distribution instead,

    P(genus g) = sum over species s in g of P(s)

is consistent by construction, and was *asserted* to be at least as accurate. That assertion
was an expectation, not a measurement, and the shipped artifact would be ~5 M parameters
lighter if it held -- so it needs proving before anything is deleted.

What this measures, per level (genus, family), on the held-out fold:
  - `direct`:      argmax of that level's own head (what dev/032 reports today)
  - `marginal`:    argmax of the species posterior summed over children
  - `consistency`: how often the direct heads' argmaxes actually disagree with the species
                   argmax's true parent -- the size of the problem the change would fix

Marginalization is done in log-space (`logsumexp` over children) rather than by summing
softmax probabilities: with 12,041 classes the tail underflows fp32 and the sum quietly
becomes dominated by rounding, which would understate the marginal path for the wrong reason.

Usage:
    python dev/042_marginalize.py --checkpoint <path/to/*.pt> -n 20000
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing

torch.multiprocessing.set_start_method("fork", force=True)  # see dev/030 for why

from sklearn.metrics import f1_score

v4 = importlib.import_module("028_lepi_hierarchical_multihead_v4")
mod032 = importlib.import_module("032_hierarchical_heads_test")
mod040 = importlib.import_module("040_onnx_export")

HIERARCHY_LEVELS = v4.HIERARCHY_LEVELS
LEVEL_NAMES = ["species", "genus", "family"]


def scatter_logsumexp(log_probs, parent_idx, n_parents):
    """log P(parent) = logsumexp over its children, numerically stable, batched.

    torch has no scatter-logsumexp, so this is the standard shift-by-max formulation done
    manually: subtract each parent's max child logit (scatter_reduce amax), exponentiate,
    scatter-add, log, add the max back.
    """
    n, _ = log_probs.shape
    device = log_probs.device
    idx = parent_idx.unsqueeze(0).expand(n, -1)

    maxes = torch.full((n, n_parents), float("-inf"), device=device, dtype=log_probs.dtype)
    maxes = maxes.scatter_reduce(1, idx, log_probs, reduce="amax", include_self=True)
    shifted = (log_probs - maxes.gather(1, idx)).exp()
    sums = torch.zeros((n, n_parents), device=device, dtype=log_probs.dtype)
    sums = sums.scatter_add(1, idx, shifted)
    return sums.log() + maxes


def evaluate(checkpoint_path, parquet_path, img_dir, n, test_set, min_img_per_spc,
             batch_size, num_workers, img_size, aug_img_size, seed, out_json=None):
    checkpoint_path = mod032.resolve_model_path(str(checkpoint_path))
    img_dir = Path(img_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, vocabs, n_classes = mod040.build_model(checkpoint)
    tax = mod040.build_taxonomy(checkpoint, vocabs)

    sp2gn = torch.tensor(tax["parents"]["species_to_genus"], device=device)
    gn2fm = torch.tensor(tax["parents"]["genus_to_family"], device=device)
    if (sp2gn < 0).any() or (gn2fm < 0).any():
        raise SystemExit("Parent map has unresolved entries; marginalization would be wrong.")
    n_gn, n_fm = n_classes[1], n_classes[2]

    print(f"Loading test fold '{test_set}' from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = v4.filter_df(df, keep_in=[test_set], min_img_per_spc=min_img_per_spc, family_filter=[])
    df["image_path"] = df["speciesKey"].astype(str) + "/" + df["filename"]
    present = df["image_path"].map(lambda p: (img_dir / p).exists())
    df = df[present]
    df = mod032.filter_known_species(df, vocabs)
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    df = df.reset_index(drop=True)
    for level in HIERARCHY_LEVELS:
        df[level] = df[level].astype(str)
    df["is_valid"] = np.arange(len(df)) % 5 == 0  # dummy split; test_dl ignores it
    test_df = df[["image_path", "is_valid", *HIERARCHY_LEVELS]]
    print(f"Evaluating on {len(test_df)} images.")

    dls = v4.make_dls(test_df, vocabs, img_dir, aug_img_size, img_size, batch_size,
                      num_workers, lowmem=False)
    dl = dls.test_dl(test_df, num_workers=num_workers)
    model.to(device).eval()

    sp, gn_d, fm_d, gn_m, fm_m = [], [], [], [], []
    with torch.no_grad():
        for batch in dl:
            xb = batch[0].to(device)
            logits = model(xb)
            log_sp = torch.log_softmax(logits[0].float(), dim=1)
            log_gn_marg = scatter_logsumexp(log_sp, sp2gn, n_gn)
            log_fm_marg = scatter_logsumexp(log_gn_marg, gn2fm, n_fm)

            sp.append(log_sp.argmax(1).cpu())
            gn_d.append(logits[1].float().argmax(1).cpu())
            fm_d.append(logits[2].float().argmax(1).cpu())
            gn_m.append(log_gn_marg.argmax(1).cpu())
            fm_m.append(log_fm_marg.argmax(1).cpu())

    sp, gn_d, fm_d, gn_m, fm_m = (torch.cat(t).numpy() for t in (sp, gn_d, fm_d, gn_m, fm_m))

    idx = {level: {str(k): i for i, k in enumerate(vocabs[level])} for level in HIERARCHY_LEVELS}
    y = {name: np.array([idx[level][k] for k in test_df[level]])
         for name, level in zip(LEVEL_NAMES, HIERARCHY_LEVELS)}

    sp2gn_np = np.asarray(tax["parents"]["species_to_genus"])
    gn2fm_np = np.asarray(tax["parents"]["genus_to_family"])

    def scores(true, pred):
        return {
            "acc": float((pred == true).mean()),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        }

    results = {
        "n_images": int(len(test_df)),
        "checkpoint": str(checkpoint_path),
        "species": scores(y["species"], sp),
        "genus": {"direct": scores(y["genus"], gn_d), "marginal": scores(y["genus"], gn_m)},
        "family": {"direct": scores(y["family"], fm_d), "marginal": scores(y["family"], fm_m)},
        "consistency": {
            # How often the *direct* heads contradict the species argmax's true parent. This
            # is the user-visible defect the change removes, and it is worth reporting even if
            # accuracy came out neutral: an incoherent triple looks like a bug to a user
            # regardless of which of the three names happens to be right.
            "genus_disagrees_with_species_parent": float((gn_d != sp2gn_np[sp]).mean()),
            "family_disagrees_with_genus_parent": float((fm_d != gn2fm_np[gn_d]).mean()),
        },
    }

    print("\n" + "=" * 72)
    print(f"Marginal vs direct heads  ({results['n_images']} images)")
    print("=" * 72)
    print(f"species (reference): acc={results['species']['acc']:.4f}  "
          f"macroF1={results['species']['macro_f1']:.4f}")
    for level in ("genus", "family"):
        d, m = results[level]["direct"], results[level]["marginal"]
        print(f"\n{level}:")
        print(f"  direct    acc={d['acc']:.4f}  macroF1={d['macro_f1']:.4f}")
        print(f"  marginal  acc={m['acc']:.4f}  macroF1={m['macro_f1']:.4f}")
        print(f"  delta     acc={m['acc']-d['acc']:+.4f}  macroF1={m['macro_f1']-d['macro_f1']:+.4f}")
    c = results["consistency"]
    print(f"\ninconsistency of the direct heads:")
    print(f"  genus  != parent(species argmax): {c['genus_disagrees_with_species_parent']:.4f}")
    print(f"  family != parent(genus argmax):   {c['family_disagrees_with_genus_parent']:.4f}")
    print("=" * 72)

    if out_json:
        Path(out_json).write_text(json.dumps(results, indent=2))
        print(f"Wrote {out_json}")
    return results


def cli():
    p = argparse.ArgumentParser(description="Marginal vs direct genus/family heads (Phase B1).")
    p.add_argument("-c", "--checkpoint", required=True)
    p.add_argument("--parquet", default="../data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet")
    p.add_argument("--img-dir", default="../data/global/images")
    p.add_argument("-n", type=int, default=20000, help="Images to evaluate (0 = all available).")
    p.add_argument("--test-set", default="0")
    # 0, matching dev/032's test path -- the one that produced the project's 0.9148 headline.
    # A non-zero value here silently restricts the test fold to well-represented species (50
    # gives 3,696 species / 484k images instead of 12,632 / 633k) and inflates every number.
    p.add_argument("--min-img-per-spc", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--aug-img-size", type=int, default=460)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default=None)
    a = p.parse_args()
    evaluate(a.checkpoint, a.parquet, a.img_dir, a.n, a.test_set, a.min_img_per_spc,
             a.batch_size, a.num_workers, a.img_size, a.aug_img_size, a.seed, a.out_json)


if __name__ == "__main__":
    cli()
