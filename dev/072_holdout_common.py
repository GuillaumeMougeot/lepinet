"""C3b — build a training set with **common** taxa deliberately held out.

C3 measured novelty detection stratified by taxonomic distance and found it monotone: `far` is easy,
`near` is hard. But its novel taxa were the *rare* ones — every species below the `min_img_per_spc=50`
training floor — so "novel" was confounded with "few images, and probably worse ones". A rare species
might be detected as novel because the model has not seen it, or because its photographs are unusual.
C3 cannot tell those apart, and said so.

This script builds the controlled version. It removes taxa that are **common** (>= `--min-count`
training images, well above the 50-image floor) at three taxonomic ranks, so that after retraining
they are novel for exactly one reason: they were withheld.

    near  individual species, whose genus keeps at least one other species
    mid   whole genera, whose family keeps at least one other genus
    far   whole families

The removal is from **all** splits, not just train, so the checkpoint's vocabulary genuinely excludes
them; a species present in the frame but absent from training would otherwise enter the vocab with
zero examples and train a dead prototype. Evaluation then runs `dev/059` against the *unmodified*
parquet, whose `stratify()` labels each row by what the checkpoint's vocab contains — so the held-out
taxa reappear as near/mid/far with no further wiring.

    python dev/072_holdout_common.py --parquet global.parquet --out c3b_train.parquet \\
        --manifest c3b_holdout.json

Cost note: this needs a full retraining (~6.4 h at 20 M), which is why it sat in the backlog behind
things that reuse an existing checkpoint. The result it buys is a clean answer to "is the monotonicity
a property of the taxonomy, or of the long tail?", and C3's headline claim rests on that distinction.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

LEVELS = ["speciesKey", "genusKey", "familyKey"]


def pick(df: pd.DataFrame, a) -> tuple[dict, dict, pd.Index]:
    """Choose held-out taxa at three ranks.

    Returns `(taxa, score_species, drop)`. The distinction matters: **whole taxa are removed** (a
    family is only unseen if all of it is gone) but **only their common members are scored**. This
    dataset has no small family made of common species -- families are either rare singletons or
    enormous (see the note in `main`) -- so a whole-family hold-out inevitably drags in the family's
    own long tail. Scoring the common members only keeps all three strata matched on image count,
    which is the entire point of C3b.
    """
    train = df[df["set"].astype(str) != a.test_set]
    per_sp = train.groupby(LEVELS, observed=True).size().rename("n").reset_index()
    common = per_sp[per_sp["n"] >= a.min_count]
    print(f"{len(per_sp)} species in train | {len(common)} with >= {a.min_count} images")

    sp_per_gn = per_sp.groupby("genusKey", observed=True).size()
    gn_per_fm = per_sp.groupby("familyKey", observed=True).size()
    taxa = {"far": [], "mid": [], "near": []}

    # far -- whole families, the smallest ones whose *typical* species is common. The median filter
    # is what rules out the tiny all-rare families; without it, sorting by size picks families of
    # 40-image singletons and `far` becomes a rare-taxa benchmark again.
    fam = per_sp.groupby("familyKey", observed=True)["n"].agg(["sum", "median", "size"])
    fam_ok = fam[(fam["median"] >= a.min_count) & (fam["sum"] <= a.max_images_far)
                 & (fam["size"] >= a.min_species_far)]
    taxa["far"] = list(fam_ok.sort_values("sum").index[: a.n_far])

    rest = per_sp[~per_sp["familyKey"].isin(taxa["far"])]

    # mid -- whole genera, only where the family survives without them.
    gen = rest.groupby(["familyKey", "genusKey"], observed=True)["n"].agg(["sum", "median", "size"])
    gen_ok = gen[(gen["median"] >= a.min_count) & (gen["size"] >= a.min_species_mid)]
    gen_ok = gen_ok[[gn_per_fm.get(f, 0) - 1 >= 1 for f, _ in gen_ok.index]]
    taxa["mid"] = [g for _, g in gen_ok.sort_values("sum").index[: a.n_mid]]

    rest = rest[~rest["genusKey"].isin(taxa["mid"])]

    # near -- single common species whose genus keeps a sibling. Sampled rather than taken from one
    # end of the size range, so `near` is not secretly "the smallest common species".
    near_ok = rest[(rest["n"] >= a.min_count) & (rest["genusKey"].map(sp_per_gn).fillna(0) >= 2)]
    taxa["near"] = list(near_ok.sample(min(a.n_near, len(near_ok)), random_state=0)["speciesKey"])

    drop = (df["familyKey"].isin(taxa["far"])
            | df["genusKey"].isin(taxa["mid"])
            | df["speciesKey"].isin(taxa["near"]))

    score = {
        "far": list(common[common["familyKey"].isin(taxa["far"])]["speciesKey"]),
        "mid": list(common[common["genusKey"].isin(taxa["mid"])]["speciesKey"]),
        "near": list(taxa["near"]),
    }
    return taxa, score, drop


def main(a):
    df = pd.read_parquet(a.parquet)
    for lv in LEVELS:
        df[lv] = df[lv].astype(str)
    taxa, score, drop = pick(df, a)

    kept = df[~drop]
    held = df[drop]
    print(f"\nheld out: {len(taxa['far'])} families, {len(taxa['mid'])} genera, "
          f"{len(taxa['near'])} species")
    test = held["set"].astype(str) == a.test_set
    for name in ("far", "mid", "near"):
        sc = held[held["speciesKey"].isin(score[name])]
        print(f"  {name:5s} removed {held[held['speciesKey'].isin(
            score[name])].shape[0]:7d} | scored {len(score[name]):4d} species, "
              f"{sc[sc['set'].astype(str) == a.test_set].shape[0]:6d} test images")
    print(f"\n{len(df)} rows -> {len(kept)} kept ({100 * len(held) / len(df):.2f} % removed); "
          f"{int(test.sum())} held-out test images in total")

    # The perturbation has to be small, or C3b measures "a worse model" instead of "a fair hold-out".
    frac = len(held) / len(df)
    if frac > a.max_frac:
        raise SystemExit(f"removed {frac:.1%} of the data, over the {a.max_frac:.0%} budget -- "
                         "lower --n-near/--n-mid or --max-images-far")
    # Every rank must survive its own removal, or `stratify` cannot assign the stratum it was
    # built for: a `near` species whose genus vanished is really `mid`.
    gone_gn = set(df[df["speciesKey"].isin(taxa["near"])]["genusKey"]) - set(kept["genusKey"])
    if gone_gn:
        raise SystemExit(f"{len(gone_gn)} genera were emptied by the `near` hold-out; "
                         "those species are mid, not near")
    gone_fm = set(df[df["genusKey"].isin(taxa["mid"])]["familyKey"]) - set(kept["familyKey"])
    if gone_fm:
        raise SystemExit(f"{len(gone_fm)} families emptied by the `mid` hold-out")

    kept.to_parquet(a.out, index=False)
    with open(a.manifest, "w") as f:
        json.dump({"taxa": {k: [str(x) for x in v] for k, v in taxa.items()},
                   "score_species": {k: [str(x) for x in v] for k, v in score.items()},
                   "min_count": a.min_count}, f, indent=2)
    print(f"wrote {a.out} and {a.manifest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--test-set", default="0")
    p.add_argument("--min-count", type=int, default=200,
                   help="training images a species needs to count as 'common' (floor is 50)")
    p.add_argument("--n-near", type=int, default=120)
    p.add_argument("--n-mid", type=int, default=15)
    p.add_argument("--n-far", type=int, default=3)
    p.add_argument("--min-species-far", type=int, default=2)
    p.add_argument("--min-species-mid", type=int, default=2)
    p.add_argument("--max-images-far", type=int, default=30000)
    p.add_argument("--max-frac", type=float, default=0.05)
    main(p.parse_args())
