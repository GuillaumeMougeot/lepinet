"""Grouped, leak-free splits of the trap dataset — the prerequisite for B3 (self-training).

B3 pseudo-labels unlabelled trap images and mixes them into training. Two ways that produces a
number that is not real, both of which this module exists to prevent:

**1. Near-duplicate leakage.** The trap is a timelapse. Consecutive frames of the same night, and
every crop of a single snapshot, are near-identical moths. A random split puts those on both sides
and can manufacture a double-digit phantom gain. The filenames carry what is needed to avoid it::

    crop_TRAPNAME_LV3_IMAGENAME_20220811015000-88-snapshot_CROPNUMBER_5_UUID_<uuid>.jpg
                   ^^^          ^^^^^^^^^^^^^^ ^^                      ^
                   trap         timestamp      snapshot                crop

so the grouping key is **(trap, night)** — stronger than (trap, snapshot), because frames minutes
apart on the same night are still near-duplicates. A "night" runs midday-to-midday, since moths fly
across midnight and a calendar date would cut a single night in half.

**2. Benchmark contamination.** Every shifted number in `RESULTS.md` was measured on *all* 47,905
trap images. If B3 trains on any of them, that benchmark is burnt and no later comparison is honest.
So this splits the trap data into `adapt` (pseudo-labelled, trained on) and `probe` (never touched
by training, the only set B3 may report on) — and **the baselines must be re-scored on `probe`
alone** to have a comparable reference. `--emit-probe-parquet` writes that eval set.

A third guard, for the risk the owner named: `--holdout-species` keeps a fraction of species out of
`adapt` entirely, so `probe` contains taxa the adaptation never saw. If B3's gain comes only from
specialising on the ~500 trap species, it will show up as a gap between the two probe subsets.

    python dev/064_flemming_groups.py --img-dir data/flemming/images --out data/flemming/splits
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

#: trap id, then the 14-digit timestamp, then the snapshot index
FNAME = re.compile(r"crop_TRAPNAME_(?P<trap>[^_]+)_IMAGENAME_(?P<ts>\d{14})-(?P<snap>\d+)-snapshot")


def parse_images(img_dir: Path) -> pd.DataFrame:
    """One row per crop, with the fields the grouping needs. Unparseable names are reported, not
    silently dropped — a rename upstream would otherwise quietly shrink the dataset."""
    rows, bad = [], []
    for p in sorted(img_dir.rglob("*.jpg")):
        m = FNAME.search(p.name)
        if not m:
            bad.append(p.name)
            continue
        ts = pd.to_datetime(m["ts"], format="%Y%m%d%H%M%S")
        # A moth night spans midnight, so shift 12 h before taking the date: everything from noon
        # to noon counts as one night and cannot be split across the boundary.
        rows.append({"image_path": str(p.relative_to(img_dir)), "species_folder": p.parent.name,
                     "trap": m["trap"], "timestamp": ts, "snapshot": int(m["snap"]),
                     "night": (ts - pd.Timedelta(hours=12)).date().isoformat()})
    if bad:
        print(f"WARNING: {len(bad)} filenames did not match the expected pattern, e.g. {bad[:2]}")
    df = pd.DataFrame(rows)
    df["group"] = df["trap"] + "|" + df["night"]
    return df


def split_groups(df: pd.DataFrame, adapt_frac: float, holdout_species_frac: float, seed: int):
    """Assign whole groups to adapt/probe, then remove held-out species from adapt only."""
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(df["group"].unique()))
    rng.shuffle(groups)
    n_adapt = int(round(len(groups) * adapt_frac))
    adapt_groups = set(groups[:n_adapt].tolist())

    species = np.array(sorted(df["species_folder"].unique()))
    rng.shuffle(species)
    held = set(species[:int(round(len(species) * holdout_species_frac))].tolist())

    df = df.copy()
    df["split"] = np.where(df["group"].isin(adapt_groups), "adapt", "probe")
    df["species_heldout"] = df["species_folder"].isin(held)
    # A held-out species must not appear in adapt at all -- otherwise "unseen" means nothing.
    df.loc[(df["split"] == "adapt") & df["species_heldout"], "split"] = "unused"
    return df, held


def check(df: pd.DataFrame) -> list[str]:
    """Assertions that would each, if violated, silently inflate B3's result."""
    problems = []
    both = set(df[df.split == "adapt"]["group"]) & set(df[df.split == "probe"]["group"])
    if both:
        problems.append(f"{len(both)} groups appear in BOTH adapt and probe (near-duplicate leak)")
    leaked = set(df[df.split == "adapt"]["species_folder"]) & set(df[df.species_heldout]["species_folder"])
    if leaked:
        problems.append(f"{len(leaked)} held-out species still present in adapt")
    if not (df[df.split == "probe"]["species_heldout"]).any():
        problems.append("probe contains no held-out species -- the specialisation check is inert")
    for s in ("adapt", "probe"):
        if (df.split == s).sum() == 0:
            problems.append(f"split {s!r} is empty")
    return problems


def main(a):
    img_dir = Path(a.img_dir)
    df = parse_images(img_dir)
    print(f"{len(df)} crops | {df.trap.nunique()} traps | {df.night.nunique()} nights | "
          f"{df.group.nunique()} (trap,night) groups | {df.species_folder.nunique()} species")
    print(f"crops per group: median {df.groupby('group').size().median():.0f}, "
          f"max {df.groupby('group').size().max()}")

    df, held = split_groups(df, a.adapt_frac, a.holdout_species_frac, a.seed)
    for s in ("adapt", "probe", "unused"):
        d = df[df.split == s]
        print(f"  {s:7s} {len(d):6d} crops  {d.group.nunique():4d} groups  "
              f"{d.species_folder.nunique():3d} species")
    print(f"  held-out species: {len(held)}  "
          f"(probe crops of held-out species: {int((df.split=='probe') & df.species_heldout).sum() if False else len(df[(df.split=='probe') & df.species_heldout])})")

    problems = check(df)
    if problems:
        print("\nLEAKAGE CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nleakage check: OK (no group spans adapt/probe; held-out species absent from adapt)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "flemming_groups.parquet")
    (out / "split_summary.json").write_text(json.dumps({
        "n_crops": len(df), "n_groups": int(df.group.nunique()),
        "grouping": "(trap, night) with nights running midday-to-midday",
        "adapt_frac": a.adapt_frac, "holdout_species_frac": a.holdout_species_frac, "seed": a.seed,
        "counts": {s: int((df.split == s).sum()) for s in ("adapt", "probe", "unused")},
        "held_out_species": sorted(held),
    }, indent=2))
    print(f"wrote {out/'flemming_groups.parquet'} and {out/'split_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapt-frac", type=float, default=0.7)
    ap.add_argument("--holdout-species-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
