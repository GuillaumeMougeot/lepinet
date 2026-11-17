"""Should the 50-image species floor go? The ToL part (owner's question, 2026-11-17; global_lepi: dev/110).

    python dev/111_tol_species_floor.py        # CPU, two streaming passes over metadata.parquet, about 3 GB

With hierarchical supervision a species under the floor could still be trained if its genus (or family) has enough
images. Keys are full paths, as `train/`. Held-out taxa stay out, including below-floor species inside a held-out
genus (mid) or family (far). "Usable" = `exclude` empty at species rank (train/'s rows).

(a) tol256-v1 on disk: below-floor species (exclude = below_floor) whose genus / family has >= 50 usable images,
    and how the rows that would come in fare against the screens the freeze applied only to rows not yet excluded.
(b) the TreeOfLife-200M catalog (dev/102's catalog_taxa, rank tuples corrected by its Fixer; no COL XR renames, so
    Lepidoptera keys are approximate): species under 50 catalog rows, not trained, not held out, under the same rule,
    and how many of their rows are on disk. Species already trained on disk but under 50 catalog rows (668; 663 under the family
    rule, 646 of those Lepidoptera: our GBIF download and the renames) are not candidates.
(c) no floor at all: images per family (species capped at 2,000; a path with no family name is not a family).

Result (2026-11-17, tol256-v1):

    (a) 9,662 below-floor species (76,244 rows) outside held-out taxa
        genus >= 50 usable:   +4,442 species, +43,849 images (4,797 below-floor species have no trained genus)
        family >= 50 usable:  +7,712 species, +66,022 images (1,658 have no trained family)
        family-rule rows: benchmark overlap 34; pixel duplicate of another row 45; URL duplicate 5;
        no content hash (unscreened for pixels) 1,671; Lepidoptera 1,993 (need the lepinet test-fold check)
    (b) 655,084 catalog species under 50, not trained, not held out (6,324,263 rows)
        genus >= 50 usable on disk:   431,026 species, 4,756,070 rows; on disk 40,678 (0.9 %) in 4,193 species
        family >= 50 usable on disk:  598,779 species, 6,026,357 rows; on disk 61,897 (1.0 %) in 7,357 species
    (c) families, min / p1 / median / under 50
        disk, species rank (usable + below_floor)   5,558: 1 / 1 / 453 / 975 (17.5 %)
        disk, plus genus- and family-rank records   5,897: 1 / 1 / 436 / 1,059 (18.0 %)
        catalog, species rank                      11,849: 1 / 1 / 62 / 5,604 (47.3 %)
        catalog, plus genus- and family-rank rows  21,495: 1 / 1 / 17 / 13,821 (64.3 %)

(b)'s on-disk counts sit a little under (a)'s because a few below-floor keys match no catalog key under 50 (renames,
merges). Catalog family paths split little: 11,849 paths are 11,517 names (disk 5,558 / 5,500).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TOL = Path("/data/datasets/tol256")
CATALOG_TAXA = Path("/data/au761367/tol256s/fix_build/catalog_taxa.parquet")   # dev/102 catalog
BENCH = Path("/data/au761367/tol256s/freeze_build/bench")                      # dev/103 bench
C = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
FLOOR, CAP, BENCH_HAMMING = 50, 2000, 6


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "dev" / file)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def keys(df):
    """Family, genus and species full paths."""
    v = df[C].fillna("").astype(str)
    k, out = v["kingdom"], {}
    for i, c in enumerate(C[1:], 1):
        k = k + "|" + v[c]
        out[i] = k
    return out[4], out[5], out[6]


def outside_heldout(d, ho):
    return d[~d["gk"].isin(ho["genus"]) & ~d["fk"].isin(ho["family"]) & ~d["sk"].isin(ho["species"])]


def fam_stats(per_family: pd.Series, name: str):
    s = per_family[~per_family.index.str.endswith("|")]
    print(f"(c) {name}: {len(s):,} families ({s.index.str.split('|').str[4].nunique():,} names); min {s.min():,}, "
          f"p1 {np.percentile(s, 1):,.0f}, median {s.median():,.0f}, under {FLOOR}: {(s < FLOOR).sum():,} ({(s < FLOOR).mean():.1%})")


def main():
    t = pd.read_csv(TOL / "heldout_taxa.csv", dtype=str, keep_default_na=False)
    f, g, s = keys(t)
    ho = {"genus": set(g[t["depth"] == "mid"]), "family": set(f[t["depth"] == "far"]), "species": set(s)}

    # ---- pass 1: rows per (species path, rank, exclude); the below-floor rows' ids
    pf = pq.ParquetFile(TOL / "metadata.parquet")
    parts, bf = [], []
    for i in range(pf.metadata.num_row_groups):
        x = pf.read_row_group(i, columns=[*C, "taxon_rank", "exclude", "uuid", "content_hash", "url"]).to_pandas()
        x["sk"], x["ex"] = keys(x)[2], x["exclude"].fillna("")
        parts.append(x.groupby(["sk", "taxon_rank", "ex"]).size().rename("n").reset_index())
        bf.append(x.loc[x["ex"] == "below_floor", ["sk", "uuid", "content_hash", "url"]])
        if len(parts) > 40:
            parts = [pd.concat(parts).groupby(["sk", "taxon_rank", "ex"], as_index=False)["n"].sum()]
    T = pd.concat(parts).groupby(["sk", "taxon_rank", "ex"], as_index=False)["n"].sum()
    bf = pd.concat(bf, ignore_index=True)
    p = T["sk"].str.split("|", expand=True)
    T["fk"] = p[0] + "|" + p[1] + "|" + p[2] + "|" + p[3] + "|" + p[4]
    T["gk"] = T["fk"] + "|" + p[5]
    print(f"dataset: {T['n'].sum():,} rows; below_floor {int(T.loc[T['ex'] == 'below_floor', 'n'].sum()):,}", flush=True)

    sp = T[T["taxon_rank"] == "species"]
    use = sp[sp["ex"] == ""]
    usable = {"gk": use.groupby("gk")["n"].sum(), "fk": use.groupby("fk")["n"].sum()}
    trained = set(use["sk"])

    # ---- (a)
    cand = sp[(sp["ex"] == "below_floor") & ~sp["sk"].isin(trained)].groupby(["sk", "gk", "fk"], as_index=False)["n"].sum()
    cand = outside_heldout(cand, ho)
    print(f"(a) below-floor species outside held-out taxa: {len(cand):,} species, {cand['n'].sum():,} rows")
    added = {}
    for rule, col in (("genus", "gk"), ("family", "fk")):
        have = cand[col].map(usable[col]).fillna(0)
        added[rule] = cand[have >= FLOOR]
        print(f"(a) {rule} >= {FLOOR} usable: +{len(added[rule]):,} species, +{added[rule]['n'].sum():,} images; "
              f"{rule} not trained at all: {int((have == 0).sum()):,}")
    # the freeze set benchmark, duplicate and held-out exclusions only on rows not yet excluded: screen these now
    add = bf[bf["sk"].isin(set(added["family"]["sk"]))]
    bench = pd.concat([pd.read_parquet(q) for q in sorted(BENCH.glob("*.parquet"))])
    bench_u = set(bench.loc[bench["dist"] <= BENCH_HAMMING, "uuid"])
    hset, uset = set(add["content_hash"].dropna()) - {""}, set(add["url"].dropna())
    hit_h, hit_u = set(), set()
    for i in range(pf.metadata.num_row_groups):   # pass 2
        x = pf.read_row_group(i, columns=["content_hash", "url", "exclude"]).to_pandas()
        o = x["exclude"].fillna("") != "below_floor"
        hit_h.update(x.loc[o & x["content_hash"].isin(hset), "content_hash"])
        hit_u.update(x.loc[o & x["url"].isin(uset), "url"])
    ch = add["content_hash"].replace("", None)
    print(f"    family-rule rows: benchmark overlap {add['uuid'].isin(bench_u).sum():,}; pixel duplicate of another row "
          f"{add['content_hash'].isin(hit_h).sum():,} (among themselves {ch.dropna().duplicated().sum():,}); URL duplicate "
          f"{add['url'].isin(hit_u).sum():,}; no content hash {ch.isna().sum():,}; Lepidoptera "
          f"{add['sk'].str.split('|').str[3].eq('Lepidoptera').sum():,}", flush=True)

    # ---- (b) catalog, rank tuples corrected
    f102 = _load("f102", "102_tol_fix_ranks.py")
    taxa = pd.read_parquet(CATALOG_TAXA)
    fix = f102.Fixer(taxa)
    new = pd.DataFrame([fix(r) for r in taxa[f102.RANKS].fillna("").itertuples(index=False, name=None)], columns=C)
    is_sp = (new["genus"] != "") & (new["species"] != "")
    new["species"] = np.where(is_sp, new["genus"] + " " + new["species"], "")
    cf, cg, cs = keys(new)
    cat = pd.DataFrame({"sk": cs, "gk": cg, "fk": cf, "n": taxa["n"].to_numpy(), "is_sp": is_sp.to_numpy()})
    csp = outside_heldout(cat[cat["is_sp"]].groupby(["sk", "gk", "fk"], as_index=False)["n"].sum(), ho)
    print(f"catalog: {cat['n'].sum():,} rows, {csp['n'].sum():,} at species rank outside held-out taxa, {len(csp):,} species")
    below = csp[(csp["n"] < FLOOR) & ~csp["sk"].isin(trained)]
    print(f"(b) catalog species under {FLOOR}, not trained, not held out: {len(below):,} species, {below['n'].sum():,} rows "
          f"(left out, trained on disk though under {FLOOR} in the catalog: "
          f"{int(((csp['n'] < FLOOR) & csp['sk'].isin(trained)).sum()):,})")
    on_disk = sp[sp["ex"] == "below_floor"].groupby("sk")["n"].sum()
    for rule, col in (("genus", "gk"), ("family", "fk")):
        b = below[below[col].map(usable[col]).fillna(0) >= FLOOR]
        on = b["sk"].map(on_disk).fillna(0)
        print(f"(b) {rule} >= {FLOOR} usable on disk: {len(b):,} species, {b['n'].sum():,} rows; on disk {on.sum():,.0f} "
              f"({on.sum() / b['n'].sum():.1%}) in {(on > 0).sum():,} species")

    # ---- (c) no floor
    keep = outside_heldout(sp[sp["ex"].isin(["", "below_floor"])], ho)
    fam_stats(keep.groupby("fk")["n"].sum(), "disk, species rank (usable + below_floor)")
    recs = T[T["taxon_rank"].isin(["genus", "family"]) & (T["ex"] == "")]
    fam_stats(pd.concat([keep[["fk", "n"]], recs[["fk", "n"]]]).groupby("fk")["n"].sum(),
              "disk, plus usable genus- and family-rank records")
    capped = csp.assign(n=csp["n"].clip(upper=CAP))
    fam_stats(capped.groupby("fk")["n"].sum(), f"catalog, species rank, {CAP:,} cap per species")
    crec = cat[~cat["is_sp"] & ~cat["gk"].isin(ho["genus"]) & ~cat["fk"].isin(ho["family"])]
    fam_stats(pd.concat([capped[["fk", "n"]], crec[["fk", "n"]]]).groupby("fk")["n"].sum(),
              "catalog, plus genus- and family-rank rows")


if __name__ == "__main__":
    main()
