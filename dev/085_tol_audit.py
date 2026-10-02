"""Audit the TreeOfLife subset for duplicates and for leakage of lepinet's held-out test fold.

Everything queued for the crawl comes from four sources -- the capped ToL selection, two
substitution passes, and our own `global_lepi` download (fill/extend, `dev/084`). Before any of it
is trained on, check:

1. **duplicate uuids** -- the same row queued twice;
2. **duplicate image URLs** -- the same photograph under two uuids (scheme-insensitive);
3. **GBIF occurrences arriving from two sources** -- a ToL row and a `global_lepi` row for the same
   occurrence, which can be the same photograph under different URLs;
4. **test-fold leakage** -- any image whose GBIF occurrence is in lepinet's held-out fold
   (`global_lepi` set '0'): 413,865 of those 629,742 test images are inside TreeOfLife, so any
   training set that includes ToL Lepidoptera must drop them;
5. **pixel duplicates and placeholders** -- identical decoded pixels (`content_hash` in the metas)
   under different uuids; and hashes shared across many species, which are the "image unavailable"
   placeholders some servers return with HTTP 200.

Writes `exclude.parquet` (uuid, reason) -- the rows to drop from training, keeping one canonical
copy of each duplicate group -- and `audit_report.json`. Images are not touched: exclusion is a
selection, applied when training parquets are built.

    python dev/085_tol_audit.py --manifest <snapshot of manifest/> \\
        --global-lepi data/global/<...>_quality_filtered.parquet --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def tol_slug(h: str) -> str:
    """dev/082's host-folder slug, reproduced exactly (including its missing lower())."""
    return re.sub(r"[^a-z0-9]+", "_", h)[:80] or "unknown"


def _types(t: pa.DataType) -> pd.ArrowDtype:
    # 94 M uuids and urls exceed the 2 GB offset limit of arrow's 32-bit `string` when pandas
    # factorizes them (duplicated, groupby); `large_string` has 64-bit offsets.
    return pd.ArrowDtype(pa.large_string() if pa.types.is_string(t) else t)


def norm_url(s: pd.Series) -> pd.Series:
    return s.fillna("").str.replace(r"^[a-zA-Z]+://", "", regex=True).str.strip()


def main(a):
    root = Path(a.manifest); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    parts = sorted(p for p in root.glob("host=*/part-*.parquet") if not p.name.endswith(".meta.parquet"))
    metas = sorted(root.glob("host=*/part-*.meta.parquet"))
    print(f"{len(parts):,} manifest parts, {len(metas):,} meta files")

    frames = []
    for p in parts:
        cols = pq.ParquetFile(p).schema_arrow.names
        want = [c for c in ("uuid", "url", "species", "data_source", "source_id", "gbif_id",
                            "global_lepi_set", "kind") if c in cols]
        df = pq.read_table(p, columns=want).to_pandas(types_mapper=_types)
        df["source"] = ("global_lepi" if p.name.startswith("part-g") else
                        "substitute" if p.name.startswith(("part-s", "part-t")) else "selected")
        df["host"] = p.parent.name.removeprefix("host=")
        frames.append(df)
    m = pd.concat(frames, ignore_index=True)
    del frames
    # One GBIF occurrence id per row: ToL's source_id when its source is GBIF, else dev/084's tag.
    gid = pd.Series(pd.NA, index=m.index, dtype="string")
    if "source_id" in m:
        is_gbif = m.get("data_source", pd.Series("", index=m.index)).eq("gbif")
        gid = gid.mask(is_gbif, m["source_id"].astype("string"))
    if "gbif_id" in m:
        gid = gid.fillna(m["gbif_id"].astype("string"))
    m["gbif"] = gid.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
    print(f"{len(m):,} rows: " + ", ".join(f"{k} {v:,}" for k, v in m["source"].value_counts().items()))

    report: dict = {"rows": len(m), "rows_by_source": m["source"].value_counts().to_dict()}
    excl: list[pd.DataFrame] = []
    kind = m["kind"].astype("string").fillna("") if "kind" in m else pd.Series("", index=m.index)
    is_fill = kind.str.startswith("fill")
    # Host folders are the crawler's slug of the hostname; unreachable rows never download.
    blocked = {tol_slug(h) for h in json.loads((root / "blocked_hosts.json").read_text())}
    reachable = ~m["host"].isin(blocked)
    report["rows_on_unreachable_hosts"] = int((~reachable).sum())

    # 1. duplicate uuids ----------------------------------------------------------------------------
    # A fill row deliberately reuses the uuid of the unreachable ToL row it stands in for.
    dup_u = m["uuid"].duplicated(keep="first")
    report["duplicate_uuids"] = int(dup_u.sum())
    report["duplicate_uuids_not_fills"] = int(m.loc[~is_fill, "uuid"].duplicated().sum())
    fill_twin = m.loc[~is_fill & m["uuid"].isin(set(m.loc[is_fill, "uuid"]))]
    report["fills_standing_in_for_an_unreachable_row"] = int((~fill_twin["host"].isin(blocked)).sum() == 0
                                                             and len(fill_twin) == int(is_fill.sum()))

    # 2. duplicate URLs -----------------------------------------------------------------------------
    m["nurl"] = norm_url(m["url"])
    dup_url = m["nurl"].duplicated(keep="first") & ~dup_u
    report["duplicate_urls"] = int(dup_url.sum())
    report["duplicate_urls_by_source"] = m.loc[dup_url, "source"].value_counts().to_dict()
    excl.append(pd.DataFrame({"uuid": m.loc[dup_url & reachable, "uuid"], "reason": "duplicate_url"}))

    # 3. the same occurrence from two sources ---------------------------------------------------------
    # Only `extend` rows can duplicate ToL: fills *are* the unreachable ToL photograph, by
    # construction. And only a reachable ToL row is a real second copy.
    occ = m[m["gbif"].notna() & ((m["source"].eq("global_lepi") & ~is_fill) |
                                  (m["source"].ne("global_lepi") & reachable))]
    occ_n = occ.groupby(["gbif", "source"]).size().unstack(fill_value=0)
    gl = occ_n["global_lepi"] if "global_lepi" in occ_n else pd.Series(0, index=occ_n.index)
    others = occ_n.drop(columns=["global_lepi"], errors="ignore").sum(axis=1)
    report["extend_occurrences_also_in_reachable_tol"] = int(((gl > 0) & (others > 0)).sum())
    # Keep the ToL copy (crawled at source), drop ours -- but only where each side has exactly one
    # image: an occurrence with several photographs legitimately has several rows.
    single = occ_n.index[(gl == 1) & (others == 1)]
    drop_gl = m["source"].eq("global_lepi") & ~is_fill & m["gbif"].isin(set(single))
    report["cross_source_duplicates_dropped"] = int(drop_gl.sum())
    same_occ = pd.DataFrame({"uuid": m.loc[drop_gl, "uuid"], "reason": "same_occurrence_as_tol"})

    # 4. test-fold leakage --------------------------------------------------------------------------
    g = pq.read_table(a.global_lepi, columns=["gbifID", "set"]).to_pandas()
    test_ids = set(g.loc[g["set"].astype(str) == "0", "gbifID"].astype("int64").astype(str))
    train_ids = set(g.loc[g["set"].astype(str) != "0", "gbifID"].astype("int64").astype(str))
    split_occ = test_ids & train_ids
    report["lepinet_test_occurrences"] = len(test_ids)
    report["occurrences_split_across_lepinet_folds"] = len(split_occ)
    leak = m["gbif"].isin(test_ids)
    if "global_lepi_set" in m:
        leak |= m["global_lepi_set"].astype("string").eq("0")
    report["rows_from_lepinet_test_fold"] = int(leak.sum())
    report["test_fold_rows_by_source"] = m.loc[leak, "source"].value_counts().to_dict()
    excl.insert(0, pd.DataFrame({"uuid": m.loc[leak, "uuid"], "reason": "lepinet_test_fold"}))
    excl.append(same_occ)

    # 5. pixel duplicates and placeholders ------------------------------------------------------------
    mf = []
    for p in metas:
        try:
            mf.append(pq.read_table(p, columns=["uuid", "species", "status", "content_hash"])
                      .to_pandas(types_mapper=_types))
        except Exception as e:                                   # a meta being rewritten mid-snapshot
            print(f"  skip {p.name}: {type(e).__name__}")
    meta = pd.concat(mf, ignore_index=True); del mf
    ok = meta[meta["status"].eq("ok") & meta["content_hash"].fillna("").ne("")]
    report["meta_rows"] = len(meta); report["ok_with_hash"] = len(ok)
    report["ok_without_hash"] = int((meta["status"].eq("ok")).sum() - len(ok))
    per_hash = ok.groupby("content_hash").agg(n=("uuid", "size"), species=("species", "nunique"))
    placeholders = set(per_hash.index[(per_hash["species"] >= a.placeholder_species)])
    is_ph = ok["content_hash"].isin(placeholders)
    report["placeholder_hashes"] = len(placeholders)
    report["placeholder_images"] = int(is_ph.sum())
    excl.append(pd.DataFrame({"uuid": ok.loc[is_ph, "uuid"], "reason": "placeholder"}))
    rest = ok[~is_ph].sort_values("uuid")
    pix_dup = rest["content_hash"].duplicated(keep="first")
    report["pixel_duplicates"] = int(pix_dup.sum())
    report["pixel_duplicates_cross_species"] = int(
        rest[rest["content_hash"].isin(set(rest.loc[pix_dup, "content_hash"]))]
        .groupby("content_hash")["species"].nunique().gt(1).sum())
    excl.append(pd.DataFrame({"uuid": rest.loc[pix_dup, "uuid"], "reason": "pixel_duplicate"}))

    ex = pd.concat(excl, ignore_index=True).drop_duplicates("uuid", keep="first")
    report["excluded_total"] = len(ex)
    report["excluded_by_reason"] = ex["reason"].value_counts().to_dict()
    ex.to_parquet(out / "exclude.parquet", index=False)
    (out / "audit_report.json").write_text(json.dumps(report, indent=2, default=int))
    print(json.dumps(report, indent=2, default=int))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--global-lepi", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--placeholder-species", type=int, default=20,
                   help="a hash shared across this many species is a placeholder, not a photo")
    main(p.parse_args())
