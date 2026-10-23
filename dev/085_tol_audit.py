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

Memory stays flat whatever the crawl size: every check compares rows sharing a key (uuid, URL,
occurrence id, pixel hash), so rows are written to `--shards` parquet files by a hash of that key
and each shard is checked alone. Only two bytes per row (attributes, flags) live in memory for the
whole run; uuids are read back as strings only for the rows that end up excluded. The pandas
version held 94 M rows of strings and needed ~130 GB (2026-10-22).

    python dev/085_tol_audit.py --manifest <snapshot of manifest/> \\
        --global-lepi data/global/<...>_quality_filtered.parquet --out <dir> [--log-mem]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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


SOURCES = ("selected", "substitute", "global_lepi")
GL = SOURCES.index("global_lepi")
# Per-row attributes (bits of one uint8), and the flags the checks set (bits of another).
FILL, REACH, GL0, HAS_GBIF = 4, 8, 16, 32                      # bits 0-1 hold the source index
DUP_UUID, DUP_URL, SAME_OCC, LEAK = 1, 2, 4, 8


def _h(values) -> np.ndarray:
    """64-bit hash per string. A collision among 94 M values has probability ~2e-4."""
    return pd.util.hash_array(np.asarray(values, dtype=object), categorize=False)


def drop_cache(path, sync: bool = False) -> None:
    """Release `path` from the page cache. Reading tens of GB filled it, and reclaiming it showed up
    as memory pressure on the owner's session -- what systemd-oomd acts on (2026-10-23)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        if sync:
            os.fsync(fd)                                   # dirty pages cannot be dropped
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _row_groups(path, columns=None, rows: int = 1_000_000):
    """Tables of whole row groups, ~`rows` rows each. ParquetFile.iter_batches retained ~95 MB per
    million rows read from one 6 GB file (pyarrow 25, 2026-10-23); reading row groups stays flat."""
    pf = pq.ParquetFile(path)
    group, n = [], 0
    for i in range(pf.metadata.num_row_groups):
        group.append(i)
        n += pf.metadata.row_group(i).num_rows
        if n >= rows:
            yield pf.read_row_groups(group, columns=columns)
            drop_cache(path)
            group, n = [], 0
    if group:
        yield pf.read_row_groups(group, columns=columns)
        drop_cache(path)


def _part_kind(name: str) -> int:
    return GL if name.startswith("part-g") else 1 if name.startswith(("part-s", "part-t")) else 0


def log_mem(stage: str, on: bool) -> None:
    if on:
        rss = next(int(x.split()[1]) for x in open("/proc/self/status") if x.startswith("RssAnon"))
        print(f"  [mem] {stage}: {rss / 1024:,.0f} MB", flush=True)


class Shards:
    """Rows partitioned by `key % k` into k parquet files, buffered so row groups stay large."""

    def __init__(self, root: Path, k: int, flush_rows: int = 2_000_000):
        self.root, self.k, self.flush_rows = root, k, flush_rows
        root.mkdir(parents=True, exist_ok=True)
        self.buf: list[dict[str, np.ndarray]] = []
        self.n = 0
        self.writers: dict[int, pq.ParquetWriter] = {}

    def add(self, **cols: np.ndarray) -> None:
        self.buf.append(cols)
        self.n += len(cols["key"])
        if self.n >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        t = {c: np.concatenate([b[c] for b in self.buf]) for c in self.buf[0]}
        self.buf, self.n = [], 0
        sid = t["key"] % np.uint64(self.k)
        order = np.argsort(sid, kind="stable")
        bounds = np.searchsorted(sid[order], np.arange(self.k + 1))
        for s in range(self.k):
            idx = order[bounds[s]:bounds[s + 1]]
            if len(idx):
                tab = pa.table({c: v[idx] for c, v in t.items()})
                if s not in self.writers:
                    self.writers[s] = pq.ParquetWriter(self.root / f"shard-{s:03d}.parquet", tab.schema)
                self.writers[s].write_table(tab)

    def close(self) -> Shards:
        self.flush()
        for w in self.writers.values():
            w.close()
        return self

    def read(self, s: int) -> dict[str, np.ndarray] | None:
        if s not in self.writers:
            return None
        t = pq.read_table(self.root / f"shard-{s:03d}.parquet")
        return {c: t[c].to_numpy(zero_copy_only=False) for c in t.column_names}

    def __iter__(self):
        for s in sorted(self.writers):
            yield self.read(s)


def _first(key: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows sorted by (key, row), and whether each is its key's first: `duplicated(keep="first")`."""
    o = np.lexsort((row, key))
    k, r = key[o], row[o]
    return k, r, np.r_[True, k[1:] != k[:-1]] if len(k) else np.zeros(0, bool)


def _read_rows(files: list[Path], col: str, file_idx: np.ndarray, row_idx: np.ndarray) -> list[str]:
    """`col` at (file, row) positions, in file then row order."""
    o = np.lexsort((row_idx, file_idx))
    f, r = file_idx[o], row_idx[o]
    out: list[str] = []
    for fi in np.unique(f):
        out += pq.read_table(files[fi], columns=[col])[col].take(pa.array(r[f == fi])).to_pylist()
    return out


def main(a):
    root = Path(a.manifest); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = out / "shards"
    shutil.rmtree(work, ignore_errors=True)
    parts = sorted(p for p in root.glob("host=*/part-*.parquet") if not p.name.endswith(".meta.parquet"))
    metas = sorted(root.glob("host=*/part-*.meta.parquet"))
    print(f"{len(parts):,} manifest parts, {len(metas):,} meta files")
    blocked = {tol_slug(h) for h in json.loads((root / "blocked_hosts.json").read_text())}
    K = a.shards

    # Stage 1: stream the manifest. Each part goes through the same pandas code as the in-memory
    # version (cheap at 50 k rows); its rows go to shards by uuid, by URL and by occurrence id.
    by_uuid, by_url, by_gbif = (Shards(work / k, K) for k in ("uuid", "url", "gbif"))
    attrs, sizes, row0 = [], [], 0
    for p in parts:
        have = pq.ParquetFile(p).schema_arrow.names
        want = [c for c in ("uuid", "url", "species", "data_source", "source_id", "gbif_id",
                            "global_lepi_set", "kind") if c in have]
        df = pq.read_table(p, columns=want).to_pandas(types_mapper=_types)
        n = len(df)
        # One GBIF occurrence id per row: ToL's source_id when its source is GBIF, else dev/084's tag.
        gid = pd.Series(pd.NA, index=df.index, dtype="string")
        if "source_id" in df:
            is_gbif = df.get("data_source", pd.Series("", index=df.index)).eq("gbif")
            gid = gid.mask(is_gbif, df["source_id"].astype("string"))
        if "gbif_id" in df:
            gid = gid.fillna(df["gbif_id"].astype("string"))
        gid = gid.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
        kind = df["kind"].astype("string").fillna("") if "kind" in df else pd.Series("", index=df.index)
        gl0 = (df["global_lepi_set"].astype("string").eq("0").fillna(False).to_numpy(bool)
               if "global_lepi_set" in df else np.zeros(n, bool))
        has_gbif = gid.notna().to_numpy(bool)
        at = np.full(n, _part_kind(p.name), np.uint8)
        at |= kind.str.startswith("fill").to_numpy(bool) * np.uint8(FILL)
        at |= np.uint8(REACH * (p.parent.name.removeprefix("host=") not in blocked))
        at |= gl0 * np.uint8(GL0) | has_gbif * np.uint8(HAS_GBIF)
        rows = np.arange(row0, row0 + n, dtype=np.int64)
        by_uuid.add(key=_h(df["uuid"].astype(object)), row=rows)
        by_url.add(key=_h(norm_url(df["url"]).astype(object)), row=rows)
        by_gbif.add(key=_h(gid[has_gbif].astype(object)), row=rows[has_gbif])
        attrs.append(at); sizes.append(n); row0 += n
    for sh in (by_uuid, by_url, by_gbif):
        sh.close()
    at = np.concatenate(attrs); del attrs
    offsets = np.r_[0, np.cumsum(sizes)]
    src, is_fill, reach = at & 3, (at & FILL) > 0, (at & REACH) > 0
    flag = np.zeros(len(at), np.uint8)
    log_mem("manifest sharded", a.log_mem)

    def by_source(mask, src=src) -> dict[str, int]:  # value_counts() order: largest first
        n = np.bincount(src[mask], minlength=len(SOURCES))
        return {SOURCES[i]: int(n[i]) for i in np.argsort(-n, kind="stable") if n[i]}
    print(f"{len(at):,} rows: " + ", ".join(f"{k} {v:,}" for k, v in by_source(slice(None)).items()))
    report: dict = {"rows": len(at), "rows_by_source": by_source(slice(None))}
    report["rows_on_unreachable_hosts"] = int((~reach).sum())

    # 1. duplicate uuids ----------------------------------------------------------------------------
    # A fill row deliberately reuses the uuid of the unreachable ToL row it stands in for.
    dup_nf = twin_n = twin_reach = 0
    for t in by_uuid:
        k, r, first = _first(t["key"], t["row"])
        flag[r[~first]] |= DUP_UUID
        nf = ~is_fill[r]
        k_nf = k[nf]
        dup_nf += len(k_nf) - int(np.count_nonzero(np.r_[True, k_nf[1:] != k_nf[:-1]])) if len(k_nf) else 0
        twin = nf & np.isin(k, np.unique(k[~nf]))
        twin_n += int(twin.sum()); twin_reach += int(reach[r[twin]].sum())
    dup_u = (flag & DUP_UUID) > 0
    report["duplicate_uuids"] = int(dup_u.sum())
    report["duplicate_uuids_not_fills"] = dup_nf
    report["fills_standing_in_for_an_unreachable_row"] = int(twin_reach == 0 and twin_n == int(is_fill.sum()))

    # 2. duplicate URLs -----------------------------------------------------------------------------
    for t in by_url:
        _, r, first = _first(t["key"], t["row"])
        flag[r[~first]] |= DUP_URL
    dup_url = ((flag & DUP_URL) > 0) & ~dup_u
    report["duplicate_urls"] = int(dup_url.sum())
    report["duplicate_urls_by_source"] = by_source(dup_url)

    # 3 + 4. the same occurrence from two sources, and test-fold leakage --------------------------------
    # Only `extend` rows can duplicate ToL: fills *are* the unreachable ToL photograph, by
    # construction. And only a reachable ToL row is a real second copy.
    test_h, train_h = [], []
    for b in _row_groups(a.global_lepi, ["gbifID", "set"]):
        g = b.to_pandas()
        ids = g["gbifID"].astype("int64").astype(str)
        is_test = g["set"].astype(str) == "0"
        test_h.append(_h(ids[is_test])); train_h.append(_h(ids[~is_test]))
    test_h, train_h = np.unique(np.concatenate(test_h)), np.unique(np.concatenate(train_h))
    report_ext = n_drop = 0
    for t in by_gbif:
        k, r = t["key"], t["row"]
        is_gl = src[r] == GL
        occ = (is_gl & ~is_fill[r]) | (~is_gl & reach[r])
        u, inv = np.unique(k[occ], return_inverse=True)
        gl = np.bincount(inv, weights=is_gl[occ], minlength=len(u))
        others = np.bincount(inv, minlength=len(u)) - gl
        report_ext += int(((gl > 0) & (others > 0)).sum())
        # Keep the ToL copy (crawled at source), drop ours -- but only where each side has exactly
        # one image: an occurrence with several photographs legitimately has several rows.
        drop = is_gl & ~is_fill[r] & np.isin(k, u[(gl == 1) & (others == 1)])
        flag[r[drop]] |= SAME_OCC; n_drop += int(drop.sum())
        flag[r[np.isin(k, test_h)]] |= LEAK
    flag[(at & GL0) > 0] |= LEAK
    leak = (flag & LEAK) > 0
    report["extend_occurrences_also_in_reachable_tol"] = report_ext
    report["cross_source_duplicates_dropped"] = n_drop
    report["lepinet_test_occurrences"] = len(test_h)
    report["occurrences_split_across_lepinet_folds"] = len(np.intersect1d(test_h, train_h))
    report["rows_from_lepinet_test_fold"] = int(leak.sum())
    report["test_fold_rows_by_source"] = by_source(leak)
    del test_h, train_h
    log_mem("manifest checks", a.log_mem)

    def uuids(mask: np.ndarray) -> list[str]:  # manifest order
        rows = np.flatnonzero(mask)
        part = np.searchsorted(offsets, rows, side="right") - 1
        return _read_rows(parts, "uuid", part, rows - offsets[part])
    excl = [pd.DataFrame({"uuid": uuids(leak), "reason": "lepinet_test_fold"}),
            pd.DataFrame({"uuid": uuids(dup_url & reach), "reason": "duplicate_url"}),
            pd.DataFrame({"uuid": uuids((flag & SAME_OCC) > 0), "reason": "same_occurrence_as_tol"})]
    del at, flag, src, is_fill, reach, dup_u, dup_url, leak

    # 5. pixel duplicates and placeholders ------------------------------------------------------------
    by_ch = Shards(work / "pixel", K)
    sp_code: dict[str, int] = {}
    meta_rows = n_ok = n_hash = 0
    for mi, p in enumerate(metas):
        try:
            t = pq.read_table(p, columns=["species", "status", "content_hash"]).to_pandas(types_mapper=_types)
        except Exception as e:                                   # a meta being rewritten mid-snapshot
            print(f"  skip {p.name}: {type(e).__name__}")
            continue
        meta_rows += len(t)
        st_ok = t["status"].eq("ok").fillna(False).to_numpy(bool)
        n_ok += int(st_ok.sum())
        ok = st_ok & t["content_hash"].fillna("").ne("").to_numpy(bool)
        n_hash += int(ok.sum())
        sp = [-1 if pd.isna(x) else sp_code.setdefault(x, len(sp_code)) for x in t["species"][ok]]
        by_ch.add(key=_h(t["content_hash"][ok].astype(object)), sp=np.array(sp, np.int32),
                  mf=np.full(int(ok.sum()), mi, np.int32), mr=np.flatnonzero(ok).astype(np.int32))
    by_ch.close(); del sp_code
    report["meta_rows"] = meta_rows; report["ok_with_hash"] = n_hash
    report["ok_without_hash"] = n_ok - n_hash
    n_ph_hash = 0
    ph_pos, cand = [], []
    for t in by_ch:
        k, sp = t["key"], t["sp"]
        known = sp >= 0                                    # nunique ignores a missing species
        pairs = pd.DataFrame({"k": k[known], "sp": sp[known]}).drop_duplicates()
        n_sp = pairs.groupby("k").size()
        ph = n_sp.index[n_sp >= a.placeholder_species].to_numpy(np.uint64)
        n_ph_hash += len(ph)
        is_ph = np.isin(k, ph)
        ph_pos.append((t["mf"][is_ph], t["mr"][is_ph]))
        rest = ~is_ph
        u, c = np.unique(k[rest], return_counts=True)
        rep = rest & np.isin(k, u[c > 1])                  # rows whose pixels repeat
        cand.append({c_: t[c_][rep] for c_ in ("key", "sp", "mf", "mr")})
    report["placeholder_hashes"] = n_ph_hash
    mf, mr = (np.concatenate(x) for x in zip(*ph_pos)) if ph_pos else (np.zeros(0, np.int32),) * 2
    report["placeholder_images"] = len(mf)
    excl.append(pd.DataFrame({"uuid": _read_rows(metas, "uuid", mf, mr), "reason": "placeholder"}))
    # Among the rest, keep the smallest uuid of each pixel-identical group (the in-memory version
    # sorted on uuid). Only rows whose hash repeats are read back.
    c = {k: np.concatenate([x[k] for x in cand]) for k in cand[0]} if cand else {}
    o = np.lexsort((c["mr"], c["mf"])) if c else np.zeros(0, int)
    grp = pd.DataFrame({"uuid": _read_rows(metas, "uuid", c["mf"], c["mr"]) if c else [],
                        "ch": c["key"][o] if c else [], "sp": c["sp"][o] if c else []})
    grp = grp.sort_values("uuid")
    pix_dup = grp["ch"].duplicated(keep="first")
    report["pixel_duplicates"] = int(pix_dup.sum())
    dup_groups = grp[grp["ch"].isin(set(grp.loc[pix_dup, "ch"]))]
    report["pixel_duplicates_cross_species"] = int(
        dup_groups[dup_groups["sp"] >= 0].groupby("ch")["sp"].nunique().gt(1).sum())
    excl.append(pd.DataFrame({"uuid": grp.loc[pix_dup, "uuid"], "reason": "pixel_duplicate"}))
    log_mem("pixel checks", a.log_mem)

    ex = pd.concat(excl, ignore_index=True).drop_duplicates("uuid", keep="first")
    report["excluded_total"] = len(ex)
    report["excluded_by_reason"] = ex["reason"].value_counts().to_dict()
    ex.to_parquet(out / "exclude.parquet", index=False)
    (out / "audit_report.json").write_text(json.dumps(report, indent=2, default=int))
    if not a.keep_shards:
        shutil.rmtree(work)
    print(json.dumps(report, indent=2, default=int))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--global-lepi", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--placeholder-species", type=int, default=20,
                   help="a hash shared across this many species is a placeholder, not a photo")
    p.add_argument("--shards", type=int, default=64, help="key-hash partitions; memory per shard ~1/k")
    p.add_argument("--keep-shards", action="store_true", help="keep <out>/shards/ for inspection")
    p.add_argument("--log-mem", action="store_true", help="print resident memory after each stage")
    main(p.parse_args())
