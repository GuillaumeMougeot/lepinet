"""Publish the TreeOfLife crawl (dev/082) as a shared, gbifxdl-style dataset.

The crawl leaves a working layout: 1,834 manifest/meta parquet pairs, three retry manifests whose
metas supersede the main ones for the hosts they re-tried, and species folders named by a lossy
slug (`_cacia_dealbata/`, 158 folders shared by two species). Other users get instead:

    <dataset>/metadata.parquet          one row per usable image (status ok in the latest attempt),
                                        with every field TreeOfLife-200M publishes for it
    <dataset>/images/<Genus_epithet>/<uuid>.jpg

Steps, all reading the crawl directory (`--crawl`) and writing to `--build`:

    python dev/090_tol_share.py snapshot --crawl D --build B   # manifest view with retries merged
    python dev/085_tol_audit.py --manifest B/snapshot --global-lepi <gl.parquet> --out B/audit
    python dev/090_tol_share.py plan  --crawl D --build B      # B/metadata.parquet + the moves
    python dev/090_tol_share.py apply --crawl D --build B      # rename folders in D/images, in place
    (move D/images and B/metadata.parquet to the shared <dataset>)
    python dev/082_tol_crawler.py fetch --offline --manifest D/manifest --images D/orphans ...
    (snapshot and dev/085 again, so the recovered rows are merged and audited)
    python dev/090_tol_share.py recover --build B              # files the offline crawl recorded
    python dev/090_tol_share.py enrich  --build B              # + ToL catalog/provenance columns
    python dev/090_tol_share.py verify  --build B --metadata B/metadata_full.parquet

`apply` journals every rename to B/applied.tsv (old<TAB>new) before making it, so it can be
resumed or reversed; files on disk that no usable row points to go to D/orphans/.

Like dev/085, `plan` and the verification in `apply` keep memory flat: rows and on-disk files are
written to shards by a hash of their path and each shard is matched alone; only a keep flag per row
(66 MB for 66 M rows) spans the whole run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RETRIES = ("retry1", "retry2", "retry3")  # oldest first: a later attempt supersedes an earlier one
PART_COLS = ("uuid", "url", "species", "genus", "family", "order", "data_source", "source_id",
             "gbif_id")
META_COLS = ("uuid", "status", "path", "content_hash", "width", "height", "bytes")


def _load_audit():
    """dev/085's Shards, _h and log_mem (dev scripts share code by file, as dev/051 does)."""
    spec = importlib.util.spec_from_file_location("dev085_audit", Path(__file__).with_name("085_tol_audit.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AUDIT = _load_audit()
Shards, _h, log_mem, _row_groups, drop_cache = (AUDIT.Shards, AUDIT._h, AUDIT.log_mem, AUDIT._row_groups,
                                                 AUDIT.drop_cache)


def host_slug(h: str) -> str:
    """dev/082's host-folder slug."""
    return re.sub(r"[^a-z0-9]+", "_", h)[:80] or "unknown"


def folder_name(species: str) -> str:
    """The exact species key as a folder name: spaces -> '_', nothing else changed."""
    name = re.sub(r"[/\x00]", "-", species.strip()).replace(" ", "_")
    return name.encode()[:255].decode(errors="ignore") or "unknown"


# --------------------------------------------------------------------------------------------- #
def snapshot(a):
    crawl, snap = Path(a.crawl), Path(a.build) / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    blocked = json.loads((crawl / "manifest/blocked_hosts.json").read_text())
    merged = 0
    for host in sorted((crawl / "manifest").glob("host=*")):
        out = snap / host.name
        out.mkdir(exist_ok=True)
        for part in sorted(p for p in host.glob("part-*.parquet") if not p.name.endswith(".meta.parquet")):
            link = out / part.name
            if not link.exists():
                link.symlink_to(part)
            meta = part.with_suffix(".meta.parquet")
            later = [crawl / r / "manifest" / host.name / meta.name for r in RETRIES]
            later = [m for m in later if m.exists()]
            dst = out / meta.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if not later:
                if meta.exists():
                    dst.symlink_to(meta)
                continue
            # Per uuid: the latest ok row if any attempt succeeded, else the latest attempt.
            tabs = [pq.read_table(m) for m in ([meta] if meta.exists() else []) + later]
            rows = {}
            for t in tabs:
                for r in t.to_pylist():
                    if r["status"] == "ok" or rows.get(r["uuid"], {}).get("status") != "ok":
                        rows[r["uuid"]] = r
            pq.write_table(pa.Table.from_pylist(list(rows.values()), schema=tabs[0].schema), dst)
            merged += 1
    # A host re-tried later is blocked only if its latest retry still found it blocked.
    for r in RETRIES:
        rm = crawl / r / "manifest"
        if not rm.is_dir():
            continue
        retried = {h.name.removeprefix("host=") for h in rm.glob("host=*")}
        now = json.loads((rm / "blocked_hosts.json").read_text())
        blocked = {h: v for h, v in blocked.items() if host_slug(h) not in retried} | {
            h: v for h, v in now.items() if host_slug(h) in retried}
    (snap / "blocked_hosts.json").write_text(json.dumps(blocked, indent=1))
    print(f"snapshot: {merged} metas merged with retries, {len(blocked)} hosts still blocked -> {snap}")


# --------------------------------------------------------------------------------------------- #
def _part_rows(part: Path) -> pa.Table | None:
    """The usable (status ok) rows of one manifest part, with its manifest columns, in path order."""
    meta = part.with_suffix(".meta.parquet")
    if not meta.exists():
        return None
    m = pq.read_table(meta, columns=list(META_COLS))
    m = m.filter(pc.equal(m["status"], "ok")).drop_columns(["status"])
    cols = pq.ParquetFile(part).schema_arrow.names
    p = pq.read_table(part, columns=[c for c in PART_COLS if c in cols])
    for c in PART_COLS:
        if c not in p.column_names:
            p = p.append_column(c, pa.nulls(p.num_rows, pa.string()))
    p = p.select(list(PART_COLS)).cast(pa.schema([(c, pa.string()) for c in PART_COLS]))
    src = ("global_lepi" if part.name.startswith("part-g") else
           "substitute" if part.name.startswith(("part-s", "part-t")) else "selected")
    t = m.join(p, "uuid", join_type="inner", use_threads=False).sort_by("path")
    return t.append_column("source", pa.array([src] * t.num_rows, pa.string()))


def _list_dir(d: Path) -> list[str]:
    with os.scandir(d) as it:
        return [f"{d.name}/{e.name}" for e in it if e.is_file(follow_symlinks=False)]


def _scan_to_shards(dirs: list[Path], shards: Shards, with_paths: bool) -> int:
    """Hash every file under `dirs` into `shards`, 2,000 folders at a time (bounded memory)."""
    n = 0
    with ThreadPoolExecutor(32) as pool:
        for i in range(0, len(dirs), 2000):
            for paths in pool.map(_list_dir, dirs[i:i + 2000]):
                if paths:
                    cols = {"key": _h(paths)}
                    if with_paths:
                        cols["path"] = np.array(paths, dtype=object)
                    shards.add(**cols)
                    n += len(paths)
    shards.close()
    return n


SRC_RANK = {"substitute": 0, "selected": 1, "global_lepi": 2}  # fill twins: keep the first
INT_COLS = ("width", "height", "bytes")
PLAN_COLS = ["path", "uuid", "species", "genus", "family", "order", "exclude", "source",
             "data_source", "source_id", "gbif_id", "url", "width", "height", "bytes", "content_hash"]
PLAN_SCHEMA = pa.schema([(c, pa.int64() if c in INT_COLS else pa.string()) for c in PLAN_COLS])


def plan(a):
    crawl, build = Path(a.crawl), Path(a.build)
    snap, img, work = build / "snapshot", crawl / "images", build / "shards"
    shutil.rmtree(work, ignore_errors=True)
    parts = sorted(p for p in snap.glob("host=*/part-*.parquet") if not p.name.endswith(".meta.parquet"))
    K = a.shards

    # Pass 1: shard each usable row by the hash of its file; note which folder each species is in.
    by_path = Shards(work / "rows", K)
    sizes, species, row0 = [], set(), 0
    pairs: set[tuple[str, str]] = set()                           # (old folder, species)
    for part in parts:
        t = _part_rows(part)
        n = 0 if t is None else t.num_rows
        sizes.append(n)
        if not n:
            continue
        paths = t["path"].to_pylist()
        by_path.add(key=_h(paths), row=np.arange(row0, row0 + n, dtype=np.int64),
                    rank=np.full(n, SRC_RANK[t["source"][0].as_py()], np.int8))
        sp = t["species"].fill_null("unknown").to_pylist()
        species.update(sp)
        pairs.update(zip((x.split("/", 1)[0] for x in paths), sp))
        row0 += n
    by_path.close()
    print(f"{row0:,} ok rows from {len(parts):,} parts")
    log_mem("rows sharded", a.log_mem)

    # Folders: the exact species key. Check the mapping is one-to-one before touching anything.
    names = {s: folder_name(s) for s in species}
    if len(set(names.values())) != len(names):
        raise SystemExit("two species map to one folder name; fix folder_name()")
    dir_map: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    for od, sp in pairs:
        dir_map.setdefault(od, set()).add(names[sp])
        sources.setdefault(names[sp], set()).add(od)
    whole = {od: next(iter(n)) for od, n in dir_map.items() if len(n) == 1 and len(sources[next(iter(n))]) == 1}
    print(f"{len(dir_map):,} folders: {len(whole):,} rename as a whole, {len(dir_map) - len(whole):,} split")

    # What is on disk, sharded the same way, then matched shard by shard.
    dirs = [d for d in img.iterdir() if d.is_dir()]
    n_disk = _scan_to_shards(dirs, by_disk := Shards(work / "disk", K), with_paths=True)
    keep = np.zeros(row0, bool)
    twins = missing = 0
    orphans: list[str] = []
    for s in range(K):
        r, d = by_path.read(s), by_disk.read(s)
        kept = np.zeros(0, np.uint64)
        if r is not None:
            # A global_lepi fill row reuses the uuid of the unreachable ToL row it replaces; if a
            # retry later fetched that ToL row too, both point at the same file. Keep one row per file.
            o = np.lexsort((r["row"], r["rank"], r["key"]))
            k, rows = r["key"][o], r["row"][o]
            first = np.r_[True, k[1:] != k[:-1]]
            have = np.isin(k, d["key"]) if d is not None else np.zeros(len(k), bool)
            twins += int((~first).sum()); missing += int((first & ~have).sum())
            keep[rows[first & have]] = True
            kept = k[first & have]
        if d is not None:
            orphans += d["path"][~np.isin(d["key"], kept)].tolist()
    orphans.sort()
    print(f"{twins:,} rows share a file with another row (fill twins)")
    print(f"{n_disk:,} files on disk in {len(dirs):,} folders; {missing:,} ok rows have no file (dropped)")
    print(f"{len(orphans):,} files on disk that no ok row points to (-> orphans/)")
    log_mem("matched against disk", a.log_mem)

    # Exclusions by uuid hash (a few million rows at most), reasons as small codes.
    ex_h, ex_c = [], []
    codes: dict[str, int] = {}
    for b in _row_groups(build / "audit/exclude.parquet"):
        ex_h.append(_h(b.column("uuid").to_pylist()))
        ex_c.append(np.array([codes.setdefault(x, len(codes)) for x in b.column("reason").to_pylist()], np.int8))
    ex_h, ex_c = (np.concatenate(x) if x else np.zeros(0, t) for x, t in ((ex_h, np.uint64), (ex_c, np.int8)))
    o = np.argsort(ex_h)
    ex_h, ex_c = ex_h[o], ex_c[o]
    reason_of = np.array([*codes, None], dtype=object)                # code -1 -> no exclusion

    # Pass 2: write the kept rows, part by part.
    cols, schema = PLAN_COLS, PLAN_SCHEMA
    mw = pq.ParquetWriter(build / "metadata.parquet", schema, compression="zstd")
    fw = pq.ParquetWriter(build / "file_moves.parquet", pa.schema([("old", pa.string()), ("new", pa.string())]))
    offset = n_rows = n_ex = n_bytes = 0
    for part, n in zip(parts, sizes):
        if not n:
            continue
        t = _part_rows(part).filter(pa.array(keep[offset:offset + n]))
        offset += n
        old = t["path"].to_pylist()
        sp = t["species"].fill_null("unknown").to_pylist()
        new = [f"{names[s]}/{o.rsplit('/', 1)[-1]}" for s, o in zip(sp, old)]
        is_gbif = pc.equal(t["data_source"], "gbif").fill_null(False)
        gbif = pc.if_else(is_gbif, t["source_id"], t["gbif_id"])
        uh = _h(t["uuid"].to_pylist())
        i = np.minimum(np.searchsorted(ex_h, uh), max(len(ex_h) - 1, 0))
        hit = ex_h[i] == uh if len(ex_h) else np.zeros(len(uh), bool)
        excl = pa.array(reason_of[np.where(hit, ex_c[i] if len(ex_h) else -1, -1)], pa.string())
        t = (t.drop_columns(["path", "gbif_id"]).append_column("path", pa.array(new, pa.string()))
             .append_column("gbif_id", gbif).append_column("exclude", excl))
        mw.write_table(t.select(cols).cast(schema))
        split = [k for k, o in enumerate(old) if o.split("/", 1)[0] not in whole]
        if split:
            fw.write_table(pa.table({"old": [old[k] for k in split], "new": [new[k] for k in split]}))
        n_rows += t.num_rows; n_ex += int(hit.sum()); n_bytes += pc.sum(t["bytes"]).as_py() or 0
    mw.close(); fw.close()
    pq.write_table(pa.table({"old": pa.array(sorted(whole), pa.string()),
                             "new": pa.array([whole[k] for k in sorted(whole)], pa.string())}),
                   build / "dir_moves.parquet")
    pq.write_table(pa.table({"old": pa.array(orphans, pa.string())}), build / "orphans.parquet")
    shutil.rmtree(work)
    log_mem("metadata written", a.log_mem)
    print(f"metadata: {n_rows:,} images, {len(names):,} species folders, {n_ex:,} flagged exclude, "
          f"{n_bytes / 1e12:.2f} TB -> {build}")


# --------------------------------------------------------------------------------------------- #
def apply(a):
    crawl, build = Path(a.crawl), Path(a.build)
    img = crawl / "images"
    log = (build / "applied.tsv").open("a", buffering=1)

    def mv(src: Path, dst: Path) -> bool:
        if not src.exists() and dst.exists():
            return False  # done on an earlier run
        log.write(f"{src.relative_to(crawl)}\t{dst.relative_to(crawl)}\n")
        os.rename(src, dst)
        return True

    # Orphans first, so a whole-folder rename carries only usable files.
    orphans = pq.read_table(build / "orphans.parquet")["old"].to_pylist()
    for o in orphans:
        dst = crawl / "orphans" / o
        dst.parent.mkdir(parents=True, exist_ok=True)
        mv(img / o, dst)
    n_dir = n_file = 0
    for od, nd in zip(*pq.read_table(build / "dir_moves.parquet").to_pydict().values()):
        if od == nd or ((img / nd).exists() and not (img / od).exists()):
            continue
        if (img / nd).exists():
            raise SystemExit(f"{nd} exists already; refusing to rename {od} onto it")
        n_dir += mv(img / od, img / nd)
    split_dirs = set()
    for batch in _row_groups(build / "file_moves.parquet", rows=100_000):
        for o, n in zip(*batch.to_pydict().values()):
            split_dirs.add(o.split("/", 1)[0])
            if o == n:
                continue
            (img / n).parent.mkdir(exist_ok=True)
            n_file += mv(img / o, img / n)
    for od in split_dirs:  # emptied shared slugs
        if (img / od).is_dir() and not any((img / od).iterdir()):
            (img / od).rmdir()
    log.close()
    print(f"renamed {n_dir:,} folders, moved {n_file:,} files, {len(orphans):,} orphans -> {crawl / 'orphans'}")

    # Verify, shard by shard: every metadata path exists, and every file is in the metadata.
    work = build / "shards"
    shutil.rmtree(work, ignore_errors=True)
    dirs = [d for d in img.iterdir() if d.is_dir()]
    n_disk = _scan_to_shards(dirs, by_disk := Shards(work / "disk", a.shards), with_paths=False)
    by_meta = Shards(work / "meta", a.shards)
    used: set[str] = set()
    for batch in _row_groups(build / "metadata.parquet", ["path"]):
        paths = batch.column(0).to_pylist()
        by_meta.add(key=_h(paths))
        used.update(p.split("/", 1)[0] for p in paths)
    by_meta.close()
    n = missing = 0
    for s in range(a.shards):
        m, d = by_meta.read(s), by_disk.read(s)
        if m is not None:
            n += len(m["key"])
            missing += int((~np.isin(m["key"], d["key"] if d is not None else np.zeros(0, np.uint64))).sum())
    shutil.rmtree(work)
    print(f"verify: {n:,} metadata rows, {missing:,} paths missing, "
          f"{len({d.name for d in dirs} - used):,} folders not in metadata, {n_disk - n + missing:,} extra files")


# --------------------------------------------------------------------------------------------- #
def recover(a):
    """Add the images `082 fetch --offline` recorded from <crawl>/orphans/ to the shared dataset:
    move each to <dataset>/images/<Genus_epithet>/ (journalled in applied.tsv) and write its row to
    <build>/recovered.parquet, which `enrich` merges in."""
    crawl, build, img = Path(a.crawl), Path(a.build), Path(a.dataset) / "images"
    orph = crawl / "orphans"
    on_disk = {str(p.relative_to(orph)) for p in orph.rglob("*.jpg")}
    parts = sorted(p for p in (build / "snapshot").glob("host=*/part-*.parquet") if not p.name.endswith(".meta.parquet"))
    rows: dict[str, dict] = {}                                    # old path -> row; fill twins: best rank wins
    for part in parts:
        t = _part_rows(part)
        if t is None:
            continue
        t = t.filter(pa.array([x in on_disk for x in t["path"].to_pylist()], pa.bool_()))
        for r in t.to_pylist():
            if r["path"] not in rows or SRC_RANK[r["source"]] < SRC_RANK[rows[r["path"]]["source"]]:
                rows[r["path"]] = r
    log = (build / "applied.tsv").open("a", buffering=1)
    out = []
    for old, r in sorted(rows.items()):
        new = f"{folder_name(r['species'] or 'unknown')}/{old.rsplit('/', 1)[-1]}"
        if (img / new).exists():
            raise SystemExit(f"{new} exists already; refusing to overwrite it")
        (img / new).parent.mkdir(exist_ok=True)
        log.write(f"orphans/{old}\t{img / new}\n")
        os.rename(orph / old, img / new)
        gbif = r["source_id"] if r["data_source"] == "gbif" else r["gbif_id"]
        out.append(r | {"path": new, "gbif_id": gbif, "exclude": None})
    log.close()
    pq.write_table(pa.Table.from_pylist(out, schema=PLAN_SCHEMA) if out else PLAN_SCHEMA.empty_table(),
                   build / "recovered.parquet")
    print(f"recovered {len(out):,} images into {img}; {len(on_disk) - len(out):,} files stay in {orph}")


class TableShards:
    """Arrow tables split into k shards by a shard id per row; strings never leave Arrow.

    Each flush writes one file per shard and closes it: k open ParquetWriters held the written
    data until close (~100 MB per million rows at k=128, 2026-10-23)."""

    def __init__(self, root: Path, k: int, flush_rows: int = 2_000_000):
        self.root, self.k, self.flush_rows = root, k, flush_rows
        root.mkdir(parents=True, exist_ok=True)
        self.buf: list[pa.Table] = []
        self.n = self.flushes = 0
        self.used: set[int] = set()

    def add(self, table: pa.Table, sid: np.ndarray) -> None:
        self.buf.append(table.append_column("_sid", pa.array(sid.astype(np.int32))))
        self.n += table.num_rows
        if self.n >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        t = pa.concat_tables(self.buf)
        self.buf, self.n = [], 0
        sid = t["_sid"].to_numpy()
        t = t.drop_columns(["_sid"])
        order = np.argsort(sid, kind="stable")
        bounds = np.searchsorted(sid[order], np.arange(self.k + 1))
        for s in range(self.k):
            if bounds[s + 1] > bounds[s]:
                d = self.root / f"shard-{s:03d}"
                d.mkdir(exist_ok=True)
                f = d / f"flush-{self.flushes:05d}.parquet"
                pq.write_table(t.take(pa.array(order[bounds[s]:bounds[s + 1]])), f)
                drop_cache(f, sync=True)
                self.used.add(s)
        self.flushes += 1

    def close(self) -> TableShards:
        self.flush()
        (self.root / "_done").touch()                      # a finished stage, for `enrich --resume`
        return self

    @classmethod
    def reopen(cls, root: Path) -> TableShards | None:
        """A stage finished by an earlier run, or None (and any partial output removed)."""
        if not (root / "_done").exists():
            shutil.rmtree(root, ignore_errors=True)
            return None
        ts = cls.__new__(cls)
        ts.root, ts.used = root, {int(d.name.removeprefix("shard-")) for d in root.glob("shard-*")}
        return ts

    def read(self, s: int) -> pa.Table | None:
        if s not in self.used:
            return None
        files = sorted((self.root / f"shard-{s:03d}").glob("*.parquet"))
        t = pa.concat_tables(pq.read_table(f) for f in files)
        for f in files:
            drop_cache(f)
        return t


# What TreeOfLife-200M publishes per image (dataset/catalog.parquet, dataset/metadata/provenance.parquet),
# and the same fields in global_lepi's gbifxdl parquet for the rows that came from our own download.
CATALOG_COLS = ["kingdom", "phylum", "class", "scientific_name", "common", "publisher", "basis_of_record", "img_type"]
PROVENANCE_COLS = {"license_name": "license_name", "license_link": "license_link",
                   "copyright_owner": "copyright_owner", "title": "title",
                   "bibliographicCitation": "bibliographic_citation"}
GL_COLS = {"kingdom": "kingdom", "phylum": "phylum", "class": "class", "scientificName": "scientific_name",
           "publisher": "publisher", "basisOfRecord": "basis_of_record", "license": "license_link",
           "rightsHolder": "copyright_owner"}


def _cc_name(url: str | None) -> str | None:
    """global_lepi's licence URL in ToL's vocabulary: .../licenses/by-nc/4.0/ -> cc-by-nc-4.0."""
    m = re.search(r"creativecommons\.org/(?:licenses/([a-z-]+)|publicdomain/(zero))/([0-9.]+)", url or "")
    return None if not m else f"cc-{m[1] or '0'}-{m[3]}"


def _gl_licence(t: pa.Table) -> pa.Table:
    links = [x or None for x in t["gl_license_link"].to_pylist()]
    return (t.drop_columns(["gl_license_link"]).append_column("gl_license_link", pa.array(links, pa.string()))
            .append_column("gl_license_name", pa.array([_cc_name(x) for x in links], pa.string())))
FINAL_COLS = ["path", "uuid", "species", "genus", "family", "order", "class", "phylum", "kingdom",
              "scientific_name", "common", "exclude", "license_name", "license_link", "copyright_owner",
              "title", "bibliographic_citation", "publisher", "basis_of_record", "img_type", "source",
              "data_source", "source_id", "gbif_id", "url", "width", "height", "bytes", "content_hash"]
FINAL_SCHEMA = pa.schema([(c, pa.int64() if c in INT_COLS else pa.string()) for c in FINAL_COLS])


def _first_per_uuid(t: pa.Table) -> pa.Table:
    _, first = np.unique(np.asarray(t["uuid"].to_pylist(), dtype=object).astype(str), return_index=True)
    return t.take(pa.array(np.sort(first))) if len(first) < t.num_rows else t


def enrich(a):
    """Rebuild <dataset>/metadata.parquet (written to <build>/metadata_full.parquet; `verify`, then
    swap it in): the plan's columns, recovered rows, every field ToL publishes per image, `exclude`
    from the latest audit, sorted by species. Joined shard by shard on a hash of the uuid."""
    build, ds, hf = Path(a.build), Path(a.dataset), Path(a.hf)
    work, K = build / "enrich", a.shards
    if not a.resume:
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    # 1. Our rows, sharded by uuid hash. The sorted hashes say which source rows to keep.
    meta = TableShards.reopen(work / "meta")
    hashes, species_n = [], {}

    def count(t: pa.Table) -> None:
        hashes.append(_h(t["uuid"].to_pylist()))
        for x in pc.value_counts(t["species"].fill_null("")).to_pylist():
            species_n[x["values"]] = species_n.get(x["values"], 0) + x["counts"]
    if meta is None:
        meta = TableShards(work / "meta", K)
        for f in [ds / "metadata.parquet", build / "recovered.parquet"]:
            for t in _row_groups(f, rows=500_000):
                t = t.drop_columns([c for c in ("exclude",) if c in t.column_names])
                count(t)
                meta.add(t, hashes[-1] % np.uint64(K))
        meta.close()
    else:                                                 # resumed: re-derive the hashes and counts
        for s in sorted(meta.used):
            count(pa.concat_tables(pq.read_table(f, columns=["uuid", "species"])
                                   for f in sorted((work / "meta" / f"shard-{s:03d}").glob("*.parquet"))))
    ours = np.sort(np.concatenate(hashes)); hashes.clear()
    print(f"{len(ours):,} rows, {len(species_n):,} species")
    log_mem("rows sharded", a.log_mem)

    def ours_mask(h: np.ndarray) -> np.ndarray:
        i = np.minimum(np.searchsorted(ours, h), len(ours) - 1)
        return ours[i] == h

    def shard_source(path: Path, cols: list[str], rename: dict[str, str], name: str, key=None,
                     transform=None) -> TableShards:
        if (done := TableShards.reopen(work / name)) is not None:
            print(f"  {path.name}: done in an earlier run")
            return done
        out = TableShards(work / name, K)
        kept = 0
        slices = (big.slice(o, 1_000_000) for big in _row_groups(path, cols)  # provenance row groups
                  for o in range(0, big.num_rows, 1_000_000))                   # hold up to 4.7 M rows
        for t in slices:
            uuid = key(t) if key else t["uuid"]
            h = _h(uuid.to_pylist())
            m = ours_mask(h)
            if m.any():
                t = pa.table({"uuid": uuid.cast(pa.string()), **{rename[c]: t[c].cast(pa.string()) for c in rename}})
                t = t.filter(pa.array(m))
                out.add(transform(t) if transform else t, h[m] % np.uint64(K))
                kept += int(m.sum())
        print(f"  {path.name}: {kept:,} rows matched")
        return out.close()

    cat = shard_source(hf / "dataset/catalog.parquet", ["uuid", *CATALOG_COLS],
                       {c: c for c in CATALOG_COLS}, "catalog")
    prov = shard_source(hf / "dataset/metadata/provenance.parquet", ["uuid", *PROVENANCE_COLS],
                        PROVENANCE_COLS, "provenance")
    gl = shard_source(Path(a.global_lepi), ["filename", *GL_COLS], {c: f"gl_{v}" for c, v in GL_COLS.items()},
                      "global_lepi", key=lambda t: pc.binary_join_element_wise(
                          "gl-", pc.replace_substring_regex(t["filename"], r"\.[^.]*$", ""), ""), transform=_gl_licence)
    ex = shard_source(build / "audit/exclude.parquet", ["uuid", "reason"], {"reason": "exclude"}, "exclude")
    log_mem("sources sharded", a.log_mem)

    # 2. Join shard by shard; route rows to species-range buckets so the output is sorted by species.
    names = sorted(species_n)
    cum = np.cumsum([species_n[n] for n in names])
    bounds = np.array([names[int(np.searchsorted(cum, cum[-1] * q / K))] for q in range(1, K)], dtype=object)
    shutil.rmtree(work / "out", ignore_errors=True)
    out = TableShards(work / "out", K)
    for s in range(K):
        t = meta.read(s)
        if t is None:
            continue
        for src in (cat, prov, gl, ex):
            x = src.read(s)
            if x is not None:
                t = t.join(_first_per_uuid(x), "uuid", join_type="left outer", use_threads=False)
        for v in {*GL_COLS.values(), "license_name"}:
            if f"gl_{v}" in t.column_names:
                base = t[v] if v in t.column_names else pa.nulls(t.num_rows, pa.string())
                t = t.drop_columns([c for c in (v, f"gl_{v}") if c in t.column_names]).append_column(
                    v, pc.coalesce(base, t[f"gl_{v}"]))
        for c in FINAL_COLS:
            if c not in t.column_names:
                t = t.append_column(c, pa.nulls(t.num_rows, pa.string()))
        sp = np.asarray(t["species"].fill_null("").to_pylist(), dtype=object)
        out.add(t.select(FINAL_COLS).cast(FINAL_SCHEMA), np.searchsorted(bounds, sp, side="right"))
    out.close()
    log_mem("joined", a.log_mem)

    w = pq.ParquetWriter(build / "metadata_full.parquet", FINAL_SCHEMA, compression="zstd")
    n = 0
    for s in range(K):
        t = out.read(s)
        if t is not None:
            w.write_table(t.sort_by([("species", "ascending"), ("path", "ascending")]), row_group_size=100_000)
            n += t.num_rows
    w.close()
    shutil.rmtree(work)
    log_mem("written", a.log_mem)
    print(f"metadata_full: {n:,} rows, {len(FINAL_COLS)} columns -> {build / 'metadata_full.parquet'}")


def verify(a):
    """Every metadata path exists under <dataset>/images, and every file there has a row."""
    build, img = Path(a.build), Path(a.dataset) / "images"
    meta_path = Path(a.metadata) if a.metadata else Path(a.dataset) / "metadata.parquet"
    work = build / "verify"
    shutil.rmtree(work, ignore_errors=True)
    dirs = [d for d in img.iterdir() if d.is_dir()]
    n_disk = _scan_to_shards(dirs, by_disk := Shards(work / "disk", a.shards), with_paths=False)
    by_meta = Shards(work / "meta", a.shards)
    used: set[str] = set()
    for batch in _row_groups(meta_path, ["path"]):
        paths = batch.column(0).to_pylist()
        by_meta.add(key=_h(paths))
        used.update(p.split("/", 1)[0] for p in paths)
    by_meta.close()
    n = missing = dup = 0
    for s in range(a.shards):
        m, d = by_meta.read(s), by_disk.read(s)
        if m is not None:
            n += len(m["key"])
            dup += len(m["key"]) - len(np.unique(m["key"]))
            missing += int((~np.isin(m["key"], d["key"] if d is not None else np.zeros(0, np.uint64))).sum())
    shutil.rmtree(work)
    print(f"verify {meta_path.name}: {n:,} rows, {missing:,} paths missing, {dup:,} duplicate paths, "
          f"{len({d.name for d in dirs} - used):,} folders not in metadata, {n_disk - n + missing + dup:,} extra files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["snapshot", "plan", "apply", "recover", "enrich", "verify"])
    p.add_argument("--crawl", default="/data/au761367/tol256s", help="crawl directory (manifest/, retry*/, images/)")
    p.add_argument("--build", required=True, help="working directory for this script's outputs")
    p.add_argument("--shards", type=int, default=32, help="path-hash partitions (plan, apply's verify)")
    p.add_argument("--log-mem", action="store_true", help="print resident memory after each stage")
    p.add_argument("--dataset", default="/data/datasets/tol256", help="the shared copy (recover, enrich, verify)")
    p.add_argument("--hf", default="/data/au761367/tol256s/hf", help="local TreeOfLife-200M metadata (enrich)")
    p.add_argument("--global-lepi", default="/data/au761367/lepinet/global/"
                   "0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet")
    p.add_argument("--metadata", default="", help="verify this file instead of <dataset>/metadata.parquet")
    p.add_argument("--resume", action="store_true", help="enrich: reuse stages an interrupted run finished")
    a = p.parse_args()
    {"snapshot": snapshot, "plan": plan, "apply": apply, "recover": recover, "enrich": enrich,
     "verify": verify}[a.step](a)
