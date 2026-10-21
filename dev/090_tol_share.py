"""Publish the TreeOfLife crawl (dev/082) as a shared, gbifxdl-style dataset.

The crawl leaves a working layout: 1,834 manifest/meta parquet pairs, three retry manifests whose
metas supersede the main ones for the hosts they re-tried, and species folders named by a lossy
slug (`_cacia_dealbata/`, 158 folders shared by two species). Other users get instead:

    <dataset>/metadata.parquet          one row per usable image (status ok in the latest attempt)
    <dataset>/images/<Genus_epithet>/<uuid>.jpg

Steps, all reading the crawl directory (`--crawl`) and writing to `--build`:

    python dev/090_tol_share.py snapshot --crawl D --build B   # manifest view with retries merged
    python dev/085_tol_audit.py --manifest B/snapshot --global-lepi <gl.parquet> --out B/audit
    python dev/090_tol_share.py plan  --crawl D --build B      # B/metadata.parquet, B/moves.parquet
    python dev/090_tol_share.py apply --crawl D --build B      # rename folders in D/images, in place

`apply` journals every rename to B/applied.tsv (old<TAB>new) before making it, so it can be
resumed or reversed; files on disk that no usable row points to go to D/orphans/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RETRIES = ("retry1", "retry2", "retry3")  # oldest first: a later attempt supersedes an earlier one
PART_COLS = ("uuid", "url", "species", "genus", "family", "order", "data_source", "source_id",
             "gbif_id")
META_COLS = ("uuid", "status", "path", "content_hash", "width", "height", "bytes")


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
    t = m.join(p, "uuid", join_type="inner")
    return t.append_column("source", pa.array([src] * t.num_rows, pa.string()))


def _scan(d: Path) -> list[str]:
    with os.scandir(d) as it:
        return [f"{d.name}/{e.name}" for e in it if e.is_file(follow_symlinks=False)]


def plan(a):
    crawl, build = Path(a.crawl), Path(a.build)
    snap = build / "snapshot"
    parts = sorted(p for p in snap.glob("host=*/part-*.parquet") if not p.name.endswith(".meta.parquet"))
    with ThreadPoolExecutor(8) as pool:
        t = pa.concat_tables([x for x in pool.map(_part_rows, parts) if x is not None])
    print(f"{t.num_rows:,} ok rows from {len(parts):,} parts")

    # A global_lepi fill row reuses the uuid of the unreachable ToL row it replaces; if a retry
    # later fetched that ToL row too, both point at the same file. Keep one row per file.
    t = t.sort_by([("path", "ascending"), ("source", "descending")])
    keep = pc.invert(pc.equal(t["path"], pa.concat_arrays([pa.array([None], pa.string()),
                                                          t["path"].combine_chunks()[:-1]])).fill_null(False))
    print(f"{t.num_rows - pc.sum(keep).as_py():,} rows share a file with another row (fill twins)")
    t = t.filter(keep)

    # One GBIF occurrence id per row, as in dev/085.
    is_gbif = pc.equal(t["data_source"], "gbif").fill_null(False)
    gbif = pc.if_else(is_gbif, t["source_id"], t["gbif_id"])
    t = t.drop_columns(["gbif_id"]).append_column("gbif_id", gbif)

    ex = pq.read_table(build / "audit/exclude.parquet").select(["uuid", "reason"])
    ex = ex.rename_columns(["uuid", "exclude"]).cast(pa.schema([("uuid", pa.string()), ("exclude", pa.string())]))
    t = t.join(ex, "uuid", join_type="left outer")

    # Folders: the exact species key. Check the mapping is one-to-one before touching anything.
    species = t["species"].fill_null("unknown")
    uniq = pc.unique(species).to_pylist()
    names = {s: folder_name(s) for s in uniq}
    clash = len(uniq) - len(set(names.values()))
    if clash:
        raise SystemExit(f"{clash} species map to an already-used folder name; fix folder_name()")
    old = t["path"].to_pylist()
    new = [f"{names[s]}/{p.rsplit('/', 1)[-1]}" for s, p in zip(species.to_pylist(), old)]
    t = t.rename_columns(["old_path" if c == "path" else c for c in t.column_names])
    t = t.append_column("path", pa.array(new, pa.string()))

    # Check against what is on disk.
    dirs = [d for d in (crawl / "images").iterdir() if d.is_dir()]
    with ThreadPoolExecutor(32) as pool:
        on_disk = {f for fs in pool.map(_scan, dirs) for f in fs}
    have = pa.array([p in on_disk for p in old])
    print(f"{len(on_disk):,} files on disk in {len(dirs):,} folders; "
          f"{t.num_rows - pc.sum(have).as_py():,} ok rows have no file (dropped)")
    t = t.filter(have)
    orphans = sorted(on_disk - set(t["old_path"].to_pylist()))
    print(f"{len(orphans):,} files on disk that no ok row points to (-> orphans/)")

    order = ["path", "uuid", "species", "genus", "family", "order", "exclude", "source",
             "data_source", "source_id", "gbif_id", "url", "width", "height", "bytes", "content_hash"]
    t = t.sort_by([("species", "ascending"), ("path", "ascending")])
    pq.write_table(t.select(order), build / "metadata.parquet", compression="zstd")
    pq.write_table(t.select(["old_path", "path"]).rename_columns(["old", "new"]), build / "moves.parquet")
    pq.write_table(pa.table({"old": pa.array(orphans, pa.string())}), build / "orphans.parquet")
    n_ex = t.num_rows - t["exclude"].null_count
    print(f"metadata: {t.num_rows:,} images, {len(uniq):,} species folders, "
          f"{n_ex:,} flagged exclude, {pc.sum(t['bytes']).as_py() / 1e12:.2f} TB -> {build}")


# --------------------------------------------------------------------------------------------- #
def apply(a):
    crawl, build = Path(a.crawl), Path(a.build)
    img = crawl / "images"
    moves = pq.read_table(build / "moves.parquet").to_pylist()
    journal = build / "applied.tsv"
    log = journal.open("a", buffering=1)

    # A folder whose files all go to one new folder, and which is that folder's only source,
    # moves as a whole. Everything else (the 158 shared slugs) moves file by file.
    targets: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    for m in moves:
        od, nd = m["old"].split("/", 1)[0], m["new"].split("/", 1)[0]
        targets.setdefault(od, set()).add(nd)
        sources.setdefault(nd, set()).add(od)
    whole = {od for od, nds in targets.items() if len(nds) == 1 and len(sources[next(iter(nds))]) == 1}

    def mv(src: Path, dst: Path) -> None:
        if not src.exists() and dst.exists():
            return  # done on an earlier run
        log.write(f"{src.relative_to(crawl)}\t{dst.relative_to(crawl)}\n")
        os.rename(src, dst)

    # Orphans first, so a whole-folder rename carries only usable files.
    orphans = pq.read_table(build / "orphans.parquet")["old"].to_pylist()
    for o in orphans:
        dst = crawl / "orphans" / o
        dst.parent.mkdir(parents=True, exist_ok=True)
        mv(img / o, dst)
    n_dir = n_file = 0
    for od in sorted(whole):
        nd = next(iter(targets[od]))
        if od != nd:
            if (img / nd).exists() and not (img / od).exists():
                continue
            if (img / nd).exists():
                raise SystemExit(f"{nd} exists already; refusing to rename {od} onto it")
            mv(img / od, img / nd)
            n_dir += 1
    for m in moves:
        od = m["old"].split("/", 1)[0]
        if od in whole:
            continue
        if m["old"] == m["new"]:
            continue
        dst = img / m["new"]
        dst.parent.mkdir(exist_ok=True)
        mv(img / m["old"], dst)
        n_file += 1
    for od in targets:  # emptied shared slugs
        if od not in whole and od not in sources and (img / od).is_dir() and not any((img / od).iterdir()):
            (img / od).rmdir()
    log.close()
    print(f"renamed {n_dir:,} folders, moved {n_file:,} files, {len(orphans):,} orphans -> {crawl/'orphans'}")

    # Verify: every metadata path exists.
    newp = [m["new"] for m in moves]
    dirs = sorted({p.split("/", 1)[0] for p in newp})
    with ThreadPoolExecutor(32) as pool:
        on_disk = {f for fs in pool.map(lambda d: _scan(img / d), dirs) for f in fs}
    missing = sum(p not in on_disk for p in newp)
    extra = sum(1 for _ in img.iterdir()) - len(dirs)
    print(f"verify: {missing:,} metadata paths missing, {extra:,} folders not in metadata")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["snapshot", "plan", "apply"])
    p.add_argument("--crawl", required=True, help="crawl directory (manifest/, retry*/, images/)")
    p.add_argument("--build", required=True, help="working directory for this script's outputs")
    a = p.parse_args()
    {"snapshot": snapshot, "plan": plan, "apply": apply}[a.step](a)
