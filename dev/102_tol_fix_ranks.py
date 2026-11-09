"""Undo the catalog's rank shift in the shared ToL dataset (/data/datasets/tol256), in place.

TreeOfLife-200M left-packs its rank columns, so 2.6 M of our rows sit in bare-epithet folders that merge
genera (`images/butleri/`), with every rank one column too high. `dev/082` now undoes it (`realign`);
this script repairs what the crawl already wrote, and fetches what the corrected keys select.

    python dev/102_tol_fix_ranks.py catalog --build B   # distinct rank tuples of the 233 M catalog rows
    python dev/102_tol_fix_ranks.py plan    --build B   # dry run: B/metadata.parquet, B/file_moves.parquet
    python dev/102_tol_fix_ranks.py delta   --build B   # rows the corrected keys select that were never queued
    # fetch them with dev/082 fetch --manifest <out> (fix/run.sh), then:
    python dev/102_tol_fix_ranks.py add     --build B   # their rows: B/metadata_final.parquet, B/added_files.parquet
    python dev/102_tol_fix_ranks.py colxr   --build C   # dry run of the Lepidoptera labels' move to COL XR (dev/101); apply --build C runs it
    python dev/102_tol_fix_ranks.py apply   --build B   # link new paths and added files, swap metadata.parquet, unlink old
    python dev/102_tol_fix_ranks.py check   --build B   # tree, keys and counts of the shared copy

What `plan` changes, per row:
- shifted rows: ranks realigned (`dev/082 realign`, with each name's rank as the complete rows give it),
  empty ranks filled from the genus's path where all complete rows of that genus agree, class filled
  for fishes and reptiles (`CLASS_OF`, the only names not taken from the catalog), `species` =
  "Genus epithet", file moved to images/Genus_epithet/.
- records not identified to species (one-word keys, e.g. `Megaselia`): `species` empty, `taxon_rank`
  genus (or family where the source copied the family into `genus`). Their folders stay.
- species that the corrected key puts under the crawl's floor (50 images in the whole catalog):
  `exclude` = below_floor, as the crawl would not have taken them. Files stay.

`apply` keeps every metadata row pointing at an existing file throughout: it hard-links each moved file
at its new path, replaces metadata.parquet atomically (the old one kept as B/metadata_before.parquet),
then removes the old links. Each step is journalled in B/applied.tsv, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                            # dev/082's dataclasses look themselves up here
    spec.loader.exec_module(mod)
    return mod


CRAWLER = _load("dev082_crawler", "082_tol_crawler.py")
SHARE = _load("dev090_share", "090_tol_share.py")
RANKS = CRAWLER.RANKS
realign, TableShards, folder_name, drop_cache, _row_groups, log_mem = (
    CRAWLER.realign, SHARE.TableShards, SHARE.folder_name, SHARE.drop_cache, SHARE._row_groups, SHARE.log_mem)

SHARE_GL = ("/data/au761367/lepinet/global/"
            "0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet")
MIN_IMG = 50  # dev/082 plan --min-img, as crawled
# Chordate orders the catalog gives no class (GBIF's backbone has none above them). The only rank values
# this script does not take from the catalog.
REPTILE_ORDERS = {"Squamata", "Testudines", "Crocodylia", "Sphenodontia"}


def CLASS_OF(phylum: str, order: str) -> str:
    if phylum != "Chordata":
        return ""
    if order in REPTILE_ORDERS:
        return "Reptilia"
    return "Actinopterygii" if order.endswith("iformes") else ""


# --------------------------------------------------------------------------------------------- #
def catalog(a):
    """B/catalog_taxa.parquet: one row per distinct (kingdom..species) in the catalog, with its count."""
    path = Path(a.hf) / "dataset/catalog.parquet"
    pf = pq.ParquetFile(path)
    parts = []
    for i in range(pf.metadata.num_row_groups):
        d = pf.read_row_group(i, columns=RANKS).to_pandas().fillna("")
        parts.append(d.value_counts().rename("n").reset_index())
        if len(parts) > 64:
            parts = [pd.concat(parts).groupby(RANKS, as_index=False)["n"].sum()]
            drop_cache(path)
    t = pd.concat(parts).groupby(RANKS, as_index=False)["n"].sum()
    t.to_parquet(Path(a.build) / "catalog_taxa.parquet")
    print(f"{len(t):,} distinct rank tuples over {t['n'].sum():,} catalog rows")


class Fixer:
    """Raw rank tuple -> corrected ranks, learned from the catalog's complete rows."""

    def __init__(self, taxa: pd.DataFrame):
        v = taxa[RANKS].to_numpy()
        complete = (v != "").all(axis=1)
        cnt: dict[str, Counter] = defaultdict(Counter)
        for i in range(5):
            for name, n in taxa[complete].groupby(RANKS[i])["n"].sum().items():
                cnt[name][i] += n
        self.kinds = {name: c.most_common(1)[0][0] for name, c in cnt.items()}
        # genus -> (phylum, class, order, family) where every complete row of that genus and kingdom agrees
        paths = taxa[complete].groupby(["kingdom", "genus"])[["phylum", "class", "order", "family"]].nunique()
        one = paths[(paths == 1).all(axis=1)].index
        first = taxa[complete].drop_duplicates(["kingdom", "genus"]).set_index(["kingdom", "genus"])
        self.genus_path = first.loc[first.index.intersection(one), ["phylum", "class", "order", "family"]] \
            .apply(tuple, axis=1).to_dict()
        self.cache: dict[tuple, tuple] = {}

    def __call__(self, raw: tuple) -> tuple:
        r = self.cache.get(raw)
        if r is None:
            r = list(realign(raw, self.kinds))
            if r != list(raw):                              # shifted: fill what the genus's path settles
                known = self.genus_path.get((r[0], r[5]))
                if known:
                    for i, x in zip(range(1, 5), known):
                        r[i] = r[i] or x
                r[2] = r[2] or CLASS_OF(r[1], r[3])
            r = self.cache[raw] = tuple(r)
        return r


def _raw(d: pd.DataFrame) -> list[tuple]:
    """The catalog's rank tuple of each metadata row: its `species` column holds dev/082's key."""
    ep = d["species"].str.split(" ", n=1).str[1].fillna("")
    return list(zip(*(d[c] for c in RANKS[:6]), ep))


def _gl_ranks(path: str) -> pd.DataFrame:
    """order, family, genus of our own GBIF rows by uuid: dev/090 enrich took only kingdom..class from
    global_lepi, so 1.41 M rows had no order, family or genus (2026-11-08)."""
    t = pq.read_table(path, columns=["filename", "order", "family", "genus"]).to_pandas()
    t.index = "gl-" + t.pop("filename").str.replace(r"\.[^.]*$", "", regex=True)
    return t[~t.index.duplicated()]


def _transform(d: pd.DataFrame, fix: Fixer, catalog_n: pd.Series, gl: pd.DataFrame):
    """Corrected rows, which ones moved, and which fell under the floor."""
    own = d["uuid"].str.startswith("gl-").to_numpy()
    if own.any():
        g = gl.reindex(d["uuid"][own])
        for c in ("order", "family", "genus"):
            d.loc[own, c] = d.loc[own, c].fillna(pd.Series(g[c].to_numpy(), index=d.index[own]))
    d[RANKS[:6] + ["species"]] = d[RANKS[:6] + ["species"]].fillna("")
    raw = _raw(d)
    new = [fix(r) for r in raw]
    shifted = np.fromiter((n != r for n, r in zip(new, raw)), bool, len(raw))
    new = pd.DataFrame(new, columns=RANKS, index=d.index)
    key = d["species"].where(~shifted, (new["genus"] + " " + new["species"]).str.strip())
    one_word = ~key.str.contains(" ")
    genus = new["genus"].where(~(one_word & (new["genus"] == "")), key)
    rank = np.where(~one_word, "species", np.where(genus == new["family"], "family", "genus"))
    under = shifted & ~one_word & (key.map(catalog_n).fillna(0).to_numpy() < MIN_IMG)
    for c in RANKS[:5]:
        d[c] = new[c].replace("", None)
    d["genus"] = genus.where(rank != "family", "").replace("", None)
    d["species"] = key.where(~one_word, None)
    d["taxon_rank"] = rank
    d["path"] = d["path"].where(~shifted, key.map(folder_name) + "/" + d["uuid"] + ".jpg")
    d["exclude"] = d["exclude"].where(d["exclude"].notna(), pd.Series(np.where(under, "below_floor", None), index=d.index))
    return d, shifted, under


def plan(a):
    build, dataset = Path(a.build), Path(a.dataset)
    taxa = pd.read_parquet(build / "catalog_taxa.parquet")
    fix = Fixer(taxa)
    keys = [CRAWLER.species_key(r) for r in taxa[RANKS].itertuples(index=False)]
    catalog_n = taxa.groupby(pd.Series(keys, index=taxa.index))["n"].sum()
    gl = _gl_ranks(a.global_lepi)
    log_mem("fixer", a.log_mem)

    meta = dataset / "metadata.parquet"
    schema = pq.read_schema(meta)
    out_schema = schema.insert(schema.get_field_index("species") + 1, pa.field("taxon_rank", pa.string()))
    # Pass 1: rows per new folder, so pass 2 can route rows to folder ranges and write them sorted.
    # It also collects each species' complete paths, to fill rows that lack ranks (our own rows that
    # global_lepi's filtered parquet does not cover) where all complete rows of the species agree.
    folders: Counter = Counter()
    paths: set[tuple] = set()
    for t in _row_groups(meta):
        d, _, _ = _transform(t.to_pandas(), fix, catalog_n, gl)
        folders.update(d["path"].str.split("/", n=1).str[0].value_counts().to_dict())
        full = d[d["species"].notna() & d[RANKS[:6]].notna().all(axis=1)]
        paths.update(full[["kingdom", "species", *RANKS[1:6]]].drop_duplicates().itertuples(index=False, name=None))
    by_species = pd.DataFrame(list(paths), columns=["kingdom", "species", *RANKS[1:6]])
    by_species = by_species[~by_species.duplicated(["kingdom", "species"], keep=False)].set_index(["kingdom", "species"])
    log_mem("pass 1", a.log_mem)
    K = 64
    names = sorted(folders)
    cum = np.cumsum([folders[n] for n in names])
    bounds = np.array([names[int(np.searchsorted(cum, cum[-1] * q / K))] for q in range(1, K)], dtype=object)

    out = TableShards(build / "work_out", K)
    moves = TableShards(build / "work_moves", 1)
    stats: Counter = Counter()
    below: Counter = Counter()
    for t in _row_groups(meta):
        old_path = t["path"].to_numpy(zero_copy_only=False)
        d, shifted, under = _transform(t.to_pandas(), fix, catalog_n, gl)
        gap = (d["species"].notna() & d[RANKS[1:6]].isna().any(axis=1)).to_numpy()
        if gap.any():
            known = by_species.reindex(pd.MultiIndex.from_frame(d.loc[gap, ["kingdom", "species"]]))
            for c in RANKS[1:6]:
                d.loc[gap, c] = d.loc[gap, c].fillna(pd.Series(known[c].to_numpy(), index=d.index[gap]))
            stats["filled_from_species"] += int(known.notna().all(axis=1).sum())
        stats["rows"] += len(d); stats["moved"] += int(shifted.sum()); stats["below_floor"] += int(under.sum())
        for r, n in d["taxon_rank"].value_counts().items():
            stats[f"rank_{r}"] += int(n)
        below.update(d["species"][under])
        folder = d["path"].str.split("/", n=1).str[0].to_numpy(dtype=object)
        out.add(pa.Table.from_pandas(d[out_schema.names], schema=out_schema, preserve_index=False),
                np.searchsorted(bounds, folder, side="right"))
        if shifted.any():
            moves.add(pa.table({"old": old_path[shifted], "new": d["path"].to_numpy(dtype=object)[shifted]}),
                      np.zeros(int(shifted.sum())))
        log_mem(f"rows {stats['rows']:,}", a.log_mem)
    out.close(); moves.close()

    w = pq.ParquetWriter(build / "metadata.parquet", out_schema, compression="zstd")
    n = 0
    for s in range(K):
        t = out.read(s)
        if t is not None:
            w.write_table(t.sort_by([("path", "ascending")]), row_group_size=100_000)
            n += t.num_rows
    w.close()
    m = moves.read(0)
    pq.write_table(m if m is not None else pa.table({"old": pa.array([], pa.string()), "new": pa.array([], pa.string())}),
                   build / "file_moves.parquet")
    shutil.rmtree(build / "work_out"); shutil.rmtree(build / "work_moves")
    stats["below_floor_species"] = len(below)
    stats["folders"] = len(folders)
    (build / "plan_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"plan: {n:,} rows -> {build / 'metadata.parquet'}; {stats['moved']:,} moves -> {build / 'file_moves.parquet'}")


def delta(a):
    """A dev/082 manifest of what the corrected keys select and the old crawl never queued.

    Selection as dev/082 `plan` + `substitute`: per species, the first `--cap` rows in catalog order on a
    host that is not blocked, for species with at least MIN_IMG rows in the catalog. Only species whose
    key the correction changes are considered (a shifted row among them); for the rest, nothing moves."""
    build, crawl = Path(a.build), Path(a.crawl)
    taxa = pd.read_parquet(build / "catalog_taxa.parquet")
    old_key = (taxa["genus"].str.strip() + " " + taxa["species"].str.strip()).str.strip()
    keys = pd.Series([CRAWLER.species_key(r) for r in taxa[RANKS].itertuples(index=False)], index=taxa.index)
    total = taxa.groupby(keys)["n"].sum()
    changed = set(keys[keys != old_key]) - {""}
    want = {k for k in changed if total.get(k, 0) >= MIN_IMG}
    blocked = set(json.loads((crawl / "share_build/snapshot/blocked_hosts.json").read_text()))
    # uuids the old crawl queued for these species, under their old keys or their own
    old_keys = set(old_key[keys.isin(want)]) | want
    queued: set[str] = set()
    for part in (crawl / "manifest").glob("host=*/part-*.parquet"):
        if part.name.endswith(".meta.parquet"):
            continue
        t = pq.read_table(part, columns=["uuid", "species"])
        queued.update(t.filter(pc.is_in(t["species"], pa.array(sorted(old_keys))))["uuid"].to_pylist())
    print(f"{len(want):,} corrected species with >= {MIN_IMG} catalog rows; {len(queued):,} of their rows queued before")

    path = Path(a.hf) / "dataset/catalog.parquet"
    pf = pq.ParquetFile(path)
    out = Path(a.out)
    emitted: Counter = Counter()
    rows: dict[str, list] = defaultdict(list)
    n_new = 0
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=CRAWLER.PLAN_COLS).to_pydict()
        for j in range(len(t["uuid"])):
            r = realign([t[c][j] for c in RANKS])
            sp = f"{r[5]} {r[6]}".strip()
            if sp not in want or emitted[sp] >= a.cap:
                continue
            url = t["source_url"][j]
            if not url:
                continue
            host = CRAWLER.host_of(url)
            if host in blocked:
                continue
            emitted[sp] += 1
            if t["uuid"][j] in queued:
                continue
            rows[host].append({"uuid": t["uuid"][j], "url": url, "species": sp, "genus": r[5], "family": r[4],
                               "order": r[3], "data_source": t["data_source"][j] or "",
                               "source_id": str(t["source_id"][j] or "")})
            n_new += 1
        if i % 200 == 0:
            drop_cache(path)
            print(f"  row group {i}: {n_new:,} new rows", flush=True)
    for host, rr in rows.items():
        d = out / f"host={CRAWLER.slug(host)}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rr), d / "part-f00000.parquet")
    for f in ("blocked_contacts.json",):
        shutil.copyfile(crawl / "manifest" / f, out / f)
    (out / "blocked_hosts.json").write_text(json.dumps({}))
    per = pd.Series({h: len(r) for h, r in rows.items()}).sort_values(ascending=False)
    summary = {"species": len(want), "new_rows": n_new, "hosts": len(rows),
               "species_with_new_rows": len({r["species"] for rr in rows.values() for r in rr}),
               "top_hosts": per.head(10).to_dict()}
    (out / "delta_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _write_sorted(tables: Callable, out: Path, schema: pa.Schema, work: Path, K: int = 64) -> int:
    """Write an iterable of tables as one parquet sorted by path, holding one path range at a time."""
    counts: Counter = Counter()
    for t in tables():                                 # `tables` is called twice: count, then route
        counts.update({x["values"]: x["counts"] for x in pc.value_counts(pc.utf8_slice_codeunits(t["path"], 0, 3)).to_pylist()})
    names = sorted(counts)
    cum = np.cumsum([counts[n] for n in names])
    bounds = np.array([names[int(np.searchsorted(cum, cum[-1] * q / K))] for q in range(1, K)], dtype=object)
    shards = TableShards(work, K)
    for t in tables():
        pre = np.asarray(pc.utf8_slice_codeunits(t["path"], 0, 3).to_pylist(), dtype=object)
        shards.add(t.cast(schema), np.searchsorted(bounds, pre, side="right"))
    shards.close()
    w = pq.ParquetWriter(out, schema, compression="zstd")
    n = 0
    for s in range(K):
        t = shards.read(s)
        if t is not None:
            w.write_table(t.sort_by([("path", "ascending")]), row_group_size=100_000)
            n += t.num_rows
    w.close()
    shutil.rmtree(work)
    return n


def _norm_url(urls) -> np.ndarray:
    """dev/085's URL identity: scheme-insensitive."""
    return SHARE.AUDIT._h([re.sub(r"^[a-z]+://", "", u or "") for u in urls])


def _genus_paths(meta: Path) -> dict:
    """(kingdom, genus) -> (phylum, class, order, family) where every complete row of the genus agrees."""
    seen: dict = {}
    for t in _row_groups(meta, RANKS[:6]):
        u = t.to_pandas().dropna().drop_duplicates()
        for k, p, c, o, f, g in u.itertuples(index=False):
            seen.setdefault((k, g), set()).add((p, c, o, f))
    return {kg: next(iter(v)) for kg, v in seen.items() if len(v) == 1}


def add(a):
    """Rows for the images the delta crawl fetched, merged into the planned metadata.

    Columns as dev/090 enrich (catalog and provenance by uuid), ranks through the same Fixer, and
    `exclude` checked against the whole dataset as dev/085 does: lepinet's test fold by GBIF
    occurrence, the audit's placeholder hashes, then URLs and pixels already present (first copy kept)."""
    build, dataset, hf, fix_dir = Path(a.build), Path(a.dataset), Path(a.hf), Path(a.out).parent
    parts = [p for p in sorted(Path(a.out).glob("host=*/part-*.parquet")) if not p.name.endswith(".meta.parquet")]
    rows = []
    for p in parts:
        m = p.with_suffix(".meta.parquet")
        if not m.exists():
            continue
        t = pq.read_table(p).to_pandas().merge(
            pq.read_table(m, columns=["uuid", "status", "path", "content_hash", "width", "height", "bytes"]).to_pandas(),
            on="uuid")
        rows.append(t[t["status"] == "ok"])
    new = pd.concat(rows, ignore_index=True).drop_duplicates("uuid")
    print(f"{len(new):,} images fetched ok, from {sum(p.with_suffix('.meta.parquet').exists() for p in parts)} of {len(parts)} parts")
    want = pa.array(new["uuid"].tolist())

    def by_uuid(path: Path, cols: list[str]) -> pd.DataFrame:
        got = []
        for t in _row_groups(path, ["uuid", *cols]):
            got.append(t.filter(pc.is_in(t["uuid"], want)).to_pandas())
        return pd.concat(got).drop_duplicates("uuid").set_index("uuid")
    cat = by_uuid(hf / "dataset/catalog.parquet", [*RANKS, *SHARE.CATALOG_COLS[3:]])
    prov = by_uuid(hf / "dataset/metadata/provenance.parquet", list(SHARE.PROVENANCE_COLS)).rename(
        columns=SHARE.PROVENANCE_COLS)
    fix = Fixer(pd.read_parquet(build / "catalog_taxa.parquet"))
    r = pd.DataFrame([fix(tuple("" if pd.isna(x) else x for x in row))
                      for row in cat.reindex(new["uuid"])[RANKS].itertuples(index=False)], columns=RANKS)
    d = pd.DataFrame({"uuid": new["uuid"].to_numpy(), "species": new["species"].to_numpy()})
    for c in RANKS[:6]:
        d[c] = r[c].replace("", None).to_numpy()
    # Rows with two or more missing ranks (never crawled before) can name a genus that has no
    # complete catalog row at all (Riodinidae). Fill their empty ranks from the genus's path in the
    # planned dataset, where that path is unique; what no source gives stays empty.
    paths = _genus_paths(build / "metadata.parquet")
    gaps = d[RANKS[1:5]].isna().any(axis=1)
    got = [paths.get((k, g)) for k, g in zip(d.loc[gaps, "kingdom"], d.loc[gaps, "genus"])]
    for i, c in enumerate(RANKS[1:5]):
        fill = pd.Series([x[i] if x else None for x in got], index=d.index[gaps])
        d.loc[gaps, c] = d.loc[gaps, c].fillna(fill)
    print(f"{int(gaps.sum()):,} added rows lacked a rank; {sum(x is not None for x in got):,} filled from "
          f"their genus; still lacking one: {int(d[RANKS[1:5]].isna().any(axis=1).sum()):,}")
    d["taxon_rank"] = "species"
    d["path"] = (d["species"].map(folder_name) + "/" + d["uuid"] + ".jpg").to_numpy()
    for c in SHARE.CATALOG_COLS[3:]:
        d[c] = cat[c].reindex(d["uuid"]).to_numpy()
    for c in SHARE.PROVENANCE_COLS.values():
        d[c] = prov[c].reindex(d["uuid"]).to_numpy() if c in prov else None
    d["source"] = "fix_ranks"
    for c in ("data_source", "source_id", "url", "content_hash", "width", "height", "bytes"):
        d[c] = new[c].to_numpy()
    d["gbif_id"] = np.where(d["data_source"] == "gbif", d["source_id"], None)

    # exclude, in dev/085's order of precedence
    test = set()
    for t in _row_groups(a.global_lepi, ["gbifID", "set"]):
        g = t.to_pandas()
        test.update(g.loc[g["set"].astype(str) == "0", "gbifID"].astype("int64").astype(str))
    seen_url, seen_px, placeholder = [], [], set()
    for t in _row_groups(build / "metadata.parquet", ["url", "content_hash", "exclude"]):
        seen_url.append(np.unique(_norm_url(t["url"].to_pylist())))
        ch = t["content_hash"].to_pylist()
        seen_px.append(np.unique(SHARE.AUDIT._h([x or "" for x in ch])))
        placeholder.update(x for x, e in zip(ch, t["exclude"].to_pylist()) if e == "placeholder")
    seen_url, seen_px = np.unique(np.concatenate(seen_url)), np.unique(np.concatenate(seen_px))
    hu, hp = _norm_url(d["url"].tolist()), SHARE.AUDIT._h(d["content_hash"].fillna("").tolist())
    dup_url = np.isin(hu, seen_url) | pd.Series(hu).duplicated().to_numpy()
    dup_px = (np.isin(hp, seen_px) | pd.Series(hp).duplicated().to_numpy()) & d["content_hash"].notna().to_numpy()
    d["exclude"] = np.select(
        [d["gbif_id"].isin(test).to_numpy(), d["content_hash"].isin(placeholder).to_numpy(), dup_url, dup_px],
        ["lepinet_test_fold", "placeholder", "duplicate_url", "pixel_duplicate"], None)
    d = d.astype({c: object for c in d.columns if c not in SHARE.INT_COLS})
    print("exclude:", d["exclude"].value_counts(dropna=False).to_dict())

    schema = pq.read_schema(build / "metadata.parquet")
    added = pa.Table.from_pandas(d[schema.names], schema=schema, preserve_index=False)
    pq.write_table(pa.table({"src": [str(fix_dir / "images" / p) for p in new["path"]],
                             "new": d["path"].tolist()}), build / "added_files.parquet")

    def tables():
        yield from _row_groups(build / "metadata.parquet")
        yield added
    n = _write_sorted(tables, build / "metadata_final.parquet", schema, build / "work_final")
    print(f"metadata_final: {n:,} rows ({len(d):,} added) -> {build / 'metadata_final.parquet'}")


def _binomial(name: str, genus: str) -> str:
    """COL XR's "Anarta (Calocestra) trifolii (Hufnagel, 1766)" as this dataset's "Anarta trifolii"."""
    words = re.sub(r"\([^)]*\)", " ", name).split()
    return f"{genus} {next(w for w in words[1:] if w[0].islower())}"


def colxr(a):
    """Dry run of PLAN row TAX on the Lepidoptera rows: dev/101's frozen tables, B/metadata.parquet + B/file_moves.parquet.

    An occurrence that is in our GBIF download (by file, else gbifID) takes dev/101's decision for its own
    (speciesKey, taxonKey). A catalog row carries no taxonKey: it takes its class's name change, where the
    occurrences above tie its name to one legacy class (MAJORITY of them), so a species does not split
    between "our rows" and "catalog rows". Classes dev/101 leaves unresolved keep their name. A row
    without a class keeps its name and takes its genus's COL XR family when that is unique. Nothing else
    changes; `exclude` follows the floor for the species a rename touches. `apply --build B` then runs it."""
    relabel = _load("dev101_colxr", "101_colxr_relabel.py")
    MAJORITY = 0.9
    names, tax = relabel.load_tables()
    build, meta = Path(a.build), Path(a.dataset) / "metadata.parquet"
    ranks = {"genus": "genus_name", "family": "family_name", "order": "order_name", "class": "class_name",
             "phylum": "phylum_name", "kingdom": "kingdom_name"}
    resolved = {k for k in tax if not k.startswith("gbif:")}
    binom = {k: _binomial(tax[k]["species_name"], tax[k]["genus_name"]) for k in resolved}

    rows = []
    for t in _row_groups(meta, ["uuid", "species", "genus", "family", "order", "gbif_id", "exclude", "path", "class", "phylum", "kingdom"]):
        rows.append(t.filter(pc.equal(t["order"], "Lepidoptera")))
    d = pa.concat_tables(rows).to_pandas()
    del rows
    g = pq.read_table(a.global_lepi, columns=["gbifID", "filename", "speciesKey", "taxonKey"]).to_pandas()
    g["pair"] = g["speciesKey"].astype("int64").astype(str) + "|" + g["taxonKey"].astype("int64").astype(str)
    by_file = pd.Series(g["pair"].to_numpy(), index="gl-" + g["filename"].str.rsplit(".", n=1).str[0])
    by_id = pd.Series(g["pair"].to_numpy(), index=g["gbifID"].astype("int64").astype(str))
    del g
    pair = d["uuid"].map(by_file)
    pair = pair.where(pair.notna(), d["gbif_id"].map(by_id))
    del by_file, by_id
    log_mem("joined", a.log_mem)

    decided = {f"{k[0]}|{k[1]}": v for k, v in names.items()}
    key = pair.map(lambda x: decided[x]["new_species_key"], na_action="ignore")
    why = pair.map(lambda x: decided[x]["reason"], na_action="ignore")
    is_sp = d["species"].notna()
    # A catalog row: its name -> the legacy class most of its occurrences are in -> that class's decision.
    seen = pd.DataFrame({"species": d["species"], "legacy": pair.str.split("|").str[0]}).dropna()
    n = seen.groupby(["species", "legacy"]).size().reset_index(name="n").sort_values("n", ascending=False)
    share = n.groupby("species")["n"].transform(lambda x: x / x.sum())
    top = n[(share >= MAJORITY)].drop_duplicates("species").set_index("species")["legacy"]
    primary = {k[0]: v["new_species_key"] for k, v in names.items() if v["reason"] != "other_species"}
    by_name = pair.isna() & is_sp
    key[by_name] = d.loc[by_name, "species"].map(top).map(primary)
    why[by_name & key.notna()] = "by_name"
    del pair, seen, n, top

    usable = is_sp & key.isin(resolved)
    after = d["species"].where(~usable, key.map(binom))
    renamed = after.notna() & (after != d["species"])
    out = d.copy()
    out["species"] = after
    for c, col in ranks.items():
        out.loc[usable, c] = key[usable].map(lambda k: tax[k][col])
    # Rows without a class: only the family, so the genus keeps one parent.
    col_rows = [r for r in tax.values() if r["parents_from"] == "col_xr"]
    fam: dict = defaultdict(set)
    for r in col_rows:
        fam[r["genus_name"]].add(r["family_name"])
    fam = {g: next(iter(v)) for g, v in fam.items() if len(v) == 1}
    rest = ~usable & d["genus"].isin(set(fam))
    out.loc[rest, "family"] = d.loc[rest, "genus"].map(fam)
    # The floor, for the species a rename touches. Counted in this dataset: the catalog has no such key.
    count = out["species"].value_counts()
    touched = set(d.loc[renamed, "species"]) | set(after[renamed])
    t = out["species"].isin(touched)
    big = out["species"].map(count).fillna(0) >= MIN_IMG
    out.loc[t & ~big & out["exclude"].isna(), "exclude"] = "below_floor"
    out.loc[t & big & (out["exclude"] == "below_floor"), "exclude"] = None
    out.loc[renamed, "path"] = out.loc[renamed, "species"].map(folder_name) + "/" + out.loc[renamed, "uuid"] + ".jpg"

    cols = ["species", "genus", "family", "order", "class", "phylum", "kingdom", "exclude", "path"]
    same = (out[cols].fillna("") == d[cols].fillna("")).all(axis=1)
    chg = out.loc[~same, ["uuid", *cols]].set_index("uuid")
    chg = {u: {c: None if pd.isna(x) else x for c, x in zip(cols, v)} for u, v in zip(chg.index, chg.itertuples(index=False, name=None))}
    stats = {"lepidoptera_rows": len(d), "rows_changed": len(chg), "rows_renamed": int(renamed.sum()), "by_decision": why[renamed].value_counts().to_dict(),
             "mapped": why.value_counts().to_dict(), "no_class_rows": int((is_sp & key.isna()).sum()), "no_class_species": int(d.loc[is_sp & key.isna(), "species"].nunique()),
             "species_before": int(d["species"].nunique()), "species_after": int(out["species"].nunique()),
             "old_names_renamed": int(d.loc[renamed, "species"].nunique()), "new_names": int(after[renamed].nunique()),
             "new_folders": len(set(after[renamed]) - set(d["species"].dropna())),
             "family_changed": int((out["family"].fillna("") != d["family"].fillna("")).sum()), "genus_changed": int((out["genus"].fillna("") != d["genus"].fillna("")).sum()),
             "order_changed": int((out["order"].fillna("") != d["order"].fillna("")).sum()),
             "over_cap_species": int((count > a.cap).sum()), "over_cap_rows_above_cap": int((count[count > a.cap] - a.cap).sum()), "largest": int(count.max()),
             "below_floor_set": int(((out["exclude"] == "below_floor") & d["exclude"].isna()).sum()), "below_floor_cleared": int(((d["exclude"] == "below_floor") & out["exclude"].isna()).sum())}
    moves = pa.table({"old": d.loc[renamed, "path"].tolist(), "new": out.loc[renamed, "path"].tolist()})
    pq.write_table(moves, build / "file_moves.parquet")
    (build / "plan_summary.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    del d, out, after, key, why
    log_mem("decided", a.log_mem)

    schema = pq.read_schema(meta)
    wanted = pa.array(list(chg), pa.string())

    def tables():
        for t in _row_groups(meta):
            hit = pc.is_in(t["uuid"], value_set=wanted)
            if not pc.any(hit).as_py():
                yield t
                continue
            where = np.flatnonzero(hit.to_numpy(zero_copy_only=False))
            uuids = t["uuid"].take(pa.array(where)).to_pylist()
            for c in cols:
                v = t[c].to_pylist()
                for i, u in zip(where, uuids):
                    v[i] = chg[u][c]
                t = t.set_column(t.schema.get_field_index(c), t.schema.field(c), pa.array(v, t.schema.field(c).type))
            yield t
    n = _write_sorted(tables, build / "metadata.parquet", schema, build / "work")
    print(f"colxr plan: {n:,} rows -> {build / 'metadata.parquet'}; {moves.num_rows:,} moves")


# --------------------------------------------------------------------------------------------- #
def apply(a):
    build, dataset = Path(a.build), Path(a.dataset)
    img, meta = dataset / "images", dataset / "metadata.parquet"
    log = (build / "applied.tsv").open("a", buffering=1)
    moves = pq.read_table(build / "file_moves.parquet")
    old, new = moves["old"].to_pylist(), moves["new"].to_pylist()

    log.write("# link\n")
    linked = 0
    for o, n in zip(old, new):
        dst = img / n
        if dst.exists():
            continue
        dst.parent.mkdir(exist_ok=True)
        os.link(img / o, dst)
        linked += 1
    print(f"linked {linked:,} files at their new paths")

    final = build / "metadata_final.parquet"           # `add` ran: the plan plus the re-fetched rows
    if final.exists():
        added = pq.read_table(build / "added_files.parquet")
        log.write("# add\n")
        linked = 0
        for s, n in zip(added["src"].to_pylist(), added["new"].to_pylist()):
            dst = img / n
            if dst.exists():
                continue
            dst.parent.mkdir(exist_ok=True)
            os.link(s, dst)                            # the source stays in the fix crawl's folder
            log.write(f"+\t{n}\n")
            linked += 1
        print(f"linked {linked:,} re-fetched files")
    else:
        final = build / "metadata.parquet"

    if not (build / "metadata_before.parquet").exists():
        os.link(meta, build / "metadata_before.parquet")
    staged = dataset / ".metadata.parquet.new"
    shutil.copyfile(final, staged)
    os.chmod(staged, 0o644)
    os.replace(staged, meta)
    log.write("# swapped metadata.parquet\n")
    print(f"swapped {meta} (previous: {build / 'metadata_before.parquet'})")

    removed = 0
    emptied = set()
    for o, n in zip(old, new):
        src = img / o
        if src.exists() and (img / n).exists():
            os.unlink(src)
            log.write(f"{o}\t{n}\n")
            removed += 1
            emptied.add(src.parent)
    for d in emptied:
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    log.close()
    print(f"removed {removed:,} old links, {sum(not d.exists() for d in emptied):,} emptied folders")


def check(a):
    """The shared copy: files vs rows (dev/090 verify), keys, the tree, and counts."""
    dataset = Path(a.dataset)
    meta = Path(a.metadata) if a.metadata else dataset / "metadata.parquet"
    SHARE.verify(argparse.Namespace(build=a.build, dataset=a.dataset, metadata=str(meta), shards=a.shards))
    cols = ["species", "taxon_rank", *RANKS[:6], "exclude", "path"]
    parts = []
    for t in _row_groups(meta, cols):
        d = t.to_pandas()
        d["folder"] = d["path"].str.split("/", n=1).str[0]
        d = d.drop(columns="path").fillna("")
        parts.append(d.value_counts().rename("n").reset_index())
    d = pd.concat(parts).groupby(cols[:-1] + ["folder"], as_index=False)["n"].sum()
    fold = d.groupby("folder")["n"].sum()
    one_low = fold[~fold.index.str.contains("_") & fold.index.str[:1].str.islower()]
    sp = d[d["taxon_rank"] == "species"]
    print(f"rows {d['n'].sum():,}; species rows {sp['n'].sum():,} in {sp['species'].nunique():,} species; "
          f"genus-level {d[d.taxon_rank == 'genus']['n'].sum():,}, family-level {d[d.taxon_rank == 'family']['n'].sum():,}")
    print(f"folders {len(fold):,}; lowercase one-word folders {len(one_low):,} ({one_low.sum():,} rows)")
    print("exclude:", d.groupby("exclude")["n"].sum().to_dict())
    # Tree: each node keyed by its full path has one parent by construction, so what can break is a
    # name: a species or genus under two paths (homonyms across kingdoms, or the source disagreeing).
    for child, cols_up in [("species", RANKS[:6]), ("genus", RANKS[:5]), ("family", RANKS[:4]), ("order", RANKS[:3]),
                           ("class", RANKS[:2])]:
        x = d[d[child] != ""].groupby([child, *cols_up])["n"].sum().reset_index()
        multi = x.groupby(child).filter(lambda z: len(z) > 1)
        if multi.empty:
            print(f"{child}: every name has one path")
            continue
        cross = multi.groupby(child)["kingdom"].nunique() > 1
        minority = multi.groupby(child)["n"].apply(lambda s: s.sum() - s.max())
        print(f"{child}: {len(cross):,} names under several paths ({int(cross.sum()):,} across kingdoms); "
              f"rows off the majority path {int(minority.sum()):,}; with an empty rank in one path "
              f"{int(multi.groupby(child).apply(lambda z: (z[cols_up] == '').any(axis=None)).sum()):,}")
    d.to_parquet(Path(a.build) / "check_taxa.parquet")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["catalog", "plan", "delta", "add", "colxr", "apply", "check"])
    p.add_argument("--build", required=True, help="working directory for this script's outputs")
    p.add_argument("--dataset", default="/data/datasets/tol256", help="the shared copy")
    p.add_argument("--hf", default="/data/au761367/tol256s/hf", help="local TreeOfLife-200M metadata")
    p.add_argument("--global-lepi", default=SHARE_GL, help="our own GBIF download's parquet (plan)")
    p.add_argument("--metadata", default="", help="check: this file instead of <dataset>/metadata.parquet")
    p.add_argument("--shards", type=int, default=32, help="check: path-hash partitions")
    p.add_argument("--crawl", default="/data/au761367/tol256s", help="delta: the crawl (manifest/, share_build/)")
    p.add_argument("--out", default="/data/au761367/tol256s/fix/manifest", help="delta: the new manifest")
    p.add_argument("--cap", type=int, default=2000, help="delta: images per species, as crawled")
    p.add_argument("--log-mem", action="store_true", help="print resident memory after each stage")
    a = p.parse_args()
    Path(a.build).mkdir(parents=True, exist_ok=True)
    {"catalog": catalog, "plan": plan, "apply": apply, "check": check, "delta": delta, "add": add, "colxr": colxr}[a.step](a)
