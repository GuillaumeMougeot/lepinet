"""Freeze the shared ToL dataset (/data/datasets/tol256) for training: one dry run, one apply, one version.

    python dev/103_tol_freeze.py names     --build F   # GBIF COL XR lookups, cached: F/gbif_match.jsonl -> F/names.parquet
    python dev/103_tol_freeze.py shortfall --build F   # what the crawl selected and we do not hold, by host and reason
    python dev/103_tol_freeze.py decode    --build F   # full decode of every image: F/decode/ (resumable)
    python dev/103_tol_freeze.py sizes     --build F   # width/height of rows the crawler recovered from disk: F/sizes.parquet
    python dev/103_tol_freeze.py bench     --build F   # benchmark photographs in the dataset: F/bench/<name>.parquet
    python dev/103_tol_freeze.py plan      --build F   # the single dry run -> F/plan/ for `dev/102 apply --build F/plan`
    python dev/103_tol_freeze.py train     --build F --out O  # after the apply: O/train.parquet, O/class_spec.json (W3)
    python dev/103_tol_freeze.py shards    --build F --train-dir O --out S [--pilot]  # mini_trainer shards v1 -> S

The lookup is the same frozen checklist as dev/101. A name is accepted when the match is an exact one, at
species rank, with the canonical name equal to ours, not an ambiguous synonym, and in the kingdom (for
Lepidoptera, the order) the dataset says. Anything else keeps its name and is listed with why. Plan rows TF, W3.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import re
import time
from collections import Counter
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COL_XR = "7ddf754f-d193-4cc9-b351-99906754a03b"  # dev/101's checklist
TOL = Path("/data/datasets/tol256")
UA = "lepinet-taxonomy-snapshot"                 # generic client name: no contact details
RATE = 6.0                                       # requests per second, as dev/101
UPPER = ["phylum", "class", "order", "family"]
TAXA = ["kingdom", *UPPER, "genus", "species", "taxon_rank"]
GOOD_KINGDOMS = {"Animalia", "Plantae", "Fungi", "Chromista", "Protozoa", "Bacteria", "Viruses", "Archaea"}


def distinct_taxa(meta: Path) -> pd.DataFrame:
    """Every distinct rank tuple of the dataset, with its row count, '' for an empty rank."""
    n: Counter = Counter()
    pf = pq.ParquetFile(meta)
    for i in range(pf.metadata.num_row_groups):
        n.update(pf.read_row_group(i, columns=TAXA).to_pandas().fillna("").value_counts().to_dict())
    return pd.DataFrame([(*k, v) for k, v in n.items()], columns=[*TAXA, "n"])


def fill_inside(d: pd.DataFrame) -> pd.DataFrame:
    """Empty ranks from the rows of the same genus, then the same family, where those agree on one value."""
    d = d.copy()

    def one(s: pd.Series) -> str:
        v = [x for x in s.unique() if x]
        return v[0] if len(v) == 1 else ""
    for keys, cols in ((["kingdom", "genus"], UPPER), (["kingdom", "family"], UPPER[:3])):
        got = d[d[keys[1]] != ""].groupby(keys)[cols].agg(one)
        j = d[keys].merge(got, how="left", left_on=keys, right_index=True)
        for c in cols:
            hit = (d[c] == "") & j[c].fillna("").ne("")
            d.loc[hit, c] = j.loc[hit, c]
    return d


def match(name: str) -> dict:
    q = urllib.parse.urlencode({"checklistKey": COL_XR, "scientificName": name, "strict": "true"})
    request = urllib.request.Request(f"https://api.gbif.org/v2/species/match?{q}", headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as r:
                d = json.load(r)
            u, acc = d.get("usage") or {}, d.get("acceptedUsage") or {}
            return {"name": name, "type": (d.get("diagnostics") or {}).get("matchType", ""), "canonical": u.get("canonicalName", ""),
                    "rank": u.get("rank", ""), "status": u.get("status", ""), "key": u.get("key", ""),
                    "accepted": acc.get("canonicalName", ""), "accepted_key": acc.get("key", ""),
                    "chain": {c["rank"].lower(): c["name"] for c in d.get("classification") or []}}
        except Exception:  # noqa: BLE001 - retried with backoff; a final failure stops the run, the cache keeps the rest
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


def names(a):
    build = Path(a.build)
    build.mkdir(parents=True, exist_ok=True)
    d = fill_inside(distinct_taxa(Path(a.dataset) / "metadata.parquet"))
    sp = d[d["taxon_rank"] == "species"]
    gap = sp[(sp[UPPER] == "").any(axis=1)]["species"]
    tax = pd.read_csv(Path(__file__).with_name("colxr") / "taxonomy.csv", dtype=str, keep_default_na=False)
    known = set(tax["species_name"].str.replace(r"\([^)]*\)", " ", regex=True).str.split().str[:2].str.join(" "))
    lep = sp[(sp["order"] == "Lepidoptera") & ~sp["species"].isin(known)]["species"]
    want = sorted(set(gap) | set(lep))
    print(f"{gap.nunique():,} species with an empty rank after the in-dataset fill; {lep.nunique():,} Lepidoptera outside dev/101's tables; {len(want):,} names", flush=True)
    cache_path = build / "gbif_match.jsonl"
    cache = {r["name"]: r for r in map(json.loads, cache_path.read_text().splitlines())} if cache_path.exists() else {}
    todo = [n for n in want if n not in cache]
    started = time.monotonic()

    def paced(item):
        i, n = item
        time.sleep(max(0.0, started + i / RATE - time.monotonic()))
        return match(n)
    with cache_path.open("a") as sink, ThreadPoolExecutor(8) as pool:
        for i, r in enumerate(pool.map(paced, enumerate(todo)), 1):
            sink.write(json.dumps(r) + "\n")
            sink.flush()
            cache[r["name"]] = r
            if i % 1000 == 0:
                print(f"  {i:,}/{len(todo):,}", flush=True)
    d.to_parquet(build / "taxa_filled.parquet")
    r = pd.DataFrame([{**{k: v for k, v in cache[n].items() if k != "chain"}, "chain": json.dumps(cache[n]["chain"])} for n in want])
    r["ok"] = (r["type"] == "EXACT") & (r["rank"] == "SPECIES") & (r["canonical"].str.replace(r"\s*\([^)]*\)", "", regex=True) == r["name"]) & (r["status"] != "AMBIGUOUS_SYNONYM")
    r.to_parquet(build / "names.parquet")
    print(r.groupby(["type", "status"], dropna=False).size().to_string())
    print(f"accepted {int(r['ok'].sum()):,} of {len(r):,}")


def _h(values) -> np.ndarray:
    return pd.util.hash_array(np.asarray(values, dtype=object), categorize=False)


def shortfall(a):
    """Rows the crawl selected that are not in the dataset, counted by host and last recorded reason."""
    build, crawl = Path(a.build), Path(a.crawl)
    meta = Path(a.dataset) / "metadata.parquet"
    pf = pq.ParquetFile(meta)
    have, now = [], Counter()
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=["uuid", "species", "exclude"]).to_pandas()
        have.append(np.unique(_h(t["uuid"].tolist())))
        now.update(t.loc[t["species"].notna(), "species"].value_counts().to_dict())
    have = np.unique(np.concatenate(have))
    print(f"{len(have):,} uuids held", flush=True)
    blocked = json.loads((crawl / "manifest/blocked_hosts.json").read_text())
    by, per_sp, total = Counter(), Counter(), Counter()
    for part in sorted((crawl / "manifest").glob("host=*/part-*.parquet")):
        if part.name.endswith(".meta.parquet"):
            continue
        host = part.parent.name[5:]
        t = pq.read_table(part, columns=["uuid", "species"]).to_pandas()
        total["selected"] += len(t)
        q = _h(t["uuid"].tolist())  # have is sorted: searchsorted, not np.isin (which re-sorts 65 M per part)
        j = np.minimum(np.searchsorted(have, q), len(have) - 1)
        miss = have[j] != q
        total["held"] += int((~miss).sum())
        t = t[miss]
        if t.empty:
            continue
        m = part.with_suffix(".meta.parquet")
        st = pd.Series(dtype=object)
        if m.exists():
            st = pq.read_table(m, columns=["uuid", "status"]).to_pandas().drop_duplicates("uuid").set_index("uuid")["status"]
        reason = t["uuid"].map(st).fillna("no_record").str.split(":").str[0]
        by.update((host, r) for r in reason)
        per_sp.update(t["species"].tolist())
    out = pd.DataFrame([(h, r, n) for (h, r), n in by.items()], columns=["host", "reason", "n"])
    blocked_names = {re.sub(r"[^a-z0-9]+", "_", k.lower()): v for k, v in (blocked.items() if isinstance(blocked, dict) else [(b, "") for b in blocked])}
    out["blocked"] = out["host"].map(lambda h: str(blocked_names.get(h, ""))[:60])
    out.to_parquet(build / "shortfall.parquet")
    sp = pd.DataFrame({"species": list(per_sp), "missing": list(per_sp.values())})
    sp["held"] = sp["species"].map(now).fillna(0).astype(int)
    sp.to_parquet(build / "shortfall_species.parquet")
    print(json.dumps(total))
    print(out.groupby("reason")["n"].sum().sort_values(ascending=False).head(25).to_string())
    print(out.groupby("host")["n"].sum().sort_values(ascending=False).head(25).to_string())
    print(f"missing rows on blocked hosts: {int(out.loc[out['blocked'] != '', 'n'].sum()):,}; elsewhere: {int(out.loc[out['blocked'] == '', 'n'].sum()):,}")


META, IMG, OUT = None, None, None


def _decode_init(meta, img, out):
    global META, IMG, OUT
    META, IMG, OUT = meta, Path(img), Path(out)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None


def _decode_group(i: int) -> tuple[int, int, int]:
    from PIL import Image
    t = pq.ParquetFile(META).read_row_group(i, columns=["uuid", "path"]).to_pydict()
    bad = []
    for u, p in zip(t["uuid"], t["path"]):
        try:
            with open(IMG / p, "rb") as f:
                data = f.read()
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)  # 2 TB through the page cache
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                if im.format != "JPEG" or min(im.size) < 1:
                    bad.append((u, p, f"format {im.format} {im.size}"))
        except Exception as e:  # noqa: BLE001 - the error is the result
            bad.append((u, p, type(e).__name__ + ": " + str(e)[:120]))
    tmp = OUT / f".rg-{i:05d}.tmp"
    tmp.write_text("".join(f"{a}\t{b}\t{c}\n" for a, b, c in bad))
    tmp.rename(OUT / f"rg-{i:05d}.tsv")
    return i, len(t["uuid"]), len(bad)


def decode(a):
    build, meta = Path(a.build), Path(a.dataset) / "metadata.parquet"
    out = build / "decode"
    out.mkdir(parents=True, exist_ok=True)
    n = pq.ParquetFile(meta).metadata.num_row_groups
    todo = [i for i in range(n) if not (out / f"rg-{i:05d}.tsv").exists()]
    print(f"{n} row groups, {len(todo)} to do on {a.cpus} processes", flush=True)
    seen, bad, t0 = 0, 0, time.time()
    with mp.get_context("spawn").Pool(a.cpus, _decode_init, (str(meta), str(Path(a.dataset) / "images"), str(out)), maxtasksperchild=20) as pool:
        for k, (_, rows, b) in enumerate(pool.imap_unordered(_decode_group, todo), 1):
            seen, bad = seen + rows, bad + b
            if k % 25 == 0:
                print(f"  {k}/{len(todo)} groups, {seen:,} images, {bad:,} bad, {seen / (time.time() - t0):.0f}/s", flush=True)
    print(f"decode done: {seen:,} images this run, {bad:,} bad")


def _sizes(rows: list[tuple[str, str]]) -> list[tuple[str, int, int]]:
    from PIL import Image
    out = []
    for u, p in rows:
        with Image.open(TOL / "images" / p) as im:      # header only
            out.append((u, *im.size))
    return out


def sizes(a):
    """F/sizes.parquet: width, height of the rows the crawler recovered from disk (width 0 in the metadata:
    it records such a row without reopening the file; 1.64 M rows). `plan` fills them in."""
    meta = Path(a.dataset) / "metadata.parquet"
    rows = []
    for t in _load_102()._row_groups(meta, ["uuid", "path", "width"]):
        x = t.to_pandas()
        rows += list(zip(*x.loc[x["width"].fillna(0).eq(0), ["uuid", "path"]].T.to_numpy()))
    chunks = [rows[i:i + 5000] for i in range(0, len(rows), 5000)]
    print(f"{len(rows):,} rows without a size", flush=True)
    with mp.get_context("spawn").Pool(a.cpus) as pool:
        got = [r for c in pool.imap_unordered(_sizes, chunks) for r in c]
    pd.DataFrame(got, columns=["uuid", "width", "height"]).to_parquet(Path(a.build) / "sizes.parquet")
    print(f"sizes: {len(got):,} read")


# --------------------------------------------------------------------------------------------- #
# Benchmark overlap. Each benchmark lists its images with a genus; a dataset image in the same genus
# whose difference hash is within BENCH_HAMMING bits of a benchmark image is the same photograph
# (re-encoded or resized on the way, which defeats content_hash). Ids are matched too where the
# dataset holds that source (Rare Species: EOL content ids; ToL's EOL host refused us, so none today).
# A new benchmark is one entry here and a re-run of `bench`; the freeze's `exclude` then gains its rows
# under a new version, without touching the split.
BENCH_HAMMING = 6


def _rare_species() -> pd.DataFrame:
    root = Path("/data/au761367/benchmarks/rare_species/images")
    files = sorted(root.glob("*/*.jpg"))
    return pd.DataFrame({"file": [str(f) for f in files], "genus": [f.parent.name.split("_")[-2] for f in files],
                         "kingdom": [f.parent.name.split("_")[0] for f in files]})


BENCHMARKS = {"rare_species": _rare_species}


def dhash(data: bytes | str) -> int:
    from PIL import Image
    with Image.open(io.BytesIO(data) if isinstance(data, bytes) else data) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int(np.packbits(bits).view(">u8")[0])


def _hash_rows(rows: list[tuple[str, str]]) -> list[tuple[str, int]]:
    out = []
    for u, p in rows:
        try:
            with open(TOL / "images" / p, "rb") as f:
                data = f.read()
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            out.append((u, dhash(data)))
        except Exception:  # noqa: BLE001 - unreadable: no hash, decode step reports it
            pass
    return out


def bench(a):
    """F/bench/<name>.parquet: dataset uuids that are a benchmark's photographs, with distance."""
    build, meta = Path(a.build), Path(a.dataset) / "metadata.parquet"
    out = build / "bench"
    out.mkdir(parents=True, exist_ok=True)
    for name, load in BENCHMARKS.items():
        b = load()
        with mp.get_context("spawn").Pool(a.cpus) as pool:
            b["h"] = pool.map(dhash, b["file"], chunksize=64)
        want = set(zip(b["kingdom"], b["genus"]))
        rows = []
        pf = pq.ParquetFile(meta)
        for i in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(i, columns=["uuid", "path", "kingdom", "genus"]).to_pandas()
            rows.append(t[[kg in want for kg in zip(t["kingdom"], t["genus"])]])
        d = pd.concat(rows, ignore_index=True)
        print(f"{name}: {len(b):,} images in {len(want):,} genera; {len(d):,} dataset images in those genera", flush=True)
        pairs = list(zip(d["uuid"], d["path"]))
        chunks = [pairs[i:i + 2000] for i in range(0, len(pairs), 2000)]
        got, t0 = [], time.time()
        with mp.get_context("spawn").Pool(a.cpus) as pool:
            for k, r in enumerate(pool.imap_unordered(_hash_rows, chunks), 1):
                got += r
                if k % 100 == 0:
                    print(f"  {len(got):,} hashed, {len(got) / (time.time() - t0):.0f}/s", flush=True)
        h = pd.DataFrame(got, columns=["uuid", "h"]).merge(d[["uuid", "kingdom", "genus"]], on="uuid")
        hits = []
        for (k, g), x in h.groupby(["kingdom", "genus"]):
            bb = b[(b["kingdom"] == k) & (b["genus"] == g)]
            if bb.empty:
                continue
            xs = x["h"].to_numpy(np.uint64)[:, None] ^ bb["h"].to_numpy(np.uint64)[None, :]
            dist = np.bitwise_count(xs)
            j = dist.argmin(axis=1)
            m = dist[np.arange(len(x)), j]
            keep = m <= 12
            hits.append(pd.DataFrame({"uuid": x["uuid"].to_numpy()[keep], "dist": m[keep],
                                      "bench_file": bb["file"].to_numpy()[j[keep]]}))
        hits = pd.concat(hits, ignore_index=True) if hits else pd.DataFrame(columns=["uuid", "dist", "bench_file"])
        hits["benchmark"] = name
        hits.to_parquet(out / f"{name}.parquet")
        print(f"{name}: hashed {len(h):,}; matches by distance:", hits["dist"].value_counts().sort_index().to_dict())
        print(f"{name}: within {BENCH_HAMMING} bits: {int((hits['dist'] <= BENCH_HAMMING).sum()):,} dataset images, "
              f"{hits.loc[hits['dist'] <= BENCH_HAMMING, 'bench_file'].nunique():,} benchmark images")


# --------------------------------------------------------------------------------------------- #
# The dry run. Every label change is decided once per distinct rank tuple, then mapped onto rows in one
# streaming pass. Outputs are what `dev/102 apply --build B` takes.
SPLIT_KEY = "tol256-freeze-v1"                      # pandas hash_key (16 bytes): the split's seed, recorded
SPLIT = (90, 95)                                    # hash % 100 < 90 train, < 95 val, else test


def _key(df: pd.DataFrame) -> pd.Series:
    return df[TAXA].fillna("").astype(str).agg("|".join, axis=1)


def _binom(name: str) -> str:
    return " ".join(re.sub(r"\([^)]*\)", " ", name).split()[:2])


def decide(d: pd.DataFrame, names: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Old rank tuple -> new rank tuple and a row-level `bad` flag; with counts of what each rule did."""
    old = d.copy()
    d = fill_inside(d)
    st = {"tuples": len(d), "filled_inside_rows": int(((old[UPPER] == "") & (d[UPPER] != "")).any(axis=1).mul(d["n"]).sum())}
    # Junk kingdoms (a source's left-packing that the rank fix could not read): the genus's one clean path, else bad.
    d["bad"] = False
    junk = ~d["kingdom"].isin(GOOD_KINGDOMS) & (d["kingdom"] != "")
    clean = d[~junk & (d["genus"] != "") & (d[UPPER] != "").all(axis=1)]
    paths = clean.groupby("genus")[["kingdom", *UPPER]].agg(lambda s: tuple(sorted(set(s))))
    for i in d.index[junk]:
        g = d.at[i, "genus"]
        if g in paths.index and all(len(v) == 1 for v in paths.loc[g]):
            for c in ["kingdom", *UPPER]:
                d.at[i, c] = paths.at[g, c][0]
        else:
            d.at[i, "bad"] = True
    st["junk_kingdom_rows_repaired"] = int(d.loc[junk & ~d["bad"], "n"].sum())
    st["junk_kingdom_rows_excluded"] = int(d.loc[d["bad"], "n"].sum())
    # COL XR: accepted names only. Lepidoptera take the whole chain (and the accepted name for a synonym);
    # every other name only fills its empty ranks. The kingdom (and order) must already agree.
    ok = names[names["final"]].set_index("name")
    chain = ok["chain"].map(json.loads)
    sp = (d["taxon_rank"] == "species") & d["species"].isin(ok.index)
    renamed = filled = relabelled = 0
    for i in d.index[sp]:
        name = d.at[i, "species"]
        c = chain[name]
        if c.get("kingdom") != d.at[i, "kingdom"]:
            continue
        if d.at[i, "order"] == "Lepidoptera" and c.get("order") == "Lepidoptera":
            new = _binom(ok.at[name, "accepted"]) if ok.at[name, "status"] == "SYNONYM" and ok.at[name, "accepted"] else name
            before = tuple(d.loc[i, UPPER + ["genus"]])
            for r in UPPER + ["genus"]:
                if c.get(r):
                    d.at[i, r] = c[r]
            if new != name:
                d.at[i, "species"] = new
                d.at[i, "genus"] = new.split()[0] if not c.get("genus") else d.at[i, "genus"]
                renamed += d.at[i, "n"]
            elif tuple(d.loc[i, UPPER + ["genus"]]) != before:
                relabelled += d.at[i, "n"]
        else:
            gap = [r for r in UPPER if d.at[i, r] == "" and c.get(r)]
            for r in gap:
                d.at[i, r] = c[r]
            filled += d.at[i, "n"] if gap else 0
    st.update({"lepidoptera_rows_renamed": int(renamed), "lepidoptera_rows_reranked": int(relabelled),
               "rows_filled_from_colxr": int(filled)})
    st["rows_with_an_empty_rank_left"] = int(((d[UPPER] == "").any(axis=1) & (d["taxon_rank"] == "species")).mul(d["n"]).sum())
    d["old"] = _key(old)
    return d, st


def tree(d: pd.DataFrame) -> dict:
    """Names under more than one full path, after the decisions (training keys by full path, so these stay apart)."""
    out = {}
    R = ["kingdom", *UPPER, "genus", "species"]
    for j, child in enumerate(R[1:], 1):
        x = d[d[child] != ""].groupby(R[:j + 1])["n"].sum().reset_index()
        multi = x.groupby(child).filter(lambda z: len(z) > 1)
        out[child] = {"names_on_several_paths": int(multi[child].nunique()),
                      "across_kingdoms": int((multi.groupby(child)["kingdom"].nunique() > 1).sum()),
                      "rows_off_majority_path": int(multi.groupby(child)["n"].apply(lambda s: s.sum() - s.max()).sum())}
    return out


def plan(a):
    F = _load_102()
    build, out = Path(a.build), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.base) if a.base else Path(a.dataset) / "metadata.parquet"
    added_src = base.parent / "added_files.parquet" if a.base else None
    d = distinct_taxa(base)
    names = pd.read_parquet(build / "names_checked.parquet")
    d, st = decide(d, names)
    st["tree_after"] = tree(d)
    m = d.set_index("old")
    new_of = m[TAXA].astype(str).agg("|".join, axis=1).to_dict()
    bad_of = set(m.index[m["bad"]])
    # Held-out taxa: the owner's draw, by full path (drawn from complete non-Lepidoptera paths, which no rule changes).
    ho = pd.read_csv(build / "heldout_taxa.csv", dtype=str, keep_default_na=False)
    ho_keys = set(ho[["kingdom", *UPPER, "genus", "species"]].agg("|".join, axis=1) + "|species")
    bench = pd.concat([pd.read_parquet(p) for p in sorted((build / "bench").glob("*.parquet"))])
    bench_uuid = set(bench.loc[bench["dist"] <= BENCH_HAMMING, "uuid"])
    # The floor, for species a rename touches (as dev/102 colxr): counted in this dataset, by name.
    new_sp = m.assign(new=m.index.map(new_of).str.split("|").str[6])
    count = new_sp.groupby("new")["n"].sum()
    old_names = m.index.str.split("|").str[6]
    touched = set(old_names[new_sp["new"].to_numpy() != old_names]) | set(new_sp.loc[new_sp["new"].to_numpy() != old_names, "new"])
    touched.discard("")

    schema = pq.read_schema(base)
    schema = schema.append(pa.field("split", pa.string())) if "split" not in schema.names else schema
    moves, ex, sp_n = [], Counter(), Counter()
    sz = pd.read_parquet(build / "sizes.parquet").set_index("uuid") if (build / "sizes.parquet").exists() else None

    def rows(t: pa.Table) -> pd.DataFrame:
        x = t.to_pandas()
        # Rows the crawler recovered from disk: "" is not a hash (null), size from `sizes`.
        x["content_hash"] = x["content_hash"].replace("", None)
        if sz is not None:
            w = x["uuid"].map(sz["width"])
            x["width"] = x["width"].where(w.isna(), w).astype(x["width"].dtype)
            x["height"] = x["height"].where(w.isna(), x["uuid"].map(sz["height"])).astype(x["height"].dtype)
        k = _key(x)
        nk = k.map(new_of)
        assert nk.notna().all(), "a rank tuple the plan did not decide"
        v = nk.str.split("|", expand=True)
        for j, c in enumerate(TAXA):
            x[c] = v[j].replace("", None).to_numpy()
        e = x["exclude"].copy()
        free = e.isna()
        for mask, why in ((x["uuid"].isin(bench_uuid), "benchmark_overlap"), (k.isin(ho_keys), "heldout_taxa"),
                          (k.isin(bad_of), "bad_kingdom")):
            e[free & mask] = why
            free = e.isna()
        t_ = x["species"].isin(touched)
        big = x["species"].map(count).fillna(0) >= F.MIN_IMG
        e[t_ & ~big & e.isna()] = "below_floor"
        e[t_ & big & (e == "below_floor")] = None
        x["exclude"] = e
        ren = (x["species"].fillna("") != k.str.split("|").str[6]) & (x["taxon_rank"] == "species")
        newp = x.loc[ren, "species"].map(F.folder_name) + "/" + x.loc[ren, "uuid"] + ".jpg"
        moves.append(pd.DataFrame({"old": x.loc[ren, "path"].to_numpy(), "new": newp.to_numpy()}))
        x.loc[ren, "path"] = newp
        obs = x["gbif_id"].where(x["gbif_id"].notna(), x["data_source"].fillna("") + ":" + x["source_id"])
        obs = obs.where(obs.notna(), x["uuid"])
        h = pd.util.hash_array(obs.to_numpy(dtype=object), hash_key=SPLIT_KEY, categorize=False) % 100
        x["split"] = np.select([h < SPLIT[0], h < SPLIT[1]], ["train", "val"], "test")
        x["_obs"], x["_h"] = obs.to_numpy(), h
        return x

    # A species whose observations all hash into val or test would be a class with nothing to train on:
    # its lowest-hash observation goes to train instead.
    has_train, best = set(), {}
    for t in F._row_groups(base):
        x = rows(t)
        u = x[(x["taxon_rank"] == "species") & x["exclude"].isna()]
        # The class is the full path, as `train` keys it: by kingdom|species, a name on two paths
        # (Dianella, Lycoperdon) counted one path's train rows for the other.
        v = u[TAXA[:7]].fillna("").astype(str)
        sk = v["kingdom"]
        for c in TAXA[1:7]:
            sk = sk + "|" + v[c]
        has_train.update(sk[u["split"] == "train"])
        lo = u.assign(sk=sk).sort_values(["_h", "_obs"]).drop_duplicates("sk")
        for s_, h_, o_ in zip(lo["sk"], lo["_h"], lo["_obs"]):
            if s_ not in best or (h_, o_) < best[s_]:
                best[s_] = (h_, o_)
    force = {best[s_][1] for s_ in set(best) - has_train}
    st["species_moved_to_train"] = len(set(best) - has_train)

    def tables():
        for t in F._row_groups(base):
            x = rows(t)
            x.loc[x["_obs"].isin(force), "split"] = "train"
            yield pa.Table.from_pandas(x[schema.names], schema=schema, preserve_index=False)

    first = [True]

    def counted():
        for t in tables():
            if first[0]:
                x = t.select(["kingdom", "species", "exclude", "split"]).to_pandas()
                ex.update(x["exclude"].fillna("").value_counts().to_dict())
                u = x[x["exclude"].isna()]
                sp_n.update(u.groupby(["kingdom", "split"]).size().to_dict())
            yield t
        first[0] = False
    final = out / ("metadata_final.parquet" if added_src else "metadata.parquet")
    n = F._write_sorted(counted, final, schema, out / "work")
    mv = pd.concat(moves, ignore_index=True).drop_duplicates()  # `rows` runs on every pass
    if added_src:
        add = pd.read_parquet(added_src)
        remap = dict(zip(mv["old"], mv["new"]))
        add["new"] = add["new"].map(lambda p: remap.get(p, p))
        add.to_parquet(out / "added_files.parquet")
        mv = mv[~mv["old"].isin(set(pd.read_parquet(added_src)["new"]))]
    mv.to_parquet(out / "file_moves.parquet")
    st.update({"rows": n, "moves": len(mv), "exclude": {k or "usable": v for k, v in ex.items()},
               "usable_by_kingdom_split": {f"{k}|{s}": v for (k, s), v in sorted(sp_n.items())},
               "split_key": SPLIT_KEY, "split": "obs = gbif_id, else data_source:source_id, else uuid; hash % 100 < 90 train, < 95 val, else test; a species with no train observation gets its lowest-hash one in train"})
    (out / "plan_summary.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


# --------------------------------------------------------------------------------------------- #
# Training parquet in mini_trainer's format (W3), from the frozen metadata: usable species rows only.
LEVELS = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]   # finest first, as cls2idx "0".."6"
SET_OF = {"train": "2", "val": "1", "test": "0"}                                  # mini_trainer: "0" test, "1" val
MISSING = {"", "nan", "NaN", "None", "none", "null", "NULL", "NA", "N/A", "n/a", "<NA>"}


def keys_of(t: tuple) -> tuple:
    """(kingdom, phylum, class, order, family, genus, species) -> seven full-path keys, finest first.

    An empty rank becomes `none:<rank>:<nearest named rank below>`, a parent of its own (the convention
    agreed with mini_trainer); whitespace becomes '_'; a key never contains '/' and is never a pandas
    missing string."""
    top = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    v = [re.sub(r"\s+", "_", x.strip()) for x in t]
    for i in range(len(v)):
        if v[i] in MISSING:
            below = next((w for w in v[i + 1:] if w not in MISSING), "")
            v[i] = f"none:{top[i]}:{below}"
    assert not any("/" in x for x in v), t
    path = ["|".join(v[: i + 1]) for i in range(len(v))]
    return tuple(reversed(path))


def train(a):
    meta = Path(a.dataset) / "metadata.parquet"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    cols = ["uuid", "path", "taxon_rank", "exclude", "split", *R]
    pf = pq.ParquetFile(meta)
    seen: dict = {}
    sets = Counter()
    # image_path (relative to <dataset>/images): lepinet keeps a given image_path and otherwise builds
    # <speciesKey>/<filename>, which does not exist for full-path keys (0 of 2,636 smoke rows found).
    schema = pa.schema([("filename", pa.string()), ("image_path", pa.string()), ("uuid", pa.string()), ("set", pa.string()),
                        *[(f"{lv}Key", pa.string()) for lv in LEVELS]])
    w = pq.ParquetWriter(out / "train.parquet.tmp", schema, compression="zstd")
    for i in range(pf.metadata.num_row_groups):
        x = pf.read_row_group(i, columns=cols).to_pandas()
        x = x[(x["taxon_rank"] == "species") & x["exclude"].isna()]
        tup = list(zip(*[x[c].fillna("").astype(str) for c in R]))
        k = [seen.get(t) or seen.setdefault(t, keys_of(t)) for t in tup]
        kk = list(zip(*k)) if k else [[] for _ in LEVELS]
        s = x["split"].map(SET_OF)
        assert s.notna().all()
        sets.update(zip(s, (kk[0] if k else [])))
        w.write_table(pa.table({"filename": x["path"].tolist(), "image_path": x["path"].tolist(), "uuid": x["uuid"].tolist(), "set": s.tolist(),
                                **{f"{lv}Key": list(kk[j]) for j, lv in enumerate(LEVELS)}}, schema=schema))
    w.close()
    os.replace(out / "train.parquet.tmp", out / "train.parquet")
    # Class spec: every class used, each child exactly one parent (true by construction of full paths; asserted).
    labels = {}
    for kk in seen.values():
        labels[kk[0]] = list(kk)
    cls2idx = {}
    for j in range(len(LEVELS)):
        names_j = sorted({v[j] for v in labels.values()})
        cls2idx[str(j)] = {n: i for i, n in enumerate(names_j)}
        if j:
            parent = {}
            for v in labels.values():
                assert parent.setdefault(v[j - 1], v[j]) == v[j], ("two parents", v[j - 1])
    spec = {"cls2idx": cls2idx, "labels": labels, "num_classes": [len(cls2idx[str(j)]) for j in range(len(LEVELS))]}
    (out / "class_spec.json").write_text(json.dumps(spec))
    per = pd.Series(sets).rename_axis(["set", "species"]).reset_index(name="n")
    no_train = set(per["species"]) - set(per.loc[per["set"] == "2", "species"])
    st = {"rows": int(per["n"].sum()), "by_set": per.groupby("set")["n"].sum().to_dict(),
          "num_classes": dict(zip(LEVELS, spec["num_classes"])), "species_without_train_rows": len(no_train),
          "none_keys": {lv: sum(1 for v in labels.values() if f"none:{lv}:" in v[j]) for j, lv in enumerate(LEVELS)}}
    (out / "train_summary.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


# --------------------------------------------------------------------------------------------- #
# Shards in mini_trainer's format v1 (ShardWriter at the pinned commit), shuffled across species by a
# seeded hash of the uuid: rows are bucketed by the hash's top bits, each bucket sorted, so the stream is
# one global permutation without holding it. The pilot keeps the PILOT lowest-hash rows per species and
# set. The build waits while the disk is short, so `dev/104` can upload and free shards as it goes.
PILOT = {"2": 50, "1": 5, "0": 5}
SPLIT_NAME = {"2": "train", "1": "val", "0": "test"}
BUCKETS = 64


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(2**24):
            h.update(chunk)
    return h.hexdigest()


def _read_image(p: str) -> bytes:
    with open(TOL / "images" / p, "rb") as f:
        data = f.read()
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return data


def shards(a):
    import shutil
    import sys
    sys.path.insert(0, a.mt)
    from mini_trainer.data.shards import ShardWriter, write_manifest
    src, out = Path(a.train_dir), Path(a.out)
    tp = src / "train.parquet"
    spec = json.loads((src / "class_spec.json").read_text())
    labels = spec["labels"]
    sp_idx = spec["cls2idx"]["0"]
    sp_names = sorted(sp_idx, key=sp_idx.get)
    key16 = f"shards{a.seed:010d}"
    # The pilot picks rows by a second, independent hash. Picking by the order hash kept the lowest-order
    # rows of every large species, so the stream ran from head species to tail species (first bucket
    # 131 k species, last 21 k) and each shard held a different mix (2026-11-12).
    pick16 = f"pilot{a.seed:011d}"
    pf = pq.ParquetFile(tp)
    r, q, sp, st = [], [], [], []
    for i in range(pf.metadata.num_row_groups):
        x = pf.read_row_group(i, columns=["uuid", "set", "speciesKey"]).to_pandas()
        u = x["uuid"].to_numpy(dtype=object)
        r.append(pd.util.hash_array(u, hash_key=key16, categorize=False))
        if a.pilot:
            q.append(pd.util.hash_array(u, hash_key=pick16, categorize=False))
        sp.append(x["speciesKey"].map(sp_idx).to_numpy(np.int32))
        st.append(x["set"].astype(np.int8).to_numpy())
    r, sp, st = np.concatenate(r), np.concatenate(sp), np.concatenate(st)
    keep = np.ones(len(r), bool)
    if a.pilot:
        q = np.concatenate(q)
        o = np.lexsort((q, sp, st))
        del q
        g = st[o].astype(np.int64) * len(sp_idx) + sp[o]
        start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
        rank = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
        cap = np.vectorize(lambda s: PILOT[str(s)])(st[o])
        keep[o] = rank < cap
        del o, g, rank, cap
    print(f"{int(keep.sum()):,} of {len(r):,} rows; by set {dict(zip(*np.unique(st[keep], return_counts=True)))}", flush=True)
    work = out.parent / f".{out.name}.buckets"
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "done").exists():
        writers: dict = {}
        off = 0
        for i in range(pf.metadata.num_row_groups):
            x = pf.read_row_group(i, columns=["filename", "uuid"])
            n = x.num_rows
            sl = slice(off, off + n)
            off += n
            k = keep[sl]
            t = pa.table({"r": r[sl][k], "filename": x["filename"].filter(pa.array(k)), "uuid": x["uuid"].filter(pa.array(k)),
                          "sp": sp[sl][k], "set": st[sl][k]})
            b = (t["r"].to_numpy() >> np.uint64(64 - 6)).astype(np.int64)
            for s in np.unique(t["set"].to_numpy()):
                for bb in np.unique(b):
                    m = (t["set"].to_numpy() == s) & (b == bb)
                    if m.any():
                        name = f"{SPLIT_NAME[str(s)]}-{bb:02d}"
                        if name not in writers:
                            writers[name] = pq.ParquetWriter(work / f"{name}.parquet", t.schema)
                        writers[name].write_table(t.filter(pa.array(m)))
        for w in writers.values():
            w.close()
        (work / "done").write_text("")
    del r, sp, st, keep
    LABELS = list(LEVELS)
    splits = {}
    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.readers) as pool:
        for s in ("train", "val", "test"):
            files = sorted(work.glob(f"{s}-*.parquet"))
            if not files:
                continue
            w = ShardWriter(out, s, LABELS, max_bytes=a.shard_bytes)
            for f in files:
                t = pq.read_table(f).sort_by("r")
                fn, uu, spp = t["filename"].to_pylist(), t["uuid"].to_pylist(), t["sp"].to_numpy()
                for j in range(0, len(fn), 4096):
                    while shutil.disk_usage(out).free < a.min_free_gb * 2**30:
                        print(f"  waiting: {shutil.disk_usage(out).free / 2**30:.0f} GB free", flush=True)
                        time.sleep(120)
                    for u, data, q in zip(uu[j:j + 4096], pool.map(_read_image, fn[j:j + 4096]), spp[j:j + 4096]):
                        w.add(u, data, dict(zip(LABELS, labels[sp_names[q]])))
                    done += len(fn[j:j + 4096])
                print(f"  {s} {f.name}: {done:,} samples, {done / (time.time() - t0):.0f}/s", flush=True)
            splits[s] = w.close()
    source = {"freeze": a.freeze, "metadata_sha256": _sha256(Path(a.dataset) / "metadata.parquet"),
              "train_parquet_sha256": _sha256(tp), "shuffle": f"uuid hash, hash_key {key16}",
              "subset": f"pilot: per species and set at most {PILOT} (set 2 train, 1 val, 0 test), the lowest by uuid hash, hash_key {pick16}" if a.pilot else "full",
              "build": "python dev/103_tol_freeze.py shards " + " ".join(sys.argv[2:])}
    write_manifest(out, splits, LABELS, source)
    shutil.rmtree(work)
    print(json.dumps({k: v["n_samples"] for k, v in splits.items()}))


def _load_102():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("dev102", Path(__file__).with_name("102_tol_fix_ranks.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dev102"] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["names", "shortfall", "decode", "sizes", "bench", "plan", "train", "shards"])
    p.add_argument("--base", default="", help="plan: start from this metadata (dev/102 add's metadata_final) instead of the dataset's")
    p.add_argument("--out", default="", help="plan: output dir for dev/102 apply (default <build>/plan)")
    p.add_argument("--build", required=True)
    p.add_argument("--train-dir", default="", help="shards: the dir with train.parquet and class_spec.json")
    p.add_argument("--mt", default="/data/au761367/tol256s/mt_6f6a3cf", help="shards: mini_trainer pinned at 6f6a3cf (git worktree)")
    p.add_argument("--pilot", action="store_true", help="shards: the pilot subset")
    p.add_argument("--seed", type=int, default=20261113, help="shards: shuffle seed")
    p.add_argument("--freeze", default="", help="shards: the freeze's version name, recorded in the manifest")
    p.add_argument("--readers", type=int, default=16, help="shards: image read threads")
    p.add_argument("--shard-bytes", type=int, default=2**30)
    p.add_argument("--min-free-gb", type=float, default=200, help="shards: pause while the disk has less")
    p.add_argument("--dataset", default=str(TOL))
    p.add_argument("--crawl", default="/data/au761367/tol256s", help="shortfall: the crawl's manifests")
    p.add_argument("--cpus", type=int, default=4, help="decode: processes (pin them with taskset)")
    a = p.parse_args()
    a.out = a.out or str(Path(a.build) / "plan")
    {"names": names, "shortfall": shortfall, "decode": decode, "sizes": sizes, "bench": bench, "plan": plan, "train": train, "shards": shards}[a.step](a)
