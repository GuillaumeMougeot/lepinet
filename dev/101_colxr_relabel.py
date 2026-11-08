#!/usr/bin/env python3
"""Move the Lepidoptera labels from the legacy GBIF Backbone to Catalogue of Life XR (PLAN row TAX).

Why: images were labelled with the legacy Backbone's accepted speciesKey, so its synonymy errors became
class definitions. journal/2026-11-03-the-training-labels-inherit-gbif-backbone-errors.md

    prepare   freeze the mapping into dev/colxr/ (needs the investigation's files and the GBIF API)
    apply     THE ONE COMMAND: writes a new dataset version, offline, from the frozen tables

        .venv/bin/python dev/101_colxr_relabel.py apply --upload

apply never changes an existing file. It writes data/global/colxr/: the relabelled parquet (every row
kept, test fold included), the trap labels, the regional lists, report.md and manifest.json. The images
stay where they are: `legacySpeciesKey` names their folder, and ucloud/stage.py copies them to
<new speciesKey>/<filename> on the node. To change a decision before applying, add a row to
dev/colxr/overrides.csv.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "dev/colxr"
NAME_MAP = TABLES / "name_map.csv"
TAXONOMY = TABLES / "taxonomy.csv"
OVERRIDES = TABLES / "overrides.csv"
SNAPSHOT = TABLES / "snapshot.json"

PARQUET_NAME = "0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"
PARQUET = ROOT / "data/global" / PARQUET_NAME
IMAGES = ROOT / "data/global/images"
TRAP = ROOT / "data/ucloud_mirror/data/flemming/flemming_names_labels.parquet"
BUNDLE = Path.home() / "codes/mini_trainer/deployment/mambo_deploy/default_bundle.json"
OUT = ROOT / "data/global/colxr"
INVESTIGATION = Path.home() / "codes/mini_trainer/.agents/local/nemo_species_colxr"
REMOTE = "/12383016/global_lepi/colxr"  # the datasets drive; jobs see it as /work/global_lepi/colxr

COL_XR = "7ddf754f-d193-4cc9-b351-99906754a03b"
RANKS = ("species", "genus", "family", "order", "class", "phylum", "kingdom")
KEYS = tuple(f"{r}Key" for r in RANKS)
FLOOR = 50  # the training floor, reported only: nothing is dropped here


# ---------------------------------------------------------------- prepare
def fetch(key: str) -> dict:
    query = urllib.parse.urlencode({"checklistKey": COL_XR, "usageKey": key})
    request = urllib.request.Request(f"https://api.gbif.org/v2/species/match?{query}",
                                     headers={"User-Agent": "lepinet-taxonomy-snapshot"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
            return {"key": key, "classification": [(c["rank"], c["key"], c["name"]) for c in data.get("classification") or []]}
        except Exception:  # noqa: BLE001 - retried with backoff; a final failure stops prepare
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def crosswalk(legacy_key: str) -> tuple[str, str]:
    """GBIF's legacy key -> accepted COL XR species, or two empty strings."""
    query = urllib.parse.urlencode({"checklistKey": COL_XR, "scientificNameID": f"gbif:{legacy_key}"})
    request = urllib.request.Request(f"https://api.gbif.org/v2/species/match?{query}", headers={"User-Agent": "lepinet-taxonomy-snapshot"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    species = next((c for c in data.get("classification") or [] if c.get("rank") == "SPECIES"), None)
    return (species["key"], species["name"]) if species else ("", "")


def lookup(keys: list[str], cache_path: Path, rate: float = 6.0) -> dict[str, dict]:
    cache = {}
    if cache_path.exists():
        cache = {r["key"]: r for r in map(json.loads, cache_path.read_text().splitlines())}
    todo = [k for k in keys if k not in cache]
    print(f"taxonomy: {len(keys):,} keys, {len(todo):,} to fetch", flush=True)

    def paced(item):
        i, key = item
        time.sleep(max(0.0, started + i / rate - time.monotonic()))
        return fetch(key)

    started = time.monotonic()
    with cache_path.open("a") as sink, ThreadPoolExecutor(8) as pool:
        for i, record in enumerate(pool.map(paced, enumerate(todo)), 1):
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            cache[record["key"]] = record
            if i % 1000 == 0:
                print(f"  {i:,}/{len(todo):,}", flush=True)
    return cache


def fill_parents(tax: list[dict], biggest: dict, legacy: dict) -> None:
    """Give every species a full chain when COL XR does not return one for its key (about 1 %).

    In order: a genus of the same name among the resolved species; the COL XR genus and family most
    species of the legacy genus and family went to; the legacy key itself. The dataset is all Lepidoptera,
    which fixes everything above family. `parents_from` records what was used.
    """
    genus_chain, by_legacy_genus, by_legacy_family = {}, collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    above = None
    for r in tax:
        if r["parents_from"] != "col_xr":
            continue
        genus_chain.setdefault(r["genus_name"], r)
        g, f = legacy[biggest[r["species_key"]]]
        by_legacy_genus[g][r["genus_key"]] += 1
        by_legacy_family[f][r["family_key"]] += 1
        above = above or r
    resolved = {r["genus_key"]: r for r in tax if r["parents_from"] == "col_xr"}
    family_names = {r["family_key"]: r["family_name"] for r in tax if r["parents_from"] == "col_xr"}
    for r in tax:
        if r["parents_from"] == "col_xr":
            continue
        g, f = legacy[biggest[r["species_key"]]]
        genus_name = r["species_name"].split()[0]
        if genus_name in genus_chain:
            source = genus_chain[genus_name]
            r["genus_key"], r["genus_name"], r["family_key"], r["family_name"] = source["genus_key"], genus_name, source["family_key"], source["family_name"]
            r["parents_from"] = "genus_name"
        elif by_legacy_genus[g]:
            genus = resolved[by_legacy_genus[g].most_common(1)[0][0]]
            r["genus_key"], r["genus_name"], r["family_key"], r["family_name"] = genus["genus_key"], genus["genus_name"], genus["family_key"], genus["family_name"]
            r["parents_from"] = "legacy_genus"
        else:
            r["genus_key"], r["genus_name"] = f"gbif:{g}", genus_name
            if by_legacy_family[f]:
                r["family_key"] = by_legacy_family[f].most_common(1)[0][0]
                r["family_name"], r["parents_from"] = family_names[r["family_key"]], "legacy_family"
            else:
                r["family_key"], r["family_name"], r["parents_from"] = f"gbif:{f}", "", "legacy_keys"
        for rank in ("order", "class", "phylum", "kingdom"):
            r[f"{rank}_key"], r[f"{rank}_name"] = above[f"{rank}_key"], above[f"{rank}_name"]


def cmd_prepare(a) -> None:
    sys.path.insert(0, str(INVESTIGATION))
    from nemo_gbif_to_colxr import same_species  # the investigation's own rule, not a copy of it

    classes = {r["gbif_species_id"]: r for r in csv.DictReader((INVESTIGATION / "nemo_species_gbif_colxr.csv").open())}
    in_lepi = {}
    for line in (INVESTIGATION / "training_names_cache.jsonl").read_text().splitlines():
        r = json.loads(line)
        in_lepi[r["taxon_key"]] = r.get("in_lepidoptera")

    def class_target(c):
        if not c["col_xr_id"] or c["manual_confidence"] == "low":
            return f"gbif:{c['gbif_species_id']}", c["scientific_name"]
        return c["col_xr_id"], c["col_xr_name"]

    rows = []
    for t in csv.DictReader((INVESTIGATION / "nemo_training_names.csv").open()):
        c = classes[t["gbif_species_id"]]
        target, name = class_target(c)
        own = {c["scientific_name"], c["col_xr_name"]} - {""}
        if not c["col_xr_id"]:
            reason = "kept_unresolved"
        elif c["manual_confidence"] == "low":
            reason = "kept_low_confidence"
        elif not t["col_xr_id"] or t["col_xr_id"] == c["col_xr_id"]:
            reason = "class"
        elif same_species({t["col_xr_name"]}, own):
            reason = "class_duplicate_record"
        elif t["taxon_rank"] == "SPECIES" and same_species({t["training_name"]}, {c["scientific_name"]}):
            reason = "class_homonym"
        elif in_lepi.get(t["taxon_key"]) is False:
            reason = "class_name_outside_lepidoptera"
        else:
            reason, target, name = "other_species", t["col_xr_id"], t["col_xr_name"]
        rows.append({"legacy_species_key": t["gbif_species_id"], "taxon_key": t["taxon_key"], "training_name": t["training_name"],
                     "n_images": t["n_images"], "new_species_key": target, "new_species_name": name, "reason": reason})

    # Trap species the training table has no class for: the labels still need a key (scored as unknown).
    trap = pq.read_table(TRAP, columns=["speciesKey", "genusKey", "familyKey", "image_path"]).to_pydict()
    trap_only = {}
    for sp, g, f, path in zip(trap["speciesKey"], trap["genusKey"], trap["familyKey"], trap["image_path"], strict=True):
        if str(sp) not in classes:
            trap_only[str(sp)] = (str(g), str(f), path.split("/")[0])
    for sp, (_, _, name) in sorted(trap_only.items()):
        match = crosswalk(sp)
        rows.append({"legacy_species_key": sp, "taxon_key": sp, "training_name": name, "n_images": "0",
                     "new_species_key": match[0] or f"gbif:{sp}", "new_species_name": match[1] or name, "reason": "trap_only"})

    TABLES.mkdir(exist_ok=True)
    with NAME_MAP.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (int(r["legacy_species_key"]), int(r["taxon_key"]))))
    if not OVERRIDES.exists():
        OVERRIDES.write_text("legacy_species_key,taxon_key,new_species_key,new_species_name,why\n")

    # Ancestors. A kept class sits under what COL XR offers: its decided species, or the higher taxon it matched.
    probes = {r["new_species_key"]: r["new_species_key"] for r in rows if not r["new_species_key"].startswith("gbif:")}
    for c in classes.values():
        kept = f"gbif:{c['gbif_species_id']}"
        if not c["col_xr_id"] and c["col_xr_matched_id"]:
            probes[kept] = c["col_xr_matched_id"]
        elif c["manual_confidence"] == "low":
            probes[kept] = c["col_xr_id"]
    cache = lookup(sorted(set(probes.values())), ROOT / "data/global/colxr_taxonomy_cache.jsonl")

    legacy = {}
    pf = pq.ParquetFile(PARQUET)
    for g in range(pf.num_row_groups):
        d = pf.read_row_group(g, columns=["speciesKey", "genusKey", "familyKey"]).to_pydict()
        legacy.update(zip(d["speciesKey"], zip(d["genusKey"], d["familyKey"], strict=True), strict=True))

    legacy.update({sp: v[:2] for sp, v in trap_only.items()})
    names = {r["new_species_key"]: r["new_species_name"] for r in rows}
    biggest = {}  # the legacy class giving a new species most of its images
    for r in sorted(rows, key=lambda r: int(r["n_images"])):
        biggest[r["new_species_key"]] = r["legacy_species_key"]

    tax = []
    for key, name in sorted(names.items()):
        chain = {rank.lower(): (k, n) for rank, k, n in cache.get(probes.get(key, ""), {}).get("classification", [])}
        if chain and chain.get("order", ("", ""))[1] != "Lepidoptera":
            chain = {}  # a homonym in another order (Capys alpheus, under Hemiptera)
        row = {"species_key": key, "species_name": name, "parents_from": "col_xr" if "genus" in chain else ""}
        for rank in RANKS[1:]:
            row[f"{rank}_key"], row[f"{rank}_name"] = chain.get(rank, ("", ""))
        tax.append(row)
    fill_parents(tax, biggest, legacy)

    with TAXONOMY.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tax[0].keys())
        w.writeheader()
        w.writerows(tax)
    SNAPSHOT.write_text(json.dumps({"checklist": COL_XR, "frozen_on": time.strftime("%Y-%m-%d"),
                                    "n_names": len(rows), "n_species": len(tax)}, indent=2) + "\n")
    print(f"{NAME_MAP}: {len(rows):,} names; {TAXONOMY}: {len(tax):,} species")


# ---------------------------------------------------------------- apply
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_tables():
    names = {(r["legacy_species_key"], r["taxon_key"]): r for r in csv.DictReader(NAME_MAP.open())}
    tax = {r["species_key"]: r for r in csv.DictReader(TAXONOMY.open())}
    for o in csv.DictReader(OVERRIDES.open()):
        hits = [k for k in names if k[0] == o["legacy_species_key"] and o["taxon_key"] in ("*", k[1])]
        if not hits or o["new_species_key"] not in tax:
            sys.exit(f"bad override (no such name, or species missing from taxonomy.csv): {o}")
        for k in hits:
            names[k].update(new_species_key=o["new_species_key"], new_species_name=o["new_species_name"], reason="override")
    return names, tax


def check_tree(tax: dict) -> list[str]:
    """Each key must have one parent, per rank. Returns the violations."""
    bad = []
    for child, parent in zip(("genus", "family", "order", "class", "phylum"), ("family", "order", "class", "phylum", "kingdom"), strict=True):
        seen = collections.defaultdict(set)
        for r in tax.values():
            seen[r[f"{child}_key"]].add(r[f"{parent}_key"])
        bad += [f"{child} {k}: {sorted(v)}" for k, v in seen.items() if len(v) > 1]
    return bad


# Strings pandas reads as missing: a key equal to one of them would vanish from predictions.csv (dev/093).
PANDAS_MISSING = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA",
                  "NULL", "NaN", "None", "n/a", "nan", "null"}


def check_keys(tax: dict) -> list[str]:
    """Keys become folder names and CSV fields: no slash or whitespace, nothing pandas reads as missing."""
    keys = {r[f"{rank}_key"] for r in tax.values() for rank in RANKS}
    return sorted(k for k in keys if k in PANDAS_MISSING or "/" in k or any(c.isspace() for c in k))


def relabel_table(table: pa.Table, targets: list[str], tax: dict) -> pa.Table:
    legacy = pa.array([str(v) for v in table["speciesKey"].to_pylist()], pa.string())
    columns = {"speciesKey": pa.array(targets, pa.string()), "legacySpeciesKey": legacy,
               "speciesName": pa.array([tax[t]["species_name"] for t in targets], pa.string())}
    for rank in RANKS[1:]:
        columns[f"{rank}Key"] = pa.array([tax[t][f"{rank}_key"] for t in targets], pa.string())
    for name in table.schema.names:
        if name not in columns:
            columns[name] = table[name]
    return pa.table(columns)


def cmd_apply(a) -> None:
    names, tax = load_tables()
    bad = check_tree(tax)
    if bad:
        sys.exit(f"taxonomy is not a tree, {len(bad)} violations; first: {bad[:5]}")
    bad = check_keys(tax)
    if bad:
        sys.exit(f"{len(bad)} unusable keys; first: {bad[:5]}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # The main parquet: one row group at a time, written beside its final name and renamed at the end.
    target_path = out / PARQUET_NAME.replace(".parquet", ".colxr.parquet")
    partial = target_path.with_suffix(".partial")
    pf = pq.ParquetFile(a.parquet)
    before, after = collections.Counter(), collections.Counter()
    train_count = collections.Counter()
    reasons = collections.Counter()
    moved = collections.Counter()
    sources = collections.defaultdict(collections.Counter)
    samples = []
    writer = None
    rng = random.Random(0)
    for g in range(pf.num_row_groups):
        table = pf.read_row_group(g)
        d = {c: table[c].to_pylist() for c in ("speciesKey", "taxonKey", "set", "filename")}
        targets = []
        for sp, tk, s, fn in zip(d["speciesKey"], d["taxonKey"], d["set"], d["filename"], strict=True):
            r = names.get((str(sp), str(tk)))
            if r is None:
                sys.exit(f"no mapping for speciesKey {sp}, taxonKey {tk}: the parquet is not the one the tables were built on")
            t = r["new_species_key"]
            targets.append(t)
            before[s] += 1
            reasons[r["reason"]] += 1
            sources[t][str(sp)] += 1
            if r["reason"] == "other_species":
                moved[str(sp)] += 1
            if s != "0":
                train_count[t] += 1
            if rng.random() < 2e-4:
                samples.append(os.path.join(str(sp), fn))
        new = relabel_table(table, targets, tax)
        if writer is None:
            writer = pq.ParquetWriter(partial, new.schema, compression="zstd")
        writer.write_table(new)
        for s in d["set"]:
            after[s] += 1
    writer.close()
    if before != after:
        sys.exit("row counts per set changed; nothing published")
    os.replace(partial, target_path)

    # Trap labels: expert names mapped through their class.
    primary = {}
    for (legacy, _), r in names.items():
        if r["reason"] != "other_species":
            primary[legacy] = r["new_species_key"]
    trap = pq.read_table(a.trap)
    trap_targets = []
    for sp in trap["speciesKey"].to_pylist():
        t = primary.get(str(sp))
        if t is None:
            sys.exit(f"trap species {sp} has no class in the mapping")
        trap_targets.append(t)
    trap_out = relabel_table(trap, trap_targets, tax)
    pq.write_table(trap_out, out / "flemming_names_labels.colxr.parquet", compression="zstd")

    # Regional lists: a new species is in a list if any legacy class that contributed images to it is.
    region_dir = out / "regions"
    region_dir.mkdir(exist_ok=True)
    metadata = json.loads(Path(a.bundle).read_text())["metadata"]
    region_sizes = {}
    for key in sorted(metadata):
        if key.startswith("regions/") and key.endswith(".classes"):
            region = key.removeprefix("regions/").removesuffix(".classes")
            legacy = set(metadata[key].split())
            new = sorted(t for t, src in sources.items() if legacy & set(src))
            (region_dir / f"{region}.txt").write_text("\n".join(new) + "\n")
            region_sizes[region] = (len(legacy), len(new))

    # Report and manifest.
    merged = {t: dict(src) for t, src in sources.items() if len(src) > 1}
    split = collections.Counter(key for src in sources.values() for key in src)
    below = sum(1 for t in sources if train_count[t] < FLOOR)
    lines = ["# Relabelling to Catalogue of Life XR", "",
             f"Rows: {sum(before.values()):,} (per set unchanged). Species: {len({key for src in sources.values() for key in src}):,} legacy classes -> {len(sources):,}.",
             f"Rows whose species changed for another species (mixed classes): {reasons['other_species']:,}, from {len(moved):,} classes.",
             f"New species fed by several legacy classes (merged): {len(merged):,}. Legacy classes split over several new species: {sum(1 for v in split.values() if v > 1):,}.",
             f"New species under the {FLOOR}-image training floor (nothing dropped here; training decides): {below:,}.",
             f"Trap crops: {trap_out.num_rows:,}; species {trap['speciesKey'].to_pandas().nunique()} -> {len(set(trap_targets))}.", "",
             "## Rows per decision", ""] + [f"- {k}: {v:,}" for k, v in reasons.most_common()] + ["", "## Regional lists (legacy -> new)", ""] + [
             f"- {k}: {v[0]:,} -> {v[1]:,}" for k, v in region_sizes.items()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    with (out / "merged_species.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["new_species_key", "new_species_name", "legacy_classes_and_rows"])
        for t, src in sorted(merged.items()):
            w.writerow([t, tax[t]["species_name"], ";".join(f"{k}:{v}" for k, v in src.items())])
    inputs = [Path(a.parquet), Path(a.trap), Path(a.bundle), NAME_MAP, TAXONOMY, OVERRIDES, SNAPSHOT]
    (out / "manifest.json").write_text(json.dumps({"snapshot": json.loads(SNAPSHOT.read_text()), "inputs": {str(p): sha256(p) for p in inputs}}, indent=2) + "\n")

    # The images: the legacy folders must still be what the parquet says.
    missing = [p for p in samples if not (Path(a.images) / p).is_file()]
    print("\n".join(lines))
    print(f"\nimage check: {len(samples) - len(missing)}/{len(samples)} sampled files found in legacy folders")
    if missing:
        sys.exit(f"missing, first: {missing[:3]}")
    print(f"written: {out}")
    if a.upload:
        subprocess.run(["ucloud", "files", "upload", str(out), a.remote, "--no-overwrite"], check=True)
        print(f"uploaded to {a.remote}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    apply = sub.add_parser("apply")
    apply.add_argument("--parquet", default=str(PARQUET))
    apply.add_argument("--images", default=str(IMAGES))
    apply.add_argument("--trap", default=str(TRAP))
    apply.add_argument("--bundle", default=str(BUNDLE))
    apply.add_argument("--out", default=str(OUT))
    apply.add_argument("--upload", action="store_true", help="also upload the new folder to UCloud (new path, nothing replaced)")
    apply.add_argument("--remote", default=REMOTE)
    apply.set_defaults(func=cmd_apply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
