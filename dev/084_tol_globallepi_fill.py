"""Complete the TreeOfLife Lepidoptera subset from our own GBIF Lepidoptera download.

Our `global_lepi` corpus (6.3 M images, downloaded earlier with gbifxdl, stored at ~512 px under
`/12383016/global_lepi/images/<speciesKey>/<url_hash>.jpg`) overlaps TreeOfLife-200M heavily: 65 % of
its images are in ToL by GBIF occurrence id (`journal/2026-08-26-bioclip2-has-seen-two-thirds-of-
our-test-fold.md`). That overlap is useful in two ways:

* **fill** -- a ToL Lepidoptera image selected under the 2,000/species cap but sitting on a server
  that refuses us or is dead, and not covered by a substitute, may be the *same photograph* we
  already hold. Match on image URL (ToL `source_url` = our `identifier`), else on GBIF occurrence
  id (ToL `source_id` = our `gbifID`) when the occurrence has exactly one image on each side.
  gbifxdl fetched these before several servers started refusing crawlers, so this recovers images
  no crawl can now reach.
* **extend** -- species still under the cap gain images from our corpus that ToL does not have at
  all; species that ToL alone left under the 50-image floor may clear it once both are combined.

The selection replay is exact for Lepidoptera alone: the cap is applied per species in catalog
order, so a species' selection does not depend on any other species' rows. Substitution is
replayed the same way, against the same list of unavailable hosts.

Outputs (to --out), in the crawler's manifest format so `dev/082 fetch` does the copying, quality
control, resizing to 256 px, metadata and resume exactly as for downloads -- the url is a
`file:///work/global_lepi/images/...` path, read from the mounted drive instead of over HTTP:

    host=file/part-gNNNNN.parquet     fills and extensions
    fill_report.json                  what was found, and what it does to Lepidoptera coverage

    python dev/084_tol_globallepi_fill.py \\
        --global-lepi data/global/<...>_quality_filtered.parquet \\
        --species-counts <manifest>/species_counts.json --blocked <manifest>/blocked_hosts.json \\
        --out <dir>

Runs on a workstation: it streams the 233 M-row catalog once and holds the 6.3 M-row global_lepi
index in memory (a few GB). No UCloud core-hours.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_spec = importlib.util.spec_from_file_location("tol", Path(__file__).with_name("082_tol_crawler.py"))
tol = importlib.util.module_from_spec(_spec); sys.modules["tol"] = tol; _spec.loader.exec_module(tol)

COLS = ["uuid", "source_url", "source_id", "data_source", "genus", "species", "order", "family"]
LOCAL_ROOT = "/work/global_lepi/images"


def norm_url(u: str) -> str:
    """Scheme-insensitive URL key: the same photograph is often listed as http on one side and
    https on the other."""
    u = (u or "").strip()
    return u.split("://", 1)[1] if "://" in u else u


def main(a):
    tol.pin_thread_pools(tol.effective_cpus())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # ---- our corpus ----------------------------------------------------------------------------
    g = pq.read_table(a.global_lepi, columns=["gbifID", "identifier", "genus", "specificEpithet",
                                              "speciesKey", "filename"]).to_pydict()
    n = len(g["gbifID"])
    g_sp = [f"{(ge or '').strip()} {(ep or '').strip()}".strip()
            for ge, ep in zip(g["genus"], g["specificEpithet"])]
    g_path = [f"{LOCAL_ROOT}/{k}/{fn}" for k, fn in zip(g["speciesKey"], g["filename"])]
    url_idx = {norm_url(u): i for i, u in enumerate(g["identifier"]) if u}
    gid_rows: dict[int, list[int]] = defaultdict(list)
    for i, x in enumerate(g["gbifID"]):
        if x is not None:
            gid_rows[int(x)].append(i)
    matched = np.zeros(n, dtype=bool)                 # our image is also somewhere in ToL
    print(f"global_lepi: {n:,} images, {len(set(g_sp)):,} species keys, {len(url_idx):,} urls")

    counts = json.loads(Path(a.species_counts).read_text())
    keep = {k for k, v in counts.items() if v >= a.min_img}
    unavailable = set(json.loads(Path(a.blocked).read_text()))
    print(f"{len(unavailable)} unavailable hosts")

    emitted, sel_unav, pool_ok, subs = Counter(), Counter(), Counter(), Counter()
    tol_occ_images: Counter = Counter()                # ToL images per GBIF occurrence (Lepidoptera)
    lost_rows: dict[str, list] = defaultdict(list)    # species -> selected rows on unavailable hosts
    tol_lepi_species: set[str] = set()
    t0 = time.monotonic()
    nrg = tol.n_row_groups(tol.CATALOG)
    if a.limit_rg:
        nrg = min(nrg, a.limit_rg)
    for i, tb in tol.iter_row_groups(tol.CATALOG, COLS, 0, nrg, a.readers, a.prefetch):
        d = tb.to_pydict()
        for j in range(len(d["uuid"])):
            if d["order"][j] != "Lepidoptera":
                continue
            url = d["source_url"][j]
            sid = d["source_id"][j] if d["data_source"][j] == "gbif" else None
            # Mark our images that ToL also has, so "extend" only adds what ToL lacks entirely.
            k = url_idx.get(norm_url(url)) if url else None
            if k is not None:
                matched[k] = True
            if sid:
                try:
                    gid = int(sid)
                    tol_occ_images[gid] += 1
                    for r in gid_rows.get(gid, ()):
                        matched[r] = True
                except ValueError:
                    pass
            sp = f"{(d['genus'][j] or '').strip()} {(d['species'][j] or '').strip()}".strip()
            tol_lepi_species.add(sp)
            if sp not in keep or not url:
                continue
            host = tol.host_of(url)
            if emitted[sp] < a.cap:                               # replay of the selection
                emitted[sp] += 1
                if host in unavailable:
                    sel_unav[sp] += 1
                    lost_rows[sp].append((d["uuid"][j], url, sid))
            elif host not in unavailable:                         # replay of the substitution
                pool_ok[sp] += 1
                if subs[sp] < sel_unav[sp]:
                    subs[sp] += 1
        if i % 100 == 0:
            print(f"  row-group {i}/{nrg} | Lepidoptera selected {sum(emitted.values()):,} | "
                  f"unavailable {sum(sel_unav.values()):,} | substituted {sum(subs.values()):,} | "
                  f"{i/max(time.monotonic()-t0,1e-9)*60:.0f} rg/min", flush=True)

    # ---- fill: lost rows we hold the same photograph for ----------------------------------------
    rows_out: list[dict] = []
    fill_by_kind: Counter = Counter()
    filled: Counter = Counter()
    used = np.zeros(n, dtype=bool)
    for sp, rows in lost_rows.items():
        need = sel_unav[sp] - subs[sp]                 # what substitution did not cover
        for uuid, url, sid in rows:
            if filled[sp] >= need:
                break
            k = url_idx.get(norm_url(url))
            kind = "url"
            if k is None and sid:
                cand = gid_rows.get(int(sid), [])
                # occurrence-level match only when it is unambiguous on both sides
                if len(cand) == 1 and tol_occ_images[int(sid)] == 1:
                    k, kind = cand[0], "gbif_occurrence"
            if k is None or used[k]:
                continue
            used[k] = True
            filled[sp] += 1; fill_by_kind[kind] += 1
            rows_out.append({"uuid": uuid, "url": "file://" + g_path[k], "species": sp,
                             "kind": f"fill:{kind}"})

    # ---- extend: species under the cap, from images ToL does not have ---------------------------
    by_sp: dict[str, list[int]] = defaultdict(list)
    for i in np.nonzero(~matched & ~used)[0]:
        by_sp[g_sp[i]].append(int(i))
    extended: Counter = Counter()
    new_species = 0
    for sp, idxs in by_sp.items():
        if " " not in sp:
            continue                                   # genus-only keys are not species
        have = (emitted[sp] - sel_unav[sp] + subs[sp] + filled[sp]) if sp in keep else 0
        # A species ToL left under the floor qualifies only if OUR images alone clear it: its ToL
        # rows were never planned, so ours are all it will have.
        if sp not in keep:
            if len(idxs) < a.min_img:
                continue
            new_species += 1
            have = 0
        room = max(0, a.cap - have)
        for i in idxs[:room]:
            extended[sp] += 1
            rows_out.append({"uuid": f"gl-{g['filename'][i].rsplit('.', 1)[0]}",
                             "url": "file://" + g_path[i], "species": sp, "kind": "extend"})

    # ---- write ---------------------------------------------------------------------------------
    d = out / "host=file"; d.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(rows_out), a.rows_per_part):
        pq.write_table(pa.Table.from_pylist(rows_out[k:k + a.rows_per_part]),
                       d / f"part-g{k // a.rows_per_part:05d}.parquet")
    lepi_keep = [s for s in emitted]
    before = sum(emitted[s] - sel_unav[s] + subs[s] for s in lepi_keep)
    report = {
        "global_lepi_images": n, "global_lepi_also_in_tol": int(matched.sum()),
        "lepidoptera_selected": sum(emitted.values()),
        "lepidoptera_on_unavailable_hosts": sum(sel_unav.values()),
        "lepidoptera_substituted": sum(subs.values()),
        "lost_after_substitution": sum(sel_unav[s] - subs[s] for s in sel_unav),
        "filled_from_global_lepi": sum(filled.values()), "fill_by_match": dict(fill_by_kind),
        "still_lost_after_fill": sum(sel_unav[s] - subs[s] - filled[s] for s in sel_unav),
        "extended_images": sum(extended.values()), "extended_species": len(extended),
        "species_new_above_floor": new_species,
        "lepidoptera_images_before": before,
        "lepidoptera_images_after": before + sum(filled.values()) + sum(extended.values()),
        "lepidoptera_species_planned": len(lepi_keep),
        "lepidoptera_species_after": len(lepi_keep) + new_species,
        "rows_written": len(rows_out),
    }
    (out / "fill_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--global-lepi", required=True)
    p.add_argument("--species-counts", required=True)
    p.add_argument("--blocked", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-img", type=int, default=50)
    p.add_argument("--cap", type=int, default=2000)
    p.add_argument("--rows-per-part", type=int, default=50_000)
    p.add_argument("--readers", type=int, default=16)
    p.add_argument("--prefetch", type=int, default=24)
    p.add_argument("--limit-rg", type=int, default=0, help="smoke test: first N row-groups")
    main(p.parse_args())
