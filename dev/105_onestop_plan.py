"""OneStop invasive-species test set: GBIF images newer than every model's training data.

Builds a manifest in the format of `dev/082_tol_crawler.py` (host-partitioned parquet with uuid, url,
species) from the GBIF occurrence API, for the insect species of the OneStop list
(`data/onestop/onestop_species_list.csv`, resolved to accepted keys in `data/onestop/taxa_resolved.csv`).

Taxonomy: occurrences are queried through GBIF's current backbone, COL XR (checklistKey below; ids from
`data/onestop/taxa_xr.csv`, matched from the list's names), not the legacy backbone. The legacy key of
Oegoconia novimundi (1845962) lumps four Oegoconia species, among them the native O. quadripuncta (388 recent
records against 171 for novimundi in COL XR); every other list species gets the same records either way
(2026-11-11). `taxon_key` keeps the legacy key, which is what the models' labels use.

Recency is the contamination guard: only occurrences with an event date on or after 2025-04-27 are kept.
Nemo (MAMBO_v3) and our own models train on GBIF download 0032836-250426092105405, made 2025-04-26;
the MILA UK & Denmark classifier was trained in April 2024. An observation made after both dates
cannot be in either training set. As a second guard, any gbifID present in the Global Lepidoptera
parquet is dropped (expected: none).

Where the species was observed does not matter (owner, 2026-11-11). At most `--cap` images per species
(2,000, owner, 2026-11-11), the training corpus's cap: each month since the start date gets a quota
proportional to its number of records, filled in GBIF's default order, all images of an occurrence kept.
The non-Lepidoptera on the list, which no moth model can name, are fetched like the rest.

    python dev/105_onestop_plan.py --out data/onestop/manifest
    python dev/082_tol_crawler.py split --manifest data/onestop/manifest --rows-per-part 20000
    taskset -c 0-3 python dev/082_tol_crawler.py fetch --manifest data/onestop/manifest \
        --images data/onestop/images --size 384 --variant medium --max-cpus 4
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

API = "https://api.gbif.org/v1/occurrence/search"
UA = {"User-Agent": "lepinet-onestop-eval/1.0 (research, Aarhus University; github.com/GuillaumeMougeot/lepinet)"}
START = dt.date(2025, 4, 27)          # day after GBIF download 0032836-250426092105405
GLOBAL = "data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"
# The list's Helicoverpa armigera key (119361453) is a new backbone key with no occurrences indexed
# under it; GBIF's occurrence index uses 8108800 (also our training key).
KEY_OVERRIDE = {119361453: 8108800}
PAGE = 300
MAX_WINDOW = 95_000                   # the search API stops at offset 100,000

spec = importlib.util.spec_from_file_location("tol", Path(__file__).with_name("082_tol_crawler.py"))
tol = importlib.util.module_from_spec(spec)
sys.modules["tol"] = tol      # dataclasses look the module up there
spec.loader.exec_module(tol)


def get(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 -- retried, then raised
            err = e
            time.sleep(min(60, 3 * 2 ** i))
    raise err


COL_XR = "7ddf754f-d193-4cc9-b351-99906754a03b"


def base(key: str, lo: dt.date, hi: dt.date) -> dict:
    return {"checklistKey": COL_XR, "taxonKey": key, "mediaType": "StillImage", "occurrenceStatus": "PRESENT",
            "eventDate": f"{lo.isoformat()},{hi.isoformat()}"}


def months(lo: dt.date, hi: dt.date):
    """Calendar-month windows from lo to hi."""
    cur = lo
    while cur <= hi:
        nxt = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        yield cur, min(hi, nxt - dt.timedelta(days=1))
        cur = nxt


def quotas(key: str, lo: dt.date, hi: dt.date, cap: int):
    """(window, quota) per month: records proportional to the month's count, summing to at most cap."""
    w = [(a, b, get({**base(key, a, b), "limit": 0})["count"]) for a, b in months(lo, hi)]
    total = sum(n for *_, n in w)
    if total <= cap:
        return [(a, b, n) for a, b, n in w if n]
    q = [(a, b, min(n, max(1, round(cap * n / total)))) for a, b, n in w if n]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxa", default="data/onestop/taxa_resolved.csv")
    ap.add_argument("--out", default="data/onestop/manifest")
    ap.add_argument("--occurrences", default="data/onestop/occurrences.parquet")
    ap.add_argument("--cap", type=int, default=2000, help="images per species")
    ap.add_argument("--exclude-hosts", nargs="*", default=["observation.org", "www.artsobservasjoner.no",
                                                           "image.laji.fi", "moth.tbn.org.tw"],
                    help="hosts that refused the crawl (403, timeouts, bad certificate, 2026-11-11); their images "
                         "are skipped and the cap is filled from the rest")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    taxa = pd.read_csv(a.taxa)
    taxa["key"] = taxa.key.map(lambda k: KEY_OVERRIDE.get(int(k), int(k)))
    taxa = taxa.drop_duplicates("key")
    xr = pd.read_csv("data/onestop/taxa_xr.csv")
    lst = pd.read_csv("data/onestop/onestop_species_list.csv")
    lst = lst[lst["GBIF Taxon ID"].notna()].drop_duplicates("Canonical name")
    name2key = dict(zip(lst["Canonical name"], lst["GBIF Taxon ID"].astype(int)))
    xr["list_key"] = xr.list_name.map(name2key)
    taxa = taxa.merge(xr[["list_key", "xr_id", "xr_name"]], on="list_key", how="left")
    assert taxa.xr_id.notna().all(), taxa[taxa.xr_id.isna()]
    seen = set(pq.read_table(GLOBAL, columns=["gbifID"]).column(0).to_pylist())
    seen = {int(x) for x in seen if x is not None}
    today = dt.date.today()
    excluded = set(a.exclude_hosts)
    rows, dropped = [], 0
    for t in taxa.itertuples():
        n_t = 0
        for lo, hi, quota in quotas(t.xr_id, START, today, a.cap):
            off, got = 0, 0
            while got < quota and n_t < a.cap and off < MAX_WINDOW:
                page = get({**base(t.xr_id, lo, hi), "limit": PAGE, "offset": off})
                for o in page["results"]:
                    if got >= quota or n_t >= a.cap:
                        break
                    gid = int(o["key"])
                    if gid in seen:
                        dropped += 1
                        continue
                    imgs = [m for m in o.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")
                            and tol.host_of(m["identifier"]) not in excluded]
                    imgs = imgs[: a.cap - n_t]
                    for i, m in enumerate(imgs):
                        rows.append({
                            "uuid": f"{gid}_{i}", "url": m["identifier"], "species": t.name,
                            "taxon_key": int(t.key), "list_key": int(t.list_key), "xr_id": t.xr_id, "xr_name": t.xr_name,
                            "gbifID": gid,
                            "occurrence_species_key": o.get("speciesKey"), "event_date": o.get("eventDate", ""),
                            "country_code": o.get("countryCode", ""), "basis_of_record": o.get("basisOfRecord", ""),
                            "dataset_key": o.get("datasetKey", ""), "publisher": o.get("publisher", ""),
                            "license": m.get("license") or o.get("license", ""),
                            "rights_holder": m.get("rightsHolder") or o.get("rightsHolder", ""),
                            "creator": m.get("creator", ""), "references": m.get("references", "")})
                    n_t += len(imgs)
                    got += len(imgs)
                if page.get("endOfRecords") or not page["results"]:
                    break
                off += PAGE
        print(f"{t.name:32s} key {t.key:>9} xr {t.xr_id:>6}: {n_t:,} images", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("uuid")
    df.to_parquet(a.occurrences, index=False)
    df["host"] = df.url.map(tol.host_of)
    for h, g in df.groupby("host"):
        d = out / f"host={tol.slug(h)}"
        d.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g.drop(columns="host"), preserve_index=False), d / "part-00000.parquet")
    summary = {"cap": a.cap, "excluded_hosts": sorted(excluded), "start": START.isoformat(), "end": today.isoformat(), "taxa": int(len(taxa)), "images": int(len(df)),
               "occurrences": int(df.gbifID.nunique()), "hosts": int(df.host.nunique()),
               "dropped_in_global_lepi": dropped, "per_species": df.groupby("species").size().to_dict()}
    (out / "plan_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_species"}, indent=2))


if __name__ == "__main__":
    main()
