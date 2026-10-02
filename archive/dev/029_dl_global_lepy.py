# download_images.py
#
# Async port of the original threaded downloader. The threaded version shared
# one `requests.Session()` across 32 worker threads, but `requests`' default
# HTTPAdapter caps its connection pool at 10 -- so most of those threads paid
# a fresh TCP+TLS handshake per request instead of reusing a keep-alive
# connection to the (single) download host. aiohttp's connector pools
# properly by default, and async I/O concurrency scales far cheaper than OS
# threads, so this can run at much higher concurrency for the same cost.

import asyncio
from pathlib import Path

import aiohttp
import pandas as pd

PARQUET_FILE = "0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"
OUTDIR = Path("images")
FAILED_LOG = Path("failed.txt")

BASE_URL = "https://anon.erda.au.dk/share_redirect/On0rdRltgS/global_lepi/images/"
# Benchmarked against the real endpoint: throughput is flat ~75-78 img/s for
# concurrency 32-96, but *drops* at 128 (~58 img/s) and drops hard by 256-512
# (~23-33 img/s) with zero errors/retries recorded -- the server throttles
# many simultaneous connections from one client rather than rejecting them.
# 64 is the middle of the flat, fast zone.
N_WORKERS = 64
QUEUE_MAXSIZE = N_WORKERS * 4  # bounds memory regardless of dataset size
N_RETRIES = 3
RETRY_BACKOFF_S = 1
REQUEST_TIMEOUT_S = 60
PRINT_EVERY = 1000


async def download_one(session: aiohttp.ClientSession, image_path: str) -> str:
    """Returns 'ok', 'skip' or 'fail'."""
    outpath = OUTDIR / image_path
    if outpath.exists():
        return "skip"

    url = BASE_URL + image_path
    for attempt in range(N_RETRIES):
        try:
            async with session.get(url) as r:
                r.raise_for_status()
                content = await r.read()
            outpath.write_bytes(content)
            return "ok"
        except Exception:
            if attempt == N_RETRIES - 1:
                return "fail"
            await asyncio.sleep(RETRY_BACKOFF_S * (attempt + 1))
    return "fail"


async def worker(queue: asyncio.Queue, session: aiohttp.ClientSession, counts: dict, failed: list):
    while True:
        image_path = await queue.get()
        if image_path is None:
            queue.task_done()
            return

        status = await download_one(session, image_path)
        counts[status] += 1
        counts["completed"] += 1
        if status == "fail":
            failed.append(image_path)

        if counts["completed"] % PRINT_EVERY == 0 or counts["completed"] == counts["total"]:
            print(
                f"processed={counts['completed']}/{counts['total']} "
                f"downloaded={counts['ok']} skipped={counts['skip']} failed={counts['fail']}",
                flush=True,
            )
        queue.task_done()


async def produce(queue: asyncio.Queue, image_paths):
    for image_path in image_paths:
        await queue.put(image_path)
    for _ in range(N_WORKERS):
        await queue.put(None)  # one stop signal per worker


async def download_all(image_paths):
    counts = {"ok": 0, "skip": 0, "fail": 0, "completed": 0, "total": len(image_paths)}
    failed = []

    queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    connector = aiohttp.TCPConnector(limit=N_WORKERS)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        workers = [asyncio.create_task(worker(queue, session, counts, failed)) for _ in range(N_WORKERS)]
        await produce(queue, image_paths)
        await asyncio.gather(*workers)

    return counts, failed


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PARQUET_FILE, columns=["speciesKey", "filename"])
    image_paths = (df["speciesKey"].astype(str) + "/" + df["filename"]).tolist()
    total = len(image_paths)
    print(f"Found {total} images to download.")

    # Pre-create every species directory once instead of stat-ing/mkdir-ing
    # it on every one of the `total` rows.
    for species in df["speciesKey"].astype(str).unique():
        (OUTDIR / species).mkdir(parents=True, exist_ok=True)

    counts, failed = asyncio.run(download_all(image_paths))

    print(
        f"finished total={counts['total']} "
        f"downloaded={counts['ok']} skipped={counts['skip']} failed={counts['fail']}"
    )

    if failed:
        FAILED_LOG.write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} failed downloads written to {FAILED_LOG}")


if __name__ == "__main__":
    main()
