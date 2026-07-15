"""Download the images referenced in the parquet metadata file into data/small/images.

Each image is downloaded into a subfolder named after the numeric identifier found
in the `image_path` column (e.g. "1801896/6c38871943...jpg" -> images/1801896/6c38871943...jpg).

Usage
-----
uv run python dev/027_download_subdataset.py
"""

import argparse
import asyncio
from pathlib import Path

import aiohttp
import pandas as pd

BASE_URL = "https://anon.erda.au.dk/share_redirect/On0rdRltgS/global_lepi/images/"
DEFAULT_PARQUET = "data/small/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.lepinet.parquet"
DEFAULT_OUTPUT_DIR = "data/small/images"
DEFAULT_MAX_CONCURRENT = 64


async def download_image(session: aiohttp.ClientSession, sem: asyncio.Semaphore, image_path: str, output_dir: Path):
    dest = output_dir / image_path
    if dest.exists():
        return None

    url = BASE_URL + image_path
    async with sem:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return f"{image_path}: HTTP {resp.status}"
                content = await resp.read()
        except Exception as e:
            return f"{image_path}: {e}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return None


async def download_all(image_paths: list[str], output_dir: Path, max_concurrent: int):
    sem = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_image(session, sem, p, output_dir) for p in image_paths]
        errors = []
        for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
            err = await coro
            if err:
                errors.append(err)
            if i % 500 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} done ({len(errors)} errors)")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="Path to the parquet metadata file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to store the downloaded images in.")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT, help="Max number of concurrent downloads.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    image_paths = df["image_path"].tolist()
    print(f"Found {len(image_paths)} images to download.")

    errors = asyncio.run(download_all(image_paths, output_dir, args.max_concurrent))

    if errors:
        print(f"\n{len(errors)} images failed to download:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("\nAll images downloaded successfully.")


if __name__ == "__main__":
    main()
