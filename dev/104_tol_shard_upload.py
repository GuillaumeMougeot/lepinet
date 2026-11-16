"""Upload a mini_trainer shard set to UCloud as it is built, verify each file's size there, and replace nothing.

    ~/.local/share/uv/tools/ucloud-api/bin/python dev/104_tol_shard_upload.py LOCAL REMOTE [--free] [--watch]

Runs with the `ucloud` tool's own Python (it imports `ucloud_api`; stdlib otherwise). Finished shards
(`<split>/NNNNNN.tar` with its `.parquet`, no `.partial`) go up in batches; a batch counts as uploaded
only when every remote file has the local byte size, and is then written to LOCAL/../<name>.uploaded.
`--free` deletes a verified local `.tar` (the full set does not fit next to the dataset; dev/103 shards
waits for disk). `manifest.json` goes up last, after every shard it lists is in the ledger, so a
reader never sees a manifest whose shards are missing. REMOTE must not exist unless the ledger does
(a resume). Nothing on UCloud is deleted, and only this script's own partial files (a dropped upload) are overwritten. The SHA-256 check is mini_trainer's
`verify --full`, run where the shards are read.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from ucloud_api.client import UCloudClient
from ucloud_api.files import Files
from ucloud_api.transfer import Transfer


def finished(root: Path) -> list[Path]:
    out = []
    for split in sorted(p for p in root.iterdir() if p.is_dir()):
        for t in sorted(split.glob("*.tar")):
            if t.with_suffix(".parquet").exists():
                out.append(t)
    return out


def main(a):
    local, remote = Path(a.local), a.remote.rstrip("/")
    ledger_path = local.parent / f"{local.name}.uploaded"
    done = set(ledger_path.read_text().split()) if ledger_path.exists() else set()
    log = open(ledger_path, "a", buffering=1)
    with UCloudClient() as client:
        files, xfer = Files(client), Transfer(client)
        if files.stat(remote) is not None and not done:
            raise SystemExit(f"{remote} exists and there is no ledger: refusing to write into it")
        made = set()

        def ensure(path: str):
            if path not in made:
                if files.stat(path) is None:
                    files.mkdir(path)
                made.add(path)
        ensure(remote)
        t_start, sent = time.time(), 0
        while True:
            todo = [t for t in finished(local) if f"{t.parent.name}/{t.stem}" not in done]
            manifest = local / "manifest.json"
            if not todo:
                if manifest.exists():
                    m = json.loads(manifest.read_text())
                    want = {s["name"] for e in m["splits"].values() for s in e["shards"]}
                    missing = want - done
                    if missing:
                        raise SystemExit(f"manifest lists {len(missing)} shards not uploaded, e.g. {sorted(missing)[:3]}")
                    if files.stat(f"{remote}/manifest.json") is None:
                        xfer.upload(manifest, f"{remote}/manifest.json", overwrite=False)
                    got = files.stat(f"{remote}/manifest.json")
                    assert got and got.size == manifest.stat().st_size, "manifest size differs"
                    log.write("manifest.json\n")
                    print(f"done: {len(want):,} shards and manifest.json at {remote}; "
                          f"{sent / 2**30:.1f} GiB this run, {sent / 2**20 / (time.time() - t_start):.1f} MiB/s", flush=True)
                    return
                if not a.watch:
                    print("nothing to upload yet")
                    return
                time.sleep(60)
                continue
            by_split: dict = {}
            for t in todo[: a.batch]:
                by_split.setdefault(t.parent.name, []).append(t)
            for split, batch in by_split.items():
                ensure(f"{remote}/{split}")
                stage = local.parent / f".{local.name}.stage" / split
                redo = local.parent / f".{local.name}.redo" / split
                for d in (stage, redo):
                    if d.exists():
                        shutil.rmtree(d)
                    d.mkdir(parents=True)
                # A resumed batch: files already there at full size are not sent again (a re-send
                # with overwrite off would land as a renamed copy). A file not in the ledger and
                # smaller remotely than here is this script's own interrupted upload (a dropped
                # connection leaves the partial file on UCloud, 2026-11-13): it alone is overwritten
                # (owner, 2026-11-15). Anything else already there stops the run.
                there = {Path(e.path).name: e.size for e in files.list_path(f"{remote}/{split}")}
                for t in batch:
                    for f in (t, t.with_suffix(".parquet")):
                        size = f.stat().st_size
                        if f.name not in there:
                            os.link(f, stage / f.name)
                        elif there[f.name] < size:
                            os.link(f, redo / f.name)
                        elif there[f.name] != size:
                            raise SystemExit(f"{split}/{f.name}: remote {there[f.name]} > local {size}: not ours, stopping")
                t0 = time.time()
                if any(stage.iterdir()):
                    xfer.upload(stage, f"{remote}/{split}", concurrency=a.j, overwrite=False)
                if any(redo.iterdir()):
                    print(f"{split}: overwriting {len(list(redo.iterdir()))} partial files of an interrupted upload", flush=True)
                    xfer.upload(redo, f"{remote}/{split}", concurrency=a.j, overwrite=True)
                shutil.rmtree(redo)
                there = {Path(e.path).name: e.size for e in files.list_path(f"{remote}/{split}")}
                nbytes = 0
                for t in batch:
                    for f in (t, t.with_suffix(".parquet")):
                        if there.get(f.name) != f.stat().st_size:
                            raise SystemExit(f"{split}/{f.name}: remote size {there.get(f.name)} != local {f.stat().st_size}")
                        nbytes += f.stat().st_size
                for t in batch:
                    name = f"{split}/{t.stem}"
                    log.write(name + "\n")
                    done.add(name)
                    if a.free:
                        t.unlink()
                shutil.rmtree(stage)
                sent += nbytes
                dt = time.time() - t0
                print(f"{split}: +{len(batch)} shards ({nbytes / 2**30:.1f} GiB in {dt:.0f} s, {nbytes / 2**20 / dt:.1f} MiB/s); "
                      f"{len(done):,} uploaded", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("local")
    p.add_argument("remote")
    p.add_argument("--free", action="store_true", help="delete each local .tar once its remote copy is verified")
    p.add_argument("--watch", action="store_true", help="keep going until the local manifest.json appears")
    p.add_argument("--batch", type=int, default=16, help="shards per upload call")
    p.add_argument("-j", type=int, default=8, help="files in parallel")
    main(p.parse_args())
