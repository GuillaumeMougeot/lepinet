"""The crawler must not deadlock when bodies fail mid-stream (journal/2026-10-01-the-crawl-resized-
the-wrong-side.md). One memory slot, one host slot, two rows whose first body read fails: before the
fix, the retrier held the memory slot while waiting for the host slot that the other row held while
waiting for the memory slot."""
import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import aiohttp

_spec = importlib.util.spec_from_file_location(
    "tol", Path(__file__).resolve().parents[1] / "dev" / "082_tol_crawler.py")
tol = importlib.util.module_from_spec(_spec); sys.modules["tol"] = tol; _spec.loader.exec_module(tol)


class _Content:
    def __init__(self, fail):
        self.fail = fail

    async def iter_chunked(self, n):
        await asyncio.sleep(0.01)
        if self.fail:
            raise aiohttp.ClientPayloadError("cut mid-body")
        yield b"\xff\xd8" + b"0" * 100


class _Resp:
    status, headers = 200, {"content-type": "image/jpeg"}

    def __init__(self, fail):
        self.content = _Content(fail)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    """First request for each url fails mid-body, later ones succeed."""
    def __init__(self):
        self.seen = set()

    def get(self, url, allow_redirects=True):
        fail = url not in self.seen
        self.seen.add(url)
        return _Resp(fail)


def test_mid_body_failures_do_not_deadlock(monkeypatch):
    monkeypatch.setattr(tol, "verify_and_encode", lambda *a, **k: (b"jpg", tol.STATUS_OK, ("h", (1, 1))))

    async def go():
        f = tol.Fetcher.__new__(tol.Fetcher)
        f.a = types.SimpleNamespace(variant="", attempts=4, max_bytes=10**6, size=256, quality=90,
                                    min_dim=1, fit="short", dead_probe=200, dead_ratio=0.97,
                                    block_probe=40, block_ratio=0.9, max_inflight=1, file_root="")
        f.inflight = asyncio.Semaphore(1)
        f.totals = tol.Counter()
        f.pool = None
        b = tol.HostBudget("example.org", 4)          # cur = MIN_CAP = 1 host slot
        s = _Session()
        rows = [{"url": f"http://example.org/{i}.jpg"} for i in range(2)]
        return await asyncio.wait_for(
            asyncio.gather(*(f.fetch_one(s, r, b) for r in rows)), timeout=20)

    res = asyncio.run(go())
    assert [r[1] for r in res] == [tol.STATUS_OK, tol.STATUS_OK]
