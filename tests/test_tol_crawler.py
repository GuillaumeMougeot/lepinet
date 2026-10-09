"""Crawler regressions (journal/engineering/2026-10-01-the-crawl-resized-the-wrong-side.md).

Deadlock: one memory slot, one host slot, two rows whose first body read fails. Before the fix, the
retrier held the memory slot while waiting for the host slot that the other row held while waiting
for the memory slot.

Throttling: a host answering 429 must be paused (queued requests included), must not spend the
rows' attempts, and must be blocked as "throttled" if it never lifts."""
import asyncio
import importlib.util
import sys
import time
import types
from pathlib import Path

import aiohttp

_spec = importlib.util.spec_from_file_location(
    "tol", Path(__file__).resolve().parents[1] / "dev" / "082_tol_crawler.py")
tol = importlib.util.module_from_spec(_spec)
sys.modules["tol"] = tol
_spec.loader.exec_module(tol)


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


def _fetcher(monkeypatch, attempts=4):
    monkeypatch.setattr(tol, "verify_and_encode", lambda *a, **k: (b"jpg", tol.STATUS_OK, ("h", (1, 1))))
    f = tol.Fetcher.__new__(tol.Fetcher)
    f.a = types.SimpleNamespace(variant="", attempts=attempts, max_bytes=10**6, size=256, quality=90,
                                min_dim=1, fit="short", dead_probe=200, dead_ratio=0.97,
                                block_probe=40, block_ratio=0.9, max_inflight=4, file_root="")
    f.inflight = asyncio.Semaphore(4)
    f.totals = tol.Counter()
    f.pool = None
    return f


class _ThrottlingSession:
    """Answers 429 to the first `n_429` requests (all of them if None), then 200."""
    def __init__(self, n_429):
        self.n_429, self.sent = n_429, []

    def get(self, url, allow_redirects=True):
        self.sent.append(time.monotonic())
        r = _Resp(False)
        if self.n_429 is None or len(self.sent) <= self.n_429:
            r.status, r.headers = 429, {}
        return r


def _fast_pauses(monkeypatch, give_up=60.0):
    monkeypatch.setattr(tol, "THROTTLE_PAUSE_MIN", 0.02)
    monkeypatch.setattr(tol, "THROTTLE_PAUSE_MAX", 0.08)
    monkeypatch.setattr(tol, "THROTTLE_GIVE_UP_S", give_up)


def test_429s_do_not_spend_attempts(monkeypatch):
    _fast_pauses(monkeypatch)

    async def go():
        f, s = _fetcher(monkeypatch, attempts=2), _ThrottlingSession(n_429=6)
        b = tol.HostBudget("example.org", 4)
        return await f.fetch_one(s, {"url": "http://example.org/0.jpg"}, b), b

    (_, status, _), b = asyncio.run(go())
    assert status == tol.STATUS_OK           # 6 x 429 with 2 attempts: still fetched
    assert b.throttle_streak == 0            # a success ends the streak


def test_queued_requests_wait_out_the_pause(monkeypatch):
    _fast_pauses(monkeypatch)

    async def go():
        f, s = _fetcher(monkeypatch), _ThrottlingSession(n_429=3)
        b = tol.HostBudget("example.org", 4)          # one slot, three rows queued on it
        rows = [{"url": f"http://example.org/{i}.jpg"} for i in range(3)]
        await asyncio.wait_for(asyncio.gather(*(f.fetch_one(s, r, b) for r in rows)), timeout=20)
        return s.sent

    sent = asyncio.run(go())
    gaps = [b - a for a, b in zip(sent, sent[1:])]
    # pauses of 0.02, 0.04, 0.08 s after the three 429s; before the fix the gaps were ~0
    assert all(g >= p * 0.9 for g, p in zip(gaps, (0.02, 0.04, 0.08))), gaps


def test_a_host_that_never_stops_throttling_is_blocked(monkeypatch):
    _fast_pauses(monkeypatch, give_up=0.3)

    async def go():
        f, s = _fetcher(monkeypatch), _ThrottlingSession(n_429=None)
        b = tol.HostBudget("example.org", 4)
        rows = [{"url": f"http://example.org/{i}.jpg"} for i in range(3)]
        res = await asyncio.wait_for(asyncio.gather(*(f.fetch_one(s, r, b) for r in rows)), timeout=20)
        return res, b, len(s.sent)

    res, b, n_sent = asyncio.run(go())
    assert b.blocked and b.reason == "throttled"
    assert [r[1] for r in res] == [tol.SKIPPED_BLOCKED] * 3   # rows stay unattempted, not exhausted
    assert n_sent < 20                                        # paused, not hammered
