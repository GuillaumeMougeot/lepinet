"""Refuse UCloud job specs whose worst-case CPU spend has not been approved.

Why this exists: on 2026-08-28 the ToL crawler was submitted on a 64-vCPU node with
`auto_extend` and `max_time = 168h`. It was network-bound and used a few cores at most, yet
it ran 63 hours and spent **4,032 core-hours -- 58 % of the project's 7,000 core-hour CPU
allocation -- in one job**. Its worst case, had the queue token not expired, was 10,752.
CPU core-hours are scarcer here than GPU hours. Nothing in the launch path computed the
cost, so nothing stopped it.

The rule this enforces, for every CPU product:

  * worst case = vCPU x (max_time if auto_extend else hours)
  * vCPU above MAX_VCPU, or a worst case above MAX_CORE_HOURS, is REFUSED unless the spec
    carries an owner sign-off line:   # budget-approved: <who/when/why>
  * auto_extend without max_time is REFUSED outright -- that is an unbounded spend.

GPU products bill GPU-hours against a separate, larger allocation and are reported but not
gated here.

Three ways in, one rule:

    python ucloud/budget_check.py ucloud/*.toml          # check specs, print worst cases
    python ucloud/budget_check.py --hook < event.json    # Claude Code PreToolUse hook
    pytest tests/test_ucloud_budget.py                   # CI: no committed spec may violate it
"""
from __future__ import annotations

import json
import re
import shlex
import sys
import tomllib
from pathlib import Path

MAX_VCPU = 8              # a network- or IO-bound job never needs more; a sized job justifies itself
MAX_CORE_HOURS = 300      # ~10 % of what was left after the crawl; above this, the owner signs off
APPROVAL = re.compile(r"^\s*#\s*budget-approved:\s*\S", re.M)


def _hours(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([hm]?)\s*", str(v))
    if not m:
        raise ValueError(f"unparseable duration {v!r}")
    n, unit = float(m.group(1)), m.group(2)
    return n / 60 if unit == "m" else n


def vcpus(product_id: str) -> int | None:
    """vCPU count for a CPU product id like `cpu-amd-zen5-64-vcpu`; None for GPU products."""
    m = re.fullmatch(r"cpu-.*?-(\d+)-vcpu", product_id or "")
    return int(m.group(1)) if m else None


def check(path: Path) -> tuple[bool, str]:
    text = path.read_text()
    spec = tomllib.loads(text)
    pid = spec.get("product", {}).get("id", "")
    n = vcpus(pid)
    if n is None:
        return True, f"{path.name}: {pid or '?'} -- GPU/other product, not gated on CPU budget"

    sched = spec.get("schedule", {})
    hours = _hours(spec.get("time_allocation", {}).get("hours"))
    auto = sched.get("auto_extend")
    max_time = sched.get("max_time")
    if auto and not max_time:
        return False, (f"{path.name}: REFUSED -- auto_extend with no max_time is an unbounded spend "
                       f"on {n} vCPU. Set max_time.")
    worst_h = _hours(max_time) if auto else hours
    worst = n * worst_h
    approved = bool(APPROVAL.search(text))
    line = f"{path.name}: {n} vCPU x {worst_h:g} h = {worst:,.0f} core-hours worst case"

    problems = []
    if n > MAX_VCPU:
        problems.append(f"{n} vCPU > {MAX_VCPU}")
    if worst > MAX_CORE_HOURS:
        problems.append(f"{worst:,.0f} core-hours > {MAX_CORE_HOURS}")
    if problems and not approved:
        return False, (f"{line} -- REFUSED ({'; '.join(problems)}). Size the node from a measured "
                       f"CPU profile, cap max_time, or get the owner's sign-off and add a line "
                       f"'# budget-approved: <who/when/why>'.")
    return True, line + (" (owner-approved)" if problems else "")


def _toml_from_command(cmd: str) -> list[Path]:
    """TOML paths passed to `ucloud q submit` / `ucloud jobs create` in a shell command."""
    out = []
    for seg in re.split(r"&&|\|\||;|\|", cmd):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        if "ucloud" not in toks:
            continue
        i = toks.index("ucloud")
        rest = toks[i + 1:]
        if rest[:2] in (["q", "submit"], ["jobs", "create"]):
            out += [Path(t) for t in rest[2:] if t.endswith(".toml")]
    return out


def hook() -> int:
    """PreToolUse hook: block the Bash call if any submitted spec fails. Exit 2 = block."""
    try:
        ev = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = (ev.get("tool_input") or {}).get("command", "")
    cwd = Path(ev.get("cwd") or ".")
    failures = []
    for p in _toml_from_command(cmd):
        p = p if p.is_absolute() else cwd / p
        if not p.exists():
            continue
        ok, msg = check(p)
        if not ok:
            failures.append(msg)
    if failures:
        print("UCloud CPU budget guard:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 2
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--hook"]:
        return hook()
    bad = 0
    for a in argv:
        ok, msg = check(Path(a))
        print(("ok    " if ok else "FAIL  ") + msg)
        bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
