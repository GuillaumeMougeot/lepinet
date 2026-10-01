"""Every committed UCloud spec must pass the CPU budget guard. See ucloud/budget_check.py."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("budget_check", ROOT / "ucloud" / "budget_check.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def test_no_committed_spec_overspends():
    failures = [m for ok, m in (bc.check(p) for p in sorted((ROOT / "ucloud").glob("*.toml"))) if not ok]
    assert not failures, "\n".join(failures)


def _toml(tmp_path, body):
    p = tmp_path / "job.toml"
    p.write_text(body)
    return p


BASE = '[product]\nid = "cpu-amd-zen5-{n}-vcpu"\n[time_allocation]\nhours = {h}\n'


def test_refuses_the_spec_that_burned_the_budget(tmp_path):
    # the 2026-08-28 crawler: 64 vCPU, auto_extend, max_time 168h
    p = _toml(tmp_path, BASE.format(n=64, h=24) + '[schedule]\nauto_extend = "1h"\nmax_time = "168h"\n')
    ok, msg = bc.check(p)
    assert not ok and "10,752" in msg


def test_refuses_unbounded_auto_extend(tmp_path):
    p = _toml(tmp_path, BASE.format(n=2, h=4) + '[schedule]\nauto_extend = "1h"\n')
    assert not bc.check(p)[0]


def test_small_bounded_job_passes(tmp_path):
    p = _toml(tmp_path, BASE.format(n=4, h=12) + '[schedule]\nauto_extend = "1h"\nmax_time = "72h"\n')
    ok, msg = bc.check(p)
    assert ok and "288" in msg


def test_owner_approval_unlocks(tmp_path):
    p = _toml(tmp_path, "# budget-approved: owner 2026-10-01, measured need\n"
              + BASE.format(n=16, h=48))
    assert bc.check(p)[0]


def test_gpu_products_not_gated(tmp_path):
    p = _toml(tmp_path, '[product]\nid = "gpu-nvidia-b200-1-gpu"\n[time_allocation]\nhours = 999\n')
    assert bc.check(p)[0]


def test_hook_blocks_submit_and_ignores_other_commands(tmp_path):
    bad = _toml(tmp_path, BASE.format(n=64, h=100))
    for cmd, code in [(f"ucloud q submit {bad}", 2), (f"cd x && ucloud jobs create {bad}", 2),
                      (f"cat {bad}", 0), ("ucloud q ls", 0)]:
        r = subprocess.run([sys.executable, str(ROOT / "ucloud" / "budget_check.py"), "--hook"],
                           input=json.dumps({"tool_input": {"command": cmd}, "cwd": str(tmp_path)}),
                           capture_output=True, text=True)
        assert r.returncode == code, (cmd, r.stderr)
