"""Wait for the margin range test, then launch the two ArcFace runs it implies (+ chained evals).

The range test (`dev/055`) reports the **knee** — the largest margin the model still absorbs. That
bounds where `m` becomes *harmful*; it does not locate the *optimum for open-set*, which can only be
found by measuring AUROC. So we bracket it:

  * ``m = 0.75 × knee``  — the recommendation (just below the knee)
  * ``m = 0.40 × knee``  — a conservative point, to see whether OOD AUROC is still climbing there

Both are scored on species macro-F1 *and* OOD AUROC and compared with the existing m = 0.3 result
(0.9069 / 0.9115). Three points on the curve is enough to say whether 0.3 was lucky or near-optimal.

    python dev/056_launch_margin_runs.py            # polls, generates configs, submits
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
REMOTE_JSON = "/12347837/repos/lepinet/data/ucloud_preds/margin_find.json"
BASE_CFG = REPO / "configs/20260729_ucloud_arcface_zscore_effnetv2s.yaml"


def sh(*args) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def wait_for_knee(poll=300, tries=120) -> float:
    """Poll the drive for the range-test output; return the knee."""
    dest = Path("/tmp/margin_find.json")
    for i in range(tries):
        sh("ucloud", "files", "download", REMOTE_JSON, str(dest))
        if dest.exists() and dest.stat().st_size > 0:
            d = json.loads(dest.read_text())
            print(f"knee={d['knee']:.3f} recommended={d['recommended_m']}")
            return float(d["knee"])
        print(f"[{time.strftime('%H:%M')}] margin_find.json not ready ({i + 1}/{tries})")
        time.sleep(poll)
    raise SystemExit("range test never produced a result")


def make_config(margin: float, tag: str) -> tuple[Path, str]:
    cfg = yaml.safe_load(BASE_CFG.read_text())
    t = cfg["train"]
    t["arcface_margin"] = [round(margin, 3), 0.0, 0.0]
    name = f"lepinet-arcface-zscore-m{tag}-effnetv2s-ucloud"
    cfg["desc"] = name
    t["model_name"] = name
    p = REPO / f"configs/20260730_ucloud_arcface_zscore_m{tag}.yaml"
    p.write_text(
        "version: 1.0\n"
        f"# ArcFace x z-score at m={margin:.3f}, chosen from the margin range test (dev/055).\n"
        "# Scored on BOTH species macro-F1 and OOD AUROC, against the m=0.3 point (0.9069/0.9115).\n"
        + yaml.safe_dump(cfg, sort_keys=False)
    )
    return p, name


def make_tomls(cfg_path: Path, name: str, tag: str) -> tuple[Path, Path]:
    train = (REPO / "ucloud/lepinet-arcface-zscore.toml").read_text()
    train = train.replace("lepinet-arcface-zscore", f"lepinet-arcface-zscore-m{tag}")
    train = train.replace("configs/20260729_ucloud_arcface_zscore_effnetv2s.yaml",
                          f"configs/{cfg_path.name}")
    tp = REPO / f"ucloud/lepinet-arcface-zscore-m{tag}.toml"
    tp.write_text(train)

    ev = (REPO / "ucloud/lepinet-arcface-zscore-eval.toml").read_text()
    ev = ev.replace("lepinet-arcface-zscore-effnetv2s-ucloud", name)
    ev = ev.replace("ucloud_preds/arcface-zscore", f"ucloud_preds/arcface-zscore-m{tag}")
    ev = ev.replace('name = "lepinet-arcface-zscore-eval"', f'name = "lepinet-arcface-zscore-m{tag}-eval"')
    ep = REPO / f"ucloud/lepinet-arcface-zscore-m{tag}-eval.toml"
    ep.write_text(ev)
    return tp, ep


def submit(toml: Path, job: str, after: str | None = None):
    args = ["ucloud", "q", "submit", str(toml), "--name", job]
    if after:
        args += ["--after", after]
    for _ in range(4):
        out = subprocess.run(args, capture_output=True, text=True).stdout.strip()
        print(f"  {job}: {out.splitlines()[-1] if out else 'no output'}")
        if "submitted job" in out or "queued" in out:
            return
        subprocess.run(["ucloud", "q", "rm", job], capture_output=True)
        time.sleep(20)


def main():
    k = wait_for_knee()
    for frac, tag in ((0.75, "hi"), (0.40, "lo")):
        m = frac * k
        cfg, name = make_config(m, tag)
        tp, ep = make_tomls(cfg, name, tag)
        print(f"m={m:.3f} ({tag}) -> {cfg.name}")
        submit(tp, f"lepi-arcface-m{tag}")
        submit(ep, f"lepi-arcface-m{tag}-eval", after=f"lepi-arcface-m{tag}")


if __name__ == "__main__":
    main()
