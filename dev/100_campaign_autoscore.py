"""Score every finished campaign run unattended (cron), so results are ready without anyone watching.

Lists ``tmp-runs/eval`` on UCloud and, for each output not yet scored, downloads it and scores it with
the same tools used by hand: mini_trainer's ``dev/lepinet/score.py`` for the test fold (whole set 0,
``--expect-instances`` so nothing is filtered), ``flemming.py`` for trap / probe / probe-HO
(calibrated by trap and night) and ``dev/099`` for open-set summaries. Appends to
``data/campaign/scores.csv`` and rewrites ``data/campaign/summary.md`` (species macro-F1 per run).
One output at a time (a few GB). A failed output is retried on the next run, three times at most.
The large test-fold CSVs are deleted after scoring; ``scores.csv`` stays next to each.

    */30 * * * * cd ~/codes/lepinet && flock -n /tmp/lepinet-autoscore.lock .venv/bin/python dev/100_campaign_autoscore.py >> data/campaign/autoscore.log 2>&1
"""
import json
import shutil
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MT = Path.home() / "codes/mini_trainer/.worktrees/lepinet-integration"
REMOTE = "/12347837/repos/mini_trainer-campaign/tmp-runs/eval"
OUT = ROOT / "data/campaign"
LOCAL = ROOT / "data/ucloud_preds/campaign"
PY = str(ROOT / ".venv/bin/python")
UCLOUD = shutil.which("ucloud") or str(Path.home() / ".local/bin/ucloud")  # cron has no ~/.local/bin on PATH
SKIP = ("ctl_",)  # the channels_last control is scored on lepinet's rows by hand (RESULTS VC)
# open-set models withheld C3b's taxa, so their whole test fold cannot be scored without filtering it
SKIP_TEST = ("os_",)
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "data/tools/mini_metrics-61db373"), "COLUMNS": "400"}


def sh(*cmd, cwd=ROOT, timeout=3600):
    r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])} exit {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


def remote(path=""):
    out = sh(UCLOUD, "files", "ls", f"{REMOTE}/{path}".rstrip("/"), cwd=ROOT / "ucloud", timeout=300)
    return [ln.split("│")[2].strip().rstrip("/") for ln in out.splitlines() if ln.count("│") > 3]


def fetch(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh(UCLOUD, "files", "download", f"{REMOTE}/{src}", str(dst), cwd=ROOT / "ucloud", timeout=1800)
    if not dst.exists():
        raise RuntimeError(f"download of {src} produced nothing")


def score_test(name, d):
    run = name.removesuffix("_test")
    csv = d / "mini_metric.csv"
    fetch(f"{name}/mini_metric.csv", csv)
    n = 31088 if run.startswith("pp_") else 632913
    sh(PY, str(MT / "dev/lepinet/score.py"), "--lepinet", str(ROOT), "--predictions", str(csv),
       "--expect-instances", str(n), "--seeds", "5")
    s = pd.read_csv(d / "scores.csv")
    s = s[s.metric == "f1"]
    for p in (csv, d / "predictions.csv"):
        p.unlink(missing_ok=True)
    return [{"run": run, "set": "test", "level": int(r.level), "zero": r.zero, "cal": r.cal_mean} for r in s.itertuples()]


def score_trap(name, d):
    run = name.removesuffix("_trap")
    csv = d / "mini_metric.csv"
    fetch(f"{name}/mini_metric.csv", csv)
    sh(PY, str(MT / "dev/lepinet/flemming.py"), "score", "--predictions", str(csv), "--lepinet", str(ROOT))
    rows = []
    for st in ("trap", "probe", "probe-ho"):
        s = pd.read_csv(d / st / "scores.csv")
        for r in s[s.metric == "f1"].itertuples():
            rows.append({"run": run, "set": st, "level": int(r.level), "zero": r.zero, "cal": r.calg_mean})
    return rows


def score_openset(name, d):
    run = name.removesuffix("_openset")
    for f in ("summary.parquet", "classes.json"):
        fetch(f"{name}/{f}", d / f)
    out = d / "auroc.csv"
    sh(PY, str(ROOT / "dev/099_openset_c3b.py"), "--run", str(d), "--out", str(out))
    a = pd.read_csv(out)
    a = a[(a["rank"] == 0) & (a.rule == "entropy")]
    return [{"run": run, "set": f"openset-{r.stratum}", "level": 0, "zero": r.auroc, "cal": None} for r in a.itertuples()]


SCORERS = {"_test": score_test, "_trap": score_trap, "_openset": score_openset}


def summary(df):
    sp = df[df.level == 0]
    t = sp.pivot_table(index="run", columns="set", values="zero", aggfunc="last")
    c = sp[sp.set.isin(["trap", "probe", "probe-ho"])].pivot_table(index="run", columns="set", values="cal", aggfunc="last")
    t = t.drop(columns=[x for x in c.columns if x in t.columns]).join(c, how="outer")
    cols = [x for x in ("test", "trap", "probe", "probe-ho", "openset-near", "openset-mid", "openset-far") if x in t.columns]
    lines = [f"# Campaign scores (auto, {time.strftime('%Y-%m-%d %H:%M')})", "",
             "Species macro-F1. test: whole set 0 at threshold 0. trap / probe / probe-ho: calibrated by "
             "(trap, night). openset: AUROC, entropy rule, species rank. Genus and family: scores.csv.", "",
             "| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for run, r in t[cols].sort_index().iterrows():
        lines.append(f"| {run} | " + " | ".join("" if pd.isna(v) else f"{v:.4f}" for v in r) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state_p = OUT / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    scores_p = OUT / "scores.csv"
    names = remote()
    only = sys.argv[1:]  # optional substrings, for a manual test
    todo = [n for n in names if n.endswith(tuple(SCORERS)) and not n.startswith(SKIP)
            and not (n.endswith("_test") and n.startswith(SKIP_TEST))
            and state.get(n, {}).get("done") is not True and state.get(n, {}).get("tries", 0) < 3 and (not only or any(o in n for o in only))]
    print(f"{time.strftime('%F %T')} {len(names)} outputs, {len(todo)} to score", flush=True)
    for name in todo:
        kind = next(k for k in SCORERS if name.endswith(k))
        files = remote(name)
        needed = {"summary.parquet", "classes.json"} if kind == "_openset" else {"mini_metric.csv"}
        if not needed <= set(files):
            continue  # still being written
        st = state.setdefault(name, {"tries": 0})
        try:
            rows = SCORERS[kind](name, LOCAL / name)
            new = pd.DataFrame(rows)
            old = pd.read_csv(scores_p) if scores_p.exists() else pd.DataFrame()
            pd.concat([old, new], ignore_index=True).to_csv(scores_p, index=False)
            st["done"] = True
            print(f"  scored {name}", flush=True)
        except Exception as e:  # report, count, carry on: one broken output must not stop the sweep
            st["tries"] += 1
            print(f"  FAILED {name} (try {st['tries']}): {e}", flush=True)
        state_p.write_text(json.dumps(state, indent=1))
    if scores_p.exists():
        summary(pd.read_csv(scores_p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
