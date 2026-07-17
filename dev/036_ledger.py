"""Experiment ledger: one table of every training run and what it scored.

Every run already leaves a complete record on disk -- dev/030 writes
`<out_dir>/<timestamp>-<model_name>/config.yaml` next to the per-epoch CSV and the exported
`.pt`, and dev/032 writes `<preds>/<model_name>/<timestamp>-<eval_name>/metrics.json` next to
the `test_config.yaml` naming the checkpoint it evaluated. Nothing ever read that record back,
so answering "what have I tried and what happened?" meant opening ~30 directories by hand and
holding the answer in your head (or in a chat log). This script is the reader.

The useful part is not the list -- it is the DIFF. Configs here are ~40 lines of which ~38 are
identical between any two runs, so printing them whole hides the experiment inside the
boilerplate. Instead each run is summarised by the config keys on which it differs from a
chosen baseline, which is the same thing as "what was this run testing?". The result reads as
the project's progress ladder, generated from disk rather than remembered:

    run                              delta vs baseline               val F1   test F1
    20260714-191001 onecycle          schedule=one_cycle             0.9014   0.8887
    20260715-073321 onecycle-10ep     schedule=one_cycle nb_epochs=10  ...    0.8976

Joins are by checkpoint path (test_config.yaml -> model_path), not by name, because model_name
is reused across timestamps and only the path identifies which run an eval actually scored.

It only ever reads `data/`; the sole thing it writes is the `--snapshot` file. That snapshot
matters more than it looks: `data/` is a symlink to machine-local storage and is gitignored, so
a clone of this repo elsewhere has no runs at all and sees nothing. `RESULTS.md` is therefore
the only durable, shareable record of what was run and what it scored -- and
`git log -p RESULTS.md` becomes the project's result history. Regenerate and commit it after a
run finishes.

Usage:
    python dev/036_ledger.py                          # all runs, auto-chosen baseline
    python dev/036_ledger.py --chain                  # the ladder: each run vs the previous
    python dev/036_ledger.py --baseline 20260714-191001
    python dev/036_ledger.py --sort test --full       # rank by test F1, show full configs
    python dev/036_ledger.py --only scored            # just the runs with a test number
    python dev/036_ledger.py --json                   # machine-readable, for further analysis
    python dev/036_ledger.py --snapshot               # write RESULTS.md (commit it)
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

# Config keys that identify *where a run wrote its output* rather than *what it tested*.
# They differ between every pair of runs, so including them in the delta would drown the one
# or two keys that actually matter.
BOOKKEEPING_KEYS = {"model_name", "out_dir", "desc", "resume_checkpoint", "resume_epochs_done"}

# mini_metrics reports per taxonomic level, keyed by string index; level 0 is species.
# Species macro-F1 is the number this project optimises, so it is what the ledger surfaces.
SPECIES = "0"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _flatten(d, prefix=""):
    """Flatten nested config dicts to dotted keys, so `aug_kwargs.max_warp` can be diffed."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def load_runs(models_dir):
    """One record per training run, from each `<timestamp>-<model_name>/config.yaml`."""
    live = live_model_names()
    runs = {}
    for cfg_path in sorted(Path(models_dir).glob("*/config.yaml")):
        run_dir = cfg_path.parent
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except yaml.YAMLError as e:
            print(f"  ! skipping {run_dir.name}: unreadable config ({e})")
            continue
        # Directory name is `<YYYYMMDD>-<HHMMSS>-<model_name>`; the first two fields are the
        # run id, which is the only thing guaranteed unique across runs.
        parts = run_dir.name.split("-", 2)
        run_id = "-".join(parts[:2])
        runs[run_dir.name] = {
            "run_id": run_id,
            "dir": run_dir,
            "name": parts[2] if len(parts) > 2 else run_dir.name,
            "config": _flatten(cfg.get("train", {})),
            "desc": cfg.get("desc", ""),
            **_run_progress(run_dir, live),
        }

    # A relaunch after a failed attempt leaves several directories sharing one model_name, and
    # the process table only names the model -- not which directory it adopted. The newest is
    # the live one by construction (dev/030 timestamps at startup); demote its predecessors.
    for name in live:
        same = sorted(k for k, r in runs.items() if r["status"] == "running" and r["name"] == name)
        for stale in same[:-1]:
            runs[stale]["status"] = "died"
    return runs


def live_model_names():
    """model_names of trainer processes running right now, from their command lines.

    Liveness cannot be inferred from file mtimes: an epoch takes ~1.5-2.5h and the CSV only
    gains a row when one *finishes*, so a healthy run in its first epoch is indistinguishable
    from one that crashed at startup -- both are a fresh directory holding just a config. The
    process table settles it. Each trainer is launched as `030_....py -c <config>`, so the
    config it names resolves to the model_name whose directory it is currently writing.
    """
    try:
        out = subprocess.run(["pgrep", "-af", "030_hierarchical_heads_benchmark"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()

    names = set()
    for line in out.splitlines():
        tokens = line.split()
        # Only the python process itself, not the wrapping shell that also matches the pattern.
        if "-c" not in tokens or not any(t.endswith(".py") for t in tokens):
            continue
        cfg_path = Path(tokens[tokens.index("-c") + 1])
        if not cfg_path.exists():
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except yaml.YAMLError:
            continue
        name = (cfg.get("train") or {}).get("model_name")
        if name:
            names.add(name)
    return names


def _run_progress(run_dir, live_names=frozenset()):
    """Final-epoch validation metrics + completion status for one run.

    Status is derived, not recorded: a run is 'done' once it has exported (its last act); else
    'running' if a trainer process is writing that model_name; else 'died'. When several
    directories share a live model_name (a relaunch after a failed attempt), only the newest
    can be the live one -- caller resolves that.

    Two export formats count: dev/030's `.pt` bundle, and the `.pkl` that the older
    multihead-v4 line (dev/028) wrote via fastai's `learn.export`. Checking only `.pt` marks
    every v4 run 'died' despite a complete export sitting next to its CSV.
    """
    exported = any(run_dir.glob("*.pt")) or any(run_dir.glob("*.pkl"))
    name = run_dir.name.split("-", 2)[-1]
    status = "done" if exported else ("running" if name in live_names else "died")

    csvs = list(run_dir.glob("*.csv"))
    rows = [ln for ln in csvs[0].read_text().splitlines() if ln.strip()] if csvs else []
    if len(rows) < 2:
        return {"epochs": 0, "val_f1": None, "status": status}

    cells = dict(zip(rows[0].split(","), rows[-1].split(",")))

    def num(key):
        try:
            return float(cells[key])
        except (KeyError, ValueError):
            return None

    return {"epochs": len(rows) - 1, "val_f1": num("f1_speciesKey"), "status": status}


def load_evals(preds_dir, runs):
    """Attach test metrics to the run whose checkpoint each eval actually scored.

    Joined via test_config.yaml's `model_path` rather than by model_name: names repeat across
    timestamps, so only the checkpoint path says which run produced the weights.
    """
    for cfg_path in sorted(Path(preds_dir).glob("*/*/test_config.yaml")):
        eval_dir = cfg_path.parent
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            metrics = json.loads(metrics_path.read_text())
        except (yaml.YAMLError, json.JSONDecodeError):
            continue

        model_path = (cfg.get("test") or {}).get("model_path", "")
        # The run directory is the checkpoint's parent (or grandparent, for the
        # SaveModelCallback checkpoints under `models/`).
        owner = next((k for k in runs if k and k in str(model_path)), None)
        if owner is None:
            continue

        record = {
            "eval_name": eval_dir.name,
            "test_set": (cfg.get("test") or {}).get("test_set"),
            "test_f1": (metrics.get("f1") or {}).get(SPECIES),
            "test_micro_acc": (metrics.get("micro_accuracy") or {}).get(SPECIES),
        }
        # Keep the newest eval per run; dir names are timestamp-prefixed so max() is chronological.
        prior = runs[owner].get("eval")
        if prior is None or eval_dir.name > prior["eval_name"]:
            runs[owner]["eval"] = record
    return runs


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------

#: Sentinel for "this key is absent from the config" -- distinct from a key explicitly set to
#: None (e.g. `hierarchy_path: null`). Rendering both as "None" would claim a run set an option
#: to nothing when in fact it predates the option and silently took the script default.
UNSET = object()


def config_delta(config, baseline, full=False):
    """The config keys where `config` differs from `baseline` -- i.e. what the run tested."""
    if full:
        return {k: v for k, v in config.items() if k not in BOOKKEEPING_KEYS}
    keys = (set(config) | set(baseline)) - BOOKKEEPING_KEYS
    return {k: config.get(k, UNSET) for k in sorted(keys)
            if config.get(k, UNSET) != baseline.get(k, UNSET)}


def _render_value(v):
    return "unset" if v is UNSET else str(v)


def format_delta(delta, max_width=None, empty="(baseline)"):
    """Render a delta as `key=value` tokens, collapsing groups that always move together.

    Nested option groups (`aug_kwargs.*`) are a single decision made once -- "light aug" flips
    six keys at the same instant -- so listing all six implies six independent experiments and
    buries the one key that actually varied. Any dotted prefix contributing 3+ changed keys
    collapses to `prefix.*(n)`; `--full` still shows them individually.
    """
    if not delta:
        return empty

    groups, flat = {}, []
    for k, v in delta.items():
        if "." in k:
            groups.setdefault(k.rsplit(".", 1)[0], []).append((k, v))
        else:
            flat.append((k, v))

    tokens = []
    for prefix, items in groups.items():
        if len(items) >= 3:
            tokens.append(f"{prefix}.*({len(items)})")
        else:
            tokens += [f"{k.split('.')[-1]}={_render_value(v)}" for k, v in items]
    tokens += [f"{k}={_render_value(v)}" for k, v in flat]

    text = " ".join(tokens)
    if max_width and len(text) > max_width:
        text = text[: max_width - 1] + "…"
    return text


def pick_baseline(runs):
    """Default baseline: the earliest completed run that was actually evaluated.

    Using the oldest scored run means deltas read as "what changed since we started", which is
    the direction the progress ladder tells its story in.
    """
    scored = [k for k, r in runs.items() if r.get("eval") and r["status"] == "done"]
    return min(scored) if scored else (min(runs) if runs else None)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_table(runs, baseline_key, sort_by, full, chain=False, width=58):
    """One row per run. `chain` diffs each run against the previous one chronologically
    (the incremental story: what did I change *next*), rather than against a fixed baseline
    (the cumulative story: how far am I from where I started)."""
    baseline = runs[baseline_key]["config"] if baseline_key else {}
    ordered = sorted(runs)
    rows = []
    for key, run in runs.items():
        ev = run.get("eval") or {}
        if chain:
            i = ordered.index(key)
            against = runs[ordered[i - 1]]["config"] if i else {}
        else:
            against = baseline
        rows.append({
            "run": f"{run['run_id']} {run['name'][:34]}",
            "ep": str(run["epochs"]),
            "delta": format_delta(config_delta(run["config"], against, full),
                                  None if full else width,
                                  empty="(unchanged)" if chain else "(baseline)"),
            "val": f"{run['val_f1']:.4f}" if run["val_f1"] is not None else "-",
            "test": f"{ev['test_f1']:.4f}" if ev.get("test_f1") is not None else "-",
            "status": run["status"] + ("" if key != baseline_key else " *base"),
            "_sort_test": ev.get("test_f1") if ev.get("test_f1") is not None else -1,
            "_sort_val": run["val_f1"] if run["val_f1"] is not None else -1,
            "_sort_run": key,
        })

    if sort_by == "test":
        rows.sort(key=lambda r: -r["_sort_test"])
    elif sort_by == "val":
        rows.sort(key=lambda r: -r["_sort_val"])
    else:
        rows.sort(key=lambda r: r["_sort_run"])

    cols = [("run", "run"), ("ep", "ep"),
            ("delta", "delta vs previous" if chain else "delta vs baseline"),
            ("val", "val F1"), ("test", "test F1"), ("status", "status")]
    widths = {c: max(len(title), *(len(r[c]) for r in rows)) for c, title in cols}

    def line(cells):
        return "  ".join(str(cells[i]).ljust(widths[c]) for i, (c, _) in enumerate(cols))

    out = [line([t for _, t in cols]), "  ".join("-" * widths[c] for c, _ in cols)]
    out += [line([r[c] for c, _ in cols]) for r in rows]
    return "\n".join(out)


def render_markdown(runs, baseline_key):
    """The committed snapshot: a markdown table of every run, newest first.

    Everything this script reads lives under `data/`, which is a symlink to machine-local
    storage and gitignored -- so a clone of this repo on any other machine has no runs at all,
    and if that disk is lost the provenance goes with it. The generated table is therefore not
    just a view: written to a tracked file it is the only durable, shareable record of what was
    run and what it scored. Committing generated output is usually a smell; here the source is
    unreachable to everyone but this machine, which makes the snapshot the artifact.

    Bonus from keeping it in git: `git log -p RESULTS.md` is the project's result history.
    """
    baseline = runs[baseline_key]["config"] if baseline_key else {}
    lines = [
        "# Results",
        "",
        "<!-- Generated by dev/036_ledger.py --snapshot. Do not edit by hand: rerun it. -->",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from `data/global/{{models,preds}}`, which is "
        "gitignored machine-local storage — this file is the only copy of this table that "
        "leaves the training box.",
        "",
        "`delta` is the config keys differing from the baseline, i.e. what the run was testing. "
        "F1 is **species (level 0) macro-F1**: `val` from the run's own CSV, `test` from "
        "mini_metrics on the held-out fold (`set == '0'`). Reasoning behind these numbers lives "
        "in [`journal/`](journal/).",
        "",
        f"Baseline: `{baseline_key}`",
        "",
        "| run | model | ep | delta vs baseline | val F1 | test F1 | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in sorted(runs, reverse=True):
        run = runs[key]
        ev = run.get("eval") or {}
        val = f"{run['val_f1']:.4f}" if run["val_f1"] is not None else "—"
        test = f"**{ev['test_f1']:.4f}**" if ev.get("test_f1") is not None else "—"
        delta = format_delta(config_delta(run["config"], baseline))
        lines.append(f"| `{run['run_id']}` | {run['name']} | {run['epochs']} | "
                     f"{delta} | {val} | {test} | {run['status']} |")

    best = max((r for r in runs.values() if (r.get("eval") or {}).get("test_f1")),
               key=lambda r: r["eval"]["test_f1"], default=None)
    if best:
        lines += ["", f"**Best test species macro-F1: {best['eval']['test_f1']:.4f}** "
                      f"(`{best['run_id']}` {best['name']}, micro-acc "
                      f"{best['eval']['test_micro_acc']:.4f})"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Table of every training run and what it scored.")
    parser.add_argument("--models-dir", default="data/global/models", help="Directory of run dirs.")
    parser.add_argument("--preds-dir", default="data/global/preds", help="Directory of eval dirs.")
    parser.add_argument("--baseline", default=None,
                        help="Run id (e.g. 20260714-191001) to diff configs against.")
    parser.add_argument("--sort", choices=["run", "val", "test"], default="run",
                        help="Order rows by run id (chronological), or by val/test species F1.")
    parser.add_argument("--full", action="store_true",
                        help="Show every config key, not just those differing from the baseline.")
    parser.add_argument("--chain", action="store_true",
                        help="Diff each run against the previous one instead of the baseline.")
    parser.add_argument("--only", choices=["done", "scored", "live"], default=None,
                        help="Filter: completed runs, runs with a test score, or running ones.")
    parser.add_argument("--json", action="store_true", help="Emit records as JSON.")
    parser.add_argument("--snapshot", nargs="?", const="RESULTS.md", default=None, metavar="PATH",
                        help="Write the table to a tracked markdown file (default RESULTS.md) "
                             "instead of printing. data/ is gitignored, so this is the only "
                             "record of these results that survives off this machine.")
    args = parser.parse_args()

    runs = load_runs(args.models_dir)
    if not runs:
        print(f"No runs found under {args.models_dir}.")
        return
    load_evals(args.preds_dir, runs)

    # Filter after loading evals so `scored` can see them, but before choosing a baseline so
    # the baseline is always a run that survives into the table.
    keep = {
        "done": lambda r: r["status"] == "done",
        "scored": lambda r: (r.get("eval") or {}).get("test_f1") is not None,
        "live": lambda r: r["status"] == "running",
    }.get(args.only)
    if keep:
        runs = {k: r for k, r in runs.items() if keep(r)}
        if not runs:
            print(f"No runs matching --only {args.only}.")
            return

    if args.baseline:
        matches = [k for k in runs if k.startswith(args.baseline)]
        if not matches:
            known = "\n  ".join(f"{r['run_id']}  {r['name']}" for r in sorted(
                runs.values(), key=lambda r: r["run_id"]))
            raise SystemExit(f"No run matching '{args.baseline}'. Known runs:\n  {known}")
        baseline_key = matches[0]
    else:
        baseline_key = pick_baseline(runs)

    if args.json:
        print(json.dumps([{**{k: v for k, v in r.items() if k != "dir"},
                           "dir": str(r["dir"]),
                           "delta": config_delta(r["config"], runs[baseline_key]["config"])}
                          for r in runs.values()], indent=2, default=str))
        return

    if args.snapshot:
        path = Path(args.snapshot)
        path.write_text(render_markdown(runs, baseline_key))
        scored = sum(1 for r in runs.values() if (r.get("eval") or {}).get("test_f1"))
        print(f"Wrote {path} — {len(runs)} runs ({scored} with a test score). "
              f"Commit it: data/ is gitignored, so this file is the record.")
        return

    print(f"\n{len(runs)} runs under {args.models_dir}")
    if not args.chain:
        print(f"baseline: {baseline_key}")
    print(f"metric:   species (level 0) macro-F1 -- val from training CSV, test from mini_metrics\n")
    print(render_table(runs, baseline_key, args.sort, args.full, args.chain))

    best = max((r for r in runs.values() if (r.get("eval") or {}).get("test_f1")),
               key=lambda r: r["eval"]["test_f1"], default=None)
    if best:
        print(f"\nbest test species macro-F1: {best['eval']['test_f1']:.4f} "
              f"({best['run_id']} {best['name']}, micro-acc {best['eval']['test_micro_acc']:.4f})")
    live = [r["name"] for r in runs.values() if r["status"] == "running"]
    if live:
        print(f"running now: {', '.join(live)}")


if __name__ == "__main__":
    main()
