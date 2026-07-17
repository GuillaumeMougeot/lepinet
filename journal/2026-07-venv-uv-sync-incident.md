# What broke the venv on 2026-07-16, and what is the known-good version set?

**Status:** RESOLVED. Environment repaired and verified.
**One-line rule: never run `uv sync` in this repo.**

## What happened

`uv sync` was run to test a `pyproject.toml` fix. There was **no `uv.lock`** — this venv is
hand-assembled with `uv pip`, not lockfile-managed. `uv sync` therefore treated every package
unreachable from `pyproject.toml`'s `dependencies` as junk and pruned it, then upgraded the
rest to a fresh resolution. Casualties:

**Removed:** `mini_trainer`, `mini_metrics`, `timm`, `tensorboard`, `psutil`, `diskcache`,
and `pip` itself.

**Upgraded into breakage:**

| package | was → became | failure |
|---|---|---|
| torch | 2.12.1 → 2.13.0 | (silent; ABI skew with torchvision) |
| torchvision | 0.27.1 → 0.28.0 | `RuntimeError: operator torchvision::nms does not exist` |
| fastcore | 1.x → 2.1.1 | `AttributeError: 'list' object has no attribute 'starmap'` — fastcore 2.x drops `L.starmap`, which fastai's own `Optimizer.set_hypers` calls, so **every `fit()` dies** |
| fastprogress | 1.0.3 → 1.1.6 | imports `fasthtml`, which requires fastcore 2.x — directly unsatisfiable with the pin above (`TypeError: mtime_policy() got an unexpected keyword argument 'arg'`) |

The fastcore break is the nastiest: fastai 2.8.7's own metadata says only `fastcore>=1.8.0`,
which does **not** exclude the version that breaks it. Nothing but a manual pin prevents it.
It also passes every import check and only fails at `learn.fit` — so an import smoke test
would have missed it.

## Known-good set (verified training-clean)

```
torch==2.12.1 (+cu130)   # must match torchvision exactly
torchvision==0.27.1 (+cu130)
fastai==2.8.7
fastcore<2               # 2.x drops L.starmap -> every fit() dies
fastprogress==1.0.3      # >=1.1 pulls fasthtml -> needs fastcore 2.x
spacy<3.8.14             # 3.8.14 ships no cp314 wheel; this venv is Python 3.14
```

Plus editable siblings and their deps:
```
uv pip install -e ../mini_trainer -e ../mini_metrics   # at ~/codes/mini_trainer, ~/codes/mini_metrics
uv pip install timm tensorboard                        # also needs psutil, diskcache (pulled in)
```

These pins are now written into `pyproject.toml` so a fresh install cannot reproduce the
breakage. **That does not make `uv sync` safe** — it will still prune.

## Unrecovered residue

`uv sync` also moved `numpy`, `pillow`, `pyarrow`, `fsspec`, `typing_extensions`. **uv keeps no
record of the versions it replaced**, so those could not be restored exactly. Everything runs,
but pillow touches image decoding — see the caveat in
[2026-07-does-longtail-help.md](2026-07-does-longtail-help.md).

## Rules learned

1. **`uv pip install`, never `uv sync`.** This venv is hand-managed. There is no lockfile
   because there has never been one.
2. **`uv run python` is also broken here** — it triggers a build of the `lepinet` project
   itself. Use `.venv/bin/python`, or `source .venv/bin/activate` inside tmux.
3. **Always `--dry-run` first.** Every failure above was listed in advance by
   `uv pip install ... --dry-run`. Running it later is what diagnosed the damage; running it
   first would have prevented all of it.
4. **Forensics via mtime** — the only trail uv leaves:
   ```
   cd .venv/lib/python3.14/site-packages
   ls -d *.dist-info | while read d; do echo "$(stat -c %Y "$d") $d"; done | sort -n
   ```
   Anything with a fresh timestamp was rewritten.

## Why `uv run` / `uv sync` failed to build at all (the original bug)

```
error: Multiple top-level packages discovered in a flat-layout: ['dev', 'bash', 'data', 'logs', 'configs']
```

`pyproject.toml` declares `name = "lepinet"`, but **there is no importable `lepinet` package** —
the code is numbered scripts in `dev/` that import each other by filename via `importlib`.
With no package to find, setuptools fell back to flat-layout auto-discovery, saw five
top-level directories, and refused to guess.

Fixed with `[tool.setuptools] packages = []` — "install the dependencies, there is no package
here". That is *true today and worth revisiting*: the four modules that became real
infrastructure (028, 030, 032, 034) are a library wearing a lab-notebook costume, and this
error is the first bill for that. Extracting them into a real `lepinet/` package would make
the declaration honest and kill the `importlib.import_module("028_...")` hack at the same time.
