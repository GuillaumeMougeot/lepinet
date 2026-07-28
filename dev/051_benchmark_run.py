"""Run lepinet train/test for a dev-registered head (hierarchical / autoregressive).

Those heads live in `dev/050_hierarchical_heads.py` and register themselves into
`lepinet.heads.HEAD_REGISTRY` on import. `lepinet train -c ...` on UCloud wouldn't import them, so
this thin runner imports the dev heads first, then dispatches to the normal lepinet entry points —
`sparse_masks` is threaded automatically by `train.evaluate` (signature-driven).

    python dev/051_benchmark_run.py train configs/<head>.yaml
    python dev/051_benchmark_run.py test  --model '...*.pt' --parquet ... --img-dir ... --out-dir ...
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _register_dev_heads() -> None:
    p = Path(__file__).with_name("050_hierarchical_heads.py")
    spec = importlib.util.spec_from_file_location("dev050_heads", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # side effect: registers HierarchicalHead (+ future ones)


def main(argv: list[str]) -> None:
    _register_dev_heads()
    if not argv:
        raise SystemExit(__doc__)
    mode, rest = argv[0], argv[1:]
    if mode == "train":
        from lepinet.train import train_from_config
        train_from_config(rest[0])
    elif mode == "test":
        # Reuse lepinet's typer CLI for `test` so all its flags work unchanged.
        from lepinet.cli import app
        app(args=["test", *rest])
    else:
        raise SystemExit(f"unknown mode {mode!r} (train|test)")


if __name__ == "__main__":
    main(sys.argv[1:])
