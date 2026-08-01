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


def _register_dev_heads():
    p = Path(__file__).with_name("050_hierarchical_heads.py")
    spec = importlib.util.spec_from_file_location("dev050_heads", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # side effect: registers the dev heads
    return module


def main(argv: list[str]) -> None:
    heads = _register_dev_heads()
    if not argv:
        raise SystemExit(__doc__)
    mode, rest = argv[0], argv[1:]
    if mode == "train":
        import yaml

        from lepinet.train import train_from_config

        # `marginal_arcface` computes its marginals inside forward, so the ArcFace margin has to be
        # applied there too -- which means forward needs the labels. Everything else keeps the
        # label-free forward and lets the loss inject the margin. See the head's docstring.
        head = (yaml.safe_load(open(rest[0])) or {}).get("train", {}).get("head")
        cbs = [heads.MarginContextCallback()] if head == "marginal_arcface" else None
        train_from_config(rest[0], extra_cbs=cbs)
    elif mode == "test":
        # Reuse lepinet's typer CLI for `test` so all its flags work unchanged.
        from lepinet.cli import app
        app(args=["test", *rest])
    else:
        raise SystemExit(f"unknown mode {mode!r} (train|test)")


if __name__ == "__main__":
    main(sys.argv[1:])
