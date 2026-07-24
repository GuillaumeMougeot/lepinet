"""lepinet — hierarchical image classification (species / genus / family / ...).

Clean, fastai-only, ``mini_trainer``-free reimplementation of the pipeline developed in ``dev/``.
It reproduces the project-best independent-head baseline (test species macro-F1 0.9148) and is
generic in the number of hierarchy levels. See ``src/lepinet/README.md`` and
``journal/2026-07-src-lepinet-baseline-port.md``.

Typical use::

    from lepinet import TrainConfig, train, evaluate, predict, export_onnx
    train(TrainConfig.from_dict(cfg["train"]))

Heavy submodules (anything pulling ``fastai.vision``) are imported lazily, so ``import lepinet``
stays cheap and side-effect-free.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0.dev0"

# Lightweight, side-effect-free submodules -- safe to import eagerly.
from . import heads, loss, metrics, schedules  # noqa: E402,F401
from .config import TrainConfig, load_config  # noqa: E402,F401

# name -> (module, attribute) for lazy top-level access.
_LAZY = {
    "train": ("lepinet.train", "train"),
    "train_from_config": ("lepinet.train", "train_from_config"),
    "evaluate": ("lepinet.test", "evaluate"),
    "predict": ("lepinet.infer", "predict"),
    "export_onnx": ("lepinet.export", "export_onnx"),
    "load_model": ("lepinet.test", "load_model"),
}

if TYPE_CHECKING:  # for type checkers / IDEs
    from .export import export_onnx  # noqa: F401
    from .infer import predict  # noqa: F401
    from .test import evaluate, load_model  # noqa: F401
    from .train import train, train_from_config  # noqa: F401


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrainConfig", "load_config", "train", "train_from_config",
    "evaluate", "predict", "export_onnx", "load_model",
    "heads", "loss", "metrics", "schedules", "__version__",
]
