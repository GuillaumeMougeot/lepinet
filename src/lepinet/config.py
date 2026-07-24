"""Training configuration: a typed dataclass loaded from YAML.

The YAML schema is unchanged from ``dev/030`` (``version`` + ``desc`` + a ``train:`` block) so
existing configs keep working, with two differences: the package **defaults precision to bf16**
(``journal/2026-07-autoregressive-fp16-instability.md``) and it **rejects the long-tail knobs that
lost** (``logit_adjust_tau``, ``class_reg_strength``) rather than silently carrying dead options —
those live on as ``dev/`` experiments (``journal/2026-07-does-longtail-help.md``).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Any

import yaml

from .data import DEFAULT_LEVELS

VALID_CONFIG_VERSIONS = [1.0]
VALID_SCHEDULES = ("one_cycle", "flat_cos", "front_loaded", "fine_tune")


@dataclass
class TrainConfig:
    """Everything :func:`lepinet.train.train` needs. Field names match the YAML ``train:`` block."""

    # --- data ---
    parquet_path: str
    img_dir: str
    out_dir: str
    model_name: str
    model_arch_name: str
    fold: str = "1"
    min_img_per_spc: int = 0
    family_filter: list = field(default_factory=list)
    hierarchy_path: str | None = None
    levels: list = field(default_factory=lambda: list(DEFAULT_LEVELS))
    num_workers: int | None = None

    # --- model ---
    head: str = "independent"
    hidden: bool | int = True  # bottleneck width; True = backbone width (the size lever)

    # --- optimisation (the winning recipe defaults) ---
    nb_epochs: int = 5
    batch_size: int = 64
    aug_img_size: int = 460
    img_size: int = 256
    base_lr: float = 1e-3
    optimizer: str = "muon"          # muon | adam
    schedule: str = "one_cycle"      # one_cycle | flat_cos | front_loaded | fine_tune
    warmup_epochs: float = 0.5
    grad_clip: float = 5.0
    freeze_epochs: int = 1
    level_weights: list | None = None
    label_smoothing: float | None = None

    # --- precision ---
    precision: str = "bf16"          # bf16 (default, safe) | fp16
    fp16: bool = True                # enable mixed precision at all

    # --- long-tail ---
    oversample_power: float = 0.0    # 0.5 = square-root oversampling (the baseline win)

    # --- augmentation ---
    aug_kwargs: dict | None = None

    # --- front_loaded schedule extras ---
    fast_decay_epochs: float = 1.0
    lr_mid_frac: float = 0.1

    # --- resume ---
    resume_checkpoint: str | None = None
    resume_epochs_done: int = 0

    def __post_init__(self):
        if self.schedule not in VALID_SCHEDULES:
            raise ValueError(f"Unknown schedule {self.schedule!r}; must be one of {VALID_SCHEDULES}.")
        if self.optimizer not in ("adam", "muon"):
            raise ValueError(f"Unknown optimizer {self.optimizer!r}; must be 'adam' or 'muon'.")
        if self.optimizer == "muon" and self.schedule == "fine_tune":
            raise ValueError(
                "optimizer='muon' requires an unfrozen schedule (one_cycle/flat_cos/front_loaded): "
                "MuonAuxAdamW re-partitions param groups, which fastai's freeze bookkeeping can't survive."
            )
        if self.precision not in ("bf16", "fp16"):
            raise ValueError(f"Unknown precision {self.precision!r}; must be 'bf16' or 'fp16'.")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        d = dict(d)
        # Reject the interventions that were measured to lose, pointing to where they live now.
        for dead, where in {"logit_adjust_tau": "logit adjustment", "class_reg_strength": "class-distribution regularization"}.items():
            if d.pop(dead, 0):
                raise ValueError(
                    f"{dead!r} ({where}) is not part of the baseline recipe — it lost "
                    f"(journal/2026-07-does-longtail-help.md). Run it as a dev/ experiment instead."
                )
            d.pop(dead, None)  # a zero value is fine; drop it silently
        # Drop autoregressive-only knobs if present (baseline is independent-only).
        for k in ("decoder_num_layers", "decoder_nhead"):
            d.pop(k, None)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            warnings.warn(f"Ignoring unknown config keys: {sorted(unknown)}", stacklevel=2)
        return cls(**{k: v for k, v in d.items() if k in known})


def load_config(config_path: str | Path) -> tuple[TrainConfig, dict]:
    """Load a YAML config → (``TrainConfig``, raw dict). Validates the version and head."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if float(raw["version"]) not in VALID_CONFIG_VERSIONS:
        raise ValueError(f"Wrong config version {raw['version']}; must be in {VALID_CONFIG_VERSIONS}.")
    return TrainConfig.from_dict(raw["train"]), raw


def stamp_out_dir(out_dir: str | Path, desc: str) -> Path:
    """Create ``<out_dir>/<YYYYmmdd-HHMMSS>-<desc>/`` and return it."""
    stamped = Path(out_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{desc}"
    stamped.mkdir(parents=True, exist_ok=True)
    return stamped


def prepare_run_dir(config_path: str | Path) -> tuple[TrainConfig, Path]:
    """Load the config, stamp a fresh run directory, copy the config into it, return both."""
    cfg, raw = load_config(config_path)
    run_dir = stamp_out_dir(cfg.out_dir, raw["desc"])
    cfg.out_dir = str(run_dir)
    copyfile(config_path, run_dir / "config.yaml")
    return cfg, run_dir
