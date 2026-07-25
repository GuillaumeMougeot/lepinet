"""Training orchestration: config → trained ``.pt`` checkpoint.

Implements the **independent-head** baseline recipe end to end (effnetv2_s, Muon, one_cycle,
warmup 0.5ep, grad_clip 5.0, light aug, bs 64, 460→256, square-root oversampling), the recipe
that reached test species macro-F1 0.9148. Everything specific to why each knob is set the way it
is lives in the journal; this module just wires the pieces together.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import data as data_mod
from . import schedules
from .callbacks import HostMemoryGuard, NaNGuard
from .config import TrainConfig, prepare_run_dir
from .heads import build_head
from .loss import FastaiLossWrapper, MultiLevelCELoss
from .metrics import default_metrics
from .model import arch_body_features, arch_is_vit, build_learner, resolve_arch


def train(cfg: TrainConfig):
    """Run one training job from a :class:`~lepinet.config.TrainConfig`. Writes ``<out_dir>/<model_name>.pt``."""
    data_mod.ensure_fork_start_method()
    levels = list(cfg.levels)
    out_dir = Path(cfg.out_dir)
    hierarchy_path = Path(cfg.hierarchy_path) if cfg.hierarchy_path else Path(cfg.parquet_path).parent / "hierarchy.csv"

    # --- data ---
    df, _hier = data_mod.gen_df(cfg.parquet_path, out_dir, cfg.min_img_per_spc, cfg.fold,
                                hierarchy_path, cfg.family_filter, levels=levels)
    vocabs = {level: sorted(df[level].unique().tolist()) for level in levels}
    n_classes = [len(vocabs[level]) for level in levels]
    print(f"Classes per level ({levels}): {n_classes}")

    sample_wgts = data_mod.sample_weights(df, level=levels[0], power=cfg.oversample_power, levels=levels)
    if sample_wgts is not None:
        print(f"Rare-class oversampling ON (power={cfg.oversample_power}, level={levels[0]}).")
    dls = data_mod.make_dls(df, vocabs, cfg.img_dir, cfg.aug_img_size, cfg.img_size, cfg.batch_size,
                            cfg.num_workers, aug_kwargs=cfg.aug_kwargs, sample_wgts=sample_wgts, levels=levels)

    # --- model ---
    arch = resolve_arch(cfg.model_arch_name)
    vit = arch_is_vit(arch, img_size=cfg.img_size)  # ViT/DINOv3 → FlatHead + manual Learner
    nf = arch_body_features(arch, img_size=cfg.img_size)
    head_kwargs = {"scale": cfg.arcface_scale, "margin": cfg.arcface_margin} if cfg.head == "arcface" else {}
    custom_head = build_head(cfg.head, nf, n_classes, hidden=cfg.hidden, pool=not vit, **head_kwargs)
    n_head_params = sum(p.numel() for p in custom_head.parameters())
    print(f"Head={cfg.head}, hidden={cfg.hidden}, backbone={'ViT' if vit else 'conv'} -> {n_head_params / 1e6:.2f} M head params")
    if cfg.head == "arcface":
        print(f"ArcFace ON (scale={cfg.arcface_scale}, margin={cfg.arcface_margin}).")

    # --- loss ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arc_margins = None
    if cfg.head == "arcface":
        arc_margins = cfg.arcface_margin if isinstance(cfg.arcface_margin, list) else [cfg.arcface_margin] * len(n_classes)
    criterion = MultiLevelCELoss(n_classes, weights=cfg.level_weights,
                                 label_smoothing=cfg.label_smoothing, device=device,
                                 arc_scale=cfg.arcface_scale if cfg.head == "arcface" else None,
                                 arc_margins=arc_margins)
    loss_func = FastaiLossWrapper(criterion)

    # --- callbacks ---
    from fastai.vision.all import CSVLogger, GradientClip, SaveModelCallback

    cbs = [
        HostMemoryGuard(),
        NaNGuard(),
        *([GradientClip(cfg.grad_clip)] if cfg.grad_clip and cfg.grad_clip > 0 else []),
        CSVLogger(out_dir / f"{cfg.model_name}.csv", append=True),
        SaveModelCallback(fname=cfg.model_name, every_epoch=True),
    ]
    # MixUp (opt-in): mix images + labels. Uses MixUpMulti (multi-target-aware) with the loss's
    # y_int=True + reduction toggling, so mixing happens through the per-level loss. A regularizer
    # for longer/bigger runs (journal/2026-07-bigger-everything.md).
    if cfg.mixup and cfg.mixup > 0:
        from .callbacks import MixUpMulti

        cbs.append(MixUpMulti(cfg.mixup))
        print(f"MixUp ON (alpha={cfg.mixup}).")
    if cfg.cutmix and cfg.cutmix > 0:
        raise NotImplementedError(
            "cutmix is not yet supported for multi-target heads (fastai's CutMix.before_batch "
            "indexes self.y, a tuple here). Use `mixup` for now; CutMixMulti is a follow-up."
        )
    # GCCallback is intentionally omitted (D3): the clean head has no per-batch reference cycle.
    # Add `GCCallback()` here if a future head reintroduces one and GPU memory climbs.

    learn = build_learner(dls, arch, custom_head, loss_func, default_metrics(levels),
                          out_dir / "models", cbs, optimizer=cfg.optimizer, vit=vit)

    if cfg.fp16:
        # bf16 by default (fp32 exponent range -> no overflow-to-NaN); fp16 only if asked.
        learn = learn.to_bf16() if cfg.precision == "bf16" else learn.to_fp16()

    if cfg.resume_checkpoint:
        state = torch.load(cfg.resume_checkpoint, map_location=device)
        state = state.get("model_state_dict", state)
        learn.model.load_state_dict(state)
        print(f"Resumed weights from {cfg.resume_checkpoint} ({cfg.resume_epochs_done}/{cfg.nb_epochs} epochs done).")

    _fit(learn, cfg)
    _save_checkpoint(learn, cfg, levels, vocabs, df, out_dir, vit=vit)


def _fit(learn, cfg: TrainConfig):
    """Drive the LR schedule (Muon-safe, LR-only) or fall back to fastai's fine_tune."""
    if cfg.schedule in ("one_cycle", "flat_cos", "front_loaded"):
        learn.unfreeze()
        if cfg.schedule == "front_loaded":
            wu = cfg.warmup_epochs if cfg.warmup_epochs and cfg.warmup_epochs > 0 else 0.15
            full_sched = schedules.front_loaded_schedule(cfg.nb_epochs, cfg.base_lr, wu,
                                                         cfg.fast_decay_epochs, cfg.lr_mid_frac)
        elif cfg.warmup_epochs and cfg.warmup_epochs > 0:
            full_sched = schedules.warmup_cos_schedule(cfg.nb_epochs, cfg.base_lr, cfg.warmup_epochs, cfg.schedule)
        else:
            full_sched = None  # built-in fit_one_cycle/fit_flat_cos: no resumable schedule fn

        if cfg.resume_checkpoint:
            if full_sched is None:
                raise ValueError("resume needs warmup_epochs > 0 or schedule='front_loaded' to rebuild the LR curve.")
            schedules.fit_resume(learn, full_sched, cfg.nb_epochs, cfg.resume_epochs_done)
        elif full_sched is not None:
            schedules.fit_scheduled(learn, cfg.nb_epochs, full_sched)
        elif cfg.schedule == "one_cycle":
            learn.fit_one_cycle(cfg.nb_epochs, cfg.base_lr)
        else:
            learn.fit_flat_cos(cfg.nb_epochs, cfg.base_lr)
    else:
        learn.fine_tune(cfg.nb_epochs, cfg.base_lr, freeze_epochs=cfg.freeze_epochs)


def _save_checkpoint(learn, cfg: TrainConfig, levels, vocabs, df, out_dir: Path, vit: bool = False):
    """Save a self-contained ``.pt``: weights + everything needed to rebuild the head at test/export.

    The hierarchy is derived from the training ``df`` (not a hierarchy.csv on disk), so the
    checkpoint's parent table can never disagree with its own class set. ``head`` / ``arcface_scale``
    / ``vit`` tell the test/export reconstruction which forward to rebuild (all default to the cosine
    conv-map path, so pre-existing effnet checkpoints load unchanged).
    """
    model_path = out_dir / f"{cfg.model_name}.pt"
    torch.save(
        {
            "model_state_dict": learn.model.state_dict(),
            "head": cfg.head,
            "arcface_scale": cfg.arcface_scale,
            "vit": vit,
            "model_arch_name": cfg.model_arch_name,
            "hidden": cfg.hidden,
            "levels": levels,
            "vocabs": vocabs,
            "hierarchy": data_mod.build_hierarchy(df, levels).to_dict(orient="list"),
        },
        model_path,
    )
    print(f"Model exported to {model_path}")
    return model_path


def train_from_config(config_path: str):
    """Load a YAML config, stamp a run dir, and train. Returns the run directory."""
    cfg, run_dir = prepare_run_dir(config_path)
    train(cfg)
    return run_dir
