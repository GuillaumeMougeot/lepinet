"""Backbone resolution and model assembly (backbone + :class:`~lepinet.heads.PooledHead`).

Handles both torchvision archs (via fastai callables) and timm names (the modern small nets for
the backbone sweep), matching exactly what ``vision_learner`` builds so a saved ``state_dict``
round-trips. ``build_learner`` wires the whole training object together.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def resolve_arch(model_arch_name: str):
    """Map a config ``model_arch_name`` to a torchvision callable or a timm name string.

    A name that is a torchvision arch (``efficientnet_v2_s``, ``resnet18``, ...) resolves to the
    fastai callable; anything else must be a real timm model (``fastvit_*``, ``repvit_*``, ...) or
    this raises — so a typo fails here, not eight hours into a run.
    """
    import fastai.vision.all as fva

    if hasattr(fva, model_arch_name):
        return getattr(fva, model_arch_name)
    import timm

    if model_arch_name in timm.list_models():
        return model_arch_name
    # Accept timm pretrained-tag names like "convnextv2_large.fcmae_ft_in22k_in1k_384": the base
    # architecture (before the dot) must be a real timm model; timm.create_model resolves the tag.
    if model_arch_name.split(".")[0] in timm.list_models():
        return model_arch_name
    raise ValueError(
        f"Unknown model_arch_name {model_arch_name!r}: not a torchvision arch and not in "
        f"timm.list_models() (base {model_arch_name.split('.')[0]!r} not found). "
        f"Check spelling or try timm.list_models('<pattern>*')."
    )


def arch_body_features(arch_spec, img_size: int = 256) -> int:
    """Feature-channel count the head will pool, for a torchvision callable or a timm name.

    For timm this forwards a dummy through the *fastai ``TimmBody``* (the exact body training
    builds), because a net with a post-stage conv head can report a different pool-less width than
    the body fastai actually wraps — detecting the wrong one silently mismatches the head.
    """
    if isinstance(arch_spec, str):
        from fastai.vision.learner import create_timm_model

        model, _ = create_timm_model(arch_spec, n_out=1, pretrained=False, custom_head=nn.Identity())
        body = model[0]
        body.eval()
        with torch.no_grad():
            out = body(torch.zeros(1, 3, img_size, img_size))
        if out.ndim != 4:
            raise ValueError(
                f"timm arch {arch_spec!r} emits a {out.ndim}D feature {tuple(out.shape)}, not a "
                f"[N,C,H,W] map -- PooledHead's AdaptiveAvgPool2d needs a spatial map. Pick a "
                f"conv-stem or hybrid net (fastvit/repvit/mobilenetv4), not a vanilla ViT."
            )
        return out.shape[1]
    from fastai.vision.all import num_features_model
    from fastai.vision.learner import create_body

    body = create_body(arch_spec(weights=None), n_in=3, pretrained=False)
    return num_features_model(body)


def build_backbone_model(arch_spec, custom_head: nn.Module) -> nn.Module:
    """``nn.Sequential(body, head)`` matching ``vision_learner``, without needing DataLoaders.

    Used by the test / export reconstruction paths (they rebuild the model from a checkpoint
    alone). For timm this defers to fastai's ``create_timm_model`` so the module tree is identical
    to training and the saved ``state_dict`` loads.
    """
    from fastai.vision.learner import create_body, create_timm_model

    if isinstance(arch_spec, str):
        model, _ = create_timm_model(arch_spec, n_out=1, pretrained=False, custom_head=custom_head)
        return model
    body = create_body(arch_spec(weights=None), n_in=3, pretrained=False)
    return nn.Sequential(body, custom_head)


def build_learner(dls, arch_spec, custom_head, loss_func, metrics, model_dir, cbs, optimizer="muon"):
    """Assemble a fastai ``Learner`` for the hierarchical head.

    ``init=None`` because the cosine head initializes itself (``weight_norm`` + spherical
    repulsion); letting fastai's default kaiming sweep run would fight that init. Muon needs its
    ``opt_func`` (backbone→Muon, head→AdamW).
    """
    from fastai.vision.all import vision_learner

    learner_kwargs = {}
    if optimizer == "muon":
        from .optim import muon_opt_func

        learner_kwargs["opt_func"] = muon_opt_func

    return vision_learner(
        dls, arch_spec,
        n_out=1,  # unused: custom_head builds the real head
        custom_head=custom_head,
        init=None,
        loss_func=loss_func,
        metrics=metrics,
        model_dir=Path(model_dir),
        cbs=cbs,
        **learner_kwargs,
    )
