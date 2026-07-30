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


class ViTBody(nn.Module):
    """Wrap a timm ViT / DINOv2 / DINOv3 model as a backbone that emits one ``[N, C]`` embedding.

    Vanilla ViTs have no ``[N,C,H,W]`` map for :class:`~lepinet.heads.PooledHead` to pool, so this
    builds the timm model headless (``num_classes=0``) and lets timm's own global pooling (CLS or
    mean over patch tokens, per the model's config) produce the pooled vector. The cosine /
    ArcFace head then runs on that via :class:`~lepinet.heads.FlatHead`. This is the seam that lets
    DINOv3 be a teacher without disturbing the conv-map path used by effnet/ConvNeXt.
    """

    def __init__(self, arch_spec: str, pretrained: bool = False):
        super().__init__()
        import timm

        self.model = timm.create_model(arch_spec, pretrained=pretrained, num_classes=0)
        self.num_features = int(self.model.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def arch_is_vit(arch_spec, img_size: int = 256) -> bool:
    """True if a timm arch emits a non-``[N,C,H,W]`` feature (a ViT that needs :class:`ViTBody`).

    Torchvision callables are always conv maps → ``False``. For timm, forwards a dummy through the
    fastai ``TimmBody``: a 4-D output is a conv map (PooledHead path); anything else is ViT-like.
    """
    if not isinstance(arch_spec, str):
        return False
    from fastai.vision.learner import create_timm_model

    model, _ = create_timm_model(arch_spec, n_out=1, pretrained=False, custom_head=nn.Identity())
    body = model[0]
    body.eval()
    with torch.no_grad():
        out = body(torch.zeros(1, 3, img_size, img_size))
    return out.ndim != 4


def arch_body_features(arch_spec, img_size: int = 256) -> int:
    """Feature-channel count the head consumes, for a torchvision callable or a timm name.

    For a conv-map timm arch this forwards a dummy through the *fastai ``TimmBody``* (the exact body
    training builds), because a net with a post-stage conv head can report a different pool-less
    width than the body fastai actually wraps. For a ViT arch it returns the pooled embedding width
    from :class:`ViTBody` (the vector :class:`~lepinet.heads.FlatHead` will receive).
    """
    if isinstance(arch_spec, str):
        from fastai.vision.learner import create_timm_model

        model, _ = create_timm_model(arch_spec, n_out=1, pretrained=False, custom_head=nn.Identity())
        body = model[0]
        body.eval()
        with torch.no_grad():
            out = body(torch.zeros(1, 3, img_size, img_size))
        if out.ndim == 4:
            return out.shape[1]
        return ViTBody(arch_spec, pretrained=False).num_features  # ViT: pooled embedding width
    from fastai.vision.all import num_features_model
    from fastai.vision.learner import create_body

    body = create_body(arch_spec(weights=None), n_in=3, pretrained=False)
    return num_features_model(body)


def build_backbone_model(arch_spec, custom_head: nn.Module, vit: bool = False) -> nn.Module:
    """``nn.Sequential(body, head)`` matching how training built the model, without DataLoaders.

    Used by the test / export reconstruction paths (they rebuild from a checkpoint alone). For a
    conv-map timm arch this defers to fastai's ``create_timm_model`` so the module tree is identical
    to training and the saved ``state_dict`` loads; for a ViT arch (``vit=True``) it pairs
    :class:`ViTBody` with the (already ``FlatHead``-wrapped) head, matching :func:`build_learner`.
    """
    from fastai.vision.learner import create_body, create_timm_model

    if isinstance(arch_spec, str):
        if vit:
            return nn.Sequential(ViTBody(arch_spec, pretrained=False), custom_head)
        model, _ = create_timm_model(arch_spec, n_out=1, pretrained=False, custom_head=custom_head)
        return model
    body = create_body(arch_spec(weights=None), n_in=3, pretrained=False)
    return nn.Sequential(body, custom_head)


def build_learner(dls, arch_spec, custom_head, loss_func, metrics, model_dir, cbs, optimizer="muon", vit=False):
    """Assemble a fastai ``Learner`` for the hierarchical head.

    ``init=None`` because the cosine head initializes itself (``weight_norm`` + spherical
    repulsion); letting fastai's default kaiming sweep run would fight that init. Muon needs its
    ``opt_func`` (backbone→Muon, head→AdamW).

    Conv-map backbones go through ``vision_learner`` (which builds the timm/torchvision body and
    attaches ``custom_head``). A **ViT backbone** (``vit=True``) can't use that path — its body has
    no conv map for a pooling head — so the model is assembled manually as
    ``nn.Sequential(ViTBody(pretrained), custom_head)`` (the head already ``FlatHead``-wrapped) and
    handed to a plain ``Learner``. Muon still works (it re-partitions params either way).
    """
    from fastai.vision.all import Learner, vision_learner

    learner_kwargs = {}
    if optimizer == "muon":
        from .optim import muon_opt_func

        learner_kwargs["opt_func"] = muon_opt_func

    if vit:
        model = nn.Sequential(ViTBody(arch_spec, pretrained=True), custom_head)
        return Learner(dls, model, loss_func=loss_func, metrics=metrics,
                       model_dir=Path(model_dir), cbs=cbs, **learner_kwargs)

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
