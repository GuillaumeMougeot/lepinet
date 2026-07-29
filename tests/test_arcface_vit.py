"""ArcFace head/loss and ViT-backbone adapter unit tests. Synthetic — no dataset or GPU needed.

These cover the two modular extensions added for the 'bigger everything' scaling work: an
ArcFace metric-learning head (angular margin, training-only) and a ViT/DINOv3 backbone path.
Both are default-off, so a companion assertion checks the cosine baseline is untouched.
"""
import pytest
import torch
import torch.nn.functional as F

from lepinet.config import TrainConfig
from lepinet.heads import ArcFaceHead, FlatHead, PooledHead, build_head
from lepinet.loss import FastaiLossWrapper, MultiLevelCELoss, apply_arcface_margin

# --------------------------------------------------------------------------- ArcFace head

def test_arcface_head_state_dict_identical_to_independent():
    """ArcFace is structurally identical to the cosine head, so weights are interchangeable."""
    ind = build_head("independent", 16, [10, 5, 2], hidden=True)
    arc = build_head("arcface", 16, [10, 5, 2], hidden=True, scale=32.0, margin=0.3)
    assert isinstance(arc, PooledHead) and isinstance(arc.head, ArcFaceHead)
    assert set(ind.state_dict()) == set(arc.state_dict())
    # an independent checkpoint loads into an arcface head and vice-versa
    arc.load_state_dict(ind.state_dict(), strict=True)


def test_arcface_forward_is_scaled_cosine():
    head = ArcFaceHead(16, [10, 5, 2], hidden=True, scale=32.0).eval()
    out = head(torch.randn(4, 16))
    assert [tuple(o.shape) for o in out] == [(4, 10), (4, 5), (4, 2)]
    # |logit| <= scale (it is scale * cos θ, cos ∈ [-1, 1])
    assert all(o.abs().max() <= 32.0 + 1e-4 for o in out)


def test_arcface_margin_lowers_only_true_class():
    emb = F.normalize(torch.randn(4, 16), dim=1)
    w = F.normalize(torch.randn(10, 16), dim=1)
    logit = 32.0 * (emb @ w.t())
    tgt = torch.tensor([0, 1, 2, 3])
    marg = apply_arcface_margin(logit, tgt, margin=0.3, scale=32.0)
    assert (marg[range(4), tgt] < logit[range(4), tgt]).all()          # true class penalised
    off = torch.ones_like(logit, dtype=torch.bool)
    off[range(4), tgt] = False
    assert torch.allclose(marg[off], logit[off], atol=1e-4)             # others untouched


def test_arcface_margin_only_in_training_phase():
    """The margin is applied when logits carry grad (train), skipped under no_grad (valid)."""
    crit = MultiLevelCELoss([10, 5, 2], arc_scale=32.0, arc_margins=[0.3, 0.0, 0.0])
    lf = FastaiLossWrapper(crit)
    yb = [torch.randint(0, c, (4,)) for c in (10, 5, 2)]
    base = [torch.randn(4, c) for c in (10, 5, 2)]

    train_preds = [b.clone().requires_grad_(True) for b in base]
    l_train = lf(train_preds, *yb)
    with torch.no_grad():
        l_eval = lf([b.clone() for b in base], *yb)
    # same logits, but the margin makes the training loss strictly larger than the margin-free eval loss
    assert float(l_train.detach()) > float(l_eval)


def test_arcface_config_rejects_mixup():
    with pytest.raises(ValueError, match="arcface.*incompatible with mixup"):
        TrainConfig(parquet_path="p", img_dir="i", out_dir="o", model_name="m",
                    model_arch_name="efficientnet_v2_s", head="arcface",
                    arcface_margin=0.3, mixup=0.2)


# --------------------------------------------------------------------------- ViT backbone

def test_flat_head_wraps_when_pool_false():
    head = build_head("independent", 24, [8, 4], hidden=True, pool=False)
    assert isinstance(head, FlatHead)
    out = head(torch.randn(3, 24))              # already-pooled [N, C]
    assert [tuple(o.shape) for o in out] == [(3, 8), (3, 4)]


def test_arch_is_vit_distinguishes_conv_and_vit():
    from lepinet.model import arch_is_vit, resolve_arch

    assert arch_is_vit(resolve_arch("efficientnet_v2_s"), img_size=224) is False
    assert arch_is_vit(resolve_arch("vit_tiny_patch16_224"), img_size=224) is True


def test_vit_backbone_builds_and_roundtrips():
    """ViT body + flat head builds, forwards, and its state_dict strict-loads (test/export path)."""
    from lepinet.model import ViTBody, arch_body_features, build_backbone_model, resolve_arch

    arch = resolve_arch("vit_tiny_patch16_224")
    nf = arch_body_features(arch, img_size=224)
    assert nf == 192
    head = build_head("independent", nf, [10, 5, 2], hidden=True, pool=False)
    model = build_backbone_model(arch, head, vit=True).eval()
    assert isinstance(model[0], ViTBody)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert [tuple(o.shape) for o in out] == [(2, 10), (2, 5), (2, 2)]

    rebuilt = build_backbone_model(arch, build_head("independent", nf, [10, 5, 2], hidden=True, pool=False), vit=True)
    rebuilt.load_state_dict(model.state_dict(), strict=True)   # keys match "1.head." reconstruction


def test_zscore_arcface_head_matches_independent_forward():
    """With zscore=True the head emits z(cos θ) — identical to the plain cosine head's forward."""
    from lepinet.heads import ArcFaceHead, IndependentHead

    ind = IndependentHead(16, [10, 5], hidden=True).eval()
    arc = ArcFaceHead(16, [10, 5], hidden=True, zscore=True).eval()
    arc.load_state_dict(ind.state_dict(), strict=True)
    x = torch.randn(4, 16)
    for a, b in zip(ind(x), arc(x)):
        assert torch.allclose(a, b, atol=1e-5)


def test_zscore_inverse_is_exact():
    """The margin composition relies on z being invertible: cos θ = sin(z / sqrt(ndim-2))."""
    import math

    from lepinet.heads import cosine_to_zscore

    d = 256
    cos = torch.linspace(-0.99, 0.99, 21)
    rec = torch.sin(cosine_to_zscore(cos, d) / math.sqrt(d - 2.0))
    assert torch.allclose(cos, rec, atol=1e-5)


def test_zscore_margin_lowers_only_true_class_and_stays_finite():
    from lepinet.loss import apply_arcface_margin_zscore

    d = 256
    logit = torch.randn(4, 10) * 2.0                      # z-scale logits
    tgt = torch.tensor([0, 1, 2, 3])
    out = apply_arcface_margin_zscore(logit, tgt, margin=0.3, ndim=d)
    assert torch.isfinite(out).all()
    assert (out[range(4), tgt] < logit[range(4), tgt]).all()      # true class penalised
    off = torch.ones_like(logit, dtype=torch.bool)
    off[range(4), tgt] = False
    assert torch.allclose(out[off], logit[off], atol=1e-4)        # others untouched


def test_zscore_margin_zero_is_identity():
    from lepinet.loss import apply_arcface_margin_zscore

    logit = torch.randn(4, 10) * 2.0
    tgt = torch.randint(0, 10, (4,))
    out = apply_arcface_margin_zscore(logit, tgt, margin=0.0, ndim=256)
    assert torch.allclose(out, logit, atol=1e-4)
