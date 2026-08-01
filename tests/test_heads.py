"""Head / masks / hidden-inference unit tests. Synthetic — no dataset or GPU needed."""
from collections import OrderedDict

import pytest
import torch

from lepinet.heads import (
    IndependentHead,
    PooledHead,
    build_head,
    cosine_to_zscore,
    infer_hidden_from_state_dict,
    sparse_masks_from_labels,
)


def test_forward_shapes_and_finiteness():
    head = IndependentHead(in_features=32, n_classes=[6, 3, 2], hidden=True).eval()
    out = head(torch.randn(4, 32))
    assert [o.shape for o in out] == [torch.Size([4, 6]), torch.Size([4, 3]), torch.Size([4, 2])]
    assert all(torch.isfinite(o).all() for o in out)


def test_cosine_head_rows_are_unit_norm():
    head = IndependentHead(32, [6, 3, 2], hidden=True)
    for layer in head.layers:
        norms = layer.weight.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_bias_frozen():
    head = IndependentHead(32, [6, 3, 2], hidden=True)
    for layer in head.layers:
        assert layer.bias.requires_grad is False


def test_hidden_int_sets_bottleneck():
    head = IndependentHead(64, [6, 3, 2], hidden=16)
    assert head.preclass_size == 16
    assert head.hidden.out_features == 16
    assert head(torch.randn(2, 64))[0].shape == torch.Size([2, 6])


def test_no_bottleneck_when_hidden_false():
    head = IndependentHead(24, [5, 2], hidden=False)
    assert head.hidden is None
    assert head(torch.randn(3, 24))[0].shape == torch.Size([3, 5])


def test_state_dict_is_minimal_and_roundtrips():
    head = IndependentHead(32, [6, 3, 2], hidden=True)
    sd = head.state_dict()
    # maximal clarity: no dead BatchNorm, no _extra_state, no mask buffers, no linear alias
    assert not any(k.endswith("_extra_state") for k in sd)
    assert not any("batch_norm" in k for k in sd)
    assert not any(k.startswith("mask_") for k in sd)
    assert not any(k.startswith("linear.") for k in sd)
    head2 = IndependentHead(32, [6, 3, 2], hidden=True)
    res = head2.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys


def test_pooled_head_pools_spatial_map():
    head = build_head("independent", 32, [6, 3, 2], hidden=True)
    assert isinstance(head, PooledHead)
    out = head(torch.randn(2, 32, 5, 5))  # [N,C,H,W]
    assert out[0].shape == torch.Size([2, 6])


def test_n_level_generalization_two_and_four_levels():
    for n in ([5, 2], [8, 4, 2, 1]):
        head = IndependentHead(16, n, hidden=True)
        out = head(torch.randn(3, 16))
        assert len(out) == len(n)
        assert [o.shape[1] for o in out] == n


def test_cosine_to_zscore_monotonic():
    x = torch.linspace(-0.9, 0.9, 50)
    z = cosine_to_zscore(x, 128)
    assert torch.all(z[1:] > z[:-1]), "z-score should increase with cosine similarity"


def test_infer_hidden_from_state_dict():
    head = IndependentHead(32, [6, 3, 2], hidden=True)
    sd = {f"head.{k}": v for k, v in head.state_dict().items()}
    assert infer_hidden_from_state_dict(sd, prefix="head.") is True
    head2 = IndependentHead(32, [6, 3, 2], hidden=8)
    sd2 = {f"head.{k}": v for k, v in head2.state_dict().items()}
    assert infer_hidden_from_state_dict(sd2, prefix="head.") == 8


def test_sparse_masks_from_labels():
    # kept as a utility for future hierarchical heads / taxonomy building
    cls2idx = {"0": {"a": 0, "b": 1, "c": 2}, "1": {"g1": 0, "g2": 1}}
    labels = OrderedDict([("a", ("a", "g1")), ("b", ("b", "g1")), ("c", ("c", "g2"))])
    masks = sparse_masks_from_labels(labels, cls2idx)
    assert masks[0].tolist() == [0, 0, 1]


def test_sparse_masks_conflict_raises():
    cls2idx = {"0": {"a": 0}, "1": {"g1": 0, "g2": 1}}
    labels = OrderedDict([("a1", ("a", "g1")), ("a2", ("a", "g2"))])
    with pytest.raises(ValueError):
        sparse_masks_from_labels(labels, cls2idx)


def test_marginalisation_is_coherent_but_not_argmax_consistent():
    """Marginalisation gives probabilistic coherence, NOT argmax agreement.

    The repo claimed the latter ("consistent by construction") in eight places including the paper's
    method section, and it is false: ``max`` and ``sum`` do not commute over a partition, so a
    confident species can be outvoted by many diffuse siblings of another genus. This pins the true
    property so the wrong one cannot creep back in.
    See journal/2026-08-01-marginalisation-is-not-argmax-consistent.md.
    """
    import torch

    # genus A = {s0}; genus B = {s1..s5}
    p_species = torch.tensor([[0.40, 0.12, 0.12, 0.12, 0.12, 0.12]])
    parent = torch.tensor([0, 1, 1, 1, 1, 1])
    p_genus = torch.zeros(1, 2).index_add_(1, parent, p_species)

    # coherence: the coarse posterior *is* the sum, and still a distribution
    assert torch.allclose(p_genus.sum(), torch.tensor(1.0))
    assert torch.allclose(p_genus, torch.tensor([[0.40, 0.60]]))

    # ...but the argmaxes disagree, which is exactly what "consistent by construction" denied
    top_species = p_species.argmax(1)
    assert parent[top_species].item() != p_genus.argmax(1).item()


def test_arcface_zscore_margin_is_correct_and_invertible():
    """Pin the ArcFace x z-score composition: the head emits Z(cos), the loss must recover cos,
    rotate the true class by m, and re-apply Z.

    Worth a test because the step everything rests on -- ``cos = sin(Z/sqrt(d-2))`` -- is only valid
    while the sine's argument stays inside +-pi/2. It does, *by construction*: cos in [-1,1] means
    arccos(-cos) in [0,pi], so |Z| <= sqrt(d-2)*pi/2 and the inverse can never fold back. That is a
    property of the transform rather than of the data, and this test says so out loud.
    """
    import math

    import torch

    from lepinet.heads import cosine_to_zscore
    from lepinet.loss import apply_arcface_margin_zscore

    d = 1280
    zvar = 1.0 / math.sqrt(d - 2)

    # invertible across the entire reachable range, with the bound where theory says it is
    cos = torch.linspace(-1 + 1e-6, 1 - 1e-6, 4001)
    z = cosine_to_zscore(cos, d)
    assert torch.allclose(torch.sin(z * zvar), cos, atol=1e-5)
    assert z.abs().max() <= math.sqrt(d - 2) * math.pi / 2 + 1e-3

    logits = cosine_to_zscore(torch.rand(8, 50) * 2 - 1, d)
    y = torch.randint(0, 50, (8,))
    rows = torch.arange(8)

    # m = 0 changes nothing
    assert torch.allclose(apply_arcface_margin_zscore(logits, y, 0.0, d), logits, atol=1e-3)

    out = apply_arcface_margin_zscore(logits, y, 0.3, d)
    # ...only the true class moves, and it moves *down* (that is the margin's entire purpose)
    assert torch.allclose(out.scatter(1, y[:, None], 0), logits.scatter(1, y[:, None], 0), atol=1e-3)
    assert bool((out[rows, y] < logits[rows, y]).all())

    # ...and it equals Z(cos(theta + m)) derived independently
    c = torch.sin(logits * zvar).clamp(-1 + 1e-7, 1 - 1e-7)
    ref = cosine_to_zscore(torch.cos(torch.acos(c) + 0.3).clamp(-1 + 1e-7, 1 - 1e-7), d)
    assert torch.allclose(out[rows, y], ref[rows, y], atol=2e-3)


def test_head_bias_stays_frozen_at_zero():
    """The z-score inverse assumes the emitted logit *is* Z(cos). A trainable bias would break that
    (Z + b can leave the invertible range), so the frozen-zero bias is load-bearing, not cosmetic."""
    from lepinet.heads import IndependentHead

    b = IndependentHead(64, [50], hidden=32).layers[0].bias
    assert float(b.abs().max()) == 0.0
    assert b.requires_grad is False
