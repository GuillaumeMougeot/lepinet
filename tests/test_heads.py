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
