"""ONNX export + marginalize tests on a synthetic tiny model (no dataset, no checkpoint)."""

import pytest
import torch
import torch.nn as nn

from lepinet.export import ExportWrapper, marginalize, scatter_logsumexp, verify_onnx
from lepinet.heads import build_head

ort = pytest.importorskip("onnxruntime")


def _tiny_model(n=(6, 3, 2)):
    body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())  # -> [N,8,H,W]
    head = build_head("independent", 8, list(n), hidden=True)
    return nn.Sequential(body, head)


def test_onnx_export_dynamo_false_parity(tmp_path):
    """The clean head must trace with the legacy exporter (dynamo=False) and match ORT."""
    model = _tiny_model().eval()
    wrapper = ExportWrapper(model).eval()
    out_names = ["logits_0", "logits_1", "logits_2"]
    onnx_path = tmp_path / "m.onnx"
    dummy = torch.rand(1, 3, 16, 16)
    torch.onnx.export(
        wrapper, (dummy,), str(onnx_path),
        input_names=["image"], output_names=out_names,
        dynamic_axes={"image": {0: "batch"}, **{n: {0: "batch"} for n in out_names}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    assert verify_onnx(wrapper, onnx_path, img_size=16, output_names=out_names, batch=2)


def test_marginalize_is_consistent():
    tax = {
        "levels": ["speciesKey", "genusKey", "familyKey"],
        "vocabs": {"speciesKey": list(range(6)), "genusKey": list(range(3)), "familyKey": list(range(2))},
        "parents": {"speciesKey_to_genusKey": [0, 0, 1, 1, 2, 2], "genusKey_to_familyKey": [0, 0, 1]},
    }
    out = marginalize(torch.randn(4, 6), tax)
    assert [o.shape[1] for o in out] == [6, 3, 2]
    for o in out:  # each level's probabilities sum to 1
        assert torch.allclose(o.exp().sum(1), torch.ones(4), atol=1e-5)


def test_scatter_logsumexp_matches_bruteforce():
    log_probs = torch.randn(3, 5)
    parent = torch.tensor([0, 0, 1, 1, 1])
    got = scatter_logsumexp(log_probs, parent, 2)
    exp0 = torch.logsumexp(log_probs[:, [0, 1]], dim=1)
    exp1 = torch.logsumexp(log_probs[:, [2, 3, 4]], dim=1)
    assert torch.allclose(got[:, 0], exp0, atol=1e-5)
    assert torch.allclose(got[:, 1], exp1, atol=1e-5)
