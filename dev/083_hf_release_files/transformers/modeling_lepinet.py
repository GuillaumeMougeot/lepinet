"""lepinet for transformers: species / genus / family of a moth or butterfly from one image.

Self-contained (torch + transformers only, no open_clip, no lepinet). Parameter names mirror the
lepinet checkpoint -- an open_clip ViT under ``0.visual`` and lepinet's cosine head under ``1.head`` --
so the repository's single ``model.safetensors`` serves transformers, lepinet and the ONNX export.

    from transformers import pipeline
    clf = pipeline("image-classification", model="gmougeot/lepinet-bioclip2-vitl14", trust_remote_code=True)
    clf("moth.jpg", top_k=3)

or, for all three ranks at once, ``LepinetForImageClassification.predict``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_lepinet import LepinetConfig

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover
    from torch.nn.utils import weight_norm


# ---------------------------------------------------------------------------
# open_clip-compatible ViT image tower (the BioCLIP-2 ViT-L/14 layout)
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.c_fc = nn.Linear(width, hidden)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden, width)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class _ResBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = _MLP(width, int(width * mlp_ratio))

    def forward(self, x):
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class _Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.resblocks = nn.ModuleList(_ResBlock(width, heads, mlp_ratio) for _ in range(layers))

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


class _VisionTransformer(nn.Module):
    def __init__(self, c: LepinetConfig):
        super().__init__()
        grid = c.image_size // c.patch_size
        self.conv1 = nn.Conv2d(3, c.width, kernel_size=c.patch_size, stride=c.patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(c.width))
        self.positional_embedding = nn.Parameter(torch.zeros(grid * grid + 1, c.width))
        self.ln_pre = nn.LayerNorm(c.width)
        self.transformer = _Transformer(c.width, c.layers, c.heads, c.mlp_ratio)
        self.ln_post = nn.LayerNorm(c.width)

    def forward(self, x):
        x = self.conv1(x).flatten(2).transpose(1, 2)                      # [N, grid^2, width]
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.transformer(self.ln_pre(x))
        return self.ln_post(x[:, 0])                                      # class-token pooling


class _Body(nn.Module):
    def __init__(self, c: LepinetConfig):
        super().__init__()
        self.visual = _VisionTransformer(c)


# ---------------------------------------------------------------------------
# lepinet's cosine head
# ---------------------------------------------------------------------------

def _cosine_to_zscore(cosine: torch.Tensor, ndim: int) -> torch.Tensor:
    """``sqrt(ndim - 2) * (acos(-cos) - pi/2)``: the head's calibrated transform of a cosine."""
    return (torch.acos(-cosine.clamp(-1 + 1e-7, 1 - 1e-7)) - math.pi / 2) * math.sqrt(ndim - 2.0)


class _CosineHead(nn.Module):
    def __init__(self, c: LepinetConfig):
        super().__init__()
        self.hidden = nn.Linear(c.width, c.head_hidden)
        self.dropout = nn.Dropout(0.0)
        self.layers = nn.ModuleList(weight_norm(nn.Linear(c.head_hidden, n), name="weight", dim=0)
                                    for n in c.n_classes)
        self.ndim = c.head_hidden

    def embed(self, x):
        return F.normalize(F.leaky_relu(self.hidden(self.dropout(x))), dim=-1)

    def forward(self, emb):
        return [_cosine_to_zscore(F.linear(emb, layer.weight), self.ndim) + layer.bias for layer in self.layers]


class _HeadWrap(nn.Module):
    def __init__(self, c: LepinetConfig):
        super().__init__()
        self.head = _CosineHead(c)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class LepinetOutput(ModelOutput):
    """``logits`` are the species logits (temperature applied), so ``softmax(logits) == prob_species``."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    prob_species: torch.FloatTensor | None = None
    prob_genus: torch.FloatTensor | None = None
    prob_family: torch.FloatTensor | None = None
    embedding: torch.FloatTensor | None = None


class LepinetPreTrainedModel(PreTrainedModel):
    config_class = LepinetConfig
    base_model_prefix = "lepinet"
    main_input_name = "pixel_values"
    _no_split_modules = ["_ResBlock"]

    def _init_weights(self, module):  # weights always come from the checkpoint
        pass


class LepinetForImageClassification(LepinetPreTrainedModel):
    def __init__(self, config: LepinetConfig):
        super().__init__(config)
        # Named "0" and "1" so the parameter names are the lepinet checkpoint's own.
        self.add_module("0", _Body(config))
        self.add_module("1", _HeadWrap(config))
        self._parents = {}
        self.post_init()

    def _parent_index(self, name: str, device) -> torch.Tensor:
        # Built from the config on first use, not registered as buffers: transformers >= 5 creates the
        # model on the meta device and leaves non-persistent buffers uninitialised.
        key = (name, str(device))
        if key not in self._parents:
            self._parents[key] = torch.tensor(getattr(self.config, name), dtype=torch.long, device=device)
        return self._parents[key]

    def _sum_children(self, p, name, n):
        return torch.zeros(p.shape[0], n, dtype=p.dtype, device=p.device).index_add_(
            1, self._parent_index(name, p.device), p)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None, **kwargs) -> LepinetOutput:
        """``pixel_values``: CLIP-normalised RGB, ``[N, 3, 224, 224]`` (what the image processor gives)."""
        feats = getattr(self, "0").visual(pixel_values)
        head = getattr(self, "1").head
        emb = head.embed(feats.float())
        logits = head(emb)[0] / self.config.temperature
        p_sp = logits.softmax(-1)
        p_g = self._sum_children(p_sp, "species_to_genus", self.config.n_classes[1])
        p_f = self._sum_children(p_g, "genus_to_family", self.config.n_classes[2])
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return LepinetOutput(loss=loss, logits=logits, prob_species=p_sp, prob_genus=p_g,
                             prob_family=p_f, embedding=emb)

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor, top_k: int = 1) -> list[dict]:
        """Per image: the backed-off ``answer``, a ``novelty`` score, and the top-``k`` at each rank.

        ``answer`` is the deepest rank whose top probability clears its threshold (species, else genus,
        else family), or ``None`` for "unknown" -- the model is not confident enough at any rank.
        The thresholds (``config.thresholds``) target 95 % precision on held-out light-trap nights.
        ``novelty`` is the entropy of the species distribution: higher means less familiar, e.g. a
        species outside the label set. "Unknown" is not a guarantee of novelty, and a species answer
        is not a guarantee that the species is in the label set.
        """
        out = self(pixel_values)
        c = self.config
        thresholds = getattr(c, "thresholds", None) or {}
        ranks = {"species": (out.prob_species, [c.id2label[i] for i in range(len(c.id2label))], c.species_keys),
                 "genus": (out.prob_genus, c.genus_labels, c.genus_keys),
                 "family": (out.prob_family, c.family_labels, c.family_keys)}
        p_sp = out.prob_species
        novelty = -(p_sp * p_sp.clamp_min(1e-12).log()).sum(-1)
        results = []
        for i in range(pixel_values.shape[0]):
            r = {"answer": None, "novelty": round(float(novelty[i]), 4)}
            for rank, (p, names, keys) in ranks.items():
                val, idx = p[i].topk(top_k)
                r[rank] = [{"name": names[j], "gbif_key": keys[j] if keys else None, "prob": round(float(v), 4)}
                           for v, j in zip(val.tolist(), idx.tolist())]
                if r["answer"] is None and float(val[0]) >= thresholds.get(rank, 0.0):
                    r["answer"] = {"rank": rank, **r[rank][0]}
            results.append(r)
        return results
