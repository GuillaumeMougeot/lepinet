"""P1 — a strong biological encoder as a frozen trunk, with our classifier stages on top.

The question T2b forced. Classifier-only adaptation worked from a trunk that had never seen the
target domain in any form, which means **adaptation needs nothing from the representation** -- and if
it needs nothing from ours, it does not need *ours*. So the pipeline may reduce to "take the best
available encoder, fit a cheap classifier, adapt it", which is a stronger and more portable claim
than anything about our backbone.

BioCLIP-2 (`imageomics/bioclip-2`) is the obvious encoder to test: a ViT-L/14 CLIP model trained on
TreeOfLife-200M, i.e. on this exact domain at a scale we cannot match. Reviewers will ask for it.

    python dev/075_pretrained_trunk.py selftest                      # loads the encoder, checks widths
    python dev/075_pretrained_trunk.py train configs/<cfg>.yaml [--freeze-body] [--init-from ...]
    python dev/075_pretrained_trunk.py test --model ... --parquet ...

## How it attaches without touching the package

`lepinet.model.build_learner` builds a ViT backbone as `ViTBody(arch_spec, pretrained=True)`, and the
test/export rebuild path calls the same class. **`ViTBody` is therefore the only seam needed**: this
module swaps it for a factory that returns a `BioCLIPBody` when the arch name is the sentinel
`bioclip2`, and defers to the original otherwise. `resolve_arch` / `arch_is_vit` /
`arch_body_features` are patched alongside so the sentinel survives validation and reports the right
width. Nothing in `src/lepinet` changes, so no published number can drift.

## Two details that would silently handicap the encoder

**Normalisation.** `make_dls` hard-codes ImageNet statistics, and CLIP models use their own. Feeding
ImageNet-normalised pixels to a CLIP tower is not a crash -- it is a quiet few-point loss that would
look like "BioCLIP-2 is not that good". `BioCLIPBody` therefore **undoes** the ImageNet normalisation
and applies CLIP's, exactly, inside `forward`. That keeps the package's data path untouched and the
correction auditable in one place.

**Which feature.** open_clip's `visual(x)` returns the *projected* 768-d embedding, which is trained
for alignment with text. For transfer we want the pooled **pre-projection** feature (width 1024), so
`proj` is set to `None` by default. `--use-proj` selects the 768-d embedding instead; it is a real
question and one flag, not a rewrite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

SENTINEL = "bioclip2"
REPO = "imageomics/bioclip-2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class BioCLIPBody(nn.Module):
    """BioCLIP-2's vision tower as a lepinet backbone emitting one ``[N, C]`` embedding."""

    def __init__(self, repo: str = REPO, use_proj: bool = False):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(f"hf-hub:{repo}")
        self.visual = model.visual
        if not use_proj and getattr(self.visual, "proj", None) is not None:
            # Drop the text-alignment projection: for transfer we want the pooled tower output.
            self.visual.proj = None

        def buf(name, v):
            self.register_buffer(name, torch.tensor(v).view(1, 3, 1, 1), persistent=False)
        buf("in_mean", IMAGENET_MEAN); buf("in_std", IMAGENET_STD)
        buf("clip_mean", CLIP_MEAN); buf("clip_std", CLIP_STD)

        self.num_features = self._probe()

    @torch.no_grad()
    def _probe(self) -> int:
        was = self.training
        self.eval()
        size = getattr(self.visual, "image_size", 224)
        size = size[0] if isinstance(size, (tuple, list)) else size
        out = self(torch.zeros(1, 3, int(size), int(size)))
        self.train(was)
        if out.ndim != 2:
            raise RuntimeError(f"expected a [N, C] embedding, got {tuple(out.shape)}")
        return int(out.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The dataloader hands us ImageNet-normalised pixels. Undo that, then apply CLIP's stats.
        x = x * self.in_std + self.in_mean
        x = (x - self.clip_mean) / self.clip_std
        return self.visual(x)


def install(use_proj: bool = False) -> None:
    """Patch the four symbols that make `bioclip2` a first-class arch name."""
    from lepinet import model as M
    from lepinet import train as T

    orig_resolve, orig_is_vit, orig_feats, OrigViT = (
        M.resolve_arch, M.arch_is_vit, M.arch_body_features, M.ViTBody)
    cache: dict = {}

    def _body(spec=SENTINEL, pretrained: bool = True):
        if spec != SENTINEL:
            return OrigViT(spec, pretrained=pretrained)
        if "b" not in cache:
            cache["b"] = BioCLIPBody(use_proj=use_proj)
        return cache["b"]

    def resolve(name):
        return SENTINEL if name == SENTINEL else orig_resolve(name)

    def is_vit(spec, img_size: int = 256):
        return True if spec == SENTINEL else orig_is_vit(spec, img_size=img_size)

    def feats(spec, img_size: int = 256):
        return _body().num_features if spec == SENTINEL else orig_feats(spec, img_size=img_size)

    M.ViTBody = _body                      # build_learner and build_backbone_model both call this
    for mod in (M, T):
        mod.resolve_arch, mod.arch_is_vit, mod.arch_body_features = resolve, is_vit, feats


def selftest() -> None:
    """Runs on the cluster before the real job, because the encoder download is the risky step."""
    print(f"loading {REPO} ...")
    body = BioCLIPBody()
    print(f"  feature width (proj stripped): {body.num_features}")
    assert body.num_features == 1024, f"expected 1024, got {body.num_features}"

    projected = BioCLIPBody(use_proj=True)
    print(f"  feature width (projected):     {projected.num_features}")
    assert projected.num_features == 768

    # The renormalisation must be exact: ImageNet-normalised input -> CLIP-normalised internally.
    raw = torch.rand(2, 3, 224, 224)
    im = (raw - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    want = (raw - torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)) / torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    got = im * body.in_std + body.in_mean
    got = (got - body.clip_mean) / body.clip_std
    err = (got - want).abs().max().item()
    print(f"  renormalisation max error: {err:.2e}")
    assert err < 1e-5, err

    n = sum(p.numel() for p in body.parameters())
    print(f"  trunk parameters: {n/1e6:.1f} M")

    install()
    from lepinet.model import arch_body_features, arch_is_vit, resolve_arch
    assert resolve_arch(SENTINEL) == SENTINEL
    assert arch_is_vit(SENTINEL) is True
    assert arch_body_features(SENTINEL) == 1024
    print("  patch installed; `bioclip2` resolves, reports ViT, width 1024")
    print("selftest: OK")


def main(argv):
    if argv and argv[0] == "selftest":
        return selftest()
    use_proj = "--use-proj" in argv
    if argv and argv[0] == "test":
        install(use_proj=use_proj)
        from lepinet.cli import app
        print("P1 eval: bioclip2 trunk registered, delegating to `lepinet test`")
        return app(["test", *[a for a in argv[1:] if a != "--use-proj"]], standalone_mode=False)
    if len(argv) < 2 or argv[0] != "train":
        raise SystemExit(__doc__)

    install(use_proj=use_proj)
    from lepinet.config import load_config
    from lepinet.train import train_from_config

    cfg_path = argv[1]
    cfg, _ = load_config(cfg_path)
    if cfg.model_arch_name != SENTINEL:
        raise SystemExit(f"config arch is {cfg.model_arch_name!r}, expected {SENTINEL!r}")
    freeze = "--freeze-body" in argv
    init_from = argv[argv.index("--init-from") + 1] if "--init-from" in argv else None
    print(f"P1: trunk={REPO} use_proj={use_proj} freeze_body={freeze} init_from={init_from}")
    train_from_config(cfg_path, init_from=init_from, freeze_body=freeze)


if __name__ == "__main__":
    main(sys.argv[1:])
