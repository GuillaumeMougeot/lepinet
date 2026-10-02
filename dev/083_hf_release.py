"""Public Hugging Face release: checkpoint -> a self-describing, framework-free model repo.

The app bundles (`lepinet bundle`, `gmougeot/lepinet-models`) are shaped for one consumer, the PWA.
This script shapes the same checkpoints for *anyone*: a user with `onnxruntime`, `numpy` and
`Pillow` should get species / genus / family from a photo in about ten lines, without installing
lepinet, torch or open_clip. So the graph carries everything that is easy to get wrong:

* **normalisation** is baked in (input: RGB float32 in [0, 1], NCHW) -- the BioCLIP-2 trunk's
  ImageNet->CLIP re-normalisation included;
* **marginalisation** is baked in: `prob_genus` / `prob_family` are the sum of their species'
  probabilities (the project's recommended readout, dev/042), so the three levels are coherent;
* the L2-normalised **embedding** is exposed, for retrieval / open-set / few-shot use.

Raw `logits_<level>` are kept under their app names, so a release graph also drops into lepinet-app.

    python dev/083_hf_release.py export  --ckpt P5.pt --out rel/p5 --img-size 224 --parquet global.parquet
    python dev/083_hf_release.py eval    --out rel/p5 --parquet flemming_probe.parquet --img-dir ...
    python dev/083_hf_release.py thresholds --out rel/p5
    python dev/083_hf_release.py upload  --out rel/p5 --repo gmougeot/lepinet-bioclip2-vitl14

Reasoning and measured numbers: journal/archive/2026-09-29-public-hf-release.md.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

LEVEL_COLS = ["speciesKey", "genusKey", "familyKey"]
DEV050 = next(p.stem for p in Path(__file__).resolve().parent.glob("050_*.py"))
LEVELS = ["species", "genus", "family"]


def ort_session(path, provider: str = "cpu", threads: int | None = None):
    """An onnxruntime session that cannot hit the fp16 NCHWc crash.

    ORT 1.27-1.30's x86 ``NchwcTransformer`` (ORT_ENABLE_ALL only) segfaults at session creation
    on the fp16 ViT graphs, with or without a Conv in them; EXTENDED skips only layout transforms.
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    if "fp16" in Path(path).name and provider != "cuda":  # CUDA sessions are unaffected, and
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED  # EXTENDED slows them
    if provider == "cuda":
        ort.preload_dlls()
        # TF32 off: evaluation numbers must be the fp32 model's, not TF32's (logit diffs ~6e-3 on).
        providers = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=providers)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load(ckpt_path: str, img_size: int):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt["model_arch_name"] == "bioclip2":
        importlib.import_module("075_pretrained_trunk").install()
    from lepinet.heads import HEAD_REGISTRY
    if ckpt["head"] not in HEAD_REGISTRY:  # marginal / marginal_arcface / hierarchical live in dev/050
        importlib.import_module(DEV050)
    from lepinet.test import load_model
    model, meta = load_model(ckpt, img_size=img_size)
    return ckpt, model.eval(), meta


def build_full_taxonomy(species_vocab: list[str], parquet: str) -> dict:
    """species -> genus -> family from the dataset, in head-index order for species.

    Species-only checkpoints (B3rep5x) carry no coarse vocab, so the tree is rebuilt from the data.
    Genus/family order is sorted by key *as a string*, which is what the multi-level checkpoints use,
    so every release shares one taxonomy.json layout and the models are drop-in interchangeable.
    """
    import pandas as pd
    df = pd.read_parquet(parquet, columns=LEVEL_COLS).dropna().astype("int64").astype(str)
    df = df.drop_duplicates("speciesKey").set_index("speciesKey")
    missing = [s for s in species_vocab if s not in df.index]
    if missing:
        raise ValueError(f"{len(missing)} species have no genus/family in {parquet}: {missing[:5]}")
    sub = df.loc[species_vocab]
    genus_vocab = sorted(sub["genusKey"].unique())
    gfam = sub.drop_duplicates("genusKey").set_index("genusKey")["familyKey"]
    family_vocab = sorted(gfam.unique())
    gi = {g: i for i, g in enumerate(genus_vocab)}
    fi = {f: i for i, f in enumerate(family_vocab)}
    return {
        "levels": LEVELS,
        "vocabs": {"species": list(species_vocab), "genus": genus_vocab, "family": family_vocab},
        "parents": {"species_to_genus": [gi[g] for g in sub["genusKey"]],
                    "genus_to_family": [fi[gfam[g]] for g in genus_vocab]},
    }


def taxonomy_for(ckpt: dict, meta: dict, parquet: str, species_only: bool = False) -> dict:
    """Multi-level checkpoints: keep their own coarse vocab order (their coarse logits index it).

    ``species_only`` publishes only the species logits and rebuilds the tree from the data, in the
    shared order -- right for a head whose coarse outputs are themselves species marginals (B8's
    ``marginal_arcface``), which then carry no information the graph's own sums do not.
    """
    if len(meta["levels"]) == 3 and not species_only:
        from lepinet.export import build_taxonomy
        tax = build_taxonomy(ckpt, meta, level_names=LEVELS)
        tax.pop("note", None)
        return tax
    return build_full_taxonomy([str(v) for v in meta["vocabs"]["speciesKey"]], parquet)


# ---------------------------------------------------------------------------
# The release graph
# ---------------------------------------------------------------------------

class PatchLinear(nn.Module):
    """A ViT patch embedding (Conv2d with kernel == stride, no padding) as reshape + Linear.

    Mathematically the same map. It exists for onnxruntime: its x86 ``NchwcTransformer`` (enabled
    only at the default ORT_ENABLE_ALL level) segfaults at session creation on fp16 ViT graphs,
    in ORT 1.27 and 1.30, whether the graph comes from a half-precision export or from
    converting the fp32 graph. It rewrites only Conv nodes, so a graph without a Conv avoids it
    without users having to pass session flags.
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        k = conv.kernel_size
        assert k == conv.stride and conv.padding in ((0, 0), 0) and conv.groups == 1, "not a patchify conv"
        self.k, self.c_out = k, conv.out_channels
        self.proj = nn.Linear(conv.in_channels * k[0] * k[1], conv.out_channels, bias=conv.bias is not None)
        with torch.no_grad():
            self.proj.weight.copy_(conv.weight.reshape(conv.out_channels, -1))
            if conv.bias is not None:
                self.proj.bias.copy_(conv.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        kh, kw = self.k
        gh, gw = h // kh, w // kw
        x = x.reshape(n, c, gh, kh, gw, kw).permute(0, 2, 4, 1, 3, 5).reshape(n, gh * gw, c * kh * kw)
        return self.proj(x).permute(0, 2, 1).reshape(n, self.c_out, gh, gw)


def replace_patchify_conv(body: nn.Module) -> bool:
    """Swap a ViT's single patch-embedding conv for :class:`PatchLinear`; False if not a ViT."""
    convs = [(n, m) for n, m in body.named_modules() if isinstance(m, nn.Conv2d)]
    if len(convs) != 1 or convs[0][1].kernel_size != convs[0][1].stride:
        return False
    parent, _, child = convs[0][0].rpartition(".")
    setattr(body.get_submodule(parent) if parent else body, child, PatchLinear(convs[0][1]))
    return True


class ReleaseWrapper(nn.Module):
    """[0,1] RGB -> raw logits (per trained head) + coherent probabilities + embedding."""

    def __init__(self, model: nn.Module, tax: dict, n_heads: int, half_body: bool = False,
                 temperature: float = 1.0, model_size: int | None = None):
        super().__init__()
        self.temperature = float(temperature)
        self.model_size = model_size
        from lepinet.infer import IMAGENET_MEAN, IMAGENET_STD
        self.body, self.wrap = model[0], model[1]
        self.head = self.wrap.head
        self.pool = getattr(self.wrap, "pool", None)
        self.n_heads = n_heads
        self.half_body = half_body
        if half_body:
            replace_patchify_conv(self.body)
            self.body.half()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("s2g", torch.tensor(tax["parents"]["species_to_genus"], dtype=torch.long))
        self.register_buffer("g2f", torch.tensor(tax["parents"]["genus_to_family"], dtype=torch.long))
        self.n_genus = len(tax["vocabs"]["genus"])
        self.n_family = len(tax["vocabs"]["family"])

    @staticmethod
    def _sum_children(p: torch.Tensor, parent: torch.Tensor, n: int) -> torch.Tensor:
        idx = parent.unsqueeze(0).expand(p.shape[0], -1)
        return torch.zeros(p.shape[0], n, dtype=p.dtype, device=p.device).scatter_add(1, idx, p)

    def forward(self, image: torch.Tensor):
        if self.model_size:
            # The training pipeline's last step: fastai resamples the item-size image to the model
            # size on the GPU, bilinear, no antialias. Doing it in the graph lets users feed the item
            # size (P5: 256) and get the training-time pixels; input already at model_size passes
            # through unchanged (a same-size bilinear resample is the identity).
            image = F.interpolate(image, size=(self.model_size, self.model_size), mode="bilinear",
                                  align_corners=False, antialias=False)
        x = (image - self.mean) / self.std
        f = self.body(x.half() if self.half_body else x).float()
        if self.pool is not None:
            f = self.pool(f)
        logits = list(self.head(f))
        # Temperature is applied to the published species logits themselves, so that
        # softmax(logits_species) == prob_species for every consumer (the app included).
        logits[0] = logits[0] / self.temperature
        emb = self.head.preclassification(f)
        p_sp = torch.softmax(logits[0], dim=1)
        p_g = self._sum_children(p_sp, self.s2g, self.n_genus)
        p_f = self._sum_children(p_g, self.g2f, self.n_family)
        return (*logits[: self.n_heads], p_sp, p_g, p_f, emb)


def output_names(n_heads: int) -> list[str]:
    return [f"logits_{lv}" for lv in LEVELS[:n_heads]] + [f"prob_{lv}" for lv in LEVELS] + ["embedding"]


def cmd_export(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt, model, meta = load(a.ckpt, a.img_size)
    tax = taxonomy_for(ckpt, meta, a.parquet, a.species_only)
    (out / "taxonomy.json").write_text(json.dumps(tax))
    from lepinet.calibrate import build_names
    build_names(a.parquet, out / "taxonomy.json", out / "names.json")

    n_heads = 1 if a.species_only else len(meta["levels"])
    names = output_names(n_heads)
    in_size = a.input_size or a.img_size
    x = torch.rand(2, 3, in_size, in_size)
    # nn.MultiheadAttention's eval fast path emits aten::_native_multi_head_attention, which has no
    # ONNX symbolic; the slow path is numerically identical and exports.
    torch.backends.mha.set_fastpath_enabled(False)
    ref = None
    # fp16 ViTs used to segfault onnxruntime at session creation; see PatchLinear for the fix.
    for precision in ("fp32",) if a.no_fp16 else ("fp32", "fp16"):
        w = ReleaseWrapper(model, tax, n_heads, half_body=(precision == "fp16"),
                           temperature=a.temperature,
                           model_size=a.img_size if a.input_size else None).eval()
        path = out / ("model.onnx" if precision == "fp32" else "model_fp16.onnx")
        with torch.no_grad():
            torch.onnx.export(w, (x,), str(path), input_names=["image"], output_names=names,
                              dynamic_axes={"image": {0: "batch", **({2: "height", 3: "width"} if a.input_size else {})},
                                            **{n: {0: "batch"} for n in names}},
                              opset_version=17, do_constant_folding=True, dynamo=False)
            if ref is None:
                ref = [o.numpy() for o in w(x)]
        got = ort_session(path).run(None, {"image": x.numpy()})
        for n, r, g in zip(names, ref, got):
            print(f"  {precision} {n:16s} max|d| {np.abs(r - g).max():.2e}  top1 agree "
                  f"{(r.argmax(1) == g.argmax(1)).mean():.0%}")
        print(f"wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")


    # Full-precision weights for fine-tuning with lepinet (the ONNX graphs are for inference).
    from safetensors.torch import save_file
    sd = {k: v.contiguous() for k, v in ckpt["model_state_dict"].items()}
    meta_json = {k: ckpt[k] for k in ("head", "model_arch_name", "levels", "hidden", "vit",
                                      "arcface_scale", "arcface_zscore") if k in ckpt}
    save_file(sd, str(out / "model.safetensors"), metadata={"lepinet": json.dumps(meta_json)})

    cfg = {
        "architecture": a.arch_desc,
        "lepinet_checkpoint": meta_json,
        "image_size": a.img_size,
        "input_size": in_size,
        "resample": a.resample,
        "species_temperature": a.temperature,
        "preprocessing": {"resize_shorter_side": in_size, "center_crop": in_size, "resample": a.resample,
                          "input": "float32 NCHW, RGB, values in [0, 1]; normalisation is inside the graph"
                                   + (f"; resampled to {a.img_size} inside the graph, as in training"
                                      if a.input_size else "")},
        "inputs": {"image": ["batch", 3, in_size, in_size]},
        "output_names": names,
        "n_classes": {lv: len(tax["vocabs"][lv]) for lv in LEVELS},
        "gbif_species_url": "https://www.gbif.org/species/{key}",
        # lepinet-app bundle keys: the release folder is also a valid app bundle.
        "name": a.name, "model": "model.onnx" if a.no_fp16 else "model_fp16.onnx", "taxonomy": "taxonomy.json", "names": "names.json",
        "thresholds": "thresholds.json", "imageSize": in_size, "inputName": "image",
        "outputs": {"species": "logits_species"}, "gbifBase": "https://www.gbif.org/species/",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"release folder ready: {sorted(p.name for p in out.iterdir())}")


# ---------------------------------------------------------------------------
# Evaluation of the *published* artefact, with the *published* preprocessing
# ---------------------------------------------------------------------------

RESAMPLE = {"bicubic": 3, "bilinear": 2}  # PIL codes; "fastai" = two-stage, see below


def preprocess(path: str, size: int, mode: str = "bicubic") -> np.ndarray:
    """Exactly the snippet in the model card: shorter side -> size, centre crop, [0, 1].

    ``mode="fastai"`` reproduces the training pipeline's validation transform instead: PIL bilinear
    crop-resize to the item size (460 for 256 models, 256 for 224), then a bilinear GPU resample to
    the model size. Kept only to measure what the card's one-step resize costs.
    """
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if mode == "fastai":
        item = 460 if size == 256 else 256
        x = preprocess(path, item, "bilinear")
        t = F.interpolate(torch.from_numpy(x)[None], size=(size, size), mode="bilinear", align_corners=False)
        return t[0].numpy()
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), RESAMPLE[mode])
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def cmd_eval(a):
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    out = Path(a.out)
    cfg = json.loads((out / "config.json").read_text())
    tax = json.loads((out / "taxonomy.json").read_text())
    size = cfg.get("input_size", cfg["image_size"])
    mode = a.resize or cfg.get("resample", "bicubic")
    df = pd.read_parquet(a.parquet)
    if a.test_set is not None:
        df = df[df["set"].astype(str) == a.test_set]
    df = df.reset_index(drop=True)
    idx = {lv: {k: i for i, k in enumerate(tax["vocabs"][lv])} for lv in LEVELS}
    sess = ort_session(out / a.model, a.provider, a.threads)
    names = [o.name for o in sess.get_outputs()]
    paths = [str(Path(a.img_dir) / p) for p in (df["image_path"] if "image_path" in df
                                                  else df["speciesKey"].astype(str) + "/" + df["filename"])]
    rows = {f"{p}_{lv}": [] for lv in LEVELS for p in ("pred", "conf")}
    rows.update({f"head_{p}_{lv}": [] for lv in LEVELS for p in ("pred", "conf")})
    rows["entropy_species"] = []  # the novelty score the cards document
    with ThreadPoolExecutor(a.workers) as pool:
        for i in range(0, len(paths), a.batch):
            batch = np.stack(list(pool.map(lambda p: preprocess(p, size, mode), paths[i:i + a.batch])))
            res = dict(zip(names, sess.run(None, {"image": batch})))
            for lv in LEVELS:
                p = res[f"prob_{lv}"]
                rows[f"pred_{lv}"] += p.argmax(1).tolist()
                rows[f"conf_{lv}"] += p.max(1).tolist()
                if lv == "species":
                    rows["entropy_species"] += (-(p * np.log(np.clip(p, 1e-12, None))).sum(1)).tolist()
                if f"logits_{lv}" in res:
                    q = torch.softmax(torch.from_numpy(res[f"logits_{lv}"]), 1).numpy()
                    rows[f"head_pred_{lv}"] += q.argmax(1).tolist()
                    rows[f"head_conf_{lv}"] += q.max(1).tolist()
            print(f"\r{i + len(batch)}/{len(paths)}", end="", flush=True)
    print()
    res = pd.DataFrame({k: v for k, v in rows.items() if len(v) == len(df)})
    for lv, col in zip(LEVELS, LEVEL_COLS):
        res[f"label_{lv}"] = df[col].astype("int64").astype(str).map(idx[lv]).fillna(-1).astype(int)
    # Trap images: a (trap, night) group, so threshold fitting and verification never share a night.
    grp = df["filename"].astype(str).str.extract(r"TRAPNAME_(\w+?)_IMAGENAME_(\d{8})")
    if grp.notna().all().all():
        res["group"] = (grp[0] + "_" + grp[1]).values
    res.to_parquet(out / f"eval_{a.name}.parquet")
    summary = summarise(res)
    print(json.dumps(summary, indent=2))
    (out / f"eval_{a.name}.json").write_text(json.dumps(summary, indent=2))


def macro_f1(pred: np.ndarray, true: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    labels = np.unique(true[true >= 0])
    return float(f1_score(true, pred, labels=labels, average="macro", zero_division=0))


def summarise(res) -> dict:
    s = {"n": len(res)}
    for lv in LEVELS:
        t = res[f"label_{lv}"].to_numpy()
        s[f"{lv}_macro_f1"] = macro_f1(res[f"pred_{lv}"].to_numpy(), t)
        s[f"{lv}_top1"] = float((res[f"pred_{lv}"].to_numpy() == t).mean())
        if f"head_pred_{lv}" in res:
            s[f"{lv}_macro_f1_own_head"] = macro_f1(res[f"head_pred_{lv}"].to_numpy(), t)
    return s


def cmd_calibrate(a):
    """Fit one species temperature by NLL on in-distribution validation images.

    Run against a T = 1 export; the result goes back into `export --temperature`. Fitted on the
    validation fold, not the shifted probe, so probabilities mean what they say on data like the
    training data; the probe-fitted thresholds then carry the precision guarantee under shift.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    out = Path(a.out)
    cfg = json.loads((out / "config.json").read_text())
    if cfg.get("species_temperature", 1.0) != 1.0:
        raise SystemExit("calibrate must run on a T = 1 export")
    tax = json.loads((out / "taxonomy.json").read_text())
    sp_idx = {k: i for i, k in enumerate(tax["vocabs"]["species"])}
    df = pd.read_parquet(a.parquet, columns=["speciesKey", "filename", "set"])
    df = df[df["set"].astype(str) == a.set].dropna()
    df["speciesKey"] = df["speciesKey"].astype("int64").astype(str)
    df = df[df["speciesKey"].isin(sp_idx)].sample(a.n, random_state=0)
    sess = ort_session(out / "model.onnx", a.provider, a.threads)
    paths = [str(Path(a.img_dir) / k / f) for k, f in zip(df["speciesKey"], df["filename"])]
    y = df["speciesKey"].map(sp_idx).to_numpy()
    chunks = []
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(paths), 32):
            batch = np.stack(list(pool.map(
                lambda p: preprocess(p, cfg.get("input_size", cfg["image_size"]), cfg.get("resample", "bicubic")),
                paths[i:i + 32])))
            chunks.append(sess.run(["logits_species"], {"image": batch})[0])
            print(f"\r{i + len(batch)}/{len(paths)}", end="", flush=True)
    print()
    z = torch.from_numpy(np.concatenate(chunks)).double()
    yt = torch.from_numpy(y)
    grid = np.round(np.exp(np.linspace(np.log(0.25), np.log(8), 301)), 4)
    nll = [float(F.cross_entropy(z / t, yt)) for t in grid]
    t = float(grid[int(np.argmin(nll))])

    def ece(p, n_bins=15):
        conf, pred = p.max(1)
        ok = (pred == yt).double()
        bins = torch.clamp((conf * n_bins).long(), max=n_bins - 1)
        return float(sum((bins == b).double().mean() * abs(ok[bins == b].mean() - conf[bins == b].mean())
                         for b in range(n_bins) if (bins == b).any()))
    p1, pt = torch.softmax(z, 1), torch.softmax(z / t, 1)
    report = {"temperature": t, "n": len(y), "fold": a.set, "nll_T1": nll[int(np.argmin(np.abs(grid - 1)))],
              "nll_T": min(nll), "ece_T1": ece(p1), "ece_T": ece(pt),
              # saturation as the float32 graph will produce it
              "saturated_T1": float((torch.softmax(z.float(), 1).max(1).values >= 1.0).float().mean()),
              "saturated_T": float((torch.softmax(z.float() / t, 1).max(1).values >= 1.0).float().mean())}
    (out / "calibration_fit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Back-off thresholds, fitted on one half of the groups and verified on the other
# ---------------------------------------------------------------------------

def cascade(res, th: dict) -> dict:
    n = len(res)
    left = np.ones(n, bool)
    out = {}
    for lv in LEVELS:
        take = left & (res[f"conf_{lv}"].to_numpy() >= th[lv])
        ok = res[f"pred_{lv}"].to_numpy()[take] == res[f"label_{lv}"].to_numpy()[take]
        out[lv] = {"coverage": float(take.mean()), "precision": float(ok.mean()) if take.any() else None}
        out[f"_{lv}_ok"] = int(ok.sum())
        left &= ~take
    answered = 1 - left.mean()
    useful = sum(out.pop(f"_{lv}_ok") for lv in LEVELS) / n
    out["abstain"] = float(left.mean())
    out["precision_answered"] = float(useful / answered) if answered else None
    out["useful_rate"] = float(useful)
    return out


def fit_thresholds(res, target: float) -> dict:
    """Smallest threshold per rank whose precision, *conditional on reaching that rank*, meets target.

    Conditional is the point (journal 2026-08-28 O1 / paper 4.6): the genus posterior of an image
    that failed the species bar is much less reliable than genus precision over all images.
    """
    # Well-calibrated models put their 95 % point within 0.01 of 1, so the grid is fine near the top.
    grid = np.unique(np.concatenate([np.linspace(0, 0.99, 199), 1 - np.logspace(-2, -5, 31)]).round(6))
    th, left = {}, np.ones(len(res), bool)
    for lv in LEVELS:
        conf = res[f"conf_{lv}"].to_numpy()
        ok = res[f"pred_{lv}"].to_numpy() == res[f"label_{lv}"].to_numpy()
        th[lv] = 1.01  # never answer at this rank unless some threshold reaches the target
        for t in grid:
            m = left & (conf >= t)
            if m.sum() >= 30 and ok[m].mean() >= target:
                th[lv] = float(t)
                break
        left &= ~(conf >= th[lv])
    return th


def cmd_thresholds(a):
    import pandas as pd
    out = Path(a.out)
    res = pd.read_parquet(out / f"eval_{a.name}.parquet")
    res = res[res["label_species"] >= 0]
    groups = np.array(sorted(res["group"].unique(), key=str))
    rng = np.random.default_rng(0)
    fit_g = set(rng.permutation(groups)[: len(groups) // 2])
    fit, held = res[res["group"].isin(fit_g)], res[~res["group"].isin(fit_g)]
    th = fit_thresholds(fit, a.target)
    th_all = fit_thresholds(res, a.target)
    report = {
        "policy": "answer species if prob_species >= t_species, else genus if prob_genus >= t_genus, "
                  "else family if prob_family >= t_family, else abstain",
        "target_precision": a.target,
        "fitted_on": f"{a.name}: half of the capture groups ({len(fit)} images)",
        "verified_on": f"the other half ({len(held)} images)",
        "thresholds_split": th, "on_held_out_half": cascade(held, th),
        "levels": {lv: {"threshold": th_all[lv]} for lv in LEVELS},
        "on_all_in_sample": cascade(res, th_all),
    }
    (out / "thresholds.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def gemm_to_matmul(model):
    """Rewrite every Gemm with a constant weight as MatMul (+ Add), transposing the weight correctly.

    onnxruntime's quantizer (1.30) does this rewrite itself during pre-processing and ignores
    ``transB=1``. Our Gemms are all square (1024x1024 attention out-projections and bottlenecks,
    1280x1280), so the untransposed weight still type-checks and the quantized model silently
    computes garbage: 0 % agreement with fp32, first misread as CLIP activation outliers.
    Doing the rewrite ourselves first leaves the quantizer nothing to get wrong.
    """
    from onnx import helper, numpy_helper
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    nodes = []
    for n in g.node:
        if n.op_type != "Gemm" or n.input[1] not in inits:
            nodes.append(n)
            continue
        at = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        assert at.get("alpha", 1.0) == 1.0 and at.get("beta", 1.0) == 1.0 and not at.get("transA", 0), n.name
        wname = n.input[1] + "_mm"
        if wname not in inits:  # a weight shared by several Gemms is transposed once
            w = numpy_helper.to_array(inits[n.input[1]])
            inits[wname] = numpy_helper.from_array((w.T if at.get("transB", 0) else w).copy(), wname)
            g.initializer.append(inits[wname])
        mm_out = n.output[0] + "_mm" if len(n.input) > 2 else n.output[0]
        nodes.append(helper.make_node("MatMul", [n.input[0], wname], [mm_out], name=n.name + "_MatMul"))
        if len(n.input) > 2:
            nodes.append(helper.make_node("Add", [mm_out, n.input[2]], [n.output[0]], name=n.name + "_Add"))
    del g.node[:]
    g.node.extend(nodes)
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    del g.value_info[:]
    return model


def cmd_quantize(a):
    """model.onnx -> model_int8.onnx, the CPU file. One recipe per architecture, because what makes
    int8 fast and what breaks it differ (measured, journal 2026-09-29, follow-up):

    ``vit`` (also B8's ConvNeXt, whose MLPs are matmuls; B8: 1.46x per image, 3.5x smaller, within
        0.35 pt): dynamic int8 (per-channel weights, activations quantized on the fly) for every matmul
        except the classifier and the 24 MLP output projections (``c_proj``). ``c_proj`` reads the
        post-GELU activations, whose outliers per-tensor int8 cannot represent; it gets 8-bit
        *weight-only* quantization (MatMulNBits: float activations) instead. P5: 1.44x faster at batch
        32, 1.65x on one image, 4x smaller. Needs onnxruntime >= 1.22 to run and `onnx-ir` to build.
    ``cnn``: static QDQ int8, calibrated on validation images (never test), activations clipped at
        the 99.999th percentile (min/max and entropy calibration: 94 % agreement vs 98 %), classifier
        fp32. Dynamic int8 is the wrong tool for a CNN: onnxruntime's ConvInteger made B3rep5x 4x
        *slower*; static QDQ uses the fused QLinearConv kernels and is 2.2x faster.

    Either way the model is first passed through :func:`gemm_to_matmul`. int8 is a CPU format: on a
    GPU its integer ops fall back to the CPU (P5: 20 img/s vs 371 fp32); GPUs get the fp16 file.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    out = Path(a.out)
    cfg = json.loads((out / "config.json").read_text())
    size = cfg["image_size"]
    mm, dst = out / "_mm.onnx", out / "model_int8.onnx"
    onnx.save(gemm_to_matmul(onnx.load(str(out / "model.onnx"))), str(mm))
    names = [n.name for n in onnx.load(str(mm), load_external_data=False).graph.node]
    head = [n for n in names if n.startswith(("/head", "/hidden"))]
    try:
        if a.arch == "vit":
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
            # the MLP output projections: open_clip ViT `c_proj`, timm ConvNeXt `mlp/fc2`
            cproj = [n for n in names if "c_proj" in n or "/mlp/fc2" in n]
            quantize_dynamic(str(mm), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                             op_types_to_quantize=["MatMul"], nodes_to_exclude=head + cproj)
            q = MatMulNBitsQuantizer(onnx.load(str(dst)), bits=8, block_size=128, is_symmetric=True,
                                     accuracy_level=4, nodes_to_include=cproj)
            q.process()
            q.model.save_model_to_file(str(dst), use_external_data_format=False)
        else:
            import pandas as pd
            from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, quantize_static
            from onnxruntime.quantization.shape_inference import quant_pre_process
            pre = out / "_pre.onnx"
            quant_pre_process(str(mm), str(pre), skip_symbolic_shape=True)
            df = pd.read_parquet(a.calib_parquet, columns=["speciesKey", "filename", "set"])
            df = df[df["set"].astype(str) == "1"].dropna().sample(a.calib_n, random_state=1)
            paths = [str(Path(a.img_dir) / str(int(k)) / f) for k, f in zip(df.speciesKey, df.filename)]

            class Reader(CalibrationDataReader):
                def __init__(self):
                    self.it = (np.stack([preprocess(p, size) for p in paths[i:i + 16]])
                               for i in range(0, len(paths), 16))

                def get_next(self):
                    x = next(self.it, None)
                    return None if x is None else {"image": x}

            quantize_static(str(pre), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            calibrate_method=CalibrationMethod.Percentile,
                            extra_options={"CalibPercentile": 99.999},
                            op_types_to_quantize=["Conv", "MatMul", "Mul", "Add"], nodes_to_exclude=head)
            pre.unlink(missing_ok=True)
    finally:
        mm.unlink(missing_ok=True)
    print(f"model_int8.onnx {dst.stat().st_size / 1e6:.0f} MB -- score it with `eval --model model_int8.onnx`")


def cmd_transformers(a):
    """Make a ViT release loadable with transformers (``trust_remote_code=True``), in place.

    The repo keeps ONE ``model.safetensors``: ``modeling_lepinet.py`` mirrors the lepinet parameter
    names, so the same file serves lepinet, transformers and (via the ONNX export) everyone else.
    ``config.json`` gains the transformers keys and keeps every ONNX / app key it already had.
    """
    import shutil

    from safetensors.torch import load_file, save_file
    out = Path(a.out)
    src = Path(__file__).resolve().parent / "083_hf_release_files" / "transformers"
    for f in ("configuration_lepinet.py", "modeling_lepinet.py"):
        shutil.copy(src / f, out / f)
    cfg = json.loads((out / "config.json").read_text())
    tax = json.loads((out / "taxonomy.json").read_text())
    names = json.loads((out / "names.json").read_text())["names"]
    sp = [n or k for n, k in zip(names["species"], tax["vocabs"]["species"])]
    cfg.update({
        "model_type": "lepinet",
        "architectures": ["LepinetForImageClassification"],
        "auto_map": {"AutoConfig": "configuration_lepinet.LepinetConfig",
                     "AutoModelForImageClassification": "modeling_lepinet.LepinetForImageClassification"},
        "image_size": cfg["image_size"], "patch_size": 14, "width": 1024, "layers": 24, "heads": 16,
        "mlp_ratio": 4.0, "head_hidden": 1024,
        "n_classes": [len(tax["vocabs"][lv]) for lv in LEVELS],
        "temperature": cfg.get("species_temperature", 1.0),
        "species_to_genus": tax["parents"]["species_to_genus"],
        "genus_to_family": tax["parents"]["genus_to_family"],
        "species_keys": tax["vocabs"]["species"],
        "genus_labels": [n or k for n, k in zip(names["genus"], tax["vocabs"]["genus"])],
        "genus_keys": tax["vocabs"]["genus"],
        "family_labels": [n or k for n, k in zip(names["family"], tax["vocabs"]["family"])],
        "family_keys": tax["vocabs"]["family"],
        "id2label": {str(i): n for i, n in enumerate(sp)},
        "label2id": {n: i for i, n in enumerate(sp)},
        "torch_dtype": "float32",
        # the 95 %-precision back-off thresholds; `predict()` applies them (> 1 means "never")
        "thresholds": {lv: v["threshold"] for lv, v in
                       json.loads((out / "thresholds.json").read_text())["levels"].items()},
    })
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    in_size = cfg.get("input_size", cfg["image_size"])
    (out / "preprocessor_config.json").write_text(json.dumps({
        "image_processor_type": "CLIPImageProcessor",
        "do_resize": True, "size": {"shortest_edge": in_size},
        "resample": 2 if cfg.get("resample") == "bilinear" else 3,
        "do_center_crop": True, "crop_size": {"height": in_size, "width": in_size},
        "do_rescale": True, "rescale_factor": 1 / 255, "do_normalize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "do_convert_rgb": True}, indent=1))
    st = out / "model.safetensors"
    from safetensors import safe_open
    with safe_open(str(st), "pt") as f:
        meta = f.metadata() or {}
    if meta.get("format") != "pt":  # transformers refuses a safetensors file without it
        save_file(load_file(str(st)), str(st), metadata={**meta, "format": "pt"})
    print("transformers files written:", sorted(p.name for p in out.iterdir() if p.suffix in (".py", ".json")))


def cmd_novelty(a):
    """AUROC of species entropy for "is this image of a species outside the label set?".

    Known = an eval parquet of in-vocabulary images; novel = one of out-of-vocabulary images (the
    test-fold species below the training floor), both produced by `eval` with the published file.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    out = Path(a.out)
    k = pd.read_parquet(out / f"eval_{a.known}.parquet")["entropy_species"].to_numpy()
    n = pd.read_parquet(out / f"eval_{a.novel}.parquet")["entropy_species"].to_numpy()
    auc = roc_auc_score(np.r_[np.zeros(len(k)), np.ones(len(n))], np.r_[k, n])
    res = {"auroc_entropy": float(auc), "n_known": len(k), "n_novel": len(n), "known": a.known, "novel": a.novel}
    (out / f"novelty_{a.novel}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def cmd_upload(a):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    info = api.upload_folder(folder_path=a.out, repo_id=a.repo, commit_message=a.message,
                             ignore_patterns=["eval_*.parquet"])
    print(info)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--img-size", type=int, required=True)
    e.add_argument("--parquet", required=True, help="dataset parquet with the GBIF key/name columns")
    e.add_argument("--name", required=True)
    e.add_argument("--arch-desc", required=True)
    e.add_argument("--no-fp16", action="store_true")
    e.add_argument("--input-size", type=int, default=None,
                   help="publish at the training item size and resample to --img-size inside the graph")
    e.add_argument("--resample", default="bicubic", choices=["bicubic", "bilinear"],
                   help="PIL resample for the shorter-side resize the card documents")
    e.add_argument("--species-only", action="store_true",
                   help="publish species logits only; genus/family come from the shared taxonomy")
    e.add_argument("--temperature", type=float, default=1.0,
                   help="divide the species logits by this inside the graph (from `calibrate`)")
    c = sub.add_parser("calibrate")
    c.add_argument("--out", required=True)
    c.add_argument("--parquet", required=True, help="in-distribution VALIDATION images, never test")
    c.add_argument("--img-dir", required=True)
    c.add_argument("--n", type=int, default=10000)
    c.add_argument("--set", default="1")
    c.add_argument("--threads", type=int, default=24)
    c.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    v = sub.add_parser("eval")
    v.add_argument("--out", required=True)
    v.add_argument("--parquet", required=True)
    v.add_argument("--img-dir", required=True)
    v.add_argument("--name", required=True)
    v.add_argument("--model", default="model.onnx")
    v.add_argument("--test-set", default=None)
    v.add_argument("--resize", default=None, choices=["bicubic", "bilinear", "fastai"],
                   help="override the config's resample (default: what the card documents)")
    v.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    v.add_argument("--workers", type=int, default=8, help="image-decoding threads")
    nv = sub.add_parser("novelty")
    nv.add_argument("--out", required=True)
    nv.add_argument("--known", required=True)
    nv.add_argument("--novel", required=True)
    tr = sub.add_parser("transformers")
    tr.add_argument("--out", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--out", required=True)
    q.add_argument("--arch", required=True, choices=["vit", "cnn"])
    q.add_argument("--calib-parquet", help="cnn: dataset parquet; calibration uses set == '1' (validation)")
    q.add_argument("--img-dir", help="cnn: image root")
    q.add_argument("--calib-n", type=int, default=256)
    v.add_argument("--batch", type=int, default=32)
    v.add_argument("--threads", type=int, default=16)
    t = sub.add_parser("thresholds")
    t.add_argument("--out", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--target", type=float, default=0.95)
    u = sub.add_parser("upload")
    u.add_argument("--out", required=True)
    u.add_argument("--repo", required=True)
    u.add_argument("--message", default="lepinet release")
    a = p.parse_args()
    {"novelty": cmd_novelty, "export": cmd_export, "quantize": cmd_quantize, "transformers": cmd_transformers, "calibrate": cmd_calibrate, "eval": cmd_eval, "thresholds": cmd_thresholds, "upload": cmd_upload}[a.cmd](a)


if __name__ == "__main__":
    main()
