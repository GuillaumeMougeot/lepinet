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

Reasoning and measured numbers: journal/2026-09-29-public-hf-release.md.
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
LEVELS = ["species", "genus", "family"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load(ckpt_path: str, img_size: int):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt["model_arch_name"] == "bioclip2":
        importlib.import_module("075_pretrained_trunk").install()
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


def taxonomy_for(ckpt: dict, meta: dict, parquet: str) -> dict:
    """Multi-level checkpoints: keep their own coarse vocab order (their coarse logits index it)."""
    if len(meta["levels"]) == 3:
        from lepinet.export import build_taxonomy
        tax = build_taxonomy(ckpt, meta, level_names=LEVELS)
        tax.pop("note", None)
        return tax
    return build_full_taxonomy([str(v) for v in meta["vocabs"]["speciesKey"]], parquet)


# ---------------------------------------------------------------------------
# The release graph
# ---------------------------------------------------------------------------

class ReleaseWrapper(nn.Module):
    """[0,1] RGB -> raw logits (per trained head) + coherent probabilities + embedding."""

    def __init__(self, model: nn.Module, tax: dict, n_heads: int, half_body: bool = False,
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        from lepinet.infer import IMAGENET_MEAN, IMAGENET_STD
        self.body, self.wrap = model[0], model[1]
        self.head = self.wrap.head
        self.pool = getattr(self.wrap, "pool", None)
        self.n_heads = n_heads
        self.half_body = half_body
        if half_body:
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
    tax = taxonomy_for(ckpt, meta, a.parquet)
    (out / "taxonomy.json").write_text(json.dumps(tax))
    from lepinet.calibrate import build_names
    build_names(a.parquet, out / "taxonomy.json", out / "names.json")

    n_heads = len(meta["levels"])
    names = output_names(n_heads)
    x = torch.rand(2, 3, a.img_size, a.img_size)
    # nn.MultiheadAttention's eval fast path emits aten::_native_multi_head_attention, which has no
    # ONNX symbolic; the slow path is numerically identical and exports.
    torch.backends.mha.set_fastpath_enabled(False)
    ref = None
    # --no-fp16: onnxruntime 1.27 segfaults at session creation (ORT_ENABLE_ALL only; BASIC and
    # EXTENDED load fine) on the fp16 BioCLIP-2 graph, with or without an fp32 patch-embedding conv.
    # A file that crashes a default InferenceSession is not publishable, so the ViT ships fp32 only.
    for precision in ("fp32",) if a.no_fp16 else ("fp32", "fp16"):
        w = ReleaseWrapper(model, tax, n_heads, half_body=(precision == "fp16"),
                           temperature=a.temperature).eval()
        path = out / ("model.onnx" if precision == "fp32" else "model_fp16.onnx")
        with torch.no_grad():
            torch.onnx.export(w, (x,), str(path), input_names=["image"], output_names=names,
                              dynamic_axes={"image": {0: "batch"}, **{n: {0: "batch"} for n in names}},
                              opset_version=17, do_constant_folding=True, dynamo=False)
            if ref is None:
                ref = [o.numpy() for o in w(x)]
        import onnxruntime as ort
        got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
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
        "species_temperature": a.temperature,
        "preprocessing": {"resize_shorter_side": a.img_size, "center_crop": a.img_size,
                          "input": "float32 NCHW, RGB, values in [0, 1]; normalisation is inside the graph"},
        "inputs": {"image": ["batch", 3, a.img_size, a.img_size]},
        "output_names": names,
        "n_classes": {lv: len(tax["vocabs"][lv]) for lv in LEVELS},
        "gbif_species_url": "https://www.gbif.org/species/{key}",
        # lepinet-app bundle keys: the release folder is also a valid app bundle.
        "name": a.name, "model": "model.onnx" if a.no_fp16 else "model_fp16.onnx", "taxonomy": "taxonomy.json", "names": "names.json",
        "thresholds": "thresholds.json", "imageSize": a.img_size, "inputName": "image",
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

    import onnxruntime as ort
    import pandas as pd
    out = Path(a.out)
    cfg = json.loads((out / "config.json").read_text())
    tax = json.loads((out / "taxonomy.json").read_text())
    size = cfg["image_size"]
    df = pd.read_parquet(a.parquet)
    if a.test_set is not None:
        df = df[df["set"].astype(str) == a.test_set]
    df = df.reset_index(drop=True)
    idx = {lv: {k: i for i, k in enumerate(tax["vocabs"][lv])} for lv in LEVELS}
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(str(out / a.model), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    paths = [str(Path(a.img_dir) / p) for p in (df["image_path"] if "image_path" in df
                                                  else df["speciesKey"].astype(str) + "/" + df["filename"])]
    rows = {f"{p}_{lv}": [] for lv in LEVELS for p in ("pred", "conf")}
    rows.update({f"head_{p}_{lv}": [] for lv in LEVELS for p in ("pred", "conf")})
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(paths), a.batch):
            batch = np.stack(list(pool.map(lambda p: preprocess(p, size, a.resize), paths[i:i + a.batch])))
            res = dict(zip(names, sess.run(None, {"image": batch})))
            for lv in LEVELS:
                p = res[f"prob_{lv}"]
                rows[f"pred_{lv}"] += p.argmax(1).tolist()
                rows[f"conf_{lv}"] += p.max(1).tolist()
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

    import onnxruntime as ort
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
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(str(out / "model.onnx"), so, providers=["CPUExecutionProvider"])
    paths = [str(Path(a.img_dir) / k / f) for k, f in zip(df["speciesKey"], df["filename"])]
    y = df["speciesKey"].map(sp_idx).to_numpy()
    chunks = []
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(paths), 32):
            batch = np.stack(list(pool.map(lambda p: preprocess(p, cfg["image_size"]), paths[i:i + 32])))
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
    e.add_argument("--temperature", type=float, default=1.0,
                   help="divide the species logits by this inside the graph (from `calibrate`)")
    c = sub.add_parser("calibrate")
    c.add_argument("--out", required=True)
    c.add_argument("--parquet", required=True, help="in-distribution VALIDATION images, never test")
    c.add_argument("--img-dir", required=True)
    c.add_argument("--n", type=int, default=10000)
    c.add_argument("--set", default="1")
    c.add_argument("--threads", type=int, default=24)
    v = sub.add_parser("eval")
    v.add_argument("--out", required=True)
    v.add_argument("--parquet", required=True)
    v.add_argument("--img-dir", required=True)
    v.add_argument("--name", required=True)
    v.add_argument("--model", default="model.onnx")
    v.add_argument("--test-set", default=None)
    v.add_argument("--resize", default="bicubic", choices=["bicubic", "bilinear", "fastai"])
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
    {"export": cmd_export, "calibrate": cmd_calibrate, "eval": cmd_eval, "thresholds": cmd_thresholds, "upload": cmd_upload}[a.cmd](a)


if __name__ == "__main__":
    main()
