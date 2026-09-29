"""Identify moths and butterflies (Lepidoptera) with a lepinet model: species, genus and family.

Needs only:  pip install onnxruntime pillow numpy huggingface_hub

    python predict.py moth.jpg                       # uses the repo this file came from
    python predict.py *.jpg --repo gmougeot/lepinet-effnetv2s --top 3

Each image gets the deepest rank the model is confident about (species, else genus, else family,
else "unknown"), using the precision-targeted thresholds in thresholds.json. Crop to one insect
first: the models were trained on single-specimen images, not whole scenes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

LEVELS = ("species", "genus", "family")


def preprocess(path: str, size: int) -> np.ndarray:
    """Shorter side -> size, centre crop, RGB float32 in [0, 1], NCHW. Normalisation is in the graph."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


class Lepinet:
    def __init__(self, folder: str | Path, model_file: str = "model.onnx"):
        folder = Path(folder)
        self.cfg = json.loads((folder / "config.json").read_text())
        self.tax = json.loads((folder / "taxonomy.json").read_text())
        self.names = json.loads((folder / "names.json").read_text())["names"]
        thr = folder / "thresholds.json"
        self.thresholds = ({lv: v["threshold"] for lv, v in json.loads(thr.read_text())["levels"].items()}
                           if thr.exists() else {lv: 0.0 for lv in LEVELS})
        self.session = ort.InferenceSession(str(folder / model_file),
                                            providers=ort.get_available_providers())
        self.outputs = [o.name for o in self.session.get_outputs()]

    def __call__(self, paths: list[str], top: int = 1) -> list[dict]:
        batch = np.stack([preprocess(p, self.cfg["image_size"]) for p in paths])
        out = dict(zip(self.outputs, self.session.run(None, {"image": batch})))
        results = []
        for i, path in enumerate(paths):
            r = {"image": path, "answer": None}
            for lv in LEVELS:
                p = out[f"prob_{lv}"][i]
                best = np.argsort(-p)[:top]
                r[lv] = [{"name": self.names[lv][j] or self.tax["vocabs"][lv][j],
                          "gbif_key": self.tax["vocabs"][lv][j], "prob": round(float(p[j]), 4)}
                         for j in best]
                if r["answer"] is None and p[best[0]] >= self.thresholds[lv]:
                    r["answer"] = {"rank": lv, **r[lv][0]}
            results.append(r)
        return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--repo", default=None, help="Hugging Face repo id (default: this file's folder)")
    ap.add_argument("--model-file", default="model.onnx")
    ap.add_argument("--top", type=int, default=1)
    a = ap.parse_args()
    if a.repo:
        from huggingface_hub import snapshot_download
        folder = snapshot_download(a.repo, allow_patterns=[a.model_file, "*.json"])
    else:
        folder = Path(__file__).resolve().parent
    model = Lepinet(folder, a.model_file)
    for r in model(a.images, top=a.top):
        ans = r["answer"]
        head = (f"{ans['rank']}: {ans['name']} ({ans['prob']:.2f})  https://www.gbif.org/species/{ans['gbif_key']}"
                if ans else "unknown (below every threshold: maybe not a lepidopteran, or not in the label set)")
        print(f"{r['image']}\n  -> {head}")
        for lv in LEVELS:
            print(f"     {lv:8s} " + ", ".join(f"{c['name']} {c['prob']:.2f}" for c in r[lv]))


if __name__ == "__main__":
    main()
