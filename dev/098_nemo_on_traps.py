"""Nemo (mini_trainer's MAMBO-v3 release, mambo-v3 0.3.1) on lepinet's reconciled trap export, in lepinet's long format.

Nemo's model card scores an older, unreconciled export of the same trap crops. This scores it the way every
lepinet model is scored ([[2026-11-03-moving-the-recipe-into-mini-trainer]], ID VN): one forward pass per view
recipe over the 47,905 crops of B1-shift's predictions.csv (same instance_id, filename and truth), written to
data/ucloud_preds/NEMO-{shift,probe,probeho}/<variant>/eval/predictions.csv, then re-scored with dev/093
(--grouped; probe-HO from NEMO-probe with --heldout). The full-vocabulary result is what predict_stream returns
with model="full"; the north_europe result is derived from the same leaf logits with the package's own
HierarchyPlan and Prediction, which is what predict_stream does for a preset. Levels 0-2 (species, genus,
family); genus and family truth from data/flemming/name2id.csv. threshold = 0, confidence = Nemo's top-1
probability at that rank, known_label = truth in Nemo's full vocabulary.

Runs in its own venv, never lepinet's (hand-managed; `uv sync` there breaks torch):

    uv venv /tmp/nemo-venv && uv pip install --python /tmp/nemo-venv "mambo-v3[onnx-cuda,hub]==0.3.1"
    /tmp/nemo-venv/bin/python -I dev/098_nemo_on_traps.py --root . --out data/ucloud_preds \
        --variants full=nemo-full ne=nemo-northeurope                      # single view
    ... --tta rotation30_pad25_3 --variants full=nemo-full-tta ne=nemo-northeurope-tta
"""
import argparse
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from mambo_deploy import Predictor
from mambo_deploy.results import Prediction

B1 = "lepinet-B1-trapaug-singlehead-arcface-effnetv2s-ucloud/eval/predictions.csv"


def long_format(base, pred, conf, vocab):
    """Rows interleaved per crop (levels 0, 1, 2), in B1's crop order."""
    n = len(base)
    truth = np.stack([base.label.astype(str), base.genus.astype(str), base.family.astype(str)], axis=1)
    out = pd.DataFrame({
        "instance_id": np.repeat(base.instance_id.values, 3),
        "filename": np.repeat(base.filename.values, 3),
        "level": np.tile([0, 1, 2], n),
        "label": truth.reshape(-1).astype(np.int64),
        "prediction": pred.reshape(-1).astype(np.int64),
        "confidence": conf.reshape(-1),
        "threshold": 0.0,
        "known_label": [t in vocab[lvl] for t, lvl in zip(truth.reshape(-1), np.tile([0, 1, 2], n))],
    })
    return out


def main(a):
    root = Path(a.root)
    base = pd.read_csv(root / "data/ucloud_preds/B1-shift" / B1, usecols=["instance_id", "filename", "level", "label"])
    assert (base.level == 0).all() and base.filename.is_unique and len(base) == 47905
    n2i = pd.read_csv(root / "data/flemming/name2id.csv").set_index("verbatimScientificName")
    folder = base.filename.str.split("/").str[0]
    assert (n2i.loc[folder, "speciesKey"].values == base.label.values).all()
    base["genus"] = n2i.loc[folder, "genusKey"].values
    base["family"] = n2i.loc[folder, "familyKey"].values
    if a.limit:
        base = base.iloc[: a.limit]
    paths = [str(root / "data/flemming/images" / f) for f in base.filename]

    p = Predictor(backend="onnx", device=a.device, model="full", precision=a.precision, tta=a.tta,
                  batch_size=a.batch_size, threads=a.threads, preprocess_workers=a.threads).load()
    ne = Predictor(bundle=p.bundle, backend="onnx", device=a.device, model="north_europe")  # selection only
    plans = {"full": None, "ne": p.hierarchy_plan(ne.selected)}
    variants = dict(v.split("=") for v in a.variants)
    vocab = [set(r) for r in p.bundle.classes["labels"]]
    n = len(paths)
    pred = {k: np.empty((n, 3), dtype=object) for k in variants}
    conf = {k: np.empty((n, 3), dtype=np.float32) for k in variants}
    print("metadata", p.metadata | {"cls2idx": None}, flush=True)
    t, off = time.time(), 0
    with closing(p.predict_stream(paths, read_workers=a.threads)) as batches:
        for r in batches:
            m = len(r)
            leaf = r.raw_logits[0]          # full plan: the (view-averaged) leaf logits, unchanged
            assert leaf.shape[1] == len(vocab[0])
            for k in variants:
                res = r if plans[k] is None else Prediction(*plans[k].numpy(leaf), 1)
                pred[k][off: off + m] = np.array(res.labels, dtype=object)[:, 0, :]
                conf[k][off: off + m] = res.confidence[:, 0, :]
            off += m
            if off // 4096 != (off - m) // 4096:
                print(f"{off}/{n} {off / (time.time() - t):.0f} img/s", flush=True)
    assert off == n, (off, n)
    print(f"done {n} crops in {time.time() - t:.0f}s", flush=True)

    out_root = Path(a.out)
    subsets = {"NEMO-shift": None, "NEMO-probe": "B1-probe", "NEMO-probeho": "B1-probeho"}
    for k, name in variants.items():
        full = long_format(base, pred[k], conf[k], vocab)
        for folder_name, ref in subsets.items():
            if ref is None:
                d = full
            else:   # the probe crop set, matched by filename, with the reference folder's instance_id and order
                r = pd.read_csv(root / "data/ucloud_preds" / ref / B1, usecols=["instance_id", "filename", "label"])
                if a.limit:
                    r = r[r.filename.isin(set(base.filename))]
                sub = full.drop(columns="instance_id").merge(r.rename(columns={"label": "ref_label"}), on="filename", how="inner")
                assert sub.filename.nunique() == len(r), (folder_name, sub.filename.nunique(), len(r))
                assert (sub[sub.level == 0].label.values == sub[sub.level == 0].ref_label.values).all()
                order = pd.Series(np.arange(len(r)), index=r.filename)
                sub = sub.assign(_o=order.loc[sub.filename].values).sort_values(["_o", "level"], kind="stable")
                d = sub[full.columns]
            dest = out_root / folder_name / name / "eval"
            dest.mkdir(parents=True, exist_ok=True)
            d.to_csv(dest / "predictions.csv", index=False)
            print(f"{dest / 'predictions.csv'}: {d.instance_id.nunique()} crops, {len(d)} rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help="folder holding NEMO-shift, NEMO-probe, NEMO-probeho")
    ap.add_argument("--tta", default="none", help="none, or a mambo_deploy TTA profile (rotation30_pad25_3 is the default recipe)")
    ap.add_argument("--variants", nargs="+", required=True, help="full=<folder> and/or ne=<folder>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
