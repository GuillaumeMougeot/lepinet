"""Pseudo-label the trap `adapt` split for B3 (self-training), with a strict confidence gate.

Self-training's failure mode is circular: the model labels images with what it already believes,
trains on that, and becomes more confident in the same errors. Nothing in the loss can detect it.
The only defences are (a) a **strict** gate so that only labels the model is nearly certain of enter
training, and (b) evaluating somewhere the training data cannot reach — which
`dev/064`'s grouped `probe` split provides.

**No trap labels are used.** The dataset happens to be labelled, and this script *reports*
pseudo-label accuracy as a diagnostic, because knowing that the retained labels are (say) 92 %
correct is what makes the downstream result interpretable. That number is printed and journalled and
never written into the training parquet.

Output paths are rewritten as ``../../flemming/images/<...>`` so a single ``img_dir`` pointed at the
global set reaches both image trees — verified against both loader paths (`ColReader` and the
`lowmem` string concat) rather than assumed.

    python dev/065_pseudolabel.py --model '...*.pt' --adapt-parquet ... --img-dir ... \\
        --out pseudo.parquet --keep-frac 0.30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def predict_species(model, dls, df, device, num_workers=32):
    """Top-1 species index and its softmax probability, for every image."""
    from lepinet.test import dl_num_workers

    nw = num_workers if num_workers is not None else dl_num_workers(dls.train)
    dl = dls.test_dl(df, num_workers=nw)
    model.to(device).eval()
    idx, conf = [], []
    for batch in dl:
        out = model(batch[0].to(device))
        z = (out[0] if isinstance(out, (list, tuple)) else out).float()
        p = torch.softmax(z, dim=1)
        c, i = p.max(dim=1)
        idx.append(i.cpu().numpy())
        conf.append(c.cpu().numpy())
    return np.concatenate(idx), np.concatenate(conf)


def main(a):
    from lepinet.data import make_dls
    from lepinet.test import load_model, resolve_checkpoint_path

    ckpt = torch.load(resolve_checkpoint_path(a.model), map_location="cpu", weights_only=False)
    levels, vocabs = list(ckpt["levels"]), ckpt["vocabs"]
    sp_vocab = [str(v) for v in vocabs[levels[0]]]

    df = pd.read_parquet(a.adapt_parquet).reset_index(drop=True)
    df["is_valid"] = True
    loader_df = df[["image_path", "is_valid"]].copy()
    for lv in levels:                      # placeholders: scoring never reads y
        loader_df[lv] = str(vocabs[lv][0])
    dls = make_dls(loader_df, vocabs, a.img_dir, int(a.img_size * 460 / 256), a.img_size,
                   128, a.num_workers, lowmem=False, levels=levels)
    model, _ = load_model(ckpt, img_size=a.img_size)
    pred_i, conf = predict_species(model, dls, loader_df,
                                   torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                   a.num_workers)
    df["pseudo_speciesKey"] = [sp_vocab[i] for i in pred_i]
    df["confidence"] = conf

    # Strict gate: keep the most confident `keep_frac`. A quantile rather than a fixed probability,
    # because the absolute scale of a softmax over 12,041 classes is not comparable across models.
    cut = float(np.quantile(conf, 1.0 - a.keep_frac))
    kept = df[df["confidence"] >= cut].copy()

    # DIAGNOSTIC ONLY -- never written to the training parquet. The trap set is labelled; knowing
    # how clean the retained pseudo-labels are is what makes the downstream number interpretable.
    truth_col = "speciesKey" if "speciesKey" in df.columns else None
    diag = {}
    if truth_col:
        acc_all = float((df["pseudo_speciesKey"] == df[truth_col].astype(str)).mean())
        acc_kept = float((kept["pseudo_speciesKey"] == kept[truth_col].astype(str)).mean())
        diag = {"pseudo_accuracy_all": acc_all, "pseudo_accuracy_kept": acc_kept}
        print(f"pseudo-label accuracy: {acc_all:.4f} over all {len(df)}, "
              f"{acc_kept:.4f} over the {len(kept)} kept  (gate bought "
              f"{acc_kept - acc_all:+.4f})")

    # Rewrite paths so one img_dir (the global set) reaches the trap tree too.
    kept["image_path"] = a.path_prefix + kept["image_path"].astype(str)
    # Genus/family from the model's own taxonomy, so the pseudo-labels are internally consistent.
    hier = {str(r[levels[0]]): r for _, r in
            pd.read_csv(a.hierarchy).astype(str).iterrows()} if a.hierarchy else {}
    out = pd.DataFrame({"image_path": kept["image_path"].values,
                        levels[0]: kept["pseudo_speciesKey"].values})
    for lv in levels[1:]:
        out[lv] = [hier.get(s, {}).get(lv, "") for s in out[levels[0]]]
    out["filename"] = [Path(p).name for p in out["image_path"]]
    out["set"] = "2"          # a third fold id: never validation ('1'), never test ('0')
    out = out[out[levels[-1]] != ""] if a.hierarchy else out

    out.to_parquet(a.out)
    summary = {"n_adapt": len(df), "keep_frac": a.keep_frac, "confidence_cut": cut,
               "n_kept": len(out), "n_distinct_species": int(out[levels[0]].nunique()),
               "confidence_percentiles": {str(q): float(np.quantile(conf, q))
                                          for q in (0.1, 0.5, 0.9, 0.99)}, **diag}
    Path(str(a.out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapt-parquet", required=True)
    ap.add_argument("--img-dir", required=True, help="Trap image root (for inference).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hierarchy", default=None, help="hierarchy.csv, to fill genus/family.")
    ap.add_argument("--keep-frac", type=float, default=0.30, help="Strict gate: fraction retained.")
    ap.add_argument("--path-prefix", default="../../flemming/images/",
                    help="Prepended so one img_dir reaches both image trees.")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--num-workers", type=int, default=32)
    main(ap.parse_args())
