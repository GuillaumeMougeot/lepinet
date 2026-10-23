"""Open-set detection on trap images, and what the back-off policy answers on species it never saw.

Inputs, from the N2/N3 jobs (`dev/061 --save-scores` on all 47,905 trap crops; higher score = more
novel): data/ucloud_preds/trap-openset/trap-openset-<model>.parquet, and C3b's `lepinet test` output
on the same crops (data/ucloud_preds/C3b-shift). Strata come from C3b's withheld taxa
(data/trap_openset/c3b_taxa.json, rebuilt with dev/072's original arguments): near = withheld species
in a genus C3b knows, mid = withheld genus, oov = outside every model's vocabulary.

    python dev/096_trap_openset.py

journal/2026-10-23-does-the-model-know-when-a-trap-image-is-too-poor.md (part B).
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "ucloud_preds"
RULES = ["msp", "entropy", "margin", "max", "energy"]
_s = importlib.util.spec_from_file_location("rel", ROOT / "dev" / "083_hf_release.py")
rel = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rel)


def auroc(novel, known):
    s = np.r_[known, novel]
    ranks = pd.Series(s).rank().to_numpy()
    rn = ranks[len(known):].sum()
    return float((rn - len(novel) * (len(novel) + 1) / 2) / (len(known) * len(novel)))


def strata(paths, labels):
    t = json.loads((ROOT / "data/trap_openset/c3b_taxa.json").read_text())
    n = pd.read_csv(ROOT / "data/flemming/name2id.csv")
    by_name = n.set_index("verbatimScientificName")
    folder = paths.str.split("/").str[0]
    gk = folder.map(by_name.genusKey).astype("Int64").astype(str)
    out = pd.Series("known", index=paths.index)
    out[labels.astype(str).isin(set(t["near"]))] = "near"
    out[gk.isin(set(t["mid"]))] = "mid"
    return out


def group_of(paths):
    m = paths.str.extract(r"TRAPNAME_([^_]+)_IMAGENAME_(\d{14})")
    night = (pd.to_datetime(m[1], format="%Y%m%d%H%M%S") - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")
    return m[0] + "|" + night


def main():
    for model in ("C3b", "B8", "P5"):
        f = D / "trap-openset" / f"trap-openset-{model}.parquet"
        if not f.exists():
            print(f"\n{model}: no scores yet")
            continue
        s = pd.read_parquet(f)
        st = strata(s.image_path, s.label)
        st[s.novel & (st == "known")] = "oov"
        print(f"\n### {model}: {int(s.novel.sum())} novel crops of {len(s)}  "
              f"({', '.join(f'{k} {v}' for k, v in st[s.novel].value_counts().items())})")
        known = ~s.novel
        rows = []
        for name in ["all novel", "near", "mid", "oov"]:
            m = s.novel if name == "all novel" else (st == name) & s.novel
            if m.sum() == 0:
                continue
            rows.append({"novel set": name, "crops": int(m.sum()), "species": s.label[m].nunique(),
                         **{r: auroc(s.loc[m, r].to_numpy(), s.loc[known, r].to_numpy()) for r in RULES}})
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        if model == "C3b":
            best = max(RULES, key=lambda r: rows[0][r])
            per = [{"species": s.image_path[m].iloc[0].split("/")[0], "stratum": st[m].iloc[0], "crops": int(m.sum()),
                    best: auroc(s.loc[m, best].to_numpy(), s.loc[known, best].to_numpy())}
                   for _, m in ((lab, (s.label == lab) & s.novel) for lab in s.label[s.novel].unique())]
            print(f"\nper novel species ({best}):")
            print(pd.DataFrame(per).sort_values("crops", ascending=False).round(3).to_string(index=False))

    # What the release cascade answers on C3b's novel crops, thresholds fitted on known crops only.
    p = next((D / "C3b-shift").rglob("predictions.csv"))
    d = pd.read_csv(p)
    lv = dict(enumerate(rel.LEVELS))
    wide = d.pivot_table(index=["instance_id", "filename"], columns="level",
                         values=["prediction", "confidence", "label"], aggfunc="first").reset_index()
    res = pd.DataFrame({"filename": wide["filename"]})
    for i, name in lv.items():
        res[f"pred_{name}"] = wide["prediction"][i].astype(str).values
        res[f"label_{name}"] = wide["label"][i].astype(str).values
        res[f"conf_{name}"] = wide["confidence"][i].astype(float).values
    s = pd.read_parquet(D / "trap-openset" / "trap-openset-C3b.parquet").set_index("image_path")
    res["novel"] = res.filename.map(s.novel).astype(bool).values
    res["stratum"] = strata(res.filename, res.label_species).where(~res.novel | (strata(res.filename, res.label_species) != "known"), "oov")
    res["group"] = group_of(res.filename)
    groups = np.array(sorted(res.group.unique()))
    fit_g = set(np.random.default_rng(0).permutation(groups)[: len(groups) // 2])
    known_fit = res[res.group.isin(fit_g) & ~res.novel]
    th = rel.fit_thresholds(known_fit, 0.95)
    print(f"\n### C3b back-off, thresholds fitted on known crops of half the nights: "
          f"{', '.join(f'{k} {v:.4f}' for k, v in th.items())}")
    out = []
    for name, m in [("known (other half)", ~res.group.isin(fit_g) & ~res.novel),
                    ("novel: near", res.stratum.eq("near")), ("novel: oov", res.stratum.eq("oov"))]:
        dd = res[m]
        c = rel.cascade(dd, th)
        out.append({"crops": name, "n": len(dd), **{f"{k} %": 100 * c[k]["coverage"] for k in rel.LEVELS},
                    **{f"{k} prec": c[k]["precision"] for k in rel.LEVELS}, "abstain %": 100 * c["abstain"]})
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
