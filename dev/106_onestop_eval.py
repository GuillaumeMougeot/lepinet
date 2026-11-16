"""OneStop: Nemo (MAMBO_v3) against MILA's UK & Denmark classifier on recent GBIF images of invasive insects.

Test set: `dev/105_onestop_plan.py` (GBIF images with an event date after both models' training data),
fetched by `dev/082_tol_crawler.py fetch`. Each model runs as deployed, with its own species list:
- Nemo: mini_trainer.deploy.Predictor(model="full"), 12,632 species, its release preprocessing
  (mambo_deploy.preprocessing.preprocess), single view, fp32.
- MILA: "UK & Denmark Species Classifier - Apr 2024" from RolnickLab/ami-data-companion
  (UKDenmarkMothSpeciesClassifier2024): timm resnet50, 2,603 species, 128x128 resize, ImageNet normalisation.

Scoring follows mini_metrics (pinned revision 70cc69a, as Asger's release charts): per OneStop Living Lab
list (Belgium, Portugal, UK, Finland), the images of that list's species only, canonical mini_metric rows at
species level, top-1 at threshold 0, known_only=False. Macro-F1 averages over every class in the truth or
the predictions, so a predicted species absent from the images counts with F1 = 0, and a listed species the
model cannot name counts with F1 = 0 too: the figure shows what the species list is worth. Uncertainty:
1,000 bootstrap resamples of occurrences (images of one occurrence move together), with a numpy copy of
the metric asserted equal to mini_metrics on the full set.

    python dev/106_onestop_eval.py predict --model nemo     # GPU; needs the mini_trainer worktree on PYTHONPATH
    python dev/106_onestop_eval.py predict --model mila
    PYTHONPATH=$MM70:<mini_trainer worktree> python dev/106_onestop_eval.py score   # MM70: git archive of mini_metrics 70cc69a
    python dev/106_onestop_eval.py figure
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/onestop")
MT = Path("/home/au761367/codes/mini_trainer/.worktrees/lepinet-integration")
MILA_DIR = ROOT / "models/mila_ukdk_2024"
MILA_W = MILA_DIR / "uk-denmark_resnet50_baseline_20240417_55250a8b.pth"
MILA_MAP = MILA_DIR / "02_ami-gbif_fine-grained_w-europe_category_map-with_names.json"
GLOBAL = "data/global/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"
LABS = ["Belgium", "Portugal", "UK", "Finland"]
KEY_OVERRIDE = {119361453: 8108800}   # as dev/105
MODELS = {"nemo": ("Nemo (MAMBO_v3, 12,632 species)", "#098e92"),
          "mila": ("MILA UK & Denmark (2,603 species)", "#8064a2")}
INTERNAL = {"otto": ("Otto (OT3, 12,632 species)", "#e8872e")}   # scored for us, not in the OneStop figure
OTTO_W = ROOT / "models/ot3/last.pt"
UA = {"User-Agent": "lepinet-onestop-eval/1.0 (research, Aarhus University)"}


def slug(s: str) -> str:   # dev/082_tol_crawler.slug
    sys.path.insert(0, "dev")
    import importlib.util
    spec = importlib.util.spec_from_file_location("tol", "dev/082_tol_crawler.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["tol"] = m
    spec.loader.exec_module(m)
    globals()["slug"] = m.slug
    return m.slug(s)


def images() -> pd.DataFrame:
    """Planned images that are on disk and complete (JPEG ends in FFD9), with their occurrence metadata. Read from
    disk rather than from the crawler's part metadata, because a part stopped by the stall guard has none; images a
    host never delivered are simply absent (owner, 2026-11-11: discard what the hosts block)."""
    occ = pd.read_parquet(ROOT / "occurrences.parquet")
    occ["path"] = [f"{slug(sp)}/{u}.jpg" for sp, u in zip(occ.species, occ.uuid)]

    def complete(p):
        f = ROOT / "images" / p
        try:
            with open(f, "rb") as h:
                h.seek(-2, 2)
                return h.read(2) == b"\xff\xd9"
        except OSError:
            return False

    occ = occ[[complete(p) for p in occ.path]]
    return occ.sort_values("uuid").reset_index(drop=True)


# --- predict --------------------------------------------------------------------------------------------------------

def predict(a):
    import torch
    from PIL import Image
    df = images()
    out = ROOT / "preds" / f"{a.model}.parquet"
    out.parent.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    if a.model == "nemo":
        sys.path[:0] = [str(MT), str(MT / "deployment")]
        from mambo_deploy.preprocessing import preprocess
        from mini_trainer.deploy import Predictor
        pred = Predictor(device="cuda", model="full", precision="fp32").load()
        assert pred.preset == "full" and pred.metadata.get("name") == "Nemo"
        labels = [k for k, _ in sorted(pred.cls2idx["0"].items(), key=lambda kv: kv[1])]

        def prep(path):
            with Image.open(path) as im:
                img = np.asarray(im.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
            return torch.from_numpy(preprocess(img))

        def forward(x):
            raw, _ = pred.forward(x.to(dev))
            return raw[0] if isinstance(raw, (list, tuple)) else raw
    elif a.model == "otto":   # OT3, mini_trainer checkpoint, single view (TTA hurts on photos); as campaign logits.py "mt"
        sys.path[:0] = [str(MT)]
        from functools import partial

        from mini_trainer.modeling import Classifier, classification_module
        from mini_trainer.utils._core.image import read_square_crop, square_crop_box
        model, pre = Classifier.build(weights=str(OTTO_W), device=dev)
        model = model.eval()
        labels = [k for k, _ in sorted(classification_module(model).metadata["cls2idx"]["0"].items(), key=lambda kv: kv[1])]
        assert isinstance(pre.reader, partial) and pre.reader.func is read_square_crop
        read = pre.reader.keywords["size"]

        def prep(path):
            with Image.open(path) as im:
                im = im.convert("RGB")
                im = im.crop(square_crop_box(*im.size, read, (0.5, 0.5))).resize((read, read), Image.Resampling.BILINEAR)
            return torch.from_numpy(np.array(im)).permute(2, 0, 1).contiguous()

        def forward(x):
            out = model(pre.transform(x.to(dev)))
            return out[0] if isinstance(out, (list, tuple)) else out
    else:
        import timm
        cmap = json.load(open(MILA_MAP))
        labels = [n for n, _ in sorted(cmap.items(), key=lambda kv: kv[1])]
        model = timm.create_model("resnet50", pretrained=False, num_classes=len(labels))
        ck = torch.load(MILA_W, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model_state_dict") or ck)
        model = model.to(dev).eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def prep(path):   # ami-data-companion Resnet50Classifier.get_transforms at input_size 128 (its pad is a no-op)
            with Image.open(path) as im:
                x = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
            x = torch.nn.functional.interpolate(x[None], size=(128, 128), mode="bilinear", antialias=True,
                                                align_corners=False)[0]
            return (x - mean) / std

        def forward(x):
            return model(x.to(dev))

    paths = [str(ROOT / "images" / p) for p in df.path]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return i, prep(paths[i])

    dl = torch.utils.data.DataLoader(DS(), batch_size=a.batch_size, num_workers=a.workers, prefetch_factor=2,
                                     multiprocessing_context="fork")
    top1 = np.zeros(len(df), np.int64)
    conf = np.zeros(len(df), np.float32)
    top5 = np.zeros((len(df), 5), np.int64)
    t0 = time.time()
    with torch.inference_mode():
        for n, (idx, x) in enumerate(dl):
            p = torch.softmax(forward(x).float(), 1)
            v, k = p.topk(5, 1)
            top1[idx] = k[:, 0].cpu().numpy()
            conf[idx] = v[:, 0].cpu().numpy()
            top5[idx] = k.cpu().numpy()
            if n % 50 == 0:
                print(f"{(n + 1) * a.batch_size:,}/{len(df):,} images, {time.time() - t0:.0f} s", flush=True)
    json.dump([str(x) for x in labels], open(ROOT / "preds" / f"{a.model}_labels.json", "w"))
    lab = np.array(labels, dtype=object)
    pd.DataFrame({"uuid": df.uuid, "pred": lab[top1], "conf": conf,
                  "top5": ["|".join(r) for r in lab[top5]]}).to_parquet(out, index=False)
    print("wrote", out, len(df))


# --- names and keys -------------------------------------------------------------------------------------------------

def gbif_json(url):
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (i + 1))
    raise err


def aliases(taxa: pd.DataFrame) -> dict[int, set[str]]:
    """Per accepted key, every name and key under which a model may know the species: the list's name and key,
    the COL XR name, the legacy backbone's accepted name and synonyms, and the name our training parquet gives
    the key (2025 backbone). The legacy key of Oegoconia novimundi lumps four Oegoconia species, so a model's
    lumped class (Nemo 1845962, MILA "Oegoconia quadripuncta") counts as naming novimundi, for both models alike."""
    cache = ROOT / "aliases.json"
    if cache.exists():
        return {int(k): set(v) for k, v in json.load(open(cache)).items()}
    import pyarrow.parquet as pq
    g = pq.read_table(GLOBAL, columns=["speciesKey", "genus", "specificEpithet"]).to_pandas().dropna()
    g = g.drop_duplicates("speciesKey")
    gname = dict(zip(g.speciesKey.astype("int64"), g.genus + " " + g.specificEpithet))
    xr = pd.read_csv(ROOT / "taxa_xr.csv")          # current GBIF backbone (COL XR) names, see dev/105
    lst = pd.read_csv(ROOT / "onestop_species_list.csv")
    lst = lst[lst["GBIF Taxon ID"].notna()].drop_duplicates("Canonical name")
    xr["list_key"] = xr.list_name.map(dict(zip(lst["Canonical name"], lst["GBIF Taxon ID"].astype(int))))
    xr_names = {int(k): [n for n in g.xr_name.dropna()] for k, g in xr.groupby("list_key")}
    out = {}
    for t in taxa.itertuples():
        k = int(t.key)
        s = {str(k), str(int(t.list_key)), t.name, t.list_name, *xr_names.get(int(t.list_key), [])}
        s |= {str(x) for x in t.list_names.split("|")}
        if k in gname:
            s.add(gname[k])
        off = 0
        while True:
            r = gbif_json(f"https://api.gbif.org/v1/species/{k}/synonyms?limit=100&offset={off}")
            for x in r["results"]:
                if x.get("canonicalName"):
                    s.add(x["canonicalName"])
                s.add(str(x["key"]))
            if r.get("endOfRecords", True):
                break
            off += 100
        out[k] = {x for x in s if x and x != "nan"}
    json.dump({k: sorted(v) for k, v in out.items()}, open(cache, "w"), indent=1)
    return out


def taxa_table() -> pd.DataFrame:
    """Accepted taxa of the list, with the Living Labs that list each one."""
    lst = pd.read_csv(ROOT / "onestop_species_list.csv")
    lst = lst[(lst["Plant or insect?"] == "Insect") & lst["GBIF Taxon ID"].notna()]
    res = pd.read_csv(ROOT / "taxa_resolved.csv")
    res["key"] = res.key.map(lambda k: KEY_OVERRIDE.get(int(k), int(k)))
    lst = lst.merge(res[["list_key", "key", "name"]], left_on=lst["GBIF Taxon ID"].astype(int), right_on="list_key")
    t = lst.groupby("key").agg(name=("name", "first"), list_key=("list_key", "first"),
                               list_name=("Canonical name", "first"),
                               list_names=("Canonical name", lambda s: "|".join(sorted(set(s)))),
                               labs=("Living Lab", lambda s: "|".join(sorted(set(s))))).reset_index()
    return t


# --- score ---------------------------------------------------------------------------------------------------------

PATTERN = r"^(micro_accuracy|accuracy|f1|recall|precision|coverage|theilU)$"


def macro_accuracy(t: np.ndarray, hit: np.ndarray, w: np.ndarray, n: int) -> float:
    """mini_metrics' macro accuracy at threshold 0: mean over truth classes of the share of their images predicted
    correctly (weighted counts, for the bootstrap). t: integer truth ids < n."""
    tot = np.bincount(t, w, n)
    ok = np.bincount(t, w * hit, n)
    present = tot > 0
    return float(np.mean(ok[present] / tot[present]))


def halves(df: pd.DataFrame) -> np.ndarray:
    """Calibration (True) / reporting (False) halves by occurrence, stratified by species, seed 0."""
    rng = np.random.default_rng(0)
    occ = df.drop_duplicates("gbifID")[["gbifID", "truth"]].sample(frac=1, random_state=0)
    occ["cal"] = occ.groupby("truth").cumcount() % 2 == (rng.integers(0, 2))
    return df.gbifID.map(dict(zip(occ.gbifID, occ.cal))).to_numpy(bool)


def vocabulary(m: str) -> set[str]:
    f = ROOT / "preds" / f"{m}_labels.json"
    if f.exists():
        return set(json.load(open(f)))
    return set(json.load(open(MILA_MAP))) if m == "mila" else nemo_vocab_all()


def score(a):
    """Canonical mini_metric rows (Asger's columns; species level) per model and list, scored with the pinned
    mini_metrics as dev/releases/mambo_v3 does: macro accuracy at threshold 0 on every image, and macro-F1 with
    the threshold chosen by OptimalConfidenceThreshold(MacroF1, eps=0.01, use_quantiles=True, n_bootstraps=0) on a
    calibration half of the occurrences and applied to the other half. Needs MM70 on PYTHONPATH."""
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroF1, OptimalConfidenceThreshold, evaluate_file
    import mini_metrics
    print("mini_metrics from", mini_metrics.__file__)
    taxa = taxa_table()
    al = aliases(taxa)
    name2key = {n: str(k) for k, ns in al.items() for n in ns}
    df = images()
    df["truth"] = df.taxon_key.astype(int).astype(str)
    cal_mask = halves(df)
    rng = np.random.default_rng(0)
    rows, per_species = [], []
    out = ROOT / "mini_metric"
    out.mkdir(exist_ok=True)
    models = [m for m in [*MODELS, *INTERNAL] if (ROOT / "preds" / f"{m}.parquet").exists()]
    for m in models:
        p = pd.read_parquet(ROOT / "preds" / f"{m}.parquet")
        d = df.assign(cal=cal_mask).merge(p, on="uuid")
        # a prediction naming a listed species (any alias) becomes its accepted key; anything else stays the
        # model's own label, prefixed so that two models' labels never collide
        d["pk"] = d.pred.astype(str).map(lambda x: name2key.get(x, f"{m}:{x}"))
        known = {str(k) for k in al if al[k] & vocabulary(m)}
        for lab in LABS + ["All lists"]:
            sp = taxa if lab == "All lists" else taxa[taxa.labs.str.split("|").map(lambda x: lab in x)]
            e = d[d.truth.isin({str(k) for k in sp.key})].reset_index(drop=True)
            if e.empty:
                continue
            tag = f"{m}_{lab.replace(' ', '_')}"

            def canonical(x, name):
                c = pd.DataFrame({"instance_id": np.arange(len(x)), "filename": x.uuid.to_numpy(), "level": 0,
                                  "label": x.truth.to_numpy(), "prediction": x.pk.to_numpy(),
                                  "confidence": x.conf.astype(float).to_numpy(), "threshold": 0,
                                  "known_label": x.truth.isin(known).astype(int).to_numpy(), "prediction_made": 1,
                                  "correct": np.where(x.truth == x.pk, 1, -1)})
                src = out / f"{tag}_{name}.csv"
                c.to_csv(src, index=False)
                return MetricDF.from_source(str(src))

            full = canonical(e, "all")
            r0 = evaluate_file(full, threshold=0, optimal=False, known_only=False, per_class=False, simple=True,
                               hierarchical=False, pattern=PATTERN, verbose=0)
            cal, rep = canonical(e[e.cal], "calibration"), canonical(e[~e.cal], "report")
            tau = float(OptimalConfidenceThreshold(crit=MacroF1, eps=0.01, use_quantiles=True, n_bootstraps=0)(cal, verbose=0)[0])
            rc = evaluate_file(rep, threshold=[tau], optimal=False, known_only=False, per_class=False, simple=True,
                               hierarchical=False, pattern=PATTERN, verbose=0)
            acc = float(r0["accuracy"][0])
            labels, inv = np.unique(e.truth.to_numpy(), return_inverse=True)
            hit = (e.truth == e.pk).to_numpy().astype(float)
            check = macro_accuracy(inv, hit, np.ones(len(e)), len(labels))
            assert abs(check - acc) < 1e-9, (m, lab, check, acc)
            # bootstrap over occurrences, stratified by species: every species keeps its own number of occurrences, so
            # the set of species averaged is fixed (an unstratified resample drops the 1-3-image species and biases up)
            uo, occ = np.unique(e.gbifID.to_numpy(), return_inverse=True)
            occ_sp = np.zeros(len(uo), np.int64)
            occ_sp[occ] = inv
            groups = [np.flatnonzero(occ_sp == k) for k in range(len(labels))]
            boots = []
            for _ in range(a.n_boot):
                wo = np.zeros(len(uo))
                for g in groups:
                    np.add.at(wo, g[rng.integers(0, len(g), len(g))], 1)
                boots.append(macro_accuracy(inv, hit, wo[occ], len(labels)))
            scored = sorted(set(e.truth))
            rows.append({"model": m, "list": lab, "macro_accuracy": acc, "acc_lo": float(np.percentile(boots, 2.5)),
                         "acc_hi": float(np.percentile(boots, 97.5)), "macro_f1_zero": float(r0["f1"][0]),
                         "threshold": tau, "macro_f1_calibrated": float(rc["f1"][0]),
                         "coverage_calibrated": float(rc["coverage"][0]),
                         "macro_accuracy_calibrated": float(rc["accuracy"][0]),
                         "micro_accuracy": float(r0["micro_accuracy"][0]),
                         "species_scored": len(scored), "species_known": len(set(scored) & known),
                         "species_in_list": len(sp), "classes_predicted_absent": int(len(set(e.pk) - set(e.truth))),
                         "images": len(e), "occurrences": len(uo), "report_images": int((~e.cal).sum())})
            if lab == "All lists":
                tk = taxa.set_index(taxa.key.astype(str))
                for k in scored:
                    tp = int(((e.truth == k) & (e.pk == k)).sum())
                    fp = int(((e.truth != k) & (e.pk == k)).sum())
                    fn = int(((e.truth == k) & (e.pk != k)).sum())
                    per_species.append({"model": m, "key": k, "species": tk.name[k], "labs": tk.labs[k],
                                        "images": int((e.truth == k).sum()), "known": k in known, "tp": tp,
                                        "recall": tp / max(1, tp + fn), "fp": fp, "fn": fn})
    s = pd.DataFrame(rows)
    s.to_csv(ROOT / "scores_per_list.csv", index=False)
    pd.DataFrame(per_species).to_csv(ROOT / "scores_per_species.csv", index=False)
    with pd.option_context("display.width", 250):
        print(s.round(3).to_string(index=False))


_NEMO_VOCAB = None


def nemo_vocab_all() -> set[str]:
    """Nemo's 12,632 species keys (as strings), from its release metadata."""
    global _NEMO_VOCAB
    if _NEMO_VOCAB is None:
        sys.path[:0] = [str(MT), str(MT / "deployment")]
        from mini_trainer.deploy import Predictor
        p = Predictor(device="cpu", model="full", precision="fp32").load()
        _NEMO_VOCAB = set(map(str, p.cls2idx["0"]))
    return _NEMO_VOCAB


# --- figure ---------------------------------------------------------------------------------------------------------

def figure(a):
    """Two panels per Living Lab list: mini_metrics macro accuracy (threshold 0, all images, 95 % bootstrap over
    occurrences) and calibrated macro-F1 (threshold fitted on half the occurrences, scored on the other half).
    Nemo and MILA only; `--with-otto` adds Otto as dotted outline bars, in a separate file for internal use."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = pd.read_csv(ROOT / "scores_per_list.csv")
    lists = [x for x in LABS + ["All lists"] if x in set(s.list)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    drawn = {**MODELS, **(INTERNAL if a.with_otto else {})}
    w = 0.8 / len(drawn)
    x = np.arange(len(lists))
    n = s[s.model == "nemo"].set_index("list").loc[lists]
    for ax, (col, lo, hi, title) in zip(axes, (
            ("macro_accuracy", "acc_lo", "acc_hi", "Macro accuracy (threshold 0, all images)"),
            ("macro_f1_calibrated", None, None, "Macro-F1, calibrated threshold (reporting half)"))):
        for j, (m, (label, color)) in enumerate(drawn.items()):
            r = s[s.model == m].set_index("list").loc[lists]
            xs = x + (j - (len(drawn) - 1) / 2) * w
            internal = m in INTERNAL
            if internal:   # dotted outline, pale fill: not part of the shared comparison
                ax.bar(xs, r[col], w - 0.04, facecolor=color + "26", edgecolor=color, linestyle=(0, (1.5, 1.5)),
                       linewidth=1.6, label=label, zorder=2)
            else:
                ax.bar(xs, r[col], w - 0.04, color=color, label=label, zorder=2)
            ink = "#222222" if internal else "white"
            top = r[col]
            if lo:
                ax.errorbar(xs, r[col], yerr=[r[col] - r[lo], r[hi] - r[col]], fmt="none", ecolor="#333333",
                            elinewidth=1, capsize=3, zorder=3)
                top = r[hi]
            for xi, v, t, (_, rr) in zip(xs, r[col], top, r.iterrows()):
                ax.text(xi, t + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="#222222")
                if col == "macro_accuracy":
                    ax.text(xi, 0.012, f"{rr.species_known}/\n{rr.species_scored}", ha="center", va="bottom",
                            fontsize=7, color=ink, fontweight="bold", linespacing=0.9)
                else:
                    ax.text(xi, 0.012, f"{rr.coverage_calibrated:.0%}", ha="center", va="bottom", fontsize=7,
                            color=ink, fontweight="bold")
        ax.set_xticks(x, [f"{l}\n{int(r.species_scored)} species\n{int(r.images):,} images" for l, r in n.iterrows()],
                      fontsize=8.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color="#e6e6e6", zorder=0)
    axes[0].set_ylabel("mini_metrics, species level")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=len(drawn), bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.text(0.01, 0.01, "Test images: GBIF photos of the OneStop invasive insect species observed after 26 Apr 2025 "
             "(after both models' training data), any country, up to 2,000 per species; scored per Living Lab list on "
             "that list's species.\nLeft: share of each species' images named correctly, averaged over species; white: "
             "listed species the model can name / species scored (unnamed species score 0); bars: 95 % bootstrap over "
             "occurrences within species.\nRight: mini_metrics macro-F1 over truth \u222a predicted species, threshold fitted on half of "
             "the occurrences (MacroF1 optimiser) and scored on the other half; white: share of images answered.",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    stem = "onestop_per_list_with_otto" if a.with_otto else "onestop_per_list"
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"{stem}.{ext}", dpi=200)
    print("wrote", ROOT / f"{stem}.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    q = sub.add_parser("predict")
    q.add_argument("--model", choices=[*MODELS, *INTERNAL], required=True)
    q.add_argument("--batch-size", type=int, default=128)
    q.add_argument("--workers", type=int, default=8)
    q.set_defaults(fn=predict)
    q = sub.add_parser("score")
    q.add_argument("--n-boot", type=int, default=1000)
    q.set_defaults(fn=score)
    q = sub.add_parser("figure")
    q.add_argument("--with-otto", action="store_true", help="add Otto (internal only; writes a separate file)")
    q.set_defaults(fn=figure)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
