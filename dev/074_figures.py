"""Paper figures: six, one per claim the paper makes.

Numbers are transcribed from journal entries (source named beside each), except figure 5, which
reads `paper/figures/fig4_deployment.json` written by `dev/086_deployment_gap.py`. Nothing is
recomputed from checkpoints here: a figure renders numbers that have already been argued.

    python dev/074_figures.py        # writes paper/figures/fig{1..6}_*.pdf and .png

File numbers follow first appearance in the paper: 1 protocol (section 1), 2 classifier (4.1),
3 open set (4.4), 4 deployment (4.6a), 5 dose curves (4.11), 6 foundation model (4.14).

Colours: the first three slots of the validated reference palette (blue, orange, aqua; safe for
colour-vision deficiency as a set), grey for references, one blue ramp for ordered categories.
Rewritten 2026-10-02: the August figures included two retracted comparisons (staged vs end-to-end
by capacity, and a best-vs-worst scoring-rule plot).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, LIGHT = "#8d8c87", "#d9d8d4"
INK, INK2 = "#0b0b0b", "#52514e"
RAMP = ["#1c5cab", "#6da7ec", "#b7d3f6"]          # species, genus, family (blue 550/300/150)
FULL, COL = 6.9, 3.3                              # two-column ML paper widths, inches

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
    "grid.color": "#ecebe8", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf/.png")


def panel(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontweight="bold", fontsize=9,
            ha="right", va="bottom")


# ------------------------------------------------------------------------------------------------
def fig1_protocol():
    """(a) the three evaluation axes; (b) in-distribution vs domain-shift accuracy for 17 models,
    drawn at equal scale on both axes so the spreads can be compared by eye."""
    fig = plt.figure(figsize=(FULL, 3.75))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 2.1], hspace=0.28)

    ax = fig.add_subplot(gs[0]); ax.set_axis_off(); panel(ax, "a")
    ax.set_xlim(0, 20); ax.set_ylim(0, 3)

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=INK2, lw=0.7))
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=7.2, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.33, body, ha="center", va="center", fontsize=5.9, color=INK2)

    box(0.1, 0.35, 3.6, 2.3, "Train", "GBIF photographs\n12,041 species, 5.7 M images", "#f4f3f1")
    tests = [(5.0, "In-distribution", "held-out fold, same source\nspecies macro-F1", "#eef4fc"),
             (10.0, "Domain shift", "light-trap crops; nights\nheld out (probe, probe-HO)", "#fdf0ea"),
             (15.0, "Open set", "unseen species, near/mid/far\nAUROC; back off to genus", "#e8f6f0")]
    for x, t, b, fc in tests:
        box(x, 0.35, 4.6, 2.3, t, b, fc)
    ax.annotate("", xy=(4.95, 1.5), xytext=(3.75, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))

    # (b) source: RESULTS.md section 2 (each row links its journal entry); full-fold in-distribution
    ax = fig.add_subplot(gs[1]); panel(ax, "b")
    groups = {
        "no target-domain data": (GREY, "o", [
            ("reference", 0.9135, 0.6270), ("B1", 0.8999, 0.6912), ("F1", 0.9219, 0.7209),
            ("L5", 0.9055, 0.7497), ("cap 250", 0.8783, 0.5776), ("cap 500", 0.8955, 0.6281),
            ("cap 1000", 0.9060, 0.6427)]),
        "self-training, end-to-end": (ORANGE, "s", [
            ("B3", 0.9003, 0.7370), ("B3rep5x", 0.9003, 0.7706), ("B6", 0.9225, 0.7699),
            ("B7", 0.9050, 0.7796), ("B8", 0.9060, 0.7798)]),
        "classifier stages on a frozen trunk": (BLUE, "D", [
            ("F2", 0.9081, 0.7541), ("F3", 0.9061, 0.7479), ("R5", 0.9074, 0.7692),
            ("G2", 0.9150, 0.7648), ("G3", 0.9138, 0.7740)]),
    }
    for name, (c, m, pts) in groups.items():
        ax.scatter([p[2] for p in pts], [p[1] for p in pts], s=26, c=c, marker=m, label=name,
                   edgecolors="white", linewidths=0.6, zorder=3)
    for lab, ind, sh, dx, dy in [("reference", 0.9135, 0.6270, 5, 2), ("B8", 0.9060, 0.7798, 5, -7),
                                 ("cap 250", 0.8783, 0.5776, 5, -2), ("B1", 0.8999, 0.6912, 5, -3),
                                 ("F1", 0.9219, 0.7209, 5, -2)]:
        ax.annotate(lab, (sh, ind), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.set_xlim(0.56, 0.80); ax.set_ylim(0.865, 0.935)
    ax.text(0.797, 0.868, "same scale on both axes: in-distribution spans 4 pt, shift 22 pt",
            ha="right", va="bottom", fontsize=6.3, color=INK2)
    ax.set_aspect("equal")
    ax.set_xlabel("domain shift: probe macro-F1")
    ax.set_ylabel("in-distribution\nspecies macro-F1")
    ax.legend(loc="lower left", handletextpad=0.3, ncol=3, columnspacing=1.2,
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1)
    save(fig, "fig1_protocol")


# ------------------------------------------------------------------------------------------------
def fig2_classifier():
    """Three questions, each answered at the classifier. Dot plots, one axis per metric."""
    rows = [
        ("a  Taxonomy: where do coarse ranks come from?",   # journal 2026-07-30-marginal-supervision
         ["genus + family heads (parameters)", "single species head, marginalised",
          "+ loss on the marginals (supervision)"],
         [0.9110, 0.9135, 0.9135], [0.6503, 0.6293, 0.6460], 2, "full trap set"),
        ("b  Long tail: rebalance the data or the classifier?",  # 2026-08-01-imbalance-methods-bench
         ["no rebalancing", "√-oversampling (data)", "balanced softmax", "both",
          "cRT: rebalance the classifier only"],
         [0.8949, 0.9135, 0.8970, 0.8689, 0.9068], [0.6445, 0.6293, 0.5726, 0.5492, 0.6539], 4,
         "full trap set"),
        ("c  Adaptation to trap images: where?",  # 2026-08-06-adaptation-is-mostly-a-classifier-problem
         ["none (B1)", "classifier only, trunk frozen (T2)", "whole network (B3rep5x)"],
         [0.6974, 0.7621, 0.7704], [0.6912, 0.7572, 0.7706], 1, "probe"),
    ]
    fig, axs = plt.subplots(3, 2, figsize=(FULL, 3.6), gridspec_kw=dict(
        height_ratios=[3, 5, 3], hspace=0.75, wspace=0.08))
    for r, (title, labels, left, right, hi, shift_name) in enumerate(rows):
        n = len(labels)
        ys = list(range(n))[::-1]
        left_name = "probe-HO (species unseen in adaptation)" if r == 2 else "in-distribution"
        for c, (vals, name) in enumerate(((left, left_name), (right, shift_name))):
            ax = axs[r, c]
            cols = [ORANGE if i == hi else GREY for i in range(n)]
            ax.hlines(ys, min(vals) - 0.02, vals, color=LIGHT, lw=0.8, zorder=1)
            ax.scatter(vals, ys, c=cols, s=26, zorder=3, edgecolors="white", linewidths=0.6)
            for v, y, col in zip(vals, ys, cols):
                ax.text(v + (max(vals) - min(vals)) * 0.04 + 0.001, y, f"{v:.4f}", va="center",
                        fontsize=6.3, color=INK if col == ORANGE else INK2)
            span = max(vals) - min(vals)
            ax.set_xlim(min(vals) - 0.15 * span - 0.004, max(vals) + 0.35 * span + 0.006)
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_yticks(ys)
            ax.set_yticklabels(labels if c == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(name, fontsize=6.8, color=INK2, labelpad=1)
            if c == 0:
                ax.set_title(title, loc="left", x=-0.62, fontweight="bold", fontsize=7.5)
    fig.text(0.99, 0.005, "Orange: the intervention that acts on the classifier. Each panel varies one factor; "
             "compare within a column.", ha="right", fontsize=6.3, color=INK2)
    save(fig, "fig2_classifier")


# ------------------------------------------------------------------------------------------------
def fig3_dose():
    """In-distribution accuracy points the wrong way: change (pt) against a reference, one axis."""
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw=dict(wspace=0.32))

    # (a) share of training that is pseudo-labelled trap images. 2026-08-04-replication-sweep
    ax = axs[0]; panel(ax, "a")
    share = [0, 0.39, 2.0, 6.1, 10.0]
    probe = [0.6912, 0.7354, 0.7706, 0.7370, 0.7159]
    ho = [0.6974, 0.7508, 0.7704, 0.7231, 0.7042]
    ind = {0: 0.8999, 2.0: 0.9003, 6.1: 0.9003}          # B1, B3rep5x, B3
    xs = list(range(len(share)))
    ax.plot(xs, [(v - probe[0]) * 100 for v in probe], "-o", color=ORANGE, ms=4, lw=1.5, label="probe")
    ax.plot(xs, [(v - ho[0]) * 100 for v in ho], "-s", color=AQUA, ms=4, lw=1.5, label="probe-HO")
    ax.plot([xs[share.index(k)] for k in ind], [(v - ind[0]) * 100 for v in ind.values()], "-D",
            color=GREY, ms=3.5, lw=1.2, label="in-distribution")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s:g}" for s in share])
    ax.set_xlabel("pseudo-labelled trap images, % of training")
    ax.set_ylabel("change vs no trap data (pt)")
    ax.set_ylim(-0.8, 10.5)
    ax.legend(loc="upper left", handlelength=1.6)

    # (b) cap on training images per species. 2026-08-28-fine-tuned-bioclip2-beats-us-and-the-head-hurts
    ax = axs[1]; panel(ax, "b")
    cap = ["250", "500", "1000", "~2000\n(ours)"]
    ind_c = [0.8783, 0.8955, 0.9060, 0.9148]
    pr_c = [0.5776, 0.6281, 0.6427, 0.6270]
    ho_c = [0.6248, 0.6371, 0.6738, 0.6412]
    xs = range(4)
    for vals, c, m, lab in ((ind_c, GREY, "D", "in-distribution"), (pr_c, ORANGE, "o", "probe"),
                            (ho_c, AQUA, "s", "probe-HO")):
        ax.plot(xs, [(v - vals[-1]) * 100 for v in vals], "-" + m, color=c, ms=4 if m != "D" else 3.5,
                lw=1.5 if c != GREY else 1.2, label=lab)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xticks(list(xs)); ax.set_xticklabels(cap)
    ax.set_xlabel("max training images per species")
    ax.set_ylabel("change vs uncapped (pt)")

    # (c) how hard the long tail is reweighted. 2026-08-01-imbalance-methods-bench
    ax = axs[2]; panel(ax, "c")
    cells = ["none", "√-over-\nsample", "balanced\nsoftmax", "both"]
    ind_l = [0.8949, 0.9135, 0.8970, 0.8689]
    sh_l = [0.6445, 0.6293, 0.5726, 0.5492]
    xs = range(4)
    ax.plot(xs, [(v - ind_l[0]) * 100 for v in ind_l], "-D", color=GREY, ms=3.5, lw=1.2, label="in-distribution")
    ax.plot(xs, [(v - sh_l[0]) * 100 for v in sh_l], "-o", color=ORANGE, ms=4, lw=1.5, label="full trap set")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xticks(list(xs)); ax.set_xticklabels(cells)
    ax.set_xlabel("reweighting toward rare species →")
    ax.set_ylabel("change vs none (pt)")
    ax.legend(loc="lower left", handlelength=1.6)
    save(fig, "fig5_dose")


# ------------------------------------------------------------------------------------------------
def fig4_openset():
    """(a) AUROC per scoring rule and model; (b) novelty by taxonomic distance."""
    fig = plt.figure(figsize=(FULL, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.42)
    ax = fig.add_subplot(gs[0]); panel(ax, "a")
    rules = ["max-logit", "energy", "MSP", "entropy", "margin"]
    # paper 4.9 (2026-08-01-the-scoring-rule-was-the-bug); plain head: data/paper_figs/ruleclamp-plain.json
    # (2026-08-06-the-arcface-open-set-claim-was-a-rule-comparison); B8, P5: rules-{B8,P5}.json (O1).
    # B8's head emits log-probabilities, so energy and MSP are degenerate for it (None).
    models = [
        ("plain cosine head, 20 M", [0.6258, 0.6149, 0.8819, 0.8990, 0.8423]),
        ("ArcFace × z-score, 20 M (A1)", [0.9068, 0.9064, 0.8953, 0.9047, 0.8979]),
        ("ArcFace × z-score, 198 M (A2)", [0.8298, 0.8287, 0.8904, 0.8813, 0.8807]),
        ("B8, 198 M", [0.8889, None, None, 0.9153, 0.9108]),
        ("P5, BioCLIP-2 ViT-L", [0.9160, 0.9156, 0.9087, 0.9161, 0.8890]),
    ]
    lo, hi = 0.60, 0.93
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#b7d3f6", "#5598e7", "#1c5cab"])
    for i, (name, vals) in enumerate(models):
        best = max(v for v in vals if v is not None)
        for j, v in enumerate(vals):
            if v is None:
                ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, fc="white", ec=LIGHT, lw=0.6, hatch="////"))
                ax.text(j + 0.48, i + 0.46, "n/a", ha="center", va="center", fontsize=6, color=INK2)
                continue
            t = (v - lo) / (hi - lo)
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, fc=cmap(t), ec="white", lw=0.6))
            ax.text(j + 0.48, i + 0.46, f"{v:.3f}", ha="center", va="center", fontsize=6.4,
                    color="white" if t > 0.62 else INK, fontweight="bold" if v == best else "normal")
            if v == best:
                ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.90, 0.86, fill=False, ec=ORANGE, lw=1.4))
    ax.set_xlim(0, 5); ax.set_ylim(len(models), 0)
    ax.set_xticks([j + 0.48 for j in range(5)]); ax.set_xticklabels(rules)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks([i + 0.46 for i in range(len(models))]); ax.set_yticklabels([m[0] for m in models])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("open-set AUROC, unseen vs known species (orange: each model's best rule)",
                  fontsize=6.5, color=INK2)

    ax = fig.add_subplot(gs[1]); panel(ax, "b")
    # ood-strat-{plain,C3b,zscore}.json; 2026-08-08-is-novelty-monotone-or-just-rare
    strata = ["near\n(seen genus)", "mid\n(seen family)", "far\n(unseen family)"]
    series = [("plain head, rare novel taxa (C3)", [0.8527, 0.9342, 0.9641], GREY, "o"),
              ("plain head, common taxa withheld (C3b)", [0.8717, 0.9463, 0.9726], ORANGE, "s"),
              ("ArcFace × z-score, rare novel taxa", [0.8680, 0.9165, 0.9444], BLUE, "D")]
    for lab, v, c, m in series:
        ax.plot(range(3), v, "-" + m, color=c, ms=4, lw=1.5, label=lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(strata)
    ax.set_ylabel("AUROC vs known species")
    ax.set_ylim(0.76, 0.99)
    ax.legend(loc="lower right", handlelength=1.4, fontsize=6.2)
    save(fig, "fig3_openset")


# ------------------------------------------------------------------------------------------------
def fig5_deployment():
    """(a) what a 95 %-precision back-off policy answers; (b) species precision vs coverage;
    (c) reliability. paper/figures/fig4_deployment.json from dev/086 (held-out trap nights)."""
    d = json.loads((OUT / "fig4_deployment.json").read_text())
    arms = [("B8, T = 1 (as in O1)", "B8, T = 1 (O1's readout)", GREY),
            ("B8, T = 1.91 (published)", "B8, T = 1.91 (published)", ORANGE),
            ("P5 (published)", "P5 (published, T = 1)", BLUE)]
    fig = plt.figure(figsize=(FULL, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.5)

    ax = fig.add_subplot(gs[0]); panel(ax, "a")
    for i, (lab, key, _) in enumerate(arms):
        h = d[key]["policy"]["held_out"]
        left = 0
        for lv, col in zip(["species", "genus", "family"], RAMP):
            cov = h[lv]["coverage"]
            ax.barh(i, cov * 100, left=left, color=col, height=0.6, edgecolor="white", linewidth=1)
            if cov > 0.05:
                ax.text(left + cov * 50, i, f"{cov * 100:.0f}", ha="center", va="center", fontsize=6.3,
                        color="white" if lv == "species" else INK)
            left += cov * 100
        ax.barh(i, h["abstain"] * 100, left=left, color=LIGHT, height=0.6, edgecolor="white", linewidth=1)
    ticks = []
    for lab, key, _ in arms:
        h = d[key]["policy"]["held_out"]
        sp = h["species"]["coverage"] * (h["species"]["precision"] or 0)
        ticks.append(f"{lab}\ncorrect: {sp * 100:.1f} species, {h['useful_rate'] * 100:.1f} any rank")
    ax.set_yticks(range(3)); ax.set_yticklabels(ticks, fontsize=6.4)
    ax.set_ylim(2.6, -0.6); ax.set_xlim(0, 100)
    ax.set_xlabel("% of trap images (95 % precision policy)")
    ax.grid(axis="y", visible=False)
    for lab, col, x in (("species", RAMP[0], 0), ("genus", RAMP[1], 23), ("family", RAMP[2], 42),
                        ("abstain", LIGHT, 62)):
        ax.add_patch(plt.Rectangle((x, -0.95), 4, 0.22, fc=col, ec="none", clip_on=False))
        ax.text(x + 5, -0.84, lab, fontsize=6.3, va="center", clip_on=False)

    ax = fig.add_subplot(gs[1]); panel(ax, "b")
    for lab, key, col in arms:
        rc = d[key]["species"]["risk_coverage"]
        ax.plot(rc["coverage"], rc["precision"], color=col, lw=1.5, label=lab.split(" (")[0])
    ax.axhline(0.95, color=INK2, lw=0.6, ls="--")
    ax.text(0.02, 0.952, "95 %", fontsize=6.2, color=INK2, va="bottom")
    ax.set_xlim(0, 1); ax.set_ylim(0.82, 1.0)
    ax.set_xlabel("species coverage"); ax.set_ylabel("species precision")
    ax.annotate("B8, T = 1: 69.7 % of\nimages at exactly p = 1.0", xy=(0.70, 0.972), xytext=(0.05, 0.85),
                fontsize=6, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))

    ax = fig.add_subplot(gs[2]); panel(ax, "c")
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=0.8)
    for lab, key, col in arms:
        r = [b for b in d[key]["species"]["reliability"] if b["n"] >= 30]
        ax.plot([b["conf"] for b in r], [b["acc"] for b in r], "-o", color=col, ms=2.8, lw=1.2,
                label=f"{lab.split(' (')[0]}  ECE {d[key]['species']['ece']:.3f}")
    ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1)
    ax.set_xlabel("species confidence"); ax.set_ylabel("species accuracy")
    ax.legend(loc="upper left", fontsize=6, handlelength=1.2)
    save(fig, "fig4_deployment")


# ------------------------------------------------------------------------------------------------
def fig6_foundation():
    """(a) contamination; (b) frozen vs fine-tuned on the decontaminated fold; (c) adaptation."""
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw=dict(width_ratios=[1.1, 1, 1.1], wspace=0.55))

    ax = axs[0]; panel(ax, "a")   # 2026-08-26-bioclip2-has-seen-two-thirds-of-our-test-fold
    labs = ["our species", "our images", "our test-fold images"]
    vals = [93.3, 65.4, 413865 / 629742 * 100]
    ax.barh(range(3), vals, color=ORANGE, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v:.1f} %", va="center", fontsize=6.4)
    ax.set_yticks(range(3)); ax.set_yticklabels(labs); ax.set_ylim(2.6, -0.6)
    ax.set_xlim(0, 115); ax.set_xticks([0, 50, 100])
    ax.set_xlabel("% inside BioCLIP-2's training data\n(exact GBIF occurrence id)")
    ax.grid(axis="y", visible=False)

    ax = axs[1]; panel(ax, "b")   # 2026-08-28-fine-tuned-bioclip2-beats-us-and-the-head-hurts
    labs = ["frozen", "1e-3", "1e-4", "1e-5"]
    vals = [0.8444, 0.8912, 0.9025, 0.9146]
    ax.bar(range(4), vals, color=[GREY, BLUE, BLUE, BLUE], width=0.6)
    ax.axhline(0.9021, color=ORANGE, lw=1.2)
    ax.text(-0.45, 0.9035, "ours 0.902", color=ORANGE, fontsize=6.2, va="bottom", ha="left")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=6.2)
    ax.set_ylim(0.82, 0.93); ax.set_xticks(range(4)); ax.set_xticklabels(labs)
    ax.set_ylabel("macro-F1, decontaminated fold")
    ax.set_xlabel("frozen, or fine-tuned at lr")
    ax.grid(axis="x", visible=False)

    ax = axs[2]; panel(ax, "c")   # 2026-08-06-adaptation-..., 2026-08-28-frozen-adaptation-...
    stages = ["none", "frozen\ntrunk", "end-to-\nend"]
    ours = [0.6270, 0.7515, 0.7706]       # reference, T2b, B3rep5x (20 M)
    bio = [0.6630, 0.7218, 0.7810]        # P3c, P4 (paper 4.14.3), P5
    ax.plot(range(3), ours, "-o", color=ORANGE, ms=4, lw=1.5, label="ours, 20 M")
    ax.plot(range(3), bio, "-s", color=BLUE, ms=4, lw=1.5, label="BioCLIP-2")
    for y, dy in ((ours, -1), (bio, 1)):
        ax.annotate(f"{y[-1]:.3f}", (2, y[-1]), xytext=(5, 5 * dy), textcoords="offset points",
                    fontsize=6, color=INK2, va="center")
    ax.set_xticks(range(3)); ax.set_xticklabels(stages)
    ax.set_xlim(-0.2, 2.6); ax.set_ylim(0.60, 0.81)
    ax.set_ylabel("probe macro-F1"); ax.set_xlabel("adaptation to trap images")
    ax.legend(loc="lower right", handlelength=1.4)
    save(fig, "fig6_foundation")


if __name__ == "__main__":
    for f in (fig1_protocol, fig2_classifier, fig4_openset, fig5_deployment, fig3_dose, fig6_foundation):
        f()
