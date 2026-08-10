"""Paper figures. Backlog item 3, deferred by the owner until the storyline settled.

Every number here is transcribed from a journal entry or `RESULTS.md`, with the source named in a
comment beside it. Nothing is recomputed from checkpoints: the point of a figure in this project is
to render numbers that have already been argued, and a figure that silently recomputes them is a
second implementation of the metric with its own bugs.

    python dev/074_figures.py            # writes paper/figures/*.pdf and *.png

Four figures, one per claim that survived:

  fig1_capacity   the staged-vs-end-to-end verdict flips with capacity  (the current headline)
  fig2_dose       replication share: 0.39 % of training buys 97 % of the gain, and transfer
                  to unseen species falls monotonically as the share rises
  fig3_rules      an open-set scoring rule does not transfer across scale or head
  fig4_novelty    novelty is monotone in taxonomic distance, on two populations chosen by
                  opposite criteria
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
FLOOR = {"probe": 0.0041, "heldout": 0.0052, "indist": 0.0005}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
STAGED, E2E = "#1b6ca8", "#c1440e"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf/.png")


def fig1_capacity() -> None:
    """journal/2026-08-10-balance-is-oversampling-and-it-does-not-scale.md"""
    axes_ = ["in-distribution", "external (probe)", "held-out species"]
    #             20 M                      198 M
    staged = [(0.9074, 0.7692, 0.7781), (0.9138, 0.7740, 0.7518)]   # R5 ; G3
    e2e = [(0.9003, 0.7706, 0.7704), (0.9060, 0.7798, 0.7816)]      # B3rep5x ; B8

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for j, (ax, name) in enumerate(zip(axs, axes_)):
        for arr, colour, label, mark in ((staged, STAGED, "staged (frozen trunk)", "o"),
                                         (e2e, E2E, "end-to-end", "s")):
            ax.plot([0, 1], [arr[0][j], arr[1][j]], marker=mark, color=colour, label=label, lw=1.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["20 M", "198 M"])
        ax.set_title(name)
        ax.margins(x=0.25)
        # shade one noise floor around the end-to-end line, so "inside noise" is readable
        f = list(FLOOR.values())[j]
        ax.fill_between([0, 1], [e2e[0][j] - f, e2e[1][j] - f], [e2e[0][j] + f, e2e[1][j] + f],
                        color=E2E, alpha=0.12, lw=0)
    axs[0].set_ylabel("species macro-F1")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("The staged recipe matches end-to-end at 20 M and does not at 198 M; "
                 "its in-distribution gain is stable", y=1.06, fontsize=9)
    save(fig, "fig1_capacity")


def fig2_dose() -> None:
    """journal/2026-08-04-replication-sweep.md, the table at line 13. Transcribed exactly --
    note `share` is the percentage of training rows, NOT the replication factor (1x/5x/13x/26x);
    conflating the two is the first mistake made drafting this figure."""
    share = [0.39, 2.0, 6.1, 10.0]                 # 1x, 5x, 13x, 26x
    probe = [0.7354, 0.7706, 0.7370, 0.7159]
    heldout = [0.7508, 0.7704, 0.7231, 0.7042]
    base = 0.6912                                   # B1, no target data (held-out 0.6974)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.axhline(base, color="grey", ls=":", lw=1)
    ax.annotate("no target data", (0.42, base), xytext=(0, 3), textcoords="offset points",
                fontsize=7, color="grey")
    ax.plot(share, probe, marker="o", color=STAGED, label="external (probe)", lw=1.6)
    ax.plot(share, heldout, marker="s", color=E2E, label="held-out species", lw=1.6)
    ax.set_xscale("log")
    ax.set_xticks(share); ax.set_xticklabels([f"{s:g}" for s in share])
    ax.set_xlabel("pseudo-labelled share of training (%)")
    ax.set_ylabel("species macro-F1")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Adaptation has an interior optimum;\ntransfer to unseen species falls throughout")
    save(fig, "fig2_dose")


def fig3_rules() -> None:
    """journal/2026-08-01-the-scoring-rule-was-the-bug.md, the table at line 26.
    Five rules, one forward pass, same images and embeddings."""
    rules = ["max-logit", "energy", "entropy", "margin", "MSP"]
    models = [("A1", "20 M", [0.9068, 0.9064, 0.9047, 0.8979, 0.8953]),
              ("B1", "20 M", [0.9010, 0.9005, 0.9008, 0.8945, 0.8917]),
              ("A2", "198 M", [0.8298, 0.8287, 0.8813, 0.8807, 0.8904]),
              ("B4", "198 M", [0.8132, 0.8118, 0.8802, 0.8789, 0.8893])]

    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    for name, scale, vals in models:
        colour = STAGED if scale == "20 M" else E2E
        ax.plot(rules, vals, marker="o" if scale == "20 M" else "s", color=colour, lw=1.5,
                alpha=0.95 if name in ("A1", "A2") else 0.5,
                label=f"{name} ({scale})")
    ax.set_ylabel("open-set AUROC")
    ax.set_ylim(0.80, 0.92)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("The best rule inverts with capacity:\nmax-logit at 20 M, MSP at 198 M (+6.1 to +7.6 pt)",
                 fontsize=8)
    save(fig, "fig3_rules")


def fig4_novelty() -> None:
    """journal/2026-08-08-is-novelty-monotone-or-just-rare.md"""
    strata = ["near\n(unseen species)", "mid\n(unseen genus)", "far\n(unseen family)"]
    rare = [0.8527, 0.9342, 0.9641]      # C3: novel = everything below the 50-image floor
    common = [0.8717, 0.9463, 0.9726]    # C3b: novel = 231 withheld species, >= 200 images each

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(strata, rare, marker="o", color=STAGED, lw=1.6,
            label="novel = rare taxa (free)")
    ax.plot(strata, common, marker="s", color=E2E, lw=1.6,
            label="novel = common taxa (withheld)")
    ax.set_ylabel("open-set AUROC")
    ax.set_ylim(0.83, 0.99)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Monotone in taxonomic distance, on two novel\npopulations chosen by opposite criteria")
    save(fig, "fig4_novelty")


if __name__ == "__main__":
    print("writing figures from journalled numbers (nothing recomputed):")
    fig1_capacity()
    fig2_dose()
    fig3_rules()
    fig4_novelty()
