"""Fill a README_*.md template's {{PLACEHOLDERS}} from a release folder's measured JSON files.

    python dev/083_hf_release_files/fill_card.py README_bioclip2.md <release_dir> key=value ...

Numbers are copied mechanically from thresholds.json / eval_*.json so the card cannot drift from
what was measured. Extra key=value pairs fill anything else.
"""
import json
import re
import sys
from pathlib import Path


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f} %"


def thresholds_table(t: dict) -> str:
    h = t["on_held_out_half"]
    th = {lv: {"threshold": v} for lv, v in t["thresholds_split"].items()}
    shipped = t["levels"]
    rows = ["| rank | threshold (fitted on half the nights) | share of held-out images answered here | precision (held-out nights) |",
            "|---|---|---|---|"]
    for lv in ("species", "genus", "family"):
        v = th[lv]["threshold"]
        shown = "never used" if v > 1 else f"{v:.3f}"
        rows.append(f"| {lv} | {shown} | {pct(h[lv]['coverage'])} | {pct(h[lv]['precision'])} |")
    rows.append(f"| *unknown* | | {pct(h['abstain'])} | |")
    rows.append("")
    rows.append(f"Overall, on the held-out nights: **{pct(1 - h['abstain'])} of images get an answer, "
                f"{pct(h['precision_answered'])} of those answers are correct**, so "
                f"{pct(h['useful_rate'])} of all images get a correct answer. "
                f"The thresholds shipped in `thresholds.json` (`levels`) are refitted on all nights: "
                + ", ".join(f"{lv} {'never' if shipped[lv]['threshold'] > 1 else round(shipped[lv]['threshold'], 3)}"
                            for lv in ("species", "genus", "family"))
                + ". The split fit and its held-out result are stored alongside them.")
    return "\n".join(rows)


def main():
    tpl, rel = Path(sys.argv[1]), Path(sys.argv[2])
    extra = dict(kv.split("=", 1) for kv in sys.argv[3:])
    s = tpl.read_text()
    thr = rel / "thresholds.json"
    if thr.exists():
        s = s.replace("{{THRESHOLDS_TABLE}}", thresholds_table(json.loads(thr.read_text())))
    for k, v in extra.items():
        s = s.replace("{{" + k + "}}", v.replace("\\n", "\n"))
    left = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", s)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (rel / "README.md").write_text(s)
    print(f"wrote {rel / 'README.md'}")


if __name__ == "__main__":
    main()
