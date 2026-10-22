"""Merge Desk: the step-by-step insertion of our results into the colleague's Overleaf draft.

Each step is one LaTeX piece in `paper/merge/NN-slug.tex`: a paragraph, table or figure, with a
header saying where it goes in the Overleaf project and why. The owner pastes one piece at a time
into Overleaf and marks it on the page; the page stores that in the artifact database
(`steps/<NN>`). `apply` then mirrors the pasted text into the local copy of the Overleaf project
(`archive/paper/asger`, gitignored), so the next pieces are written against what Overleaf holds.

    python dev/094_merge_desk.py build                 # -> data/merge_desk/index.html
    python dev/094_merge_desk.py apply --reviews DIR   # DIR/steps/<NN>.json, as saved by ArtifactData

Piece header, one `%% key: value` line each: title, target (file in the Overleaf project), anchor
(an exact line of LaTeX), action (before | after | after-paragraph), why, source, check (items
separated by ' | '), status (ready | pasted | hold).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIECES = ROOT / "paper" / "merge"
OVERLEAF = ROOT / "archive" / "paper" / "asger"
TEMPLATE = Path(__file__).with_suffix(".html")
OUT = ROOT / "data" / "merge_desk"

# The whole plan, so every step is visible before it is drafted. A drafted piece replaces its row.
PLANNED = [
    ("01", "Methods", "Global Lepidoptera, the training dataset"),
    ("02", "Methods", "Light-trap images, the out-of-domain dataset"),
    ("03", "Methods", "The trap split: adaptation set, probe set, held-out species"),
    ("04", "Methods", "Training: two recipes, one to be kept"),
    ("05", "Methods", "Self-training on unlabelled trap images"),
    ("06", "Methods", "Adapting the classifier only, on a frozen backbone"),
    ("07", "Methods", "Evaluation: three axes and the open-set benchmark"),
    ("08", "Methods", "Open-set scoring rules"),
    ("09", "Methods", "Which classes a macro average counts (predicted-only species)"),
    ("10", "Methods", "Calibration split grouped by trap and night"),
    ("11", "Methods", "Two kinds of noise: retraining spread and bootstrap"),
    ("12", "Methods", "Backing off to genus or family: thresholds per rank"),
    ("13", "Results", "Hierarchy: coarse parameters against coarse supervision"),
    ("14", "Results", "Novelty is graded by taxonomic distance (with the alignment result)"),
    ("15", "Results", "Accuracy saturates; the three axes disagree"),
    ("16", "Results", "Metric conventions change the ranking"),
    ("17", "Results", "Open-set scoring rules do not transfer across scale"),
    ("18", "Results", "The test fold is inside a foundation model's training set"),
    ("19", "Results", "Long-tail rebalancing inverts under shift; cRT removes the trade"),
    ("20", "Results", "Self-training: dose, labels against pseudo-labels, new traps"),
    ("21", "Results", "Adaptation is a classifier problem"),
    ("22", "Results", "Abstention under shift is a composition, not a score"),
    ("23", "Appendix", "Margin analysis, nuisance augmentation, head scaling, recipe ladder, noise floors"),
]


def parse(path):
    head, body = {}, []
    for line in path.read_text().splitlines():
        m = re.match(r"%%\s*(\w[\w-]*):\s*(.*)", line)
        if m and not body:
            head[m.group(1)] = m.group(2).strip()
        else:
            body.append(line)
    head["latex"] = "\n".join(body).strip("\n") + "\n"
    head["id"] = path.name[:2]
    head["file"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    head["check"] = [c.strip() for c in head.get("check", "").split("|") if c.strip()]
    head["source"] = [s.strip() for s in head.get("source", "").split(";") if s.strip()]
    return head


def anchor_state(step, earlier):
    """Where the anchor line is: in the local Overleaf copy, or created by an earlier piece."""
    target = OVERLEAF / step.get("target", "")
    anchor = step.get("anchor", "")
    if target.exists() and anchor and anchor in target.read_text():
        return "found"
    for e in earlier:
        if anchor and anchor in e["latex"]:
            return f"created by step {e['id']}"
    return "not found"


def build(a):
    drafted = {p["id"]: p for p in map(parse, sorted(PIECES.glob("[0-9][0-9]-*.tex")))}
    steps, earlier = [], []
    for sid, part, title in PLANNED:
        if sid in drafted:
            s = dict(drafted[sid], part=part)
            s["anchor_state"] = anchor_state(s, earlier)
            earlier.append(s)
        else:
            s = {"id": sid, "part": part, "title": title, "status": "planned"}
        steps.append(s)
    for sid in sorted(set(drafted) - {p[0] for p in PLANNED}):
        sys.exit(f"piece {sid} is not in PLANNED")
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"steps": steps, "built": a.stamp}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/"))
    (OUT / "index.html").write_text(html)
    n = sum(1 for s in steps if s.get("status") != "planned")
    bad = [s["id"] for s in steps if s.get("anchor_state") == "not found"]
    print(f"{n} drafted of {len(steps)} steps -> {OUT / 'index.html'}" + (f"; anchors NOT FOUND: {bad}" if bad else ""))


def insert(text, anchor, action, latex):
    i = text.find(anchor)
    if i < 0:
        raise ValueError(f"anchor not found: {anchor!r}")
    if action == "before":
        j = text.rfind("\n", 0, i) + 1
        return text[:j] + latex.rstrip("\n") + "\n\n" + text[j:]
    end = text.find("\n", i)
    end = len(text) if end < 0 else end
    if action == "after-paragraph":
        m = re.search(r"\n\s*\n", text[end:])
        end = end + m.start() if m else len(text)
    return text[:end] + "\n" + ("\n" if action == "after-paragraph" else "") + latex.rstrip("\n") + "\n" + text[end:]


def apply(a):
    """Mirror pasted pieces into the local Overleaf copy, in step order, and mark them pasted."""
    for f in sorted(Path(a.reviews).rglob("*.json")):
        r = json.loads(f.read_text())
        r = r.get("data", r)
        if r.get("status") not in ("pasted", "changed"):
            continue
        sid = f.stem
        piece = next(PIECES.glob(f"{sid}-*.tex"))
        p = parse(piece)
        if p.get("status") == "pasted":
            continue
        latex = r.get("edited") or p["latex"]
        target = OVERLEAF / p["target"]
        target.write_text(insert(target.read_text(), p["anchor"], p["action"], latex))
        src = piece.read_text()
        if r.get("edited"):
            src = "\n".join(l for l in src.splitlines() if l.startswith("%%")) + "\n" + r["edited"]
        piece.write_text(re.sub(r"%% status: \w+", "%% status: pasted", src, count=1))
        print(f"step {sid}: mirrored into {p['target']}" + (" (owner's edited version)" if r.get("edited") else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("build")
    q.add_argument("--stamp", default="")
    q.set_defaults(fn=build)
    q = sub.add_parser("apply")
    q.add_argument("--reviews", required=True)
    q.set_defaults(fn=apply)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
