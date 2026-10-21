"""Paper Desk: cut paper/DRAFT.md into short reading items, ordered by priority, as one HTML page.

The owner reads one paragraph, table or figure at a time in a published Artifact, edits it or leaves a
note, and moves on. Reviews are stored in the artifact's database (`reviews/<item id>`); the agent
reads them, applies them to DRAFT.md, then rebuilds and republishes the page.

    python dev/090_paper_reader.py build      # DRAFT.md -> data/paper_reader/index.html (+ items.json)

Item ids survive edits: a rebuilt item keeps the id of the most similar item of the same section in
the previous items.json, so a review stays attached to its paragraph after the text changes.
"""
from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "paper" / "DRAFT.md"
TEMPLATE = Path(__file__).with_suffix(".html")
OUT = ROOT / "data" / "paper_reader"
MAX_WORDS = 120

# Reading priority. Each tier lists section-number prefixes; an item belongs to the first tier whose
# prefix matches its section. Order inside a tier follows this list, then the paper.
TIERS = [
    ("Front matter", "What a reviewer reads first.", ["abstract", "1"]),
    ("Overview table", "Every model the paper names, in one table.", ["4.0"]),
    ("Self-training", "The largest lever and the newest results.", ["4.11"]),
    ("The spine", "Interventions belong in the classifier.", ["4.15"]),
    ("Evaluation protocol", "Benchmarks, scoring rules, contamination.", ["3.2", "4.9", "4.14"]),
    ("Long tail and heads", "Where rebalancing and coarse ranks belong.", ["4.13", "4.1"]),
    ("Abstention and novelty", "What the model says when it does not know.", ["4.6", "4.4", "4.5"]),
    ("Margin and other results", "Results likely to move to an appendix.", ["4.3", "4.7", "4.8", "4.10", "4.2", "4.12"]),
    ("Method and setup", "Definitions; check notation.", ["2", "3"]),
    ("Discussion", "What the paper concludes and does not claim.", ["5", "6"]),
    ("Related work", "Every citation still needs checking.", ["1b", "references"]),
]

DECISIONS = [
    {"id": "dec-scope", "title": "Scope", "md":
     "The draft makes 9 contributions over 16 results sections. A paper needs one argument. Which spine?",
     "options": [
         {"key": "classifier", "label": "Classifier stage + protocol + self-training", "rec": True,
          "md": "§4.15 as the spine, with the evaluation protocol (three axes, scoring rules, contamination) "
                "and self-training (§4.11). Margin (§4.3, §4.7) and head scaling (§4.12) go to an appendix. "
                "PLAN's proposal."},
         {"key": "selftrain", "label": "Self-training at light traps",
          "md": "§4.11 with the hold-out and label-budget results as the whole paper; the protocol as setup; "
                "the rest as a second paper. Shortest, most applied."},
         {"key": "all", "label": "Keep everything",
          "md": "One long paper for a venue without a page limit. Least work now, hardest to read."},
     ]},
    {"id": "dec-venue", "title": "Venue", "md": "Where does it go? This sets the length.",
     "options": [
         {"key": "tmlr", "label": "TMLR", "md": "ML journal, no page limit, rolling submission."},
         {"key": "mee", "label": "Methods in Ecology and Evolution",
          "md": "The light-trap and biodiversity audience. Check its word limit."},
         {"key": "ecoinf", "label": "Ecological Informatics", "md": "Applied, fewer length constraints."},
         {"key": "workshop", "label": "A vision workshop (e.g. FGVC, CV4Animals)",
          "md": "Short version first; depends on deadlines."},
     ]},
    {"id": "dec-title", "title": "Title",
     "md": "Current: *Knowing what you don't know: calibrated open-set hierarchical classification for "
           "long-tailed species identification*. It describes the project before the pivot to shift.",
     "options": [
         {"key": "keep", "label": "Keep it", "md": ""},
         {"key": "classifier", "label": "Put the classifier finding in the title",
          "md": "e.g. *Adapt the classifier, not the representation: species identification under domain shift*."},
         {"key": "selftrain", "label": "Put self-training in the title",
          "md": "e.g. *Unlabelled trap images beat labels: self-training for moth identification at light traps*."},
     ]},
    {"id": "dec-45", "title": "§4.5, open-set under shift",
     "md": "Its numbers read the plain head with its worst rule, so the comparison is invalid. "
           "Re-scoring is an evaluation-only GPU run (PLAN step 3).",
     "options": [
         {"key": "rescore", "label": "Re-score, then keep", "rec": True, "md": "One eval job per head."},
         {"key": "drop", "label": "Drop the section", "md": "Mention the gap in Limitations."},
         {"key": "caveat", "label": "Keep it with the caveat", "md": ""},
     ]},
]


def section_key(heading):
    """'### 4.13.1 The same shape' -> '4.13.1'; '## Abstract (draft)' -> 'abstract'."""
    t = heading.lstrip("#").strip()
    m = re.match(r"(\d+(?:\.\d+)*[a-z]?)\b", t)
    if m:
        return m.group(1)
    return t.split()[0].lower().strip("_(")


def tier_of(key):
    """Exact section match first ('1b' is not '1'), then prefix ('4.6a' and '4.13.1' are in '4.6', '4.13')."""
    for exact in (True, False):
        for i, (_, _, prefixes) in enumerate(TIERS):
            for j, p in enumerate(prefixes):
                if key == p or not exact and re.match(rf"{re.escape(p)}([.a-z]|$)", key):
                    return i, j
    return len(TIERS) - 1, 99


def words(s):
    return len(re.findall(r"\S+", s))


def split_long(block):
    """Split a paragraph over MAX_WORDS at sentence ends into pieces of similar size."""
    if words(block) <= MAX_WORDS or block.lstrip().startswith(("|", "$$", "!", ">")):
        return [block]
    lines = block.split("\n")
    if re.match(r"\s*([-*]|\d+\.) ", lines[0]):
        # A list: break only where a bullet (of any depth) starts, so every piece stays verbatim.
        out, cur = [], []
        for ln in lines:
            if cur and re.match(r"\s*([-*]|\d+\.) ", ln) and words("\n".join(cur + [ln])) > MAX_WORDS:
                out.append("\n".join(cur))
                cur = []
            cur.append(ln)
        return out + ["\n".join(cur)]
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z*(])", block)
    n = -(-words(block) // MAX_WORDS)
    target = words(block) / n
    out, cur = [], []
    for s in sents:
        cur.append(s)
        if words(" ".join(cur)) >= target and len(out) < n - 1:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def blocks(text):
    """(heading, block) pairs. Figures merge with their caption, display math with its paragraph."""
    heading, out = "", []
    for raw in re.split(r"\n\s*\n", text):
        b = raw.strip("\n")
        if not b.strip() or b.strip() == "---":
            continue
        if b.startswith("#"):
            first, _, rest = b.partition("\n")
            heading = first
            if not rest.strip():
                continue
            b = rest
        if out and out[-1][0] == heading and (
                b.startswith("**Figure") and out[-1][1].startswith("![")
                or b.startswith("$$") or out[-1][1].rstrip().endswith("$$") and not b.startswith("|")):
            out[-1] = (heading, out[-1][1] + "\n\n" + b)
            continue
        out.append((heading, b))
    return out


def kind_of(b):
    if b.startswith("!["):
        return "figure"
    if b.lstrip().startswith("|"):
        return "table"
    if "$$" in b:
        return "math"
    if b.lstrip().startswith((">",)):
        return "quote"
    if b.lstrip().startswith(("- ", "* ", "1. ")):
        return "list"
    return "text"


def build(a):
    text = DRAFT.read_text()
    body = text.split("\n---\n", 1)[1] if "\n---\n" in text[:3000] else text
    items = []
    for heading, b in blocks(body):
        for piece in split_long(b):
            if piece not in text:
                sys.exit(f"chunk not found verbatim in DRAFT.md: {piece[:80]!r}")
            key = section_key(heading)
            t, p = tier_of(key)
            items.append({"section": heading.lstrip("#").strip(), "key": key, "tier": t, "prio": p,
                          "kind": kind_of(piece), "md": piece, "words": words(piece)})
    for i, it in enumerate(items):
        it["pos"] = i

    # Stable ids: reuse the previous id of the most similar item in the same section.
    prev_path = OUT / "items.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else []
    free = {p["id"]: p for p in prev if not p["id"].startswith("dec-")}
    for it in items:
        cands = [p for p in free.values() if p["section"] == it["section"]] or list(free.values())
        best = max(cands, key=lambda p: difflib.SequenceMatcher(None, p["md"], it["md"]).ratio(), default=None)
        if best and difflib.SequenceMatcher(None, best["md"], it["md"]).ratio() > 0.5:
            it["id"] = best["id"]
            del free[best["id"]]
        else:
            it["id"] = "i" + hashlib.sha1(it["md"].encode()).hexdigest()[:10]

    items.sort(key=lambda it: (it["tier"], it["prio"], it["pos"]))
    ordered = [dict(d, kind="decision", tier=-1, section="Decisions") for d in DECISIONS] + items
    tiers = [{"name": "Decisions", "why": "Answer these first: they decide what gets cut."}] + \
            [{"name": n, "why": w} for n, w, _ in TIERS]
    for it in ordered:
        it["tier"] += 1

    figs = {}
    for m in re.finditer(r"!\[[^\]]*\]\((figures/[^)]+\.png)\)", text):
        f = ROOT / "paper" / m.group(1)
        figs[m.group(1)] = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()

    OUT.mkdir(parents=True, exist_ok=True)
    prev_path.write_text(json.dumps([{k: it[k] for k in ("id", "section", "md")} for it in ordered], indent=1))
    data = {"items": ordered, "tiers": tiers, "figs": figs, "built": a.stamp,
            "paperWords": words(text)}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/"))
    (OUT / "index.html").write_text(html)
    n_words = sum(it.get("words", 0) for it in items)
    print(f"{len(ordered)} items ({len(DECISIONS)} decisions), {n_words} words -> {OUT / 'index.html'} "
          f"({len(html) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("build")
    q.add_argument("--stamp", default="", help="shown on the page, e.g. the date of the rebuild")
    q.set_defaults(fn=build)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
