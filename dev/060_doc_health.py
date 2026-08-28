"""Machine-checked documentation hygiene — so drift is *detected*, not remembered.

This repository carries an unusual amount of prose: a journal, a paper draft, a guided map, three
docs pages, a results snapshot. That is deliberate (see `CLAUDE.md`), but prose rots silently —
a renamed file breaks twenty links, a finished run leaves `PLAN.md` describing a world that no
longer exists, a new journal entry never reaches the index.

Discipline does not scale to that. Checks do. Everything below is a rule that can be violated by
accident and confirmed by a machine; anything requiring judgement is deliberately *not* here.

    python dev/060_doc_health.py           # report, exit 1 on failure
    python dev/060_doc_health.py --quiet   # silent unless something is wrong

`tests/test_doc_health.py` runs this in CI, so a broken link fails the build like a broken import.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "journal"

#: Files in journal/ that are living documents: no date, kept current, never frozen.
LIVING = {"README.md", "PLAN.md", "DIRECTIONS.md"}
#: The four kinds an archival entry may declare. See CLAUDE.md for what each means.
KINDS = {"research", "subproject", "infrastructure", "incident", "living"}
DATED = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-.+\.md$")

#: Docs a newcomer or reviewer reads. The owner asked for no emoji in these; the journal is
#: historical record and is left alone.
STRUCTURAL = ["START-HERE.md", "README.md", "RESULTS.md", "CLAUDE.md", "DEVELOPER.md",
              "journal/README.md", "journal/PLAN.md", "journal/DIRECTIONS.md", "paper/DRAFT.md"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF️]")

# Skip generated/vendored trees. `data` is a symlink to machine-local storage.
SKIP = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
        "data", "archive", "lepinet.egg-info", "mini_trainer", "mini_metrics", "lepinet-app"}


def md_files() -> list[Path]:
    return [p for p in ROOT.rglob("*.md")
            if not any(part in SKIP for part in p.relative_to(ROOT).parts)]


def journal_entries() -> tuple[list[Path], list[Path]]:
    """(living, archival) — the two tiers, split by naming convention."""
    files = sorted(p for p in JOURNAL.glob("*.md"))
    return ([p for p in files if p.name in LIVING],
            [p for p in files if p.name not in LIVING])


# --------------------------------------------------------------------------- checks
# Each check appends human-readable failures. A check that cannot fail mechanically does not
# belong here -- judgement lives in CLAUDE.md, not in an assertion.

def check_journal_naming(fail):
    _, archival = journal_entries()
    for p in archival:
        if not DATED.match(p.name):
            fail(f"journal/{p.name}: not YYYY-MM-DD-question.md, and not a known living doc "
                 f"({sorted(LIVING)}). Date it by when the question was *opened*.")


def check_kind_and_status(fail):
    _, archival = journal_entries()
    for p in archival:
        head = "\n".join(p.read_text().split("\n")[:12])
        m = re.search(r"\*\*Kind:\*\*\s*([a-z]+)", head)
        if not m:
            fail(f"journal/{p.name}: no '**Kind:**' in the first 12 lines (one of {sorted(KINDS)}).")
        elif m.group(1) not in KINDS:
            fail(f"journal/{p.name}: unknown kind {m.group(1)!r}; expected one of {sorted(KINDS)}.")
        if "**Status:**" not in head:
            fail(f"journal/{p.name}: no '**Status:**' in the first 12 lines "
                 f"(OPEN / RESOLVED / SUPERSEDED + the answer).")


def check_index_complete(fail):
    """Every journal file must be reachable from journal/README.md, or it is effectively lost."""
    text = (JOURNAL / "README.md").read_text()
    living, archival = journal_entries()
    missing = [p.name for p in living + archival
               if p.name != "README.md" and p.name not in text]
    for name in missing:
        fail(f"journal/{name}: not linked from journal/README.md -- unreachable from the map.")


def check_links(fail):
    """Every relative markdown link and every [[wikilink]] must resolve."""
    link = re.compile(r"\[[^\]]*\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")
    wiki = re.compile(r"\[\[([0-9]{4}-[0-9]{2}-[0-9]{2}-[A-Za-z0-9._-]+|PLAN|DIRECTIONS)\]\]")
    for p in md_files():
        text = p.read_text()
        for target in link.findall(text):
            if target.startswith(("http://", "https://")):
                continue
            if not (p.parent / target).resolve().exists():
                fail(f"{p.relative_to(ROOT)}: broken link -> {target}")
        for name in wiki.findall(text):
            if not (JOURNAL / f"{name}.md").exists():
                fail(f"{p.relative_to(ROOT)}: broken wikilink -> [[{name}]]")


def check_no_emoji(fail):
    for rel in STRUCTURAL:
        p = ROOT / rel
        if not p.exists():
            continue
        hits = sorted(set(EMOJI.findall(p.read_text())))
        if hits:
            fail(f"{rel}: emoji {' '.join(repr(c) for c in hits)} -- the owner asked for none in "
                 f"structural docs. Use words.")


def check_plan_is_current(fail):
    """`PLAN.md` claims to be true *today*. If a journal entry is newer, it probably is not.

    This is the one check that catches the failure mode that actually matters: work happened and
    the status board was not updated. It cannot prove PLAN.md is right -- only that it has been
    touched since the most recent thing that could have invalidated it.
    """
    plan = JOURNAL / "PLAN.md"
    if not plan.exists():
        fail("journal/PLAN.md is missing -- it is the entry point for 'where are we'.")
        return
    m = re.search(r"\*\*Last updated:\*\*\s*(\d{4}-\d{2}-\d{2})", plan.read_text())
    if not m:
        fail("journal/PLAN.md: no '**Last updated:** YYYY-MM-DD' in the header.")
        return
    updated = date.fromisoformat(m.group(1))
    _, archival = journal_entries()
    newest = max((date(*map(int, DATED.match(p.name).groups()))
                  for p in archival if DATED.match(p.name)), default=updated)
    if newest > updated:
        fail(f"journal/PLAN.md last updated {updated}, but journal entries exist from {newest}. "
             f"Work landed without the status board moving.")


def check_math_renders(fail):
    """LaTeX that a terminal shows fine and GitHub silently refuses to render.

    GitHub's MathJax subset rejects some macros outright ("The following macros are not allowed:
    operatorname"), needs `$$` alone on its own line for display math, and cannot parse an inline
    `$...$` that spans a newline -- which markdown reflowing produces very easily. All three fail
    *silently* in the sense that nothing is wrong locally; the equation just does not appear.
    Found the hard way on 2026-08-28, when six equations in the paper had never rendered.
    """
    blocked = ["\\operatorname", "\\lVert", "\\rVert", "\\substack", "\\bm{", "\\\\[",
               "\\mathbb{1}", "\\overset"]
    for rel in ["paper/DRAFT.md", "docs/concepts.md", "README.md"]:
        p = ROOT / rel
        if not p.exists():
            continue
        for n, line in enumerate(p.read_text().split("\n"), 1):
            for b in blocked:
                if b in line:
                    fail(f"{rel}:{n}: {b!r} is not in GitHub's MathJax subset -- "
                         f"use \\mathrm{{}}, \\|, or plain \\\\.")
            if "$$" in line and line.strip() != "$$":
                fail(f"{rel}:{n}: `$$` must be alone on its line for GitHub to render a display "
                     f"block.")
            stripped = re.sub(r"\$\$.*?\$\$", "", line)
            if len(re.findall(r"(?<!\$)\$(?!\$)", stripped)) % 2:
                fail(f"{rel}:{n}: inline math spans a line break; GitHub renders it as literal "
                     f"text. Keep `$...$` on one line.")


CHECKS = [check_journal_naming, check_kind_and_status, check_index_complete,
          check_links, check_no_emoji, check_plan_is_current, check_math_renders]


def run() -> list[str]:
    failures: list[str] = []
    for check in CHECKS:
        check(failures.append)
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    failures = run()
    if failures:
        print(f"doc health: {len(failures)} problem(s)\n")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    if not a.quiet:
        living, archival = journal_entries()
        print(f"doc health: OK  ({len(archival)} archival entries, {len(living)} living, "
              f"{len(md_files())} markdown files checked)")


if __name__ == "__main__":
    main()
