"""Machine-checked documentation hygiene — so drift is *detected*, not remembered.

This repository carries an unusual amount of prose: a journal, a paper draft, the README, the
results registry, the docs pages. That is deliberate (see `CLAUDE.md`), but prose rots silently —
a renamed file breaks twenty links, a finished run leaves `PLAN.md` describing a world that no
longer exists, a new journal entry never reaches the index.

Discipline does not scale to that. Checks do. Everything below is a rule that can be violated by
accident and confirmed by a machine; anything requiring judgement is deliberately *not* here.

    python tests/test_doc_health.py           # report, exit 1 on failure
    python tests/test_doc_health.py --quiet   # silent unless something is wrong
    pytest -q tests/test_doc_health.py        # the same checks as a test; CI runs this

Lived in `dev/060_doc_health.py` until 2026-10-02. It is long-term repository tooling, not an
experiment (dev/) and not part of the installed package (src/), so it sits with the tests it is.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "journal"
#: Journal entries that are not research (subprojects, infrastructure, incidents) live here.
JOURNAL_ENGINEERING = JOURNAL / "engineering"
#: The status board, the one file meant to be true today. At the repo root since 2026-10-02.
PLAN = ROOT / "PLAN.md"

#: Files in journal/ that are living documents: no date, kept current, never frozen.
LIVING = {"README.md"}
#: The four kinds an archival entry may declare. See CLAUDE.md for what each means.
KINDS = {"research", "subproject", "infrastructure", "incident", "living"}
DATED = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-.+\.md$")

#: Docs a newcomer or reviewer reads. The owner asked for no emoji in these; the journal is
#: historical record and is left alone.
STRUCTURAL = ["README.md", "PLAN.md", "RESULTS.md", "CLAUDE.md", "journal/README.md",
              "paper/DRAFT.md"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF️]")

# Skip generated/vendored trees. `data` is a symlink to machine-local storage.
SKIP = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
        "data", "archive", "lepinet.egg-info", "mini_trainer", "mini_metrics", "lepinet-app"}


def md_files() -> list[Path]:
    # `archive` is skipped only at the top level: journal/engineering/ holds live, linked entries.
    return [p for p in ROOT.rglob("*.md")
            if p.relative_to(ROOT).parts[0] != "archive"
            and not any(part in SKIP - {"archive"} for part in p.relative_to(ROOT).parts)]


def journal_entries() -> tuple[list[Path], list[Path]]:
    """(living, archival) — the two tiers, split by naming convention."""
    files = sorted([*JOURNAL.glob("*.md"), *JOURNAL_ENGINEERING.glob("*.md")])
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
    wiki = re.compile(r"\[\[([0-9]{4}-[0-9]{2}-[0-9]{2}-[A-Za-z0-9._-]+|PLAN)\]\]")
    for p in md_files():
        text = p.read_text()
        # Full GitHub URLs into this repo (docs/ pages must use them) are checked like relative
        # links: moving a file breaks them just as silently.
        for path in re.findall(r"github\.com/GuillaumeMougeot/lepinet/blob/main/([^)#\s]+)", text):
            if not (ROOT / path).exists():
                fail(f"{p.relative_to(ROOT)}: broken repo URL -> {path}")
        for target in link.findall(text):
            if target.startswith(("http://", "https://")):
                continue
            if not (p.parent / target).resolve().exists():
                fail(f"{p.relative_to(ROOT)}: broken link -> {target}")
        for name in wiki.findall(text):
            if name == "PLAN":
                if not PLAN.exists():
                    fail(f"{p.relative_to(ROOT)}: broken wikilink -> [[PLAN]]")
                continue
            if not any((d / f"{name}.md").exists() for d in (JOURNAL, JOURNAL_ENGINEERING)):
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
    plan = PLAN
    if not plan.exists():
        fail("PLAN.md is missing -- it is the entry point for 'where are we'.")
        return
    m = re.search(r"\*\*Last updated:\*\*\s*(\d{4}-\d{2}-\d{2})", plan.read_text())
    if not m:
        fail("PLAN.md: no '**Last updated:** YYYY-MM-DD' in the header.")
        return
    updated = date.fromisoformat(m.group(1))
    _, archival = journal_entries()
    newest = max((date(*map(int, DATED.match(p.name).groups()))
                  for p in archival if DATED.match(p.name)), default=updated)
    if newest > updated:
        fail(f"PLAN.md last updated {updated}, but journal entries exist from {newest}. "
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


def check_docs_site_links(fail):
    """Pages in docs/ are built by `mkdocs build --strict`, which aborts on any relative link that
    leaves docs/. Link to the rest of the repo with a full GitHub URL instead. The Docs workflow
    failed on exactly this on 2026-10-02."""
    link = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
    docs = ROOT / "docs"
    for p in docs.glob("*.md"):
        for target in link.findall(p.read_text()):
            if re.match(r"^[a-z]+:", target):
                continue
            if docs not in (p.parent / target).resolve().parents and (p.parent / target).resolve() != docs:
                fail(f"docs/{p.name}: relative link leaves docs/ -> {target}; mkdocs --strict "
                     f"rejects it. Use https://github.com/GuillaumeMougeot/lepinet/blob/main/...")


CHECKS = [check_journal_naming, check_kind_and_status, check_index_complete,
          check_links, check_docs_site_links, check_no_emoji, check_plan_is_current, check_math_renders]


def run() -> list[str]:
    failures: list[str] = []
    for check in CHECKS:
        check(failures.append)
    return failures


def test_docs_are_healthy():
    failures = run()
    assert not failures, "documentation drift:\n  - " + "\n  - ".join(failures)


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
