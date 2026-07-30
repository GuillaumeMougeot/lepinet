"""Documentation hygiene is a test, not a habit.

This repo's prose (journal, paper, guided map, docs) is load-bearing: it is how a fresh contributor
-- human or agent -- reconstructs why anything is the way it is. Prose with broken links and a stale
status board is worse than no prose, because it is confidently wrong.

So the rules that a machine *can* check are checked here, and fail the build like a broken import.
See `dev/060_doc_health.py` for what they are, and `CLAUDE.md` for the ones that need judgement.
"""
import importlib.util
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parent.parent / "dev" / "060_doc_health.py"


def _load():
    # dev/ modules start with a digit, so they cannot be imported by name.
    spec = importlib.util.spec_from_file_location("doc_health", SPEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not SPEC.exists(), reason="dev/ not present (installed package only)")
def test_docs_are_healthy():
    failures = _load().run()
    assert not failures, "documentation drift:\n  - " + "\n  - ".join(failures)
