"""ucloud/stage.py: a relabelled parquet copies from the legacy folder to the new one."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

STAGE = Path(__file__).resolve().parent.parent / "ucloud/stage.py"


def stage(tmp_path, rows):
    src = tmp_path / "src"
    for r in rows:
        folder = src / str(r.get("legacySpeciesKey", r["speciesKey"]))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / r["filename"]).write_text(r["filename"])
    parquet = tmp_path / "m.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    dst = tmp_path / "dst"
    subprocess.run([sys.executable, str(STAGE), "--parquet", str(parquet), "--src", str(src), "--dst", str(dst),
                    "--min-img-per-spc", "0", "--readers", "2"], check=True, capture_output=True)
    return dst


def test_plain_parquet_keeps_its_layout(tmp_path):
    rows = [{"speciesKey": "1", "filename": "a.jpg", "set": "2", "familyKey": "9"},
            {"speciesKey": "2", "filename": "b.jpg", "set": "0", "familyKey": "9"}]
    dst = stage(tmp_path, rows)
    assert (dst / "1/a.jpg").read_text() == "a.jpg"
    assert not (dst / "2/b.jpg").exists()  # the test fold is not staged


def test_relabelled_parquet_reads_legacy_folder(tmp_path):
    rows = [{"speciesKey": "H9MH", "legacySpeciesKey": "11", "filename": "a.jpg", "set": "2", "familyKey": "F"},
            {"speciesKey": "H9MH", "legacySpeciesKey": "12", "filename": "b.jpg", "set": "2", "familyKey": "F"}]
    dst = stage(tmp_path, rows)
    assert sorted(p.name for p in (dst / "H9MH").iterdir()) == ["a.jpg", "b.jpg"]


def test_module_imports():
    spec = importlib.util.spec_from_file_location("stage", STAGE)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
