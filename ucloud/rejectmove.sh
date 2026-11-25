# Global Lepidoptera, step 2 of emptying global_lepi/images (after tailmove.sh): move the 720,072 images of kept species
# that the quality filter rejected (in the postprocessed parquet, not in the quality-filtered one; not in the shards,
# not in the workstation's copy) to global_lepi/quality_rejected/<speciesKey>/<file>. Renames inside one mount, resumable
# (the first run, 2026-11-25, hit its 2 h cap with three filesystem calls per file; now one, with progress lines).
# Afterwards images/ holds exactly the quality-filtered set, which global_lepi_shards-v1 has byte for byte.
set -euo pipefail
G=/work/global_lepi
python3 - <<'PY'
import os
G = "/work/global_lepi"
moved = already = missing = 0
made = set()
for n, line in enumerate(open(f"{G}/quality_rejected_files.txt"), 1):
    rel = line.strip()
    if not rel:
        continue
    d = f"{G}/quality_rejected/{rel.split('/')[0]}"
    if d not in made:   # one makedirs per species folder, not per file
        os.makedirs(d, exist_ok=True)
        made.add(d)
    try:                # one call per file: the rename itself; on a miss, see whether a previous run moved it
        os.rename(f"{G}/images/{rel}", f"{G}/quality_rejected/{rel}")
        moved += 1
    except FileNotFoundError:
        if os.path.exists(f"{G}/quality_rejected/{rel}"):
            already += 1
        else:
            missing += 1
    if n % 20000 == 0:
        print(f"{n:,} lines: moved {moved:,}, already moved {already:,}, missing {missing:,}", flush=True)
print(f"rejected: moved {moved}, already moved {already}, missing {missing}", flush=True)
PY
echo "quality_rejected: $(find "$G/quality_rejected" -type f | wc -l) files"
echo "images/ files: $(find "$G/images" -type f | wc -l)"
echo "images/ files not .jpg:"; find "$G/images" -type f ! -name '*.jpg' | head -5
