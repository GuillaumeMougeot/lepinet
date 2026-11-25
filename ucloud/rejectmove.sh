# Global Lepidoptera, step 2 of emptying global_lepi/images (after tailmove.sh): move the 720,072 images of kept species
# that the quality filter rejected (in the postprocessed parquet, not in the quality-filtered one; not in the shards,
# not in the workstation's copy) to global_lepi/quality_rejected/<speciesKey>/<file>. Renames inside one mount, resumable.
# Afterwards images/ holds exactly the quality-filtered set, which global_lepi_shards-v1 has byte for byte.
set -euo pipefail
G=/work/global_lepi
python3 - <<'PY'
import os
G = "/work/global_lepi"
moved = already = missing = 0
for line in open(f"{G}/quality_rejected_files.txt"):
    rel = line.strip()
    if not rel:
        continue
    src, dst = f"{G}/images/{rel}", f"{G}/quality_rejected/{rel}"
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
        moved += 1
    elif os.path.exists(dst):
        already += 1
    else:
        missing += 1
print(f"rejected: moved {moved}, already moved {already}, missing {missing}", flush=True)
PY
echo "quality_rejected: $(find "$G/quality_rejected" -type f | wc -l) files"
echo "images/ files: $(find "$G/images" -type f | wc -l)"
echo "images/ files not .jpg:"; find "$G/images" -type f ! -name '*.jpg' | head -5
