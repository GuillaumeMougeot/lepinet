# Setup for staged runs: same environment as setup.sh, plus staging onto node-local NVMe.
# Kept separate so the smoke test (one family, read straight from /work) does not pay the copy.
#
# setup.sh is sourced from the mount rather than duplicated here -- the sync puts this repo at
# /work/lepinet, so it is on disk by the time this runs. The STAGE_* vars below switch on
# setup.sh's staging step (stage.py), which copies ONLY the images the parquet references
# (filtered the same way training does) to /tmp/global_lepi/images. The training config's
# img_dir must point at that path; the test config keeps reading fold '0' from /work.
#
# STAGE_MIN_IMG_PER_SPC / STAGE_FAMILY_FILTER MUST match the training config's filtering, or the
# staged set won't match what training opens (stage.py verifies the count and aborts on a short
# copy, but it cannot catch a filter mismatch -- that is on whoever edits these together).

export STAGE_PARQUET=/work/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet
export STAGE_SRC=/work/global_lepi/images
export STAGE_DST=/tmp/global_lepi/images
export STAGE_MIN_IMG_PER_SPC=50
export STAGE_FAMILY_FILTER=          # empty = all families (the full benchmark set)

# shellcheck disable=SC1091
source /work/lepinet/ucloud/setup.sh
