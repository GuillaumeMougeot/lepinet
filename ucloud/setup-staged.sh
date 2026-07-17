# Setup for full-dataset runs: same environment as setup.sh, plus staging the images onto
# node-local NVMe. Kept separate so the smoke test (one family, read straight from /work)
# does not pay a 1 h copy.
#
# setup.sh is sourced from the mount rather than duplicated here -- the sync puts this
# repo at /work/lepinet, so it is on disk by the time this runs. STAGE_IMAGES is what
# switches its staging step on.

export STAGE_IMAGES=/work/global_lepi/images
# shellcheck disable=SC1091
source /work/lepinet/ucloud/setup.sh
