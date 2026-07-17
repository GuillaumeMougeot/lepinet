# Environment setup for a lepinet job on UCloud, embedded into the batch script by
# ucloud-api. Runs with cwd = /work/lepinet (the synced working tree).
#
# The job image (pytorch-te) ships its own PyTorch, but this builds a fresh venv from
# lepinet's pyproject instead: that pins torch==2.12.1/torchvision==0.27.1, which the
# pyproject comments say the compiled ops depend on matching exactly.

set -x

# The image has no uv. (This spec deliberately avoids ucloud-api's `python = "uv"`
# shortcut: that runs `uv sync`, which would put the venv on the network drive.)
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# Keep the venv and the download cache on the job's local disk. /work is network storage;
# unpacking ~4 GB of torch's small files onto it is slow and burns drive quota.
export UV_CACHE_DIR=/tmp/uv-cache
uv venv --python 3.14 /tmp/venv
# shellcheck disable=SC1091
source /tmp/venv/bin/activate

# One resolution for both projects so torch is pinned once. lepinet sets packages=[], so
# installing it contributes dependencies only; mini_trainer is a sibling repo locally and
# a second mount here (/work/mini_trainer).
uv pip install /work/lepinet /work/mini_trainer

# Fail loudly and early if the GPU isn't visible -- otherwise fastai silently trains on
# CPU and the run looks like a timeout instead of a misconfiguration.
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible to torch")
print("device", torch.cuda.get_device_name(0))
PY

# Stage the images onto node-local NVMe. Skipped when STAGE_IMAGES is unset, so the
# smoke test (which reads a single family) can keep reading straight from /work.
# `set -e` for this one step only: training against a half-copied tree would "work" and
# quietly report metrics on missing data.
if [ -n "${STAGE_IMAGES:-}" ]; then
  set -e
  python /work/lepinet/ucloud/stage.py "$STAGE_IMAGES" /tmp/global_lepi/images
  set +e
fi
