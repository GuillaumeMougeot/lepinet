# Environment setup for a src/lepinet job on UCloud, embedded into the batch script by ucloud-api.
# Runs with cwd = /work/lepinet (the synced working tree). Unlike the old dev/030 setup.sh, this
# needs NO mini_trainer/mini_metrics -- the package is standalone -- and builds the venv straight
# from the locked pyproject (torch==2.12.1+cu130 via the pinned PyTorch index).

set -x

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# Keep the venv and cache on node-local disk; /work is a network mount (slow for many small files).
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PROJECT_ENVIRONMENT=/tmp/venv

cd /work/lepinet
# Reproducible install from uv.lock (respects [tool.uv.sources]/[[tool.uv.index]] -> cu130 torch).
# --no-dev skips pytest/ruff/mkdocs (not needed on the node); --extra timm installs the timm
# backbones (needed by convnextv2/fastvit/... runs; harmless for torchvision-arch runs). The
# project (lepinet) is installed, so the `lepinet` CLI is on PATH after activation.
uv sync --no-dev --frozen --extra timm
# shellcheck disable=SC1091
source /tmp/venv/bin/activate

# Preflight: fail before burning GPU-hours, not after.
#
# Set LEPINET_NO_GPU=1 for jobs that legitimately have no GPU -- the TreeOfLife crawler runs on a
# cpu-amd-zen5 product because it is network-bound, and without this the preflight aborts a
# perfectly healthy job for missing hardware it never asked for.
if ! python - <<'PY'
import importlib
import os
import shutil
import torch
# Whether this job is *supposed* to have a GPU is decided by the product, and the reliable signal is
# whether the driver tooling exists at all. A cpu-amd-zen5 node has no `nvidia-smi`; a GPU node whose
# driver has broken (which has happened here -- see the NVML mismatch incident) still has it, and
# must still fail loudly rather than quietly training on CPU for eight hours.
# Setting the env var in a TOML `run =` line does not work: setup is embedded *above* it.
want_gpu = (os.environ.get("LEPINET_NO_GPU", "") not in ("1", "true", "yes")
            and shutil.which("nvidia-smi") is not None)
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if want_gpu and not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible to torch, but nvidia-smi exists -- driver problem")
print("device", torch.cuda.get_device_name(0) if want_gpu else "cpu (no nvidia-smi: CPU product)")
import lepinet  # noqa: F401
for mod in ("fastai.vision.all", "lepinet.train", "lepinet.test", "psutil"):
    importlib.import_module(mod)
print("preflight: OK --", "GPU" if want_gpu else "CPU", "+ lepinet + fastai import")
PY
then
  echo "PREFLIGHT FAILED -- aborting before the run starts (see the error above)."
  exit 1
fi
