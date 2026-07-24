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
# --no-dev skips pytest/ruff/mkdocs (not needed on the node); the project (lepinet) is installed,
# so the `lepinet` CLI is on PATH after activation.
uv sync --no-dev --frozen
# shellcheck disable=SC1091
source /tmp/venv/bin/activate

# Preflight: fail before burning GPU-hours, not after.
if ! python - <<'PY'
import importlib
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible to torch")
print("device", torch.cuda.get_device_name(0))
import lepinet  # noqa: F401
for mod in ("fastai.vision.all", "lepinet.train", "lepinet.test", "psutil"):
    importlib.import_module(mod)
print("preflight: OK -- GPU visible, lepinet + fastai import")
PY
then
  echo "PREFLIGHT FAILED -- aborting before the run starts (see the error above)."
  exit 1
fi
