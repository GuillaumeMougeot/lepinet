# Environment for the C3 broad-taxa jobs (lepinet-bt-a, lepinet-bt-c), embedded into the batch script by ucloud-api.
# cwd = /work/broad_taxa: a staged folder (dev/117 as bt_gpu.py, dev/115's index.csv, the MCC24 zip, lepinet's
# pyproject.toml / uv.lock), not the repo. Dependencies only, from lepinet's lock (torch 2.12.1+cu130, timm).
set -x
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/venv
cd /work/broad_taxa
uv sync --no-dev --frozen --extra timm --no-install-project --python 3.14
# shellcheck disable=SC1091
source /tmp/venv/bin/activate
python - <<'PY' || exit 1
import torch, timm, torchvision
assert torch.cuda.is_available(), "no GPU"
print(torch.__version__, torchvision.__version__, timm.__version__, torch.cuda.get_device_name(0),
      round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
PY
