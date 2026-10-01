# Minimal environment for dev/082_tol_crawler.py on a small CPU node. Embedded into the batch
# script by ucloud-api, like setup-lepinet.sh.
#
# Why a separate script: setup-lepinet.sh installs the full training stack -- torch, torchvision and
# the CUDA libraries, ~5 GB of venv plus ~4 GB of downloaded wheels -- into the container's /tmp.
# The crawler imports none of it (pyarrow, aiohttp, Pillow, huggingface_hub). On a 1-vCPU / 3 GB
# node that payload took ~10 minutes of every restart, and the crawl was killed twice in a row with
# no traceback -- the signature of the container exceeding its memory limit, to which written
# files' page cache can count. This installs ~200 MB in well under a minute.
#
# Versions pinned to the ones the crawler was developed and tested against.

set -x

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_CACHE_DIR=/tmp/uv-cache

uv venv /tmp/venv --python 3.14
# shellcheck disable=SC1091
source /tmp/venv/bin/activate
uv pip install "pyarrow==25.0.0" "aiohttp==3.14.3" "pillow==12.3.0" "huggingface_hub==1.24.0" "numpy==2.5.1"
rm -rf "$UV_CACHE_DIR"        # the wheels are installed; do not keep a second copy in /tmp

cd /work/lepinet

# Preflight: imports, and what the cgroup actually grants -- the numbers to read if the job dies.
if ! python - <<'PY'
import os
import aiohttp, PIL, pyarrow, huggingface_hub  # noqa: F401
def cg(name):
    try:
        return open(f"/sys/fs/cgroup/{name}").read().strip()
    except OSError:
        return "?"
print("crawler env OK | cpu.max", cg("cpu.max"), "| memory.max", cg("memory.max"),
      "| memory.current", cg("memory.current"))
PY
then
  echo "PREFLIGHT FAILED -- aborting before the run starts (see the error above)."
  exit 1
fi
