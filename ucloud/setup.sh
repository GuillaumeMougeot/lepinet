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

# One resolution for all three so torch is pinned once. lepinet sets packages=[], so
# installing it contributes dependencies only; mini_trainer and mini_metrics are sibling
# repos locally and extra mounts here (/work/mini_trainer, /work/mini_metrics).
#
# mini_metrics is needed by dev/032 (the test half) and by nothing in training, which is
# why its absence stayed invisible until a job first ran train->test in one go: the
# 2026-07-17 pipeline smoke trained happily for 4 minutes and then died on
# `ModuleNotFoundError: No module named 'mini_metrics'`. Same shape as the tensorboard
# gap -- an import that no pyproject declares, papered over locally by an editable install
# nobody remembers doing. Verified below rather than assumed.
uv pip install /work/lepinet /work/mini_trainer /work/mini_metrics

# Preflight: fail before burning GPU-hours, not after.
#
# Checks the GPU is visible (otherwise fastai silently trains on CPU and the run looks like a
# timeout rather than a misconfiguration), and imports every module the job needs -- including
# the ones only the *test* half touches. An import error that surfaces after training is the
# worst case: it costs a full run to learn something a two-second check catches at startup.
# That is exactly how mini_metrics slipped through (five jobs never reached the test step).
#
# `if ! python ...; then exit 1; fi` is load-bearing. This script has no `set -e`, so a bare
# heredoc that exits non-zero prints its error and the script cheerfully continues to training
# -- which is what happened on 2026-07-17 and what made the *existing* CUDA check toothless
# too: it could never have stopped a CPU-only job despite the comment claiming it would.
#
# Only genuinely required modules belong here. timm is NOT one: dev/014 (the archived
# multilabel line) imports it, and mini_trainer imports it lazily inside timm-architecture
# code paths, but nothing this pipeline runs touches it -- efficientnet_v2_s resolves through
# fastai/torchvision. Listing it made the preflight fail on a package the job never needs.
if ! python - <<'PY'
import importlib
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible to torch")
print("device", torch.cuda.get_device_name(0))

missing = []
for mod in ("fastai.vision.all", "fastai.callback.tensorboard", "mini_trainer.hierarchical.model",
            "mini_trainer.training.muon", "mini_metrics.metrics", "psutil"):
    try:
        importlib.import_module(mod)
    except Exception as e:
        missing.append(f"{mod}: {e}")
if missing:
    raise SystemExit("preflight import check failed:\n  " + "\n  ".join(missing))
print("preflight: OK -- GPU visible, all required modules import (training AND test halves)")
PY
then
  echo "PREFLIGHT FAILED -- aborting before the run starts (see the error above)."
  exit 1
fi

# Stage the images onto node-local NVMe. Skipped when STAGE_IMAGES is unset, so the
# smoke test (which reads a single family) can keep reading straight from /work.
# `set -e` for this one step only: training against a half-copied tree would "work" and
# quietly report metrics on missing data.
if [ -n "${STAGE_IMAGES:-}" ]; then
  set -e
  python /work/lepinet/ucloud/stage.py "$STAGE_IMAGES" /tmp/global_lepi/images
  set +e
fi
