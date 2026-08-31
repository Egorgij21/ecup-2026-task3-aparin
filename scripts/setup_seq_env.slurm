#!/bin/bash
# Build $ROOT/envs/seq -- a CUDA PyTorch environment for the `seq` approach (sequence models
# over the raw daily panel).  MUST run on `compute`: apini has no outbound internet.
#
# Why a new env rather than reusing envs/ag: that one carries torch==2.13.0+cpu as an
# AutoGluon dependency.  Upgrading it in place would silently change what run_ag.py runs,
# and e0064 (our LB best) came out of that env.  CLAUDE.md rule 8 -- no unrequested changes
# to a working pipeline.
#
#SBATCH --job-name=env_seq
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err

set -e
ROOT=/path/to/ecup
cd "$ROOT"
mkdir -p logs envs

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

echo "=== quota BEFORE ==="
/usr/lpp/mmfs/bin/mmlsquota --block-size auto gpfsFlash | tail -2 || true

BASEPY=/data/home/<user>/miniconda3/bin/python3
rm -rf "$ROOT/envs/seq"
"$BASEPY" -m venv --copies "$ROOT/envs/seq"
PY="$ROOT/envs/seq/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools

# The apini GPUs are H200 NVL (sm_90).  cu128 wheels cover them; cu126 is the fallback if
# the pinned build is missing for this Python minor.
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 \
  || "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# Same pins as envs/ecup so the panel-building code behaves identically in both.
"$PY" -m pip install "numpy==2.2.6" "polars==1.34.0" "pyarrow==21.0.0" "pyyaml" "tqdm"

echo "=== versions ==="
"$PY" - <<'PYEOF'
import sys, importlib
print("python", sys.version)
for m in ["numpy", "polars", "pyarrow", "yaml"]:
    print(f"{m:10s} {getattr(importlib.import_module(m), '__version__', '?')}")
import torch
print("torch     ", torch.__version__)
print("cuda build", torch.version.cuda)
print("arch list ", torch.cuda.get_arch_list())
print("is_available (on compute, expected False):", torch.cuda.is_available())
PYEOF

"$PY" -m pip freeze > "$ROOT/envs/requirements.seq.lock.txt"

echo "=== quota AFTER ==="
/usr/lpp/mmfs/bin/mmlsquota --block-size auto gpfsFlash | tail -2 || true
du -sh "$ROOT/envs/seq"
echo "=== SEQ ENV SETUP DONE ==="
