#!/bin/bash
# e0914 -- frozen TS foundation-model embedding (MOMENT-1-small) as a blend member (IDEAS.md §I28).
# Builds a fresh env with momentfm once (compute has internet -> weights auto-download to HF cache),
# gates on a clean import + a tiny forward pass, then runs the fold-4 feasibility. CPU (compute):
# no GPU-quota fight, and MOMENT-small is light enough for a subsampled extraction.
#SBATCH --job-name=e0914tsfm
#SBATCH --partition=compute
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag envs
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16
export HF_HOME="$ROOT/envs/hf_cache"

if ! envs/tsfm/bin/python -c "import momentfm" 2>/dev/null; then
  echo "=== building envs/tsfm (deps as wheels first, then momentfm --no-deps) ==="
  [ -x envs/tsfm/bin/python ] || /data/home/<user>/miniconda3/bin/python3 -m venv envs/tsfm
  envs/tsfm/bin/pip install -q --upgrade pip setuptools wheel
  # install deps that ship py3.13 wheels first, so momentfm never triggers an old-numpy source build
  envs/tsfm/bin/pip install -q numpy torch transformers huggingface_hub einops deprecation \
        lightgbm polars pyarrow scikit-learn 2>&1 | tail -8
  envs/tsfm/bin/pip install -q --no-deps momentfm 2>&1 | tail -8
fi
PY="$ROOT/envs/tsfm/bin/python"
# gate: import + a tiny frozen forward pass (downloads weights on first call; compute has internet)
"$PY" - <<'PYGATE' || { echo "MOMENT gate FAILED -- aborting"; exit 3; }
import torch, numpy as np
from momentfm import MOMENTPipeline
m = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small", model_kwargs={"task_name":"embedding"})
m.init(); m.eval()
x = torch.zeros(2,1,512); msk = torch.ones(2,512)
with torch.no_grad(): out = m(x_enc=x, input_mask=msk)
print("MOMENT gate OK: embedding shape", tuple(out.embeddings.shape))
PYGATE

echo "=== running e0914 fold-4 TS-FM embedding test ==="
"$PY" src/run_tsfm.py
echo "=== e0914 DONE ==="
