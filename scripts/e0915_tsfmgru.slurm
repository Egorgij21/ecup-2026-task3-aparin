#!/bin/bash
# e0915 (GRU over MOMENT per-patch tokens) + e0916 (raw-channel GRU + MOMENT embedding channels).
# IDEAS.md §I28, both per-anchor, fold-4 feasibility. Reuses envs/tsfm (momentfm + torch + polars +
# lightgbm), built by e0914; rebuilds only if momentfm is missing.
#SBATCH --job-name=e0915tsfmgru
#SBATCH --partition=compute
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%A_%a.out
#SBATCH --error=/path/to/ecup/logs/%x_%A_%a.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16
export HF_HOME="$ROOT/envs/hf_cache"
PY="$ROOT/envs/tsfm/bin/python"
if ! "$PY" -c "import momentfm" 2>/dev/null; then
  echo "=== building envs/tsfm ==="
  [ -x "$PY" ] || /data/home/<user>/miniconda3/bin/python3 -m venv envs/tsfm
  envs/tsfm/bin/pip install -q --upgrade pip setuptools wheel
  envs/tsfm/bin/pip install -q numpy torch transformers huggingface_hub einops deprecation lightgbm polars pyarrow scikit-learn 2>&1 | tail -8
  envs/tsfm/bin/pip install -q --no-deps momentfm 2>&1 | tail -8
fi
# gate: confirm the per-patch hook captures a [B,P,d] tensor before the full run
"$PY" - <<'PYGATE' || { echo "per-patch hook gate FAILED -- aborting"; exit 3; }
import torch; from momentfm import MOMENTPipeline
m=MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small",model_kwargs={"task_name":"embedding"}); m.init(); m.eval()
cap={}
m.encoder.register_forward_hook(lambda a,b,o: cap.__setitem__("h", o.last_hidden_state if hasattr(o,"last_hidden_state") else o[0]))
with torch.no_grad(): m(x_enc=torch.zeros(2,1,512), input_mask=torch.ones(2,512))
print("per-patch hook OK:", tuple(cap["h"].shape))
PYGATE

echo "=== e0915 GRU over per-patch tokens ==="
"$PY" src/run_tsfm_gru.py --mode patch --exp-id e0915
echo "=== e0916 raw-GRU + FM embedding channels ==="
"$PY" src/run_tsfm_gru.py --mode channels --exp-id e0916
echo "=== e0915/e0916 DONE ==="
