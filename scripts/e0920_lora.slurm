#!/bin/bash
# e0920 LoRA MOMENT -- SMOKE on compute (validates PEFT+MOMENT wiring, the end-to-end train loop, and
# installs peft into envs/tsfm using compute's internet). IDEAS.md §I29 fine-tune axis. Once the smoke
# prints a fold rho without crashing, the full fold-4 runs on andrena GPU (LoRA backprop is too slow on
# CPU at full scale). --smoke = 2 anchors x 3k users x 2 epochs, 6k val.
#SBATCH --job-name=e0920lora
#SBATCH --partition=compute
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 NUMEXPR_MAX_THREADS=16 HF_HOME="$ROOT/envs/hf_cache"
PY="$ROOT/envs/tsfm/bin/python"
"$PY" -c "import peft" 2>/dev/null || { echo "=== installing peft into envs/tsfm ==="; envs/tsfm/bin/pip install -q peft 2>&1 | tail -4; }
"$PY" -c "import peft, momentfm; print('peft', peft.__version__)"
"$PY" src/run_tsfm_lora.py --exp-id e0920 --fold 4 --smoke
echo "=== e0920 LoRA smoke DONE ==="
