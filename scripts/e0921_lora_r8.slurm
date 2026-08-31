#!/bin/bash
# e0920 LoRA MOMENT -- FULL fold-4 on andrena GPU (LoRA backprop is too slow on CPU). PEFT wiring
# already validated by the compute smoke (0.64% trainable, T5 q/v). IDEAS.md §I29 fine-tune axis.
# peft already installed in envs/tsfm. Reads rho_partial + corr_vs_GRU (the twin check).
#SBATCH --job-name=e0921lg8
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs runs/ag
export PYTHONUNBUFFERED=1 HF_HOME="$ROOT/envs/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$ROOT/envs/tsfm/bin/python"
"$PY" -c "import peft, torch; print('peft', peft.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
"$PY" src/run_tsfm_lora.py --exp-id e0921 --fold 4 --lora-r 8
echo "=== e0921 LoRA r=8 fold-4 DONE ==="
