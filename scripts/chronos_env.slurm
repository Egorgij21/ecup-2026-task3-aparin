#!/bin/bash
# Build envs/chronos (chronos-forecasting) + GATE the embed() API before writing the runner.
# IDEAS.md §I29 strengthen: Chronos-Bolt is the #2 frozen extractor (2510.26777) vs MOMENT (last),
# different objective (value-tokenised T5 seq2seq) + corpus -> decorrelation. `compute` has internet
# for the build + weight download. Gate prints the embed() output shape so run_tsfm_chronos.py can be
# written against the real tensor layout (the e0914 de-risk pattern).
#SBATCH --job-name=chronos_env
#SBATCH --partition=compute
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/ecup/logs/%x_%j.out
#SBATCH --error=/path/to/ecup/logs/%x_%j.err
set -e
set -o pipefail
ROOT=/path/to/ecup
cd "$ROOT"; mkdir -p logs envs
export PYTHONUNBUFFERED=1 HF_HOME="$ROOT/envs/hf_cache"

if ! envs/chronos/bin/python -c "import chronos" 2>/dev/null; then
  echo "=== building envs/chronos ==="
  [ -x envs/chronos/bin/python ] || /data/home/<user>/miniconda3/bin/python3 -m venv envs/chronos
  envs/chronos/bin/pip install -q --upgrade pip setuptools wheel
  # deps as py3.13 wheels first, then chronos-forecasting
  envs/chronos/bin/pip install -q numpy torch transformers accelerate lightgbm polars pyarrow scikit-learn 2>&1 | tail -6
  envs/chronos/bin/pip install -q chronos-forecasting 2>&1 | tail -6
fi
PY="$ROOT/envs/chronos/bin/python"

echo "=== gate: Chronos-Bolt embed() output layout ==="
"$PY" - <<'PYGATE' || { echo "chronos gate FAILED -- aborting"; exit 3; }
import torch, numpy as np
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
# two series of DIFFERENT lengths -> confirm padding/batching behaviour and per-patch layout
ctx = [torch.tensor(np.random.rand(400).astype("float32")),
       torch.tensor(np.random.rand(180).astype("float32"))]
out = pipe.embed(ctx)
emb = out[0] if isinstance(out, tuple) else out
print("EMBED_OK type(out)=", type(out).__name__,
      "| emb.shape=", tuple(emb.shape), "| dtype=", emb.dtype)
if isinstance(out, tuple):
    print("second element type/shape:", type(out[1]).__name__,
          getattr(out[1], "shape", None))
print("d_model (last dim) =", emb.shape[-1], "  n_tokens (dim1) =", emb.shape[1])
PYGATE
echo "=== chronos_env DONE ==="
