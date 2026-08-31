#!/bin/bash
#SBATCH --job-name=env
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

# `compute` HAS internet; make sure no offline flags leak in from the shell profile.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

# NOT /path/to/envs/<project>/bin/python: its bundled ensurepip is broken,
# so `python -m venv` dies with "ensurepip ... exit status 1".  This one works.
BASEPY=/data/home/<user>/miniconda3/bin/python3
rm -rf "$ROOT/envs/ecup"
"$BASEPY" -m venv --copies "$ROOT/envs/ecup"

PY="$ROOT/envs/ecup/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools

# Core stack -- pinned so the env is reproducible for the jury.
"$PY" -m pip install \
    "numpy==2.2.6" \
    "pandas==2.3.3" \
    "pyarrow==21.0.0" \
    "polars==1.34.0" \
    "scikit-learn==1.7.2" \
    "scipy==1.16.2" \
    "matplotlib==3.10.6" \
    "tqdm" "pyyaml"

# Boosters + tuning: installed non-fatally, wheel availability varies by Python minor.
for pkg in "lightgbm==4.6.0" "xgboost==3.0.5" "catboost==1.2.8" "optuna==4.5.0" "statsmodels==0.14.5"; do
    "$PY" -m pip install "$pkg" || echo "WARN: could not install $pkg"
done

echo "=== versions ==="
"$PY" - <<'PYEOF'
import importlib, sys
print("python", sys.version)
for m in ["numpy","pandas","pyarrow","polars","sklearn","scipy","lightgbm","xgboost",
          "catboost","matplotlib","optuna","statsmodels"]:
    try:
        print(f"{m:14s} {getattr(importlib.import_module(m),'__version__','?')}")
    except Exception as e:
        print(f"{m:14s} FAILED: {type(e).__name__}: {e}")
PYEOF

"$PY" -m pip freeze > "$ROOT/envs/requirements.lock.txt"
echo "=== ENV SETUP DONE ==="
