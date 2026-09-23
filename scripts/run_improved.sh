#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
export PYTHONPATH="$project_root/src"
export WINDOPS_ML_MODULE=windops.ml.plugin
export WINDOPS_EXECUTION_MODE=deterministic
export WINDOPS_OFFLINE=1
export WINDOPS_MODEL_DIR="$project_root/models/geometry_v2/active"
export WINDOPS_DATA_DIR="$project_root/data/experiments/ml_weather_geometry_v1/runtime_data"
"$project_root/.venv/bin/python" scripts/prepare_improved_runtime.py
exec "$project_root/.venv/bin/python" -m streamlit run app.py "$@"
