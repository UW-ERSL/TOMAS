#!/usr/bin/env bash
# Wrapper to run TOMAS scripts under the WSL conda env `tomas-wsl`.
#
# Why PYTHONNOUSERSITE=1:
#   ~/.local/lib/python3.10/site-packages has an OLDER torch_sparse_solve_cpp.so
#   compiled against a different torch ABI. Without isolation it shadows the
#   env-internal build and triggers `undefined symbol: ...sparse_coo_tensor...`.
#
# Why explicit env Python:
#   Avoids any conda activation side effects; same Python whether called from
#   bash or from PowerShell wsl -d Ubuntu-22.04 -- ...
#
# Usage:
#   bash scripts/run_wsl.sh <python script + args>
#   bash scripts/run_wsl.sh -m scripts.reproduce_tomas --config ...
#
# Or, e.g. from Windows PowerShell:
#   wsl -d Ubuntu-22.04 -- bash "/mnt/c/.../TOMAS/scripts/run_wsl.sh" \
#       scripts/reproduce_tomas.py --config notebooks/config_diffuser_a60.yaml ...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TOMAS_WSL_PYTHON:-$HOME/miniconda3/envs/tomas-wsl/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY not found. Activate or recreate conda env 'tomas-wsl' first." >&2
  exit 1
fi

cd "$ROOT"
export PYTHONNOUSERSITE=1
exec "$PY" -u "$@"
