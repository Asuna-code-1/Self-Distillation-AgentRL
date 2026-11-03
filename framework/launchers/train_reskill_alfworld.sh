#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/reskill_local_env.sh"
exec "${CONDA_BASE}/bin/python" "${SCRIPT_DIR}/reskill_recipe.py" "$@"
