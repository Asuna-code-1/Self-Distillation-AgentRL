#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/reskill_local_env.sh"
mkdir -p "${CONDA_ENVS_PATH}" "${CONDA_PKGS_DIRS}" "${TMPDIR}" "${PIP_CACHE_DIR}"
# The original environment is not reused; optional Search retriever is unnecessary for ALFWorld.
export CONDA_USE_COMMUNITY_CHANNELS=1
export INSTALL_RETRIEVER="${INSTALL_RETRIEVER:-0}"
exec bash "${SCRIPT_DIR}/setup_env.sh"
