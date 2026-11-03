#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${SCRIPT_DIR}/activate.sh" searchqa
WHEELS="${BENCH_ASSET_ROOT}/downloads/wheels"
STATUS="${BENCH_ASSET_ROOT}/logs/torch-wheels-retry.exit"
while [[ ! -f "${STATUS}" ]]; do sleep 5; done
[[ "$(cat "${STATUS}")" == 0 ]]
# Explicit CUDA wheel avoids accidentally selecting the CPU variant.
python -m pip install --find-links "${WHEELS}" "${WHEELS}/torch-2.10.0-3-cp310-cp310-manylinux_2_28_x86_64.whl"
python -m pip install --find-links "${WHEELS}" -r "${SCRIPT_DIR}/searchqa.txt"
python -m pip check
python -m pip freeze > "${BENCH_ASSET_ROOT}/logs/searchqa-pip-freeze.txt"
echo 'BENCHMARK_ENV_INSTALL_COMPLETED searchqa'
