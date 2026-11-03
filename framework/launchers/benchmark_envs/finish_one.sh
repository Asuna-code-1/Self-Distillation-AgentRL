#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
BENCHMARK="${1:?benchmark}"
INSTALL_STATUS="${2:?installation exit-code file}"
while [[ ! -f "${INSTALL_STATUS}" ]]; do sleep 5; done
if [[ "$(cat "${INSTALL_STATUS}")" != 0 ]]; then echo 'Installation failed; inspect its log.' >&2; exit 1; fi
source "${SCRIPT_DIR}/activate.sh" "${BENCHMARK}"
export CUDA_VISIBLE_DEVICES=''
cd "${BENCH_REPO_ROOT}"
case "${BENCHMARK}" in
  alfworld)
    while [[ ! -f "${ALFWORLD_DATA}/logic/alfred.pddl" ]]; do sleep 5; done
    python "${SCRIPT_DIR}/smoke.py" alfworld
    ;;
  sciworld)
    python launchers/prepare_scienceworld.py --data-root "${DATA_ROOT}" --download
    python "${SCRIPT_DIR}/smoke.py" sciworld
    ;;
  webshop)
    while [[ ! -f "${DATA_ROOT}/webshop/webshop-download-manifest.json" ]]; do sleep 5; done
    python "${SCRIPT_DIR}/build_webshop_indexes.py"
    python "${SCRIPT_DIR}/smoke.py" webshop
    ;;
  searchqa)
    while [[ ! -f "${DATA_ROOT}/searchR1/searchqa-index-download-manifest.json" || ! -f "${DATA_ROOT}/searchR1/searchqa-corpus-download-manifest.json" || ! -f "${BENCH_ASSET_ROOT}/models/e5-base-v2/searchqa-e5-download-manifest.json" ]]; do sleep 5; done
    python "${SCRIPT_DIR}/prepare_search_data.py"
    python "${SCRIPT_DIR}/assemble_search_assets.py"
    python "${SCRIPT_DIR}/smoke.py" searchqa
    ;;
  *) exit 2;;
esac
python -m pip check
python -m pip freeze > "${BENCH_ASSET_ROOT}/logs/${BENCHMARK}-final-pip-freeze.txt"
echo "BENCHMARK_PREPARATION_COMPLETED ${BENCHMARK}"
