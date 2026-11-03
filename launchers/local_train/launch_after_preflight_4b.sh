#!/usr/bin/env bash
set -uo pipefail
ROOT=/workspace/fudan/donghaosen/Self-Distillation-AgentRL
STATUS="${1:?preflight exit-code file}"
export EXPERIMENT_NAME="${2:?unique experiment name}"
RUN_DIR="${3:?attempt log directory}"
while [[ ! -f "${STATUS}" ]]; do sleep 5; done
if [[ "$(cat "${STATUS}")" != 0 ]]; then
    echo 'Preflight failed; no training was started.' >&2
    printf '1\n' > "${RUN_DIR}/launcher.exit"
    exit 1
fi
cd "${ROOT}"
bash launchers/local_train/run_qwen35_4b_4gpu.sh
code=$?
printf '%s\n' "${code}" > "${RUN_DIR}/launcher.exit"
exit "${code}"
