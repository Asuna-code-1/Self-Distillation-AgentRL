#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/fudan/donghaosen/Self-Distillation-AgentRL
STATUS=/data/fd/donghaosen/Self-Distillation-AgentRL/logs/training/extensions-h200.exit
while [[ ! -f "${STATUS}" ]]; do sleep 5; done
[[ "$(cat "${STATUS}")" == 0 ]]
source "${ROOT}/launchers/reskill_local_env.sh"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ROOT}/local_assets/conda/envs/agentrl-reskill"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset PIP_CONSTRAINT
cd "${ROOT}"
python launchers/local_train/preflight.py
python -m unittest discover -s tests -p test_reskill_grpo.py -v
