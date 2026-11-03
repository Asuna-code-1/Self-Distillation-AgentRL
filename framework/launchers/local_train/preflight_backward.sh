#!/usr/bin/env bash
set -uo pipefail
cd /workspace/fudan/donghaosen/Self-Distillation-AgentRL
source launchers/reskill_local_env.sh
export CUDA_VISIBLE_DEVICES=0
local_assets/conda/envs/agentrl-reskill/bin/python launchers/local_train/check_native_backward.py
code=$?
printf '%s\n' "$code" > /data/fd/donghaosen/Self-Distillation-AgentRL/logs/training/preflight-backward.exit
exit "$code"
