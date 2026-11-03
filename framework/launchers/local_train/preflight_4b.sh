#!/usr/bin/env bash
set -uo pipefail
cd /workspace/fudan/donghaosen/Self-Distillation-AgentRL
export MODEL_SIZE=4B
export MODEL_PATH="$PWD/models/Qwen3.5-4B"
source launchers/reskill_local_env.sh
export CUDA_VISIBLE_DEVICES=0
local_assets/conda/envs/agentrl-reskill/bin/python -m unittest discover -s tests -p 'test_*.py' -v && local_assets/conda/envs/agentrl-reskill/bin/python launchers/local_train/check_native_backward_4b.py
code=$?
printf '%s\n' "$code" > /data/fd/donghaosen/Self-Distillation-AgentRL/logs/training/preflight-4b-20261031.exit
exit "$code"
