#!/usr/bin/env bash
set -uo pipefail
cd /workspace/fudan/donghaosen/Self-Distillation-AgentRL
source launchers/reskill_local_env.sh
PYTHON=local_assets/conda/envs/agentrl-reskill/bin/python
export LD_LIBRARY_PATH="$PWD/local_assets/conda/envs/agentrl-reskill/lib:$LD_LIBRARY_PATH"
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v && "$PYTHON" launchers/local_train/check_fixed_game_replay.py
code=$?
printf '%s\n' "$code" > /data/fd/donghaosen/Self-Distillation-AgentRL/logs/training/preflight-rewrite-quality.exit
exit "$code"
