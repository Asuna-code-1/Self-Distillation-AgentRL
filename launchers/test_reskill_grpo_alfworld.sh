#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/reskill_local_env.sh"
MODEL_DIR="${1:?Usage: test_reskill_grpo_alfworld.sh <merged model> [Hydra overrides ...]}"
shift
export MODEL_PATH="$(realpath -e -- "${MODEL_DIR}")"
export EVAL_KIND=reskill_grpo
export RESKILL_MAX_MODEL_LEN="${RESKILL_MAX_MODEL_LEN:-9216}"
export EVAL_DESCRIPTION='merged ReSkill-GRPO model with task-only initial skills'
export EVAL_RUNNER="${SCRIPT_DIR}/run_reskill_grpo_1gpu.sh"
export TRAIN_LOG_FILE=/dev/null
exec bash "${SCRIPT_DIR}/test_pretrained_1gpu.sh" alfworld "$@"
