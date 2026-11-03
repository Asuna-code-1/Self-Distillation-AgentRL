#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/reskill_local_env.sh"
export MERGE_METHOD=reskill_grpo
exec bash "${SCRIPT_DIR}/merge_grpo_checkpoint.sh" "$@"
