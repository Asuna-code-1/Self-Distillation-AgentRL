#!/usr/bin/env bash
set -euo pipefail
[[ -n "${TMUX:-}" ]] || { echo 'Run training inside tmux.' >&2; exit 2; }
ROOT=/workspace/fudan/donghaosen/Self-Distillation-AgentRL
export MODEL_SIZE=4B
export MODEL_PATH="${ROOT}/models/Qwen3.5-4B"
source "${ROOT}/launchers/reskill_local_env.sh"
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export MODEL_SIZE=4B GPU_ID=0,1,2,3
export RESKILL_FAIL_ON_NONFINITE_GRAD=1
export SKIP_AGENT_DATA_PREP=1
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.71}"
export PPO_MICRO_BATCH_SIZE=64
export LOG_PROB_MICRO_BATCH_SIZE=256
export RAY_NUM_CPUS=64
export RESKILL_LOGGER=swanlab
export TOTAL_TRAINING_STEPS=160 TOTAL_EPOCHS=160 SAVE_FREQ=160
export SWANLAB_MODE=online
export SWANLAB_ROOT="${ROOT}/local_assets/swanlab-config"
export SWANLAB_CONFIG_DIR="${ROOT}/local_assets/swanlab-config"
export SWANLAB_LOG_DIR="/data/fd/donghaosen/Self-Distillation-AgentRL/swanlab"
export SWANLAB_API_KEY_FILE="/workspace/fudan/donghaosen/.credentials/swanlab-20261031-4b.key"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-reskill_grpo_qwen35_4b_alfworld_4gpu_160step_swanlab_$(date -u +%Y%m%dT%H%M%SZ)}"
export RESKILL_GPU_INDICES=0,1,2,3
export RESKILL_RESOURCE_TRACE="/data/fd/donghaosen/Self-Distillation-AgentRL/training/resource-traces/${EXPERIMENT_NAME}"
cd "${ROOT}"
exec bash launchers/train_reskill_alfworld.sh \
    env.disable_thinking=True \
    algorithm.reskill_grpo.rewrite_quality.enable=False \
    algorithm.reskill_grpo.rewrite_quality.loss_coef=0.1 \
    algorithm.reskill_grpo.rewrite_quality.beta=1.0 \
    algorithm.reskill_grpo.rewrite_quality.interval=5 \
    algorithm.reskill_grpo.rewrite_quality.old_success_threshold=0.75 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=131072 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=131072 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:0,video:0}" \
    +ray_init._temp_dir="${RAY_TMPDIR}" \
    +ray_init.include_dashboard=False \
    +ray_init.object_store_memory=8589934592 \
    "$@"
