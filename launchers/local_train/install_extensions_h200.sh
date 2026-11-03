#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/fudan/donghaosen/Self-Distillation-AgentRL
source "${ROOT}/launchers/reskill_local_env.sh"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ROOT}/local_assets/conda/envs/agentrl-reskill"
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
export CUDA_HOME="${ROOT}/local_assets/cuda128"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export FLASH_ATTN_CUDA_ARCHS=90
export TORCH_CUDA_ARCH_LIST=9.0
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=8
export NVCC_THREADS=2
cd "${ROOT}"
python -m pip install -c constraints.txt --no-build-isolation --no-deps /data/fd/donghaosen/Self-Distillation-AgentRL/downloads/training/flash_attn-2.8.3.tar.gz
python -m pip install --no-deps compat_wheels/decord-0.6.0-1agentrl-py3-none-manylinux2010_x86_64.whl
python - <<'PY'
import torch,transformers,vllm,flash_attn,causal_conv1d,fla,wandb
print('NATIVE_TRAINING_IMPORTS_OK',torch.__version__,transformers.__version__,vllm.__version__)
PY
