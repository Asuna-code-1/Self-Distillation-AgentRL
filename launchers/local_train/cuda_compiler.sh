#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/fudan/donghaosen/Self-Distillation-AgentRL
source "${ROOT}/launchers/benchmark_envs/common.sh"
export CONDA_PKGS_DIRS="${BENCH_ASSET_ROOT}/cache/conda-cuda-compiler"
"${CONDA_BASE}/bin/conda" create -p "${ROOT}/local_assets/cuda128" --override-channels -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 -y
