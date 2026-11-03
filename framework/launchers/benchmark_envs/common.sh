#!/usr/bin/env bash
# Source before installing or using an isolated benchmark environment.
BENCH_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export BENCH_ASSET_ROOT=/data/fd/donghaosen/Self-Distillation-AgentRL
export CONDA_BASE=/workspace/fudan/donghaosen/miniconda3
export CONDA_ENVS_PATH="${BENCH_REPO_ROOT}/local_assets/conda/envs"
export CONDA_PKGS_DIRS="${BENCH_ASSET_ROOT}/cache/conda"
export CONDA_REGISTER_ENVS=false
export PIP_CACHE_DIR="${BENCH_ASSET_ROOT}/cache/pip"
export XDG_CACHE_HOME="${BENCH_ASSET_ROOT}/cache"
export HF_HOME="${BENCH_ASSET_ROOT}/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TMPDIR="${BENCH_ASSET_ROOT}/tmp"
export DATA_ROOT="${BENCH_ASSET_ROOT}/data"
export ALFWORLD_DATA="${DATA_ROOT}/alfworld"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MAX_JOBS=4
