#!/usr/bin/env bash
# Usage: source launchers/benchmark_envs/activate.sh alfworld|webshop|searchqa|sciworld
BENCH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${BENCH_SCRIPT_DIR}/common.sh"
BENCH_SELECTED="${1:?Select alfworld, webshop, searchqa or sciworld}"
case "${BENCH_SELECTED}" in alfworld|webshop|searchqa|sciworld) ;; *) return 2;; esac
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENVS_PATH}/agentrl-${BENCH_SELECTED}"
if [[ -d "${CONDA_PREFIX}/lib/jvm" ]]; then
    export JAVA_HOME="${CONDA_PREFIX}/lib/jvm"
    export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
fi
export PYTHONPATH="${BENCH_REPO_ROOT}:${BENCH_REPO_ROOT}/agent_system/environments/env_package/webshop/webshop${PYTHONPATH:+:${PYTHONPATH}}"

# The container's inherited Python 3.12 Torch libraries break Python 3.10.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
