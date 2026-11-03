#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
BENCHMARK="${1:?alfworld|webshop|searchqa|sciworld}"
case "${BENCHMARK}" in alfworld|webshop|searchqa|sciworld) ;; *) exit 2;; esac
export CONDA_PKGS_DIRS="${BENCH_ASSET_ROOT}/cache/conda-${BENCHMARK}"
ENV_PREFIX="${CONDA_ENVS_PATH}/agentrl-${BENCHMARK}"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    extras=()
    if [[ "${BENCHMARK}" == webshop || "${BENCHMARK}" == sciworld ]]; then extras+=(openjdk=11); fi
    "${CONDA_BASE}/bin/conda" create --prefix "${ENV_PREFIX}" --override-channels -c conda-forge python=3.10.20 pip "${extras[@]}" -y
fi
export PATH="${ENV_PREFIX}/bin:${PATH}"
if [[ -d "${ENV_PREFIX}/lib/jvm" ]]; then
    export JAVA_HOME="${ENV_PREFIX}/lib/jvm"
    export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
fi
python -m pip install 'pip==26.2' 'setuptools==83.0.0' 'wheel==0.47.0' pybind11 packaging ninja
if [[ "${BENCHMARK}" == alfworld ]]; then
    python -m pip install "${BENCH_REPO_ROOT}/compat_wheels/textworld-1.7.0-1agentrl-py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
fi
if [[ "${BENCHMARK}" == alfworld || "${BENCHMARK}" == webshop || "${BENCHMARK}" == sciworld ]]; then
    # Environment workers are CPU-only; Qwen training remains in agentrl-reskill.
    cpu_packages=(torch==2.10.0)
    if [[ "${BENCHMARK}" != sciworld ]]; then cpu_packages+=(torchvision==0.25.0); fi
    python -m pip install --index-url https://download.pytorch.org/whl/cpu "${cpu_packages[@]}"
fi
if [[ "${BENCHMARK}" == searchqa ]]; then
    # CUDA build allows the native E5 retriever to use a GPU when explicitly run.
    python -m pip install torch==2.10.0
fi
python -m pip install -r "${SCRIPT_DIR}/${BENCHMARK}.txt"
if [[ "${BENCHMARK}" == webshop ]]; then
    python -m pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
fi
python "${SCRIPT_DIR}/write_activation_hooks.py"
python -m pip check
python -m pip freeze > "${BENCH_ASSET_ROOT}/logs/${BENCHMARK}-pip-freeze.txt"
echo "BENCHMARK_ENV_INSTALL_COMPLETED ${BENCHMARK}"
