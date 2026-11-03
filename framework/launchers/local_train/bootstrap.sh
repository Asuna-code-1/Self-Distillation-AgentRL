#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/fudan/donghaosen/Self-Distillation-AgentRL
source "${ROOT}/launchers/benchmark_envs/common.sh"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ROOT}/local_assets/conda/envs/agentrl-reskill"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export JAVA_HOME="${CONDA_PREFIX}/lib/jvm"
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=9.0
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd "${ROOT}"
python - "${BENCH_ASSET_ROOT}/downloads/wheels" <<'PY'
from pathlib import Path
import subprocess,sys
wheels=[str(p) for p in Path(sys.argv[1]).glob('*.whl') if '+cpu' not in p.name]
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*sorted(wheels)],check=True)
PY
python -m pip install "${ROOT}/compat_wheels/textworld-1.7.0-1agentrl-py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
python -m pip install -c constraints.txt -r launchers/local_train/requirements-alfworld.txt
MAX_JOBS=4 python -m pip install -c constraints.txt --no-build-isolation --no-deps -r requirements-kernels.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install --no-deps compat_wheels/decord-0.6.0-1agentrl-py3-none-manylinux2010_x86_64.whl
python - <<'PY'
import torch,transformers,vllm,flash_attn,causal_conv1d,fla,wandb
print('NATIVE_TRAINING_IMPORTS_OK',torch.__version__,transformers.__version__,vllm.__version__)
PY
