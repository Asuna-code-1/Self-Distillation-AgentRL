#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${SCRIPT_DIR}/activate.sh" searchqa
WHEELS="${BENCH_ASSET_ROOT}/downloads/wheels"
python - "${WHEELS}" <<'PY'
from pathlib import Path
import subprocess,sys
wheels=[str(p) for p in Path(sys.argv[1]).glob('*.whl') if '+cpu' not in p.name]
assert any(Path(p).name.startswith('torch-2.10.0-3-') for p in wheels)
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*sorted(wheels)],check=True)
PY
python -m pip install -r "${SCRIPT_DIR}/searchqa.txt"
python -m pip check
python -m pip freeze > "${BENCH_ASSET_ROOT}/logs/searchqa-pip-freeze.txt"
echo 'BENCHMARK_ENV_INSTALL_COMPLETED searchqa'
