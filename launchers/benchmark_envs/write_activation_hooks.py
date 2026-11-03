"""Install reversible Conda hooks for this project's four benchmark envs."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
values={
 'ALFWORLD_DATA':str(ASSETS/'data/alfworld'),'DATA_ROOT':str(ASSETS/'data'),
 'HF_HOME':str(ASSETS/'cache/huggingface'),'HF_HUB_DISABLE_XET':'1',
 'PIP_CACHE_DIR':str(ASSETS/'cache/pip'),'XDG_CACHE_HOME':str(ASSETS/'cache'),
 'TMPDIR':str(ASSETS/'tmp'),'PYTHONNOUSERSITE':'1',
 'PYTHONPATH':str(ROOT)+':'+str(ROOT/'agent_system/environments/env_package/webshop/webshop'),
 'JAVA_HOME':'${CONDA_PREFIX}/lib/jvm',
 'JVM_PATH':'${CONDA_PREFIX}/lib/jvm/lib/server/libjvm.so',
 'LD_LIBRARY_PATH':'${CONDA_PREFIX}/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
}
keys=' '.join(values)
activation='''# Isolate benchmark dependencies; preserve parent-shell settings for deactivate.
for agentrl_key in KEYS; do
    agentrl_saved="_AGENTRL_SAVED_${agentrl_key}"
    agentrl_flag="${agentrl_saved}_WAS_SET"
    if [[ -v "${agentrl_key}" ]]; then
        printf -v "${agentrl_flag}" '%s' 1
        printf -v "${agentrl_saved}" '%s' "${!agentrl_key}"
    else
        printf -v "${agentrl_flag}" '%s' 0
        printf -v "${agentrl_saved}" '%s' ''
    fi
    export "${agentrl_saved}" "${agentrl_flag}"
done
'''.replace('KEYS',keys)
for key,value in values.items():activation+=f'export {key}="{value}"\n'
activation+='unset agentrl_key agentrl_saved agentrl_flag\n'
deactivation='''for agentrl_key in KEYS; do
    agentrl_saved="_AGENTRL_SAVED_${agentrl_key}"
    agentrl_flag="${agentrl_saved}_WAS_SET"
    if [[ "${!agentrl_flag:-0}" == 1 ]]; then
        printf -v "${agentrl_key}" '%s' "${!agentrl_saved}"
        export "${agentrl_key}"
    else
        unset "${agentrl_key}"
    fi
    unset "${agentrl_saved}" "${agentrl_flag}"
done
unset agentrl_key agentrl_saved agentrl_flag
'''.replace('KEYS',keys)
for name in ('alfworld','webshop','searchqa','sciworld'):
 prefix=ROOT/'local_assets/conda/envs'/('agentrl-'+name)
 if not (prefix/'bin/python').exists():continue
 for kind,content in [('activate',activation),('deactivate',deactivation)]:
  directory=prefix/'etc/conda'/f'{kind}.d';directory.mkdir(parents=True,exist_ok=True)
  (directory/'agentrl_benchmark_paths.sh').write_text(content)
 print('CONDA_HOOKS_READY',name)
