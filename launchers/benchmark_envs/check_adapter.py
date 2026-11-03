"""Import the native benchmark adapter without initializing a Ray cluster."""
import importlib,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
benchmark=sys.argv[1]
name={'alfworld':'alfworld','webshop':'webshop','searchqa':'search','sciworld':'scienceworld'}[benchmark]
module=importlib.import_module('agent_system.environments.env_package.'+name+'.envs')
if 'ray' in sys.modules:assert not sys.modules['ray'].is_initialized()
report={'benchmark':benchmark,'native_adapter':module.__name__,'import_passed':True,'ray_cluster_started':False,'python':sys.executable}
path=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/logs')/(benchmark+'-adapter-check.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
