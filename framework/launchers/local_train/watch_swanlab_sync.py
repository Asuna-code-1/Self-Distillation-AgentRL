"""Read-only verification of SwanLab uploads for the current isolated run."""
import json,os,re,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
os.environ.setdefault('SWANLAB_ROOT',str(root/'local_assets/swanlab-config'))
os.environ.setdefault('SWANLAB_CONFIG_DIR',str(root/'local_assets/swanlab-config'))
import swanlab
assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
record=json.loads((assets/'training/current-run.json').read_text());dest=Path(record['attempt_directory'])
api=swanlab.Api(api_key=Path('/workspace/fudan/donghaosen/.credentials/swanlab-20261031-4b.key').read_text().strip())
path=None;last_checked=-2
while True:
 text=re.sub(r'\x1b\[[0-9;]*m','',Path(record['console_log']).read_text())
 if path is None:
  matches=re.findall(r'https://swanlab\.(?:cn|ai)/@([A-Za-z0-9_.-]+)/([A-Za-z0-9_.+-]+)/runs/([A-Za-z0-9_-]+)',text)
  if matches:path='/'.join(matches[-1])
 if path:
  stepfile=assets/'training/resource-traces'/record['name']/'step_peaks.jsonl'
  rows=[json.loads(s) for s in stepfile.read_text().splitlines() if s] if stepfile.exists() else []
  local=max([r['step'] for r in rows],default=-1)
  if local!=last_checked:
   try:
    run=api.run(path);summary=run.summary(keys=['training/global_step','val/success_rate','actor/grad_norm'])
    remote=max([int(v.get('step',-1)) for v in summary.values() if isinstance(v,dict)],default=-1)
    result={'time':time.time(),'local_completed_step':local,'remote_step':remote,'state':run.state,
            'url':run.url,'synced':remote>=local,'summary':summary}
    with (dest/'swanlab-sync-checks.jsonl').open('a') as f:f.write(json.dumps(result,default=str)+'\n')
    print(json.dumps(result,default=str),flush=True)
    if remote>=local:last_checked=local
   except Exception as error:print(type(error).__name__,flush=True)
 if (dest/'launcher.exit').exists():break
 time.sleep(30)
