"""Read-only verification that completed local steps appear in the remote project."""
import json,os,re,time
from pathlib import Path
import wandb
assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
record=json.loads((assets/'training/current-run.json').read_text());dest=Path(record['attempt_directory'])
os.environ['WANDB_API_KEY']=re.search(r'wandb_v1_[A-Za-z0-9_-]+',Path('/workspace/fudan/donghaosen/AGENTS.md').read_text())[0]
api=wandb.Api(timeout=30);run_id=None;last_checked=-2
while True:
 text=Path(record['console_log']).read_text()
 if run_id is None:
  matches=re.findall(r'https://wandb.ai/1973570506-wuhan-university/agentic_alfworld/runs/([A-Za-z0-9]+)',text)
  if matches:run_id=matches[-1]
 if run_id:
  stepfile=assets/'training/resource-traces'/record['name']/'step_peaks.jsonl'
  rows=[json.loads(s) for s in stepfile.read_text().splitlines() if s] if stepfile.exists() else []
  local=max([row['step'] for row in rows],default=-1)
  if local!=last_checked:
   try:
    api.flush();run=api.run('1973570506-wuhan-university/agentic_alfworld/'+run_id)
    remote=int(run.summary.get('_step',-1))
    result={'time':time.time(),'local_completed_step':local,'remote_step':remote,'state':run.state,
            'forge_url':'https://forge.coreweave.com/wandb/1973570506-wuhan-university/agentic_alfworld/runs/'+run_id,
            'synced':remote>=local,'metric_count':len(run.summary.keys())}
    with (dest/'wandb-sync-checks.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    if remote>=local:last_checked=local
   except Exception as error:
    print(type(error).__name__,flush=True)
 if (dest/'launcher.exit').exists():break
 time.sleep(30)
