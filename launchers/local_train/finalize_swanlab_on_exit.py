"""Close only this attempt's cloud run after its launcher has actually exited."""
import datetime,json,os,re,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
r=json.loads((assets/'training/current-run.json').read_text());d=Path(r['attempt_directory'])
os.environ.setdefault('SWANLAB_ROOT',str(root/'local_assets/swanlab-config'))
os.environ.setdefault('SWANLAB_CONFIG_DIR',str(root/'local_assets/swanlab-config'))
while not (d/'launcher.exit').exists():time.sleep(5)
code=int((d/'launcher.exit').read_text().strip())
text=Path(r['console_log']).read_text(errors='replace')
urls=re.findall(r'https://swanlab\.(?:cn|ai)/@([\w.-]+)/([\w.+-]+)/runs/([\w-]+)',text)
if not urls:raise SystemExit('No cloud run was created; no cloud mutation needed')
path='/'.join(urls[-1]);owner,project,slug=urls[-1]
state='FINISHED' if code==0 else ('ABORTED' if code in (143,148) else 'CRASHED')
import swanlab
key=Path('/workspace/fudan/donghaosen/.credentials/swanlab-20261031-4b.key').read_text().strip()
for attempt in range(4):
 try:
  api=swanlab.Api(api_key=key);run=api.run(path)
  assert run.name==r['name'],'Cloud identity mismatch; refusing to change state'
  before=run.state
  if before != state:
   response=run._ctx.client.put(f'/project/{owner}/{project}/runs/{slug}/state',data={'state':state,'finishedAt':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),'from':'sdk'})
   assert response.raw.ok,'State update failed'
  after=swanlab.Api(api_key=key).run(path).state
  assert after==state,'Cloud state does not match launcher exit'
  report={'path':path,'launcher_exit':code,'previous_state':before,'verified_state':after,'time':time.time()}
  (d/'cloud-final-state.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
  break
 except Exception as error:
  print(type(error).__name__,flush=True)
  if attempt==3:raise SystemExit('Cloud finalization failed; inspect run manually')
  time.sleep(10)
