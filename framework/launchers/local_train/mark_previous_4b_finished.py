"""Apply the user's explicit completed status to the stopped previous run."""
import datetime,json,os,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
os.environ.setdefault('SWANLAB_ROOT',str(root/'local_assets/swanlab-config'))
os.environ.setdefault('SWANLAB_CONFIG_DIR',str(root/'local_assets/swanlab-config'))
import swanlab
name='reskill_grpo_qwen35_4b_alfworld_4gpu_180step_no_kl_no_quality_patched_swanlab_20261101T071213Z_b10cf'
d=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/training/attempts')/name
key=Path('/workspace/fudan/donghaosen/.credentials/swanlab-20261031-4b.key').read_text().strip()
owner,project,slug='Dhs2004','agentic_alfworld','dvt7rcn9';path=f'{owner}/{project}/{slug}'
api=swanlab.Api(api_key=key);run=api.run(path)
assert run.name==name
before=run.state
response=run._ctx.client.put(f'/project/{owner}/{project}/runs/{slug}/state',data={'state':'FINISHED','finishedAt':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),'from':'sdk'})
assert response.raw.ok
state=swanlab.Api(api_key=key).run(path).state
assert state=='FINISHED'
report={'path':path,'previous_state':before,'verified_state':state,'reason':'User explicitly requested marking previous stopped experiment completed','completed_training_steps':135,'time':time.time()}
(d/'user-completed-state.json').write_text(json.dumps(report,indent=2)+'\n')
p=d/'stopped-run.json';r=json.loads(p.read_text());r['cloud_state']='FINISHED';r['user_marked_completed']=True;p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(report))
