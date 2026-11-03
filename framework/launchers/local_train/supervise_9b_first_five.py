"""Sample this attempt and verify five finite training steps; guard at 130 GiB."""
import json,math,re,subprocess,time
from pathlib import Path
import pynvml
assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
r=json.loads((assets/'training/current-run.json').read_text());d=Path(r['attempt_directory'])
pynvml.nvmlInit();handles=[pynvml.nvmlDeviceGetHandleByIndex(i) for i in r['gpus']]
peaks=[0.]*len(handles);high=0;last=-1;reported_five=False
first_steps=list(range(int(r.get('resume_step',0))+1,int(r.get('resume_step',0))+6))
last_first_step=first_steps[-1]
while True:
 used=[pynvml.nvmlDeviceGetMemoryInfo(h).used/2**30 for h in handles]
 peaks=[max(a,b) for a,b in zip(peaks,used)]
 with (d/'gpu-watch.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'used_gib':used})+'\n')
 high=high+1 if max(used)>130 else 0
 text=re.sub(r'\x1b\[[0-9;]*m','',Path(r['console_log']).read_text(errors='replace'))
 rows={}
 for line in text.splitlines():
  match=re.search(r'\bstep:(\d+) -',line)
  if not match:continue
  step=int(match[1]);metrics={k:float(v) for k,v in re.findall(r'([\w/.-]+):(-?(?:\d+(?:\.\d+)?|nan|inf))',line)}
  rows[step]=metrics
 completed=max(rows,default=0)
 if completed!=last:
  print(json.dumps({'completed_steps':completed,'peak_gib':peaks}),flush=True);last=completed
 if high>=2:
  (d/'gpu-alert.json').write_text(json.dumps({'peak_gib':peaks,'reason':'over 130 GiB twice'},indent=2))
  subprocess.run(['tmux','send-keys','-t',r['session'],'C-z'],check=True);time.sleep(1)
  for cmd in ['kill %1','kill -CONT %1']:
   subprocess.run(['tmux','send-keys','-t',r['session'],'-l',cmd],check=True)
   subprocess.run(['tmux','send-keys','-t',r['session'],'Enter'],check=True);time.sleep(.5)
  raise SystemExit('Stopped attempt at memory guard')
 if not reported_five and all(s in rows for s in first_steps):
  finite=all(math.isfinite(v) for s in first_steps for k,v in rows[s].items() if k.startswith('actor/'))
  report={'five_steps_completed':True,'actor_metrics_finite':finite,'sampled_peak_gib':peaks,'steps':{s:rows[s] for s in first_steps},'time':time.time()}
  if r.get('rewrite_quality_enabled') or r.get('audit_skills'):
   traces=assets/'training/checkpoints/reskill_grpo/alfworld'/r['name']/'skill_refinement'
   step_groups={step:[g for line in (traces/f'step-{step}.jsonl').read_text().splitlines() for g in json.loads(line)['groups']] for step in first_steps}
   placeholders={'revised skill','your skill','procedural skill','rewritten skill','skill'}
   label=lambda value:re.sub(r'[\W_]+',' ',str(value).casefold()).strip()
   semantic={step:{'unique_initial':len({g['initial_skill'] for g in groups if g['initial_skill']}),
                         'accepted_placeholders':sum(g['accepted'] and label(g.get('refined_skill','')) in placeholders for g in groups)} for step,groups in step_groups.items()}
   report['skill_semantics']=semantic
   report['skill_semantics_verified']=all(v['unique_initial']>1 and v['accepted_placeholders']==0 for v in semantic.values())
   if r.get('rewrite_quality_enabled'):
    evaluated=[g for g in step_groups[last_first_step] if 'rewrite_quality' in g]
    schedule=all(rows[step].get('rewrite_quality/active_step')==int(step==last_first_step) for step in first_steps)
    checks=[]
    for g in evaluated:
     q=g['rewrite_quality'];checks.append(q['p_old']<.75 and q['group_size']==r['group_size'] and q['q']==int(q['p_new']>q['p_old'] and q['edit_distance']>0) and (q['q'] or not g['accepted']) and q['paired_request_seeds'] and q['same_policy_before_update'])
    report['rewrite_quality_verified']=schedule and all(checks) and (not evaluated or rows[last_first_step].get('actor/rewrite_quality_samples',0)>0)
    report['rewrite_quality_evaluated_groups']=len(evaluated)
    report['rewrite_quality_positive_groups']=sum(g['rewrite_quality']['q'] for g in evaluated)
   else:
    report['quality_supervision_disabled_verified']=all(rows[step].get('rewrite_quality/active_step')==0 for step in first_steps)
  if r.get('save_validation_skills'):
   skill_file=assets/'training/checkpoints/reskill_grpo/alfworld'/r['name']/f'validation_skills/step-{last_first_step}.jsonl'
   skills=[json.loads(line) for line in skill_file.read_text().splitlines()] if skill_file.exists() else []
   report['validation_skills_saved']=len(skills)==r['validation_size']
   report['validation_skills_count']=len(skills)
   report['validation_valid_skills']=sum(row['skill_valid'] for row in skills)
  (d/'five-step-supervision-report.json').write_text(json.dumps(report,indent=2)+'\n')
  print('FIRST_FIVE_VERIFIED '+json.dumps({'finite':finite,'peak_gib':peaks}),flush=True)
  reported_five=True
 if (d/'launcher.exit').exists():
  exits=[]
  for pid in set(int(v) for v in re.findall(r'WorkerDict pid=(\d+)',text)):
   path=Path(f'/proc/{pid}/stat')
   try:
    values=path.read_text().split(') ',1)[1].split()
    if values[0]=='Z':exits.append({'pid':pid,'wait_status':int(values[49])})
   except (FileNotFoundError,ProcessLookupError):pass
  (d/'worker-exit-status.json').write_text(json.dumps(exits,indent=2)+'\n')
  raise SystemExit('Training exited; inspect launcher.exit and worker-exit-status.json')
 time.sleep(1)
