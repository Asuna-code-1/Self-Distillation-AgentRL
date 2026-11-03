"""Compact read-only progress from the current isolated experiment."""
import json,re
from pathlib import Path
assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
r=json.loads((assets/'training/current-run.json').read_text());dest=Path(r['attempt_directory'])
text=Path(r['console_log']).read_text();lines=text.splitlines();trace=assets/'training/resource-traces'/r['name']
result={'name':r['name'],'phase':'initializing','completed_steps':[]}
p=trace/'phases.jsonl'
if p.exists():result['phase']=json.loads(p.read_text().splitlines()[-1])
p=trace/'step_peaks.jsonl'
if p.exists():
 rows=[json.loads(s) for s in p.read_text().splitlines() if s]
 result['completed_steps']=[v['step'] for v in rows]
 peaks={}
 for key,value in rows[-1]['metrics'].items():
  phase=key.split('_gpu_')[0].replace('resources/','');peaks[phase]=max(peaks.get(phase,0),value)
 result['latest_phase_peaks_gib']=peaks
p=trace/'gpu_samples.jsonl'
if p.exists():result['gpu']=json.loads(p.read_text().splitlines()[-1])
metrics=[s for s in lines if re.search(r'step:\d+ -',s)]
if metrics:
 latest=metrics[-1];keys=['timing_s/gen','timing_s/update_actor','timing_s/step','episode/valid_action_ratio','actor/grad_norm',
 'actor/skill_refinement_sft_loss','rewrite_quality/active_step','rewrite_quality/evaluated_groups','rewrite_quality/positive_groups',
 'rewrite_quality/negative_groups','actor/rewrite_quality_loss','response_length/mean','episode/reward/mean']
 result['metrics']={}
 for key in keys:
  match=re.search(re.escape(key)+r':([^ ]+)',latest)
  if match:result['metrics'][key]=match[1]
p=dest/'wandb-sync-checks.jsonl'
if p.exists():result['wandb']=json.loads(p.read_text().splitlines()[-1])
p=dest/'swanlab-sync-checks.jsonl'
if p.exists():result['swanlab']=json.loads(p.read_text().splitlines()[-1])
for filename in ('launcher.exit','gpu-alert.json','first-five-memory.json'):
 p=dest/filename
 if p.exists():result[filename]=p.read_text()
if 'launcher.exit' in result:result['log_tail']='\n'.join(lines[-12:])
print(json.dumps(result,ensure_ascii=False))
