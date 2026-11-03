"""Maintain current-run metadata for one isolated attempt without touching training."""
import json,re,time
from pathlib import Path
pointer=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/training/current-run.json')
initial=json.loads(pointer.read_text());name=initial['name'];directory=Path(initial['attempt_directory'])
while True:
    record=json.loads(pointer.read_text())
    if record['name']!=name:break
    text=Path(record['console_log']).read_text(errors='replace')
    steps=[int(s) for s in re.findall(r'\bstep:(\d+) -',text)]
    record['completed_steps']=max(steps,default=int(record.get('resume_step',0)))
    urls=re.findall(r'https://swanlab\.(?:cn|ai)/@[\w.-]+/[\w.+-]+/runs/[\w-]+',text)
    if urls:record['swanlab_url']=urls[-1]
    record['stage']='running' if record['completed_steps']>int(record.get('resume_step',0)) else 'validating_before_training' if steps else 'initializing'
    report=directory/'five-step-supervision-report.json'
    if report.exists():
        result=json.loads(report.read_text())
        record['first_five_verified']=bool(result['five_steps_completed'] and result['actor_metrics_finite'] and result.get('skill_semantics_verified',False) and result.get('validation_skills_saved',False) and result.get('quality_supervision_disabled_verified',False))
        record['first_five_peaks_gib']=result['sampled_peak_gib']
    exitfile=directory/'launcher.exit'
    if exitfile.exists():
        code=int(exitfile.read_text().strip());record['launcher_exit']=code
        record['stage']='finished' if code==0 else 'stopped' if code in (143,148) else 'failed'
    record['last_status_update']=time.time()
    temporary=pointer.with_name('current-run-status-pending.json')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(pointer)
    if exitfile.exists():break
    time.sleep(15)
