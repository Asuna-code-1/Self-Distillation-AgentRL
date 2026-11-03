"""Read-only GPU sampling; stop only this tmux job if memory exceeds 130 GiB."""
import json,subprocess,sys,time
from pathlib import Path
import pynvml
ASSETS=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
record=json.loads((ASSETS/'training/current-run.json').read_text())
status=Path(record.get('preflight_status',str(ASSETS/'logs/training/preflight-rewrite-quality.exit')))
while not status.exists():time.sleep(1)
if status.read_text().strip()!='0':raise SystemExit('Preflight failed; watchdog not armed')
pynvml.nvmlInit()
handles=[pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(4)]
destination=Path(record['attempt_directory']);trace=ASSETS/'training/resource-traces'/record['name']
consecutive=0;peaks=[0.]*4
keep_guard="--keep-guard" in sys.argv
reported_five=False
while True:
 used=[pynvml.nvmlDeviceGetMemoryInfo(h).used/2**30 for h in handles]
 peaks=[max(a,b) for a,b in zip(peaks,used)]
 with (destination/'gpu-watch.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'used_gib':used})+'\n')
 consecutive=consecutive+1 if max(used)>130 else 0
 if consecutive>=2:
  alert={'time':time.time(),'sampled_peak_gib':peaks,'reason':'More than 130 GiB for two samples','session':record['session']}
  (destination/'gpu-alert.json').write_text(json.dumps(alert,indent=2)+'\n')
  # The interactive tmux session has exactly one foreground training job.
  subprocess.run(['/usr/bin/tmux','send-keys','-t',record['session'],'C-z'],check=True)
  time.sleep(.8)
  for command in ('kill %1','kill -CONT %1'):
   subprocess.run(['/usr/bin/tmux','send-keys','-t',record['session'],'-l',command],check=True)
   subprocess.run(['/usr/bin/tmux','send-keys','-t',record['session'],'Enter'],check=True)
   time.sleep(.4)
  raise SystemExit('Stopped this job at GPU memory guardrail; inspect gpu-alert.json')
 step_file=trace/'step_peaks.jsonl'
 if step_file.exists():
  try:
   data=[json.loads(line) for line in step_file.read_text().splitlines() if line]
   if not reported_five and any(v['step']>=5 for v in data):
    (destination/'first-five-memory.json').write_text(json.dumps({'sampled_peaks_gib':peaks,'at_least_five_steps_completed':True},indent=2)+'\n')
    print('FIRST_FIVE_RESOURCE_WINDOW_COMPLETED',peaks,flush=True)
    reported_five=True
    if not keep_guard:break
  except json.JSONDecodeError:pass
 exit_file=destination/'launcher.exit'
 if exit_file.exists():raise SystemExit('Training exited; inspect console.log')
 time.sleep(.3)
