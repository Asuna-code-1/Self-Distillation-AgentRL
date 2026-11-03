"""Start an isolated, authorized 9B attempt and its read-only observers."""
import datetime,json,shlex,subprocess,sys,uuid
from pathlib import Path
root=Path(__file__).resolve().parents[2];a=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/training');pointer=a/'current-run.json'
number=int(sys.argv[1]);old=json.loads(pointer.read_text())
old['stage']='stopped_by_memory_guard' if Path(old['attempt_directory'],'gpu-alert.json').exists() else 'stopped_for_throughput_calibration'
with (a/'attempts.jsonl').open('a') as f:f.write(json.dumps(old)+'\n')
name='reskill_grpo_qwen35_9b_alfworld_4gpu_180step_no_kl_swanlab_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:5]
d=a/'attempts'/name;d.mkdir(parents=True);session=f'reskill-qwen35-9b-swanlab-a{number:02d}'
r={**old,'name':name,'session':session,'attempt_directory':str(d),'console_log':str(d/'console.log'),'stage':'starting','completed_steps':0,'ppo_max_token_len_per_gpu':6144,'forward_max_token_len_per_gpu':98304,'gpu_memory_utilization':.62,'save_freq':50,'max_actor_ckpts':4,'optimizer_offload':True}
pointer.write_text(json.dumps(r,indent=2)+'\n')
cmd='export EXPERIMENT_NAME='+shlex.quote(name)+'; bash launchers/local_train/run_qwen35_9b_4gpu_no_kl.sh > '+shlex.quote(str(d/'console.log'))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(d/'launcher.exit'))
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root),'bash --noprofile --norc -i'],check=True)
subprocess.run(['tmux','send-keys','-t',session,'-l',cmd],check=True);subprocess.run(['tmux','send-keys','-t',session,'Enter'],check=True)
for sess,script,log in [(f'sdar-9b-supervise-a{number:02d}','supervise_9b_first_five.py','supervision.log'),(f'sdar-9b-swanlab-sync-a{number:02d}','watch_swanlab_sync.py','swanlab-watch.log'),(f'sdar-9b-finalize-a{number:02d}','finalize_swanlab_on_exit.py','cloud-finalize.log')]:
 command='local_assets/conda/envs/agentrl-reskill/bin/python -u launchers/local_train/'+script+' > '+shlex.quote(str(d/log))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',sess,'-c',str(root),command],check=True)
with (a/'attempts.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
print(json.dumps(r,indent=2))
