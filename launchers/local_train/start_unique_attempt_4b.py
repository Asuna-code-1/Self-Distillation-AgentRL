"""Start only the user-authorized, uniquely named Qwen3.5 experiment in tmux."""
import datetime,json,shlex,subprocess,sys,uuid
from pathlib import Path
root=Path(__file__).resolve().parents[2];assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
number=int(sys.argv[1]);status=Path(sys.argv[2]);pointer=assets/'training/current-run.json'
if pointer.exists():
 old=json.loads(pointer.read_text());old['stage']='previous_attempt_ended'
 with (assets/'training/attempts.jsonl').open('a') as f:f.write(json.dumps(old)+'\n')
name='reskill_grpo_qwen35_4b_alfworld_4gpu_160step_no_rewrite_quality_swanlab_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_a{number:02d}_'+uuid.uuid4().hex[:5]
dest=assets/'training/attempts'/name;dest.mkdir(parents=True)
record={'name':name,'session':f'reskill-qwen35-4b-a{number:02d}','attempt_directory':str(dest),
        'console_log':str(dest/'console.log'),'stage':'waiting_for_preflight','preflight_status':str(status),
        'gpus':[0,1,2,3],'lambda_skill':.1,'rewrite_quality_enabled':False,'total_training_steps':160,
        'logging_backend':'swanlab','swanlab_project':'agentic_alfworld',
        'model_size':'4B','model_path':str(root/'models/Qwen3.5-4B'),'disable_thinking':True,
        'ppo_max_token_len_per_gpu':20480,'forward_max_token_len_per_gpu':131072,
        'gpu_memory_utilization':.71,'fail_on_nonfinite_gradient':True}
pointer.write_text(json.dumps(record,indent=2)+'\n')
with (assets/'training/attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
subprocess.run(['/usr/bin/tmux','new-session','-d','-s',record['session'],'-c',str(root),'bash --noprofile --norc -i'],check=True)
cmd='bash launchers/local_train/launch_after_preflight_4b.sh '+shlex.quote(str(status))+' '+shlex.quote(name)+' '+shlex.quote(str(dest))+' > '+shlex.quote(str(dest/'console.log'))+' 2>&1'
subprocess.run(['/usr/bin/tmux','send-keys','-t',record['session'],'-l',cmd],check=True)
subprocess.run(['/usr/bin/tmux','send-keys','-t',record['session'],'Enter'],check=True)
watchcmd='local_assets/conda/envs/agentrl-reskill/bin/python launchers/local_train/watch_first_five.py > '+shlex.quote(str(dest/'watchdog.log'))+' 2>&1'
subprocess.run(['/usr/bin/tmux','new-session','-d','-s',f'sdar-4b-first-five-a{number:02d}','-c',str(root),watchcmd],check=True)
print(json.dumps(record,indent=2))
