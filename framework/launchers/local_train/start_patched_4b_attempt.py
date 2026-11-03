"""Launch the authorized 4B run after checks, retaining all earlier attempts."""
import datetime,json,shlex,subprocess,sys,uuid
from pathlib import Path
root=Path(__file__).resolve().parents[2];assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL');training=assets/'training'
number=int(sys.argv[1]);status=Path(sys.argv[2]);assert status.read_text().strip()=='0','Preflight did not pass'
name='reskill_grpo_qwen35_4b_alfworld_4gpu_180step_no_kl_no_quality_patched_swanlab_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:5]
d=training/'attempts'/name;d.mkdir(parents=True);session=f'reskill-qwen35-4b-patched-a{number:02d}'
r={'name':name,'session':session,'attempt_directory':str(d),'console_log':str(d/'console.log'),'stage':'starting','gpus':[0,1,2,3],
   'model_size':'4B','model_path':str(assets/'models/Qwen3.5-4B'),'total_training_steps':180,'save_freq':50,'test_freq':5,'max_actor_ckpts':4,
   'use_kl_loss':False,'use_kl_in_reward':False,'rewrite_quality_enabled':False,'lambda_skill':.1,'group_size':8,'resume_mode':'disable',
   'logging_backend':'swanlab','gpu_memory_utilization':.71,'ppo_max_token_len_per_gpu':20480,'forward_max_token_len_per_gpu':131072,
   'optimizer_offload':True,'free_cache_engine':True,'fail_on_nonfinite_gradient':False,'max_consecutive_nonfinite':3,
   'coordinated_finite_check':True,'stable_ppo_ratio':True,'audit_skills':True,'save_validation_skills':True,'validation_size':128,'completed_steps':0}
pointer=training/'current-run.json'
if pointer.exists():
 with (training/'attempts.jsonl').open('a') as f:f.write(json.dumps(json.loads(pointer.read_text()))+'\n')
pointer.write_text(json.dumps(r,indent=2)+'\n')
cmd='export EXPERIMENT_NAME='+shlex.quote(name)+'; bash launchers/local_train/run_qwen35_4b_4gpu_patched.sh > '+shlex.quote(str(d/'console.log'))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(d/'launcher.exit'))
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root),'bash --noprofile --norc -i'],check=True)
subprocess.run(['tmux','send-keys','-t',session,'-l',cmd],check=True);subprocess.run(['tmux','send-keys','-t',session,'Enter'],check=True)
for suffix,script,log in [('supervise','supervise_9b_first_five.py','supervision.log'),('swanlab-sync','watch_swanlab_sync.py','swanlab-watch.log'),('finalize','finalize_swanlab_on_exit.py','cloud-finalize.log')]:
 command='local_assets/conda/envs/agentrl-reskill/bin/python -u launchers/local_train/'+script+' > '+shlex.quote(str(d/log))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',f'sdar-4b-patched-{suffix}-a{number:02d}','-c',str(root),command],check=True)
with (training/'attempts.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
subprocess.run([sys.executable,str(root/'launchers/local_train/publish_project_run.py')],check=True)
print(json.dumps(r,indent=2))
