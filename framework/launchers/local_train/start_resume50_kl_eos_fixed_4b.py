"""Launch the authorized 4B run after checks, retaining all earlier attempts."""
import datetime,json,shlex,subprocess,sys,uuid
from pathlib import Path
root=Path(__file__).resolve().parents[2];assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL');training=assets/'training'
old_name='reskill_grpo_qwen35_4b_alfworld_4gpu_180step_kl_no_quality_userprompts_swanlab_20261102T022507Z_ba133'
checkpoint=training/'checkpoints/reskill_grpo/alfworld'/old_name/'global_step_50'
required=[checkpoint/'data.pt']+[checkpoint/'actor'/f'{kind}_world_size_4_rank_{rank}.pt' for kind in ('model','optim','extra_state') for rank in range(4)]
assert all(p.is_file() and p.stat().st_size>0 for p in required),'Incomplete resume checkpoint'
number=int(sys.argv[1]);status=Path(sys.argv[2]);assert status.read_text().strip()=='0','Preflight did not pass'
name='reskill_grpo_qwen35_4b_alfworld_4gpu_180step_kl_no_quality_resume50_kl_eos_fixed_swanlab_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:5]
d=training/'attempts'/name;d.mkdir(parents=True);session=f'reskill-qwen35-4b-resume50-a{number:02d}'
r={'name':name,'session':session,'attempt_directory':str(d),'console_log':str(d/'console.log'),'stage':'starting','gpus':[0,1,2,3],
   'model_size':'4B','model_path':str(assets/'models/Qwen3.5-4B'),'total_training_steps':180,'save_freq':50,'test_freq':5,'max_actor_ckpts':4,
   'use_kl_loss':True,'kl_loss_coef':.01,'use_kl_in_reward':False,'rewrite_quality_enabled':False,'lambda_skill':.1,'group_size':8,'resume_mode':'resume_path','resume_step':50,'resume_from_path':str(checkpoint),'parent_run':old_name,
   'logging_backend':'swanlab','gpu_memory_utilization':.71,'ppo_max_token_len_per_gpu':20480,'forward_max_token_len_per_gpu':131072,
   'optimizer_offload':True,'free_cache_engine':True,'fail_on_nonfinite_gradient':False,'max_consecutive_nonfinite':3,
   'coordinated_finite_check':True,'stable_ppo_ratio':True,'audit_skills':True,'save_validation_skills':True,'validation_size':128,'completed_steps':50}
# Archive the exact training sources before any new worker imports them.
import hashlib,shutil
snapshot=root/'local_assets/run_sources'/name
files=['verl/trainer/ppo/reskill_grpo.py','verl/trainer/ppo/core_algos.py',
       'verl/trainer/ppo/ray_trainer.py','verl/workers/actor/dp_actor.py',
       'verl/utils/numerics.py','verl/trainer/config/ppo_trainer.yaml',
       'launchers/local_train/run_qwen35_4b_resume50_kl_eos_fixed.sh',
       'launchers/local_train/start_resume50_kl_eos_fixed_4b.py',
       'launchers/run_skill_method_1gpu.sh','launchers/reskill_recipe.py',
       'tests/test_reskill_grpo.py','tests/test_numerics_stability.py',
       'tests/test_kl_eos_stability.py','verl/utils/torch_functional.py',
       'verl/workers/fsdp_workers.py','verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
       'launchers/local_train/supervise_9b_first_five.py','launchers/local_train/watch_run_status.py']
hashes={}
for relative in files:
 target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(root/relative,target);hashes[relative]=hashlib.sha256(target.read_bytes()).hexdigest()
(snapshot/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
r['source_snapshot']=str(snapshot)
r['prompt_variant']='user_exact_20261102'
r['fresh_base_model']=False
r['stable_kl_exponent']=True
r['response_mask_from_engine_lengths']=True
r['tokenizer_eos_metadata']=True
r['checkpoint_integrity_file_count']=len(required)
pointer=training/'current-run.json'
if pointer.exists():
 with (training/'attempts.jsonl').open('a') as f:f.write(json.dumps(json.loads(pointer.read_text()))+'\n')
pointer.write_text(json.dumps(r,indent=2)+'\n')
cmd='export EXPERIMENT_NAME='+shlex.quote(name)+' RESUME_CHECKPOINT='+shlex.quote(str(checkpoint))+'; bash launchers/local_train/run_qwen35_4b_resume50_kl_eos_fixed.sh > '+shlex.quote(str(d/'console.log'))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(d/'launcher.exit'))
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root),'bash --noprofile --norc -i'],check=True)
subprocess.run(['tmux','send-keys','-t',session,'-l',cmd],check=True);subprocess.run(['tmux','send-keys','-t',session,'Enter'],check=True)
subprocess.run([sys.executable,str(root/'launchers/local_train/publish_project_run.py')],check=True)
for suffix,script,log in [('supervise','supervise_9b_first_five.py','supervision.log'),('swanlab-sync','watch_swanlab_sync.py','swanlab-watch.log'),('finalize','finalize_swanlab_on_exit.py','cloud-finalize.log'),('status','watch_run_status.py','status-watch.log')]:
 command='local_assets/conda/envs/agentrl-reskill/bin/python -u launchers/local_train/'+script+' > '+shlex.quote(str(d/log))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',f'sdar-4b-resume50-{suffix}-a{number:02d}','-c',str(root),command],check=True)
with (training/'attempts.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
destination=root/'runs'/name
configuration=root/'runs'/old_name/'fixed-task-tracking.json'
(destination/'fixed-task-tracking.json').write_text(configuration.read_text())
command='local_assets/conda/envs/agentrl-reskill/bin/python -u launchers/local_train/track_fixed_task_skills.py > '+shlex.quote(str(d/'fixed-task-tracker.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',f'sdar-4b-resume50-fixed-task-a{number:02d}','-c',str(root),command],check=True)
print(json.dumps(r,indent=2))
