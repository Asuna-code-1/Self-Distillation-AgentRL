"""Stop one authorized run after a saved step and launch one fresh matched run."""
import argparse,datetime,hashlib,json,os,re,shlex,signal,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TRAINING=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/training')
POINTER=TRAINING/'current-run.json'

def write_json(path,value):
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)

def checkpoint_ready(record,target,training=TRAINING):
    text=Path(record['console_log']).read_text(errors='replace')
    if not re.search(r'\bstep:'+str(target)+r' -',text):return False
    base=training/'checkpoints/reskill_grpo/alfworld'/record['name']
    checkpoint=base/f'global_step_{target}'
    files=[checkpoint/'data.pt']+[checkpoint/'actor'/f'{kind}_world_size_4_rank_{rank}.pt' for kind in ['model','optim','extra_state'] for rank in range(4)]
    if not all(p.is_file() and p.stat().st_size>0 for p in files):return False
    validation=base/'validation_skills'/f'step-{target}.jsonl'
    if not validation.exists():return False
    try:rows=[json.loads(line) for line in validation.read_text().splitlines()]
    except json.JSONDecodeError:return False
    return len(rows)==record['validation_size'] and all(row['step']==target for row in rows)

def live_group(pgid):
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            values=(path/'stat').read_text().split(') ',1)[1].split()
            if int(values[2])==pgid and values[0]!='Z':result.append(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return result

def send_literal(session,command):
    subprocess.run(['tmux','send-keys','-t',session,'-l',command],check=True)
    subprocess.run(['tmux','send-keys','-t',session,'Enter'],check=True)

def stop_run(record,pgid):
    directory=Path(record['attempt_directory'])
    if not (directory/'launcher.exit').exists():
        subprocess.run(['tmux','send-keys','-t',record['session'],'C-z'],check=True)
        time.sleep(1)
        send_literal(record['session'],'kill %1');time.sleep(1)
        send_literal(record['session'],'kill -CONT %1')
    for attempt in range(20):
        if not live_group(pgid):break
        try:os.killpg(pgid,signal.SIGTERM);os.killpg(pgid,signal.SIGCONT)
        except ProcessLookupError:break
        time.sleep(1)
    if live_group(pgid):
        os.killpg(pgid,signal.SIGKILL)
        for _ in range(10):
            if not live_group(pgid):break
            time.sleep(1)
    assert not live_group(pgid),'Old training group has not exited; refusing a second GPU job'
    if not (directory/'launcher.exit').exists():(directory/'launcher.exit').write_text('143\n')

def finish_cloud(record,target):
    # Wait until the old finalizer has exited before applying planned completion.
    directory=Path(record['attempt_directory'])
    for _ in range(60):
        if (directory/'cloud-final-state.json').exists():break
        time.sleep(2)
    if not (directory/'cloud-final-state.json').exists():return False
    os.environ.setdefault('SWANLAB_ROOT',str(ROOT/'local_assets/swanlab-config'))
    os.environ.setdefault('SWANLAB_CONFIG_DIR',str(ROOT/'local_assets/swanlab-config'))
    import swanlab
    match=re.search(r'/@([^/]+)/([^/]+)/runs/([^/]+)',record['swanlab_url']);owner,project,slug=match.groups()
    key=Path('/workspace/fudan/donghaosen/.credentials/swanlab-20261031-4b.key').read_text().strip()
    api=swanlab.Api(api_key=key);run=api.run(f'{owner}/{project}/{slug}')
    assert run.name==record['name'],'Cloud run identity mismatch'
    response=run._ctx.client.put(f'/project/{owner}/{project}/runs/{slug}/state',data={'state':'FINISHED','finishedAt':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),'from':'sdk'})
    assert response.raw.ok
    state=swanlab.Api(api_key=key).run(f'{owner}/{project}/{slug}').state
    assert state=='FINISHED'
    write_json(directory/'planned-completion.json',{'completed_step':target,'cloud_state':state,'reason':'User requested completion at this step followed by a new fresh run'})
    return True

def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan');args=parser.parse_args()
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());record=plan['run'];directory=Path(record['attempt_directory'])
    def status(stage,**fields):
        plan.update(status=stage,updated_at=time.time(),**fields);write_json(plan_path,plan);print(json.dumps({'status':stage,**fields}),flush=True)
    status('waiting_for_step',target_step=plan['target_step'])
    last_completed=-1
    while not checkpoint_ready(record,plan['target_step']):
        current=json.loads(POINTER.read_text())
        if current['name']!=record['name']:raise RuntimeError('Current run changed before handoff; refusing to stop a different run')
        if current.get('completed_steps',0)!=last_completed:
            last_completed=current.get('completed_steps',0);status('waiting_for_step',current_completed_step=last_completed)
        if (directory/'launcher.exit').exists():raise RuntimeError('Run exited before the requested checkpoint was completed')
        time.sleep(2)
    with (directory/'handoff-claimed.json').open('x') as handle:json.dump({'target_step':plan['target_step'],'time':time.time()},handle)
    status('stopping_after_checkpoint')
    stop_run(record,plan['training_pgid']);status('old_run_stopped')
    # Never silently launch with source changes made after the handoff was armed.
    for relative,expected in plan['source_hashes'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,'Source changed while waiting: '+relative
    assert json.loads(POINTER.read_text())['name']==record['name']
    environment=dict(os.environ);environment.update(plan['safe_environment'])
    environment['GPU_MEMORY_UTILIZATION']=str(record['gpu_memory_utilization'])
    command=[str(ROOT/'local_assets/conda/envs/agentrl-reskill/bin/python'),str(ROOT/'launchers/local_train/start_fresh_kl_eos_fixed_4b.py'),str(plan['next_attempt_number']),str(ROOT/'local_assets/setup/kl-eos-resume50-checks-20261102.exit'),record['name']]
    with (directory/'next-run-launch.log').open('x') as log:
        subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    new=json.loads(POINTER.read_text())
    assert new['name']!=record['name'] and new['resume_mode']=='disable' and new['resume_from_path'] is None and new['fresh_base_model']
    status('new_run_started',new_run_name=new['name'],new_run_directory=new['project_run_directory'])
    try:cloud_finished=finish_cloud(record,plan['target_step'])
    except Exception as error:
        cloud_finished=False;print('Cloud completion update failed: '+type(error).__name__,flush=True)
    plan['old_cloud_marked_finished']=cloud_finished

    for _ in range(600):
        current=json.loads(POINTER.read_text())
        if current['name']!=new['name']:raise RuntimeError('New run pointer changed while checking startup')
        if current.get('swanlab_url'):
            status('complete',new_swanlab_url=current['swanlab_url']);return
        if (Path(new['attempt_directory'])/'launcher.exit').exists():raise RuntimeError('New run exited during startup; inspect its logs')
        time.sleep(2)
    status('new_run_started_url_pending')

if __name__=='__main__':
    try:main()
    except Exception as error:
        try:
            failure_path=Path(os.sys.argv[1]);failure=json.loads(failure_path.read_text());failure.update(status='failed',error_type=type(error).__name__,error=str(error),updated_at=time.time());write_json(failure_path,failure)
        except Exception:pass
        print('HANDOFF_FAILED '+type(error).__name__+': '+str(error),flush=True)
        raise
