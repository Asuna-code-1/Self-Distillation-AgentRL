"""Expose experiment artifacts in the project while large files stay under /data."""
import json,time
from pathlib import Path

root=Path(__file__).resolve().parents[2]
assets=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
pointer=assets/'training/current-run.json';record=json.loads(pointer.read_text())
base=root/'runs';destination=base/record['name'];destination.mkdir(parents=True,exist_ok=True)
checkpoint=assets/'training/checkpoints/reskill_grpo/alfworld'/record['name']
numerics=assets/'training/numerics'/record['name'];numerics.mkdir(parents=True,exist_ok=True)
links={'logs':Path(record['attempt_directory']),'checkpoints':checkpoint,
       'validation_skills':checkpoint/'validation_skills','training_skills':checkpoint/'skill_refinement',
       'numerics':numerics,'resource_traces':assets/'training/resource-traces'/record['name'],
       'model':Path(record['model_path']),'source':root/'local_assets/run_sources'/record['name'],
       'recipe.json':root/'local_assets/recipes'/(record['name']+'.json')}
for name,target in links.items():
 link=destination/name
 if link.is_symlink():
  assert link.resolve()==target.resolve(),f'Unexpected existing link: {link}'
 elif link.exists():raise RuntimeError(f'Refusing to replace existing file: {link}')
 else:link.symlink_to(target,target_is_directory=name!='recipe.json')
current=base/'current'
if current.is_symlink() and current.resolve()!=destination.resolve():
 current.rename(base/f'previous-current-{time.time_ns()}')
if not current.is_symlink():
 if current.exists():raise RuntimeError('Refusing to replace runs/current')
 current.symlink_to(destination,target_is_directory=True)
status=base/'current-run.json'
if not status.is_symlink() and not status.exists():status.symlink_to(pointer)
record['project_run_directory']=str(destination);pending=pointer.with_name('current-run-publish-pending.json');pending.write_text(json.dumps(record,indent=2)+'\n');pending.replace(pointer)
print(destination)
