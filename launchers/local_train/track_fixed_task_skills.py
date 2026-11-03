"""Read saved validation skills every ten steps; never invokes a model or changes training."""
import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
record=json.loads(Path('/data/fd/donghaosen/Self-Distillation-AgentRL/training/current-run.json').read_text())
configuration=root/'runs'/record['name']/'fixed-task-tracking.json'
config=json.loads(configuration.read_text());run=Path(record['attempt_directory'])
validation=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/training/checkpoints/reskill_grpo/alfworld')/record['name']/'validation_skills'
peaks=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/training/resource-traces')/record['name']/'step_peaks.jsonl'
output=run/'fixed-task-skills-every-10.jsonl';readable=run/'fixed-task-skills-every-10.txt'
seen={json.loads(line)['step'] for line in output.read_text().splitlines()} if output.exists() else set()
while True:
    try:
        completed={json.loads(line)['step'] for line in peaks.read_text().splitlines() if line} if peaks.exists() else set()
    except json.JSONDecodeError:
        time.sleep(2);continue
    for step in sorted(s for s in completed if s%10==0 and s not in seen):
        path=validation/f'step-{step}.jsonl'
        if not path.exists():continue
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        matches=[row for row in rows if row['task']==config['task']]
        exact=[row for row in matches if row['gamefile']==config['preferred_gamefile']]
        chosen=exact[0] if exact else matches[0] if matches else None
        row={'step':step,'task':config['task'],'status':'present' if chosen else 'missing',
             'same_instance':bool(exact),'record':chosen,'source_file':str(path)}
        with output.open('a') as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        text=f"Step {step}\nTask: {config['task']}\nSame task instance: {bool(exact)}\n"
        if chosen:
            text+=chosen['raw_skill']+f"\nFormat valid: {chosen['skill_valid']}; episode success: {chosen['success']}\n\n"
        else:text+='This validation did not contain the fixed task; no output was fabricated.\n\n'
        with readable.open('a') as handle:handle.write(text)
        print(text,flush=True);seen.add(step)
    if (run/'launcher.exit').exists():break
    time.sleep(5)
