"""Local matched ALFWorld recipe; delegates all training to the native launcher."""
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
from uuid import uuid4


def recipe():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    defaults = {
        'PAPER_CONFIG': '1', 'GPU_ID': '0,1,2,3', 'TRAIN_DATA_SIZE': '16',
        'VAL_DATA_SIZE': '128', 'GROUP_SIZE': '8', 'PPO_MINI_BATCH_SIZE': '256',
        'TOTAL_TRAINING_STEPS': '160', 'TOTAL_EPOCHS': '160', 'TEST_FREQ': '5',
        'MAX_ENV_STEPS': '30', 'MAX_PROMPT_LENGTH': '2048', 'MAX_RESPONSE_LENGTH': '512',
        'VAL_BEFORE_TRAIN': 'true', 'RESUME_MODE': 'disable', 'ALFWORLD_STRICT_EVAL': '0',
        'WANDB_MODE': 'online', 'SAVE_FREQ': '160',
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    env.setdefault('EXPERIMENT_NAME', f'reskill_grpo_qwen35_{env.get("MODEL_SIZE", "2B")}_{stamp}')
    name = env['EXPERIMENT_NAME']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('EXPERIMENT_NAME must be a filename-safe identifier')
    backend = env.get('RESKILL_LOGGER', 'wandb')
    if backend not in {'wandb','swanlab'}:
        raise ValueError('Unsupported experiment logging backend')
    count = len(env['GPU_ID'].split(','))
    overrides = [
        'algorithm.gamma=0.95', f'trainer.n_gpus_per_node={count}',
        'trainer.project_name=agentic_alfworld', f'trainer.logger=[console,{backend}]',
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton',
    ]
    return root, env, overrides


def main():
    root, env, overrides = recipe()
    args = sys.argv[1:]
    dry = '--print-config' in args
    args = [arg for arg in args if arg != '--print-config']
    command = ['bash', str(root / 'launchers/run_reskill_grpo_1gpu.sh'), 'alfworld', *overrides, *args]
    # Never export credentials or the whole inherited environment in a manifest.
    keys = ['MODEL_PATH', 'CONDA_ENV', 'PROJECT_ROOT', 'DATA_ROOT', 'GPU_ID', 'PAPER_CONFIG',
            'TRAIN_DATA_SIZE', 'VAL_DATA_SIZE', 'GROUP_SIZE', 'PPO_MINI_BATCH_SIZE',
            'TOTAL_TRAINING_STEPS', 'MAX_ENV_STEPS', 'MAX_PROMPT_LENGTH', 'MAX_RESPONSE_LENGTH',
            'TEST_FREQ', 'EXPERIMENT_NAME', 'WANDB_MODE', 'SAVE_FREQ', 'ALFWORLD_STRICT_EVAL', 'RESKILL_LOGGER']
    manifest = {'environment': {key: env[key] for key in keys if key in env}, 'command': command}
    if dry:
        print(json.dumps(manifest, indent=2))
        return
    if not env.get('TMUX'):
        raise SystemExit('Run authorized training inside tmux.')
    destination = root / 'checkpoints/reskill_grpo/alfworld' / env['EXPERIMENT_NAME']
    if destination.exists():
        raise SystemExit(f'Run directory already exists; choose another name: {destination}')
    # Refuse CLI overrides of identity/output paths which could bypass isolation.
    forbidden = ('trainer.experiment_name=', 'trainer.default_local_dir=', 'trainer.resume_mode=')
    if any(arg.lstrip('+').startswith(forbidden) for arg in args):
        raise SystemExit('Use EXPERIMENT_NAME or the dedicated resume workflow for run identity.')
    backend = env.get('RESKILL_LOGGER', 'wandb')
    if backend == 'swanlab':
        if not env.get('SWANLAB_API_KEY'):
            credential = Path(env['SWANLAB_API_KEY_FILE'])
            env['SWANLAB_API_KEY'] = credential.read_text().strip()
        env['SWANLAB_MODE'] = 'online'
        env.pop('SWANLAB_RUN_ID', None)
        env.pop('SWANLAB_RESUME', None)
    if backend == 'wandb' and env.get('WANDB_MODE') == 'online' and not env.get('WANDB_API_KEY'):
        rules = root.parent / 'AGENTS.md'
        match = re.search(r'wandb_v1_[A-Za-z0-9_-]+', rules.read_text()) if rules.is_file() else None
        if not match:
            raise SystemExit('Set WANDB_API_KEY in the environment for online logging.')
        env['WANDB_API_KEY'] = match[0]
    # Do not reuse a W&B identity inherited from an older experiment.
    if backend == 'wandb':
        env['WANDB_RUN_ID'] = uuid4().hex[:8]
        env['WANDB_RESUME'] = 'never'
    location = root / 'local_assets/recipes' / (env['EXPERIMENT_NAME'] + '.json')
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open('x') as handle:
        json.dump(manifest, handle, indent=2)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
