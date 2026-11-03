"""ReSkill-GRPO using this project's native veRL/Ray/FSDP training path.

Ported from the local SEED ReSkill implementation; no SEED runtime dependency.

One policy generates a skill, executes a group, and rewrites that skill. Only
task -> rewritten skill is supervised. This module never changes KL settings.
"""
import json
import os
import math
from pathlib import Path
import re
from difflib import SequenceMatcher
import time

import torch


def condition_observation(observation, skill):
    """Place an episode skill after the task, preserving the action instructions."""
    text = str(observation).strip()
    section = ('**Episode-Level Skill**\nRefer to this episode-level skill when deciding '
               'what action to take in the current episode:\n[' + skill.strip() + ']')
    for pattern in (r'^.*\bYour task is to:\s*.*(?:\n|$)',
                    r'^.*\bYour current task is:\s*.*(?:\n|$)',
                    r'^.*\bYour task is:\s*.*(?:\n|$)',
                    r'^.*\bYour question:\s*.*(?:\n|$)',
                    r'^.*\bYour goal is(?: to)?\b[:\s].*(?:\n|$)'):
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return '\n\n'.join(part for part in
                                (text[:match.end()].rstrip(), section, text[match.end():].lstrip()) if part)
    for anchor in ("Now it's your turn to", "Now it's your turn"):
        index = text.find(anchor)
        if index >= 0:
            return '\n\n'.join(part for part in (text[:index].rstrip(), section, text[index:].lstrip()) if part)
    return text + '\n\n' + section if text else section


def skill_prompt(task):
    return (
        'Before solving the task, write a concise, task-specific procedural skill. '
        'Describe a plan, pitfalls, and checks; do not execute actions or invent observations. '
        'Do not output analysis, reasoning, or thinking tags. '
        'Output only your skill enclosed in <skill>...</skill>, with no text outside these tags.\n'
        'Task:\n' + task
    )


def parse_skill(text, max_chars):
    match = re.fullmatch(r'\s*<skill>\s*([^<>]+?)\s*</skill>\s*', text, re.DOTALL)
    if match is None or not 0 < len(match[1].strip()) <= max_chars:
        raise ValueError('Expected one complete, nonempty <skill> block within the skill budget')
    skill = match[1].strip()
    label = re.sub(r'[\W_]+', ' ', skill.casefold()).strip()
    if not label or label in {
        'skill', 'your skill', 'revised skill', 'rewritten skill', 'procedural skill',
        'your revised skill', 'your rewritten skill', 'insert skill here',
    }:
        raise ValueError('Placeholder text is not a task-specific skill')
    return skill


def sft_payload(tokenizer, pairs, max_prompt_tokens, max_response_tokens, template_kwargs=None):
    """Left-pad prompts, right-pad labels; train only labels, including EOS.

    Retokenize the accepted rewrite under the ORIGINAL generation prompt. Never
    reuse the rewrite prompt or its continuation token IDs as the SFT input.
    """
    rows = []
    for task, skill in pairs:
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': skill_prompt(task)}],
            tokenize=True, return_dict=False, add_generation_prompt=True, **(template_kwargs or {}),
        )
        response = tokenizer.encode(f'<skill>{skill}</skill>', add_special_tokens=False)
        if tokenizer.eos_token_id is None:
            raise ValueError('Skill SFT requires an EOS token')
        response.append(tokenizer.eos_token_id)
        if len(prompt) <= max_prompt_tokens and len(response) <= max_response_tokens:
            rows.append((prompt, response))
    if not rows:
        return None
    pwidth = max(len(p) for p, _ in rows)
    rwidth = max(len(r) for _, r in rows)
    ids = torch.full((len(rows), pwidth + rwidth), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, (prompt, response) in enumerate(rows):
        ids[i, pwidth-len(prompt):pwidth] = torch.tensor(prompt)
        ids[i, pwidth:pwidth+len(response)] = torch.tensor(response)
        mask[i, pwidth-len(prompt):pwidth+len(response)] = 1
    return {
        'input_ids': ids, 'responses': ids[:, pwidth:].clone(),
        'attention_mask': mask, 'position_ids': (mask.cumsum(-1) - 1).clamp_min(0),
    }


def sft_shard_indices(count, rank, world_size):
    """Every FSDP rank must run the same number of forwards/backwards.

    Duplicate rows have zero weight, so uneven/small batches do not hang ranks
    or bias the global example mean. No process group or ports are created.
    """
    if count < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid SFT shard')
    slots = math.ceil(count / world_size)
    raw = torch.arange(slots) * world_size + rank
    return raw.clamp_max(count - 1), (raw < count).float()


def refinement_metrics(records, tokenizer):
    """Descriptive metrics from existing samples; no extra evaluation rollouts.

    Success refers to executions under the INITIAL skill. Edit distance and
    acceptance are not evidence that the rewritten skill improves task reward.
    """
    def mean(values):
        return sum(values) / max(len(values), 1)
    rewrites = [r for r in records if 'raw_rewrite' in r]
    valid = [r for r in rewrites if 'refined_skill' in r]
    accepted = [r for r in records if r['accepted']]
    successes = [[t['reward'] > 0 for t in r['trajectories']] for r in records]
    initial_tokens = [tokenizer.encode(r['initial_skill'], add_special_tokens=False)
                      for r in records if r['initial_skill']]
    refined_tokens = [tokenizer.encode(r['refined_skill'], add_special_tokens=False) for r in valid]
    metrics = {
        'groups': len(records), 'accepted': len(accepted),
        'invalid_initial': sum(not r['initial_skill'] for r in records),
        'initial_valid_rate': mean([bool(r['initial_skill']) for r in records]),
        'rewrite_attempt_rate': len(rewrites) / max(len(records), 1),
        'rewrite_valid_rate': len(valid) / max(len(rewrites), 1),
        'acceptance_rate': len(accepted) / max(len(records), 1),
        'initial_skill_tokens_mean': mean([len(ids) for ids in initial_tokens]),
        'refined_skill_tokens_mean': mean([len(ids) for ids in refined_tokens]),
        'task_success_rate': mean([s for group in successes for s in group]),
        'group_any_success_rate': mean([any(group) for group in successes]),
        'group_all_success_rate': mean([bool(group) and all(group) for group in successes]),
        'group_mixed_outcome_rate': mean([any(group) and not all(group) for group in successes]),
        'rewrite_changed_rate': mean([r['initial_skill'] != r['refined_skill'] for r in valid]),
        'rewrite_edit_ratio_mean': mean([
            1 - SequenceMatcher(None, r['initial_skill'], r['refined_skill'], autojunk=False).ratio()
            for r in valid]),
        'accepted_unique_skill_rate': len({r['refined_skill'] for r in accepted}) / max(len(accepted), 1),
    }
    for reason in ('invalid_rewrite', 'no_successful_evidence', 'rewrite_context_budget', 'sft_context_budget'):
        metrics['rejected_' + reason] = sum(r['reason'] == reason for r in records)
    return {'skill_refinement/' + key: value for key, value in metrics.items()}


def backward_refinement_sft(actor, payload):
    """Accumulate weighted SFT gradients; do not zero gradients or step Adam.

    FSDP averages gradients across ranks; world_size / N gives the exact global
    mean. SFT uses temperature 1 regardless of the rollout sampling temperature.
    """
    if actor.ulysses_sequence_parallel_size != 1:
        raise ValueError('Skill refinement currently requires actor SP=1')
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world = torch.distributed.get_world_size() if distributed else 1
    count = len(payload['responses'])
    indices, valid = sft_shard_indices(count, rank, world)
    device = next(actor.actor_module.parameters()).device
    tensors = {key: torch.as_tensor(payload[key]).index_select(0, indices).to(device)
               for key in ('responses', 'input_ids', 'attention_mask', 'position_ids')}
    valid = valid.to(device)
    coef = float(payload['loss_coef'])
    micro = int(payload['micro_batch_size'])
    if not math.isfinite(coef) or coef < 0 or micro < 1:
        raise ValueError('Invalid skill refinement loss configuration')
    loss_sum = 0.0
    for start in range(0, len(indices), micro):
        batch = {key: value[start:start+micro] for key, value in tensors.items()}
        _, log_probs = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=False)
        response_mask = batch['attention_mask'][:, -batch['responses'].shape[1]:].to(log_probs.dtype)
        per_example = -(log_probs * response_mask).sum(-1) / response_mask.sum(-1).clamp_min(1)
        weighted_sum = (per_example * valid[start:start+micro]).sum()
        loss = weighted_sum * (coef * world / count)
        from verl.utils.numerics import finite_loss_all_ranks
        if not finite_loss_all_ranks(loss):
            if os.environ.get('RESKILL_COORDINATED_FINITE_CHECK') != '1':
                raise FloatingPointError('Nonfinite skill refinement SFT loss')
            actor._nonfinite_forward_loss = True
            if getattr(actor, '_numerics_context', None) is not None:
                actor._numerics_context['nonfinite_auxiliary'] = {
                    'type':'skill_sft','batch':batch,'log_probs':log_probs.detach()}
            return {'actor/skill_refinement_nonfinite_loss':1.0}
        loss.backward()
        loss_sum += float(weighted_sum.detach())
    return {
        'actor/skill_refinement_sft_loss': loss_sum * world / count,
        'actor/skill_refinement_weighted_sft_loss': loss_sum * world / count * coef,
        'actor/skill_refinement_sft_coef': coef,
        'actor/skill_refinement_sft_samples': count,
        'actor/skill_refinement_sft_tokens': int(torch.as_tensor(payload['attention_mask'])[:, -payload['responses'].shape[1]:].sum()),
    }


class GRPOSkillRefinement:
    def __init__(self, trainer):
        self.trainer = trainer
        self.cfg = trainer.config.algorithm.reskill_grpo
        cfg = trainer.config
        if cfg.algorithm.adv_estimator not in ('grpo', 'gigpo'):
            raise ValueError('Skill refinement extends GRPO only')
        if cfg.algorithm.adv_estimator == 'gigpo' and cfg.algorithm.gigpo.step_advantage_w != 0:
            raise ValueError('Use zero step advantage for the matched GRPO path')
        if cfg.algorithm.filter_groups.enable:
            raise ValueError('Disable dynamic group filtering for ReSkill-GRPO')
        if cfg.actor_rollout_ref.actor.strategy not in ('fsdp', 'fsdp2'):
            raise ValueError('ReSkill-GRPO currently supports native FSDP/FSDP2 actors')
        if cfg.actor_rollout_ref.rollout.mode != 'sync' or cfg.actor_rollout_ref.rollout.name != 'vllm':
            raise ValueError('ReSkill-GRPO requires synchronous native vLLM rollout')
        if int(cfg.actor_rollout_ref.rollout.val_kwargs.n) != 1:
            raise ValueError('ReSkill-GRPO validation requires val_kwargs.n=1')
        for key in ('use_sdl_loss', 'use_sdar_loss', 'use_icsd_loss', 'use_pcsd_loss', 'use_ovcsd_loss'):
            if cfg.actor_rollout_ref.actor.get(key, False):
                raise ValueError('ReSkill-GRPO must run without other distillation losses')
        if cfg.actor_rollout_ref.actor.ulysses_sequence_parallel_size != 1:
            raise ValueError('Skill refinement currently requires actor SP=1')
        if not str(cfg.env.env_name).startswith('alfworld/'):
            raise ValueError('The initial skill refinement integration supports ALFWorld text tasks')
        if int(cfg.env.rollout.n) < 2 and not cfg.trainer.get('val_only', False):
            raise ValueError('GRPO needs at least two trajectories per task')
        if float(cfg.actor_rollout_ref.actor.get('skill_gen_loss_coef', 0)) != 0:
            raise ValueError('Disable the separate SEED skill-generation optimizer update')
        if float(cfg.actor_rollout_ref.actor.get('opd_loss_coef', 0)) != 0:
            raise ValueError('Disable OPD for the GRPO + skill SFT method')
        if not math.isfinite(float(self.cfg.sft_coef)) or float(self.cfg.sft_coef) < 0:
            raise ValueError('sft_coef must be finite and nonnegative')
        for key in ('skill_max_new_tokens', 'rewrite_max_new_tokens', 'rewrite_max_prompt_length',
                    'max_skill_chars', 'sft_micro_batch_size', 'trace_chars_per_step'):
            if int(self.cfg[key]) < 1:
                raise ValueError(f'{key} must be positive')
        max_model_len = (cfg.actor_rollout_ref.rollout.get('max_model_len') or
                         int(cfg.data.max_prompt_length) + int(cfg.data.max_response_length))
        if max_model_len and max(
            int(self.cfg.rewrite_max_prompt_length) + int(self.cfg.rewrite_max_new_tokens),
            int(cfg.data.max_prompt_length) + int(self.cfg.skill_max_new_tokens),
        ) > int(max_model_len):
            raise ValueError('Skill prompts and responses exceed rollout.max_model_len')
        self.root = Path(cfg.trainer.default_local_dir).resolve() / 'skill_refinement'
        self.pending_payload = None
        self.pending_quality_payload = None
        self.pending_metrics = {}
        self.quality_cfg = self.cfg.get('rewrite_quality', {})
        if self.quality_cfg.get('enable', False):
            for key in ('loss_coef','beta'):
                if not math.isfinite(float(self.quality_cfg[key])) or float(self.quality_cfg[key]) < 0:
                    raise ValueError('Rewrite quality coefficients must be finite and nonnegative')
            if int(self.quality_cfg.interval) < 1 or int(self.quality_cfg.micro_batch_size) < 1:
                raise ValueError('Rewrite quality interval and microbatch must be positive')
            if not 0 <= float(self.quality_cfg.old_success_threshold) <= 1:
                raise ValueError('Invalid old-skill success threshold')
        self.template_kwargs = dict(cfg.data.get('apply_chat_template_kwargs', {}))
        trainer.traj_collector.skill_refinement = self

    def quality_step(self):
        return (self.quality_cfg.get('enable', False)
                and int(self.trainer.global_steps) > 0
                and int(self.trainer.global_steps) % int(self.quality_cfg.interval) == 0)

    def _prompt_ids(self, prompt):
        return self.trainer.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=True, return_dict=False, add_generation_prompt=True,
            **self.template_kwargs)

    def _generate(self, prompts, actor, max_tokens, max_prompt_length, greedy=False):
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        if any(len(self._prompt_ids(prompt)) > max_prompt_length for prompt in prompts):
            raise ValueError('Skill prompt exceeds budget; refusing to truncate task or instructions')
        t = self.trainer
        # Use the native tokenizer/postprocessor without changing the collector's
        # action context length. A separate lightweight collector shares its model
        # tokenizer, processor and all native Qwen position-ID handling.
        from copy import deepcopy
        import numpy as np
        from verl import DataProto
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
        cfg = deepcopy(t.config)
        cfg.data.max_prompt_length = max_prompt_length
        collector = TrajectoryCollector(cfg, t.tokenizer, t.processor)
        base = DataProto.from_dict(
            tensors={'input_ids': torch.zeros((len(prompts), 1), dtype=torch.long)},
            non_tensors={'raw_prompt': np.array([[] for _ in prompts], dtype=object),
                         'data_source': np.array(['text'] * len(prompts), dtype=object)},
            meta_info={'eos_token_id': t.tokenizer.eos_token_id,
                       'pad_token_id': t.tokenizer.pad_token_id, 'recompute_log_prob': False,
                       'do_sample': True, 'validate': False, 'skill_refinement_generation': True,
                       'sampling_params': {'temperature': 0.0 if greedy else
                                           float(t.config.actor_rollout_ref.rollout.temperature),
                                           'max_tokens': max_tokens, 'n': 1}},
        )
        seed = collector.preprocess_batch(base, {'text': prompts})
        keys = [k for k in ('raw_prompt_ids', 'raw_prompt') if k in seed.non_tensor_batch]
        gen = seed.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'], non_tensor_batch_keys=keys)
        gen.meta_info = dict(seed.meta_info)  # DataProto.pop does not preserve metadata by default.
        padded, extra = pad_dataproto_to_divisor(gen, actor.world_size)
        generated = unpad_dataproto(actor.generate_sequences(padded), extra)
        # Keep exact generated target tokens (excluding stop/padding), for rewrite supervision.
        stops = {t.tokenizer.pad_token_id,t.tokenizer.eos_token_id}
        self.last_generation_token_ids = []
        for row in generated.batch['responses'].detach().cpu().tolist():
            end = next((i for i,value in enumerate(row) if value in stops),len(row))
            self.last_generation_token_ids.append(row[:end])
        return t.tokenizer.batch_decode(generated.batch['responses'], skip_special_tokens=True)

    def begin_rollout(self, obs, envs, actor, phase):
        started = time.perf_counter()
        n = int(self.trainer.config.env.rollout.n) if phase == 'train' else 1
        tasks = list(envs.tasks)
        if len(tasks) != len(obs['text']) or len(tasks) % n:
            raise ValueError('Task/group alignment is invalid')
        for start in range(0, len(tasks), n):
            if any(task != tasks[start] for task in tasks[start:start+n]):
                raise ValueError('Every GRPO group must contain the same task')
            anchors = obs.get('anchor')
            if anchors is not None and any(str(value) != str(anchors[start]) for value in anchors[start:start+n]):
                raise ValueError('A GRPO group contains different initial environments')
        group_tasks = tasks[::n]
        texts = self._generate([skill_prompt(task) for task in group_tasks], actor,
                               int(self.cfg.skill_max_new_tokens),
                               int(self.trainer.config.data.max_prompt_length), greedy=phase != 'train')
        skills = []
        for text in texts:
            try:
                skills.append(parse_skill(text, int(self.cfg.max_skill_chars)))
            except ValueError:
                skills.append('')  # Malformed initial skill: execute without guidance; retain audit text.
        if phase == 'train':
            self.pending_payload = None
            self.pending_quality_payload = None
            self.pending_metrics = {}
            self.pending_metrics['skill_refinement/initial_generation_seconds'] = time.perf_counter() - started
        return {'tasks': group_tasks, 'skills': skills, 'raw_skills': texts, 'group_size': n,
                'episode_skills': [skill for skill in skills for _ in range(n)]}

    @staticmethod
    def condition_observations(obs, context):
        result = dict(obs)
        for key in ('text', 'text_base'):
            if obs.get(key) is not None:
                result[key] = [condition_observation(text, skill)
                               if skill else text for text, skill in zip(obs[key], context['episode_skills'])]
        return result

    def _rewrite_prompt(self, task, skill, traces):
        instruction = (
            'Analyze the completed trajectories and their environment feedback. Rewrite the initial skill '
            'to improve future execution of this task. Use successful behavior and correct observed mistakes. '
            'Only state procedures usable BEFORE execution; do not invent observations, copy an answer, '
            'or output a replay of one trajectory. Do not output analysis, reasoning, or thinking tags. '
            'Output only the revised skill enclosed in <skill>...</skill>, with no text outside these tags.\n'
        )
        # Preserve every trajectory's outcome. Reduce step excerpts uniformly across
        # the group to fit the prompt, and record full traces separately on disk.
        cap = int(self.cfg.trace_chars_per_step)
        step_limit = max((len(trace['steps']) for trace in traces), default=0)
        while True:
            compact = []
            for trace in traces:
                steps = trace['steps']
                if step_limit and len(steps) > step_limit:
                    indices = sorted({round(i * (len(steps)-1) / max(step_limit-1, 1)) for i in range(step_limit)})
                    steps = [steps[i] for i in indices]
                elif not step_limit:
                    steps = []
                compact.append({**{k: v for k, v in trace.items() if k != 'steps'},
                                'steps_total': len(trace['steps']),
                                'steps': [{**s, 'observation': s['observation'][:cap],
                                           'response': s['response'][:cap],
                                           'next_observation': s.get('next_observation', '')[:cap]} for s in steps],
                                'excerpts_truncated': len(steps) < len(trace['steps']) or
                                any(max(len(s['observation']), len(s['response']),
                                        len(s.get('next_observation', ''))) > cap for s in steps)})
            prompt = instruction + json.dumps({'task': task, 'initial_skill': skill,
                                               'trajectories': compact}, ensure_ascii=False)
            if len(self._prompt_ids(prompt)) <= int(self.cfg.rewrite_max_prompt_length):
                return prompt
            if step_limit > 2:
                step_limit = max(2, step_limit // 2)
            elif cap > 64:
                cap = max(64, cap // 2)
            else:
                # Outcomes without observations/actions cannot support a rewrite.
                return None

    def save_validation_skills(self, context, episodes, rewards):
        """Save all actually used validation skills, including malformed raw output."""
        if not context.get('tasks'):
            return
        if len(rewards) != len(context['episode_skills']):
            raise ValueError('Validation skills and episode outcomes are misaligned')
        directory = self.root.parent / 'validation_skills'
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f'step-{int(self.trainer.global_steps)}.jsonl'
        gamefiles = context.get('initial_gamefiles', [None] * len(rewards))
        n = context['group_size']
        with destination.open('a') as handle:
            for group, task in enumerate(context['tasks']):
                for replica in range(n):
                    index = group * n + replica
                    record = {'step':int(self.trainer.global_steps), 'phase':'validation',
                              'task':task, 'gamefile':gamefiles[index], 'replica':replica,
                              'raw_skill':context['raw_skills'][group],
                              'skill':context['skills'][group], 'skill_valid':bool(context['skills'][group]),
                              'episode_reward':float(rewards[index]), 'success':bool(rewards[index] > 0),
                              'episode_steps':sum(bool(s['active_masks']) for s in episodes[index])}
                    handle.write(json.dumps(record,ensure_ascii=False) + '\n')
        print(f'Saved {len(rewards)} validation skills to {destination}',flush=True)

    def finish_rollout(self, context, episodes, rewards, actor, phase):
        if phase != 'train':
            self.save_validation_skills(context, episodes, rewards)
            return  # Validation is an audit only; no outcome-conditioned rewrite or gradients.
        started = time.perf_counter()
        t = self.trainer
        records, prompts, eligible = [], [], []
        n = context['group_size']
        for group, task in enumerate(context['tasks']):
            traces = []
            for index in range(group*n, (group+1)*n):
                steps = []
                for step in episodes[index]:
                    if not bool(step['active_masks']):
                        continue
                    steps.append({'step': len(steps), 'observation': str(step.get('anchor_obs', step.get('obs_text', ''))),
                                  'response': t.tokenizer.decode(step['responses'], skip_special_tokens=True),
                                  'next_observation': str(step.get('skill_refinement_next_obs', '')),
                                  'reward': float(step['rewards']),
                                  'valid_action': bool(step.get('is_action_valid', True))})
                traces.append({'reward': float(rewards[index]), 'invalid_actions': sum(not s['valid_action'] for s in steps),
                               'steps': steps})
            record = {'group_index':group, 'task': task, 'initial_skill': context['skills'][group],
                      'raw_initial_skill': context['raw_skills'][group], 'trajectories': traces,
                      'accepted': False, 'reason': 'pending_rewrite'}
            records.append(record)
            prompt = self._rewrite_prompt(task, record['initial_skill'], traces)
            if prompt is None:
                record['reason'] = 'rewrite_context_budget'
                continue
            record['rewrite_prompt'] = prompt
            prompts.append(prompt)
            eligible.append(record)
        pairs = []
        if prompts:
            rewrites = self._generate(prompts, actor, int(self.cfg.rewrite_max_new_tokens),
                                      int(self.cfg.rewrite_max_prompt_length), greedy=True)
            token_rows = getattr(self,'last_generation_token_ids',None)
            if token_rows is None or len(token_rows) != len(rewrites):
                token_rows = [t.tokenizer.encode(text,add_special_tokens=False) for text in rewrites]
            for record, text, token_ids in zip(eligible, rewrites, token_rows):
                record['raw_rewrite'] = text
                record['rewrite_token_ids'] = token_ids
                try:
                    skill = parse_skill(text, int(self.cfg.max_skill_chars))
                except ValueError:
                    record['reason'] = 'invalid_rewrite'
                    continue
                record['refined_skill'] = skill
                if self.cfg.require_success and not any(trace['reward'] > 0 for trace in record['trajectories']):
                    record['reason'] = 'no_successful_evidence'
                    continue
                pair = (record['task'], skill)
                if sft_payload(t.tokenizer, [pair], int(t.config.data.max_prompt_length),
                               int(self.cfg.rewrite_max_new_tokens), self.template_kwargs) is None:
                    record['reason'] = 'sft_context_budget'
                    continue
                pairs.append(pair)
                record.update(accepted=True, reason='format_and_evidence_filter')
        quality_started = time.perf_counter()
        self.pending_metrics['rewrite_quality/active_step'] = int(self.quality_step())
        if self.quality_step():
            from verl.trainer.ppo.rewrite_quality import evaluate_rewritten_skills, rewrite_quality_payload
            evaluate_rewritten_skills(self,records,context,actor)
            evaluated = [record for record in records if 'rewrite_quality' in record]
            masked = 0
            for record in evaluated:
                if not record['rewrite_quality']['q']:
                    masked += int(record['accepted'])
                    record['pre_quality_regular_reason'] = record['reason']
                    record.update(accepted=False,reason='rewrite_quality_rejected')
            # Rebuild, rather than retain earlier positive pairs: no invalid target can receive both losses.
            pairs = [(record['task'],record['refined_skill']) for record in records if record['accepted']]
            quality_payload = rewrite_quality_payload(t.tokenizer,evaluated,int(self.cfg.rewrite_max_prompt_length),
                                                       int(self.cfg.rewrite_max_new_tokens),self.template_kwargs)
            if quality_payload is not None:
                quality_payload.update(loss_coef=float(self.quality_cfg.loss_coef),beta=float(self.quality_cfg.beta),
                                       micro_batch_size=int(self.quality_cfg.micro_batch_size))
            self.pending_quality_payload = quality_payload
            if quality_payload is None:
                self.pending_metrics.update({'actor/rewrite_quality_loss':0.0,
                                             'actor/rewrite_quality_weighted_loss':0.0})
            def average(key):
                return sum(record['rewrite_quality'][key] for record in evaluated)/max(len(evaluated),1)
            self.pending_metrics.update({
                'rewrite_quality/evaluated_groups':len(evaluated),
                'rewrite_quality/positive_groups':sum(record['rewrite_quality']['q'] for record in evaluated),
                'rewrite_quality/negative_groups':sum(not record['rewrite_quality']['q'] for record in evaluated),
                'rewrite_quality/masked_regular_skill_sft':masked,
                'rewrite_quality/extra_trajectories':len(evaluated)*n,
                'rewrite_quality/old_success_rate':average('p_old'),
                'rewrite_quality/new_success_rate':average('p_new'),
                'rewrite_quality/edit_distance':average('edit_distance'),
                'rewrite_quality/execution_seconds':time.perf_counter()-quality_started,
            })
        payload = sft_payload(t.tokenizer, pairs, int(t.config.data.max_prompt_length),
                              int(self.cfg.rewrite_max_new_tokens), self.template_kwargs)
        if payload is not None:
            payload.update(loss_coef=float(self.cfg.sft_coef), micro_batch_size=int(self.cfg.sft_micro_batch_size))
        self.pending_payload = payload
        self.pending_metrics.update(refinement_metrics(records, t.tokenizer))
        self.pending_metrics['skill_refinement/rewrite_and_filter_seconds'] = time.perf_counter() - started
        self.root.mkdir(parents=True, exist_ok=True)
        # Append to preserve previous attempts after a checkpoint resume.
        with (self.root / f'step-{t.global_steps}.jsonl').open('a') as handle:
            handle.write(json.dumps({'step': int(t.global_steps), 'groups': records,
                                     'metrics': self.pending_metrics}, ensure_ascii=False) + '\n')

    def prepare_actor_batch(self, batch, metrics):
        metrics.update(self.pending_metrics)
        if getattr(batch, 'batch', None) is not None and 'advantages' in batch.batch:
            advantages = batch.batch['advantages']
            mask = batch.batch.get('response_mask', batch.batch['attention_mask'][:, -advantages.shape[1]:]).bool()
            metrics['skill_refinement/grpo_nonzero_token_advantage_rate'] = float(
                ((advantages != 0) & mask).sum() / mask.sum().clamp_min(1))
        if self.pending_payload is not None and float(self.cfg.sft_coef) > 0:
            batch.meta_info['skill_refinement'] = self.pending_payload
        if self.pending_quality_payload is not None and float(self.quality_cfg.loss_coef) > 0:
            batch.meta_info['rewrite_quality'] = self.pending_quality_payload
        self.pending_payload = None
        self.pending_quality_payload = None
        return batch
