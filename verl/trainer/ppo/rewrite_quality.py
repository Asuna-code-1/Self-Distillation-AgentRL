"""Periodic, paired execution feedback for ReSkill rewrite quality.

Supervision uses the ACTUAL rewrite prompt, not the task-only skill prompt.
Invalid rewrites receive nonnegative unlikelihood, never negative-weight SFT.
"""
import hashlib
import math
import re
import time
import unicodedata

import torch


def normalized_skill(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    lines = []
    for line in text.splitlines():
        if re.fullmatch(r'\s*```[\w-]*\s*', line):
            continue
        line = re.sub(r'^\s*(?:#{1,6}\s+|[-+*]\s+|\d+[.)]\s+)', '', line)
        lines.append(line)
    text = '\n'.join(lines)
    for marker in ('**', '__', '*', '_', '`'):
        pattern = re.escape(marker) + r'(.+?)' + re.escape(marker)
        text = re.sub(pattern, lambda match: match[1], text, flags=re.DOTALL)
    return re.sub(r'\s+', '', text)


def normalized_edit_distance(old, new):
    a, b = normalized_skill(old), normalized_skill(new)
    if a == b:
        return 0.0
    denominator = max(len(a), len(b), 1)
    # Exact Levenshtein distance with O(min(lengths)) memory.
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            saved = row[j]
            row[j] = min(row[j] + 1, row[j-1] + 1, previous + (ca != cb))
            previous = saved
    return row[-1] / denominator


def quality_signal(old_rewards, new_rewards, old_skill, new_skill):
    if len(old_rewards) != len(new_rewards) or not old_rewards:
        raise ValueError('Quality comparison requires equal, nonempty groups')
    old_count = sum(float(value) > 0 for value in old_rewards)
    new_count = sum(float(value) > 0 for value in new_rewards)
    distance = normalized_edit_distance(old_skill, new_skill)
    return {'p_old': old_count / len(old_rewards), 'p_new': new_count / len(new_rewards),
            'old_successes': old_count, 'new_successes': new_count, 'group_size':len(old_rewards),
            'edit_distance': distance, 'q': int(new_count > old_count and distance > 0)}


def paired_execution_seed(pair_key, turn):
    digest = hashlib.sha256(f'{pair_key}:turn={int(turn)}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % (2**31 - 1)


def unlikelihood_from_log_probs(log_probs):
    # Work in fp32 and keep BOTH torch.where branches finite near p=1.
    values = log_probs.float().clamp(max=-torch.finfo(torch.float32).eps)
    return -torch.where(values < -math.log(2),
                        torch.log1p(-torch.exp(values)),
                        torch.log(-torch.expm1(values)))


def rewrite_quality_payload(tokenizer, records, max_prompt_tokens, max_response_tokens, template_kwargs=None):
    rows = []
    for record in records:
        if 'rewrite_quality' not in record:
            continue
        prompt = tokenizer.apply_chat_template(
            [{'role':'user','content':record['rewrite_prompt']}], tokenize=True, return_dict=False,
            add_generation_prompt=True, **(template_kwargs or {}))
        response = record.get('rewrite_token_ids')
        if response is None:
            response = tokenizer.encode(record['raw_rewrite'], add_special_tokens=False)
        if not response or len(prompt) > max_prompt_tokens or len(response) > max_response_tokens:
            record['rewrite_quality']['gradient_skipped'] = 'context_budget'
            continue
        rows.append((prompt, response, int(record['rewrite_quality']['q'])))
    if not rows:
        return None
    pwidth, rwidth = max(len(p) for p,_,_ in rows), max(len(r) for _,r,_ in rows)
    ids = torch.full((len(rows), pwidth+rwidth), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i,(prompt,response,_) in enumerate(rows):
        ids[i,pwidth-len(prompt):pwidth] = torch.tensor(prompt)
        ids[i,pwidth:pwidth+len(response)] = torch.tensor(response)
        mask[i,pwidth-len(prompt):pwidth+len(response)] = 1
    return {'input_ids':ids,'responses':ids[:,pwidth:].clone(),'attention_mask':mask,
            'position_ids':(mask.cumsum(-1)-1).clamp_min(0),
            'quality_q':torch.tensor([q for _,_,q in rows],dtype=torch.bool)}


def backward_rewrite_quality(actor, payload):
    from verl.trainer.ppo.reskill_grpo import sft_shard_indices
    if actor.ulysses_sequence_parallel_size != 1:
        raise ValueError('Rewrite quality currently requires SP=1')
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world = torch.distributed.get_world_size() if distributed else 1
    count = len(payload['responses'])
    indices, valid = sft_shard_indices(count, rank, world)
    device = next(actor.actor_module.parameters()).device
    tensors = {k:torch.as_tensor(payload[k]).index_select(0,indices).to(device)
               for k in ('input_ids','responses','attention_mask','position_ids')}
    quality = torch.as_tensor(payload['quality_q']).index_select(0,indices).to(device)
    valid = valid.to(device)
    coefficient, beta = float(payload['loss_coef']), float(payload['beta'])
    micro = int(payload['micro_batch_size'])
    if not all(math.isfinite(v) and v >= 0 for v in (coefficient,beta)) or micro < 1:
        raise ValueError('Invalid rewrite-quality loss settings')
    sums = {'loss':0.,'sft_contribution':0.,'ul_contribution':0.}
    for start in range(0,len(indices),micro):
        batch = {k:v[start:start+micro] for k,v in tensors.items()}
        _, log_probs = actor._forward_micro_batch(batch,temperature=1.0,calculate_entropy=False)
        mask = batch['attention_mask'][:,-batch['responses'].shape[1]:].to(torch.float32)
        denominator = mask.sum(-1).clamp_min(1)
        nll = -(log_probs.float()*mask).sum(-1)/denominator
        ul = (unlikelihood_from_log_probs(log_probs)*mask).sum(-1)/denominator
        q, weight = quality[start:start+micro], valid[start:start+micro]
        positive = torch.where(q,nll,torch.zeros_like(nll))*weight
        negative = torch.where(q,torch.zeros_like(ul),ul)*weight
        total = positive + beta*negative
        loss = total.sum()*(coefficient*world/count)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite rewrite quality loss')
        loss.backward()
        sums['loss'] += float(total.detach().sum())
        sums['sft_contribution'] += float(positive.detach().sum())
        sums['ul_contribution'] += float(negative.detach().sum())
    result = {'actor/rewrite_quality_'+k:v*world/count for k,v in sums.items()}
    result.update({'actor/rewrite_quality_weighted_loss':sums['loss']*world/count*coefficient,
                   'actor/rewrite_quality_coef':coefficient,'actor/rewrite_quality_beta':beta,
                   'actor/rewrite_quality_samples':count})
    return result


def evaluate_rewritten_skills(controller, records, context, actor):
    """Replay only eligible fixed tasks; never advance the primary environments."""
    n = context['group_size']
    selected = [r for r in records if 'refined_skill' in r and
                sum(t['reward']>0 for t in r['trajectories'])/n < float(controller.quality_cfg.old_success_threshold)]
    if not selected:
        return 0
    from copy import deepcopy
    from pathlib import Path
    import numpy as np
    from verl import DataProto
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld.envs import AlfworldEnvs
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from verl.utils.gpu_resource_trace import phase

    trainer = controller.trainer
    n = context['group_size']
    gamefiles = context.get('initial_gamefiles')
    if gamefiles is None:
        raise ValueError('Paired quality evaluation requires original task instance IDs')
    groups = [r['group_index'] for r in selected]
    files = [str(gamefiles[group*n+i]) for group in groups for i in range(n)]
    seeds = [int(trainer.config.env.seed)+group for group in groups for _ in range(n)]
    keys = [context['pair_keys'][group*n+i] for group in groups for i in range(n)]
    expected_anchors = [str(context['initial_anchors'][group*n+i]) for group in groups for i in range(n)]
    forced_skills = [record['refined_skill'] for record in selected for _ in range(n)]
    cfg = deepcopy(trainer.config)
    cfg.data.train_batch_size = len(selected)
    config_path = Path(__file__).resolve().parents[3]/'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
    raw = AlfworldEnvs(str(config_path),int(cfg.env.seed),len(selected),n,
                       dict(cfg.env.resources_per_worker),is_train=True,
                       fixed_game_files=files,worker_seeds=seeds)
    manager = AlfWorldEnvironmentManager(raw,alfworld_projection,cfg)
    original_reset = manager.reset
    def checked_reset(kwargs=None):
        obs, infos = original_reset(kwargs)
        actual = [str(info.get('extra.gamefile')) for info in infos]
        if actual != files or list(map(str,obs['anchor'])) != expected_anchors:
            raise ValueError('Quality replay initial states differ from original task instances')
        return obs, infos
    manager.reset = checked_reset

    class FixedSkillCollector(TrajectoryCollector):
        def preprocess_batch(self,gen_batch,obs):
            conditioned = controller.condition_observations(obs,{'episode_skills':forced_skills})
            return super().preprocess_batch(gen_batch,conditioned)

    collector = FixedSkillCollector(cfg,trainer.tokenizer,trainer.processor)
    collector.rewrite_pair_keys = keys
    count = len(forced_skills)
    base = DataProto.from_dict(tensors={'input_ids':torch.zeros((count,1),dtype=torch.long)},
                              non_tensors={'raw_prompt':np.asarray([[]]*count,dtype=object),
                                           'data_source':np.asarray(['text']*count,dtype=object)},
                              meta_info=dict(context['generation_meta_info']))
    try:
        with phase('rewrite_quality_execution'):
            episodes,rewards,_,_,_,_ = collector.vanilla_multi_turn_loop(base,actor,manager)
        for index,record in enumerate(selected):
            new_rewards = [float(v) for v in rewards[index*n:(index+1)*n]]
            result = quality_signal([t['reward'] for t in record['trajectories']],new_rewards,
                                    record['initial_skill'],record['refined_skill'])
            result.update({'new_rewards':new_rewards,'paired_request_seeds':True,
                           'same_policy_before_update':True,'fixed_gamefiles':files[index*n:(index+1)*n],
                           'new_trajectories':[
                               [{'response':trainer.tokenizer.decode(s['responses'],skip_special_tokens=True),
                                 'reward':float(s['rewards']),'valid_action':bool(s.get('is_action_valid',True))}
                                for s in trajectory if bool(s['active_masks'])]
                               for trajectory in episodes[index*n:(index+1)*n]]})
            record['rewrite_quality'] = result
    finally:
        raw.close()
    return len(selected)
