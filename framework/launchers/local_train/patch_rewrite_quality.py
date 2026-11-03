from pathlib import Path
root=Path('/workspace/fudan/donghaosen/Self-Distillation-AgentRL')
def patch(name,old,new,count=1):
 p=root/name;s=p.read_text();assert s.count(old)==count,(name,s.count(old),old[:60]);p.write_text(s.replace(old,new))
patch('verl/trainer/config/ppo_trainer.yaml','  reskill_grpo:\n    enable: False\n','''  reskill_grpo:
    enable: False
    rewrite_quality:
      enable: False
      interval: 5
      old_success_threshold: 0.75
      loss_coef: 0.1
      beta: 1.0
      micro_batch_size: 1
''')
patch('agent_system/environments/env_package/alfworld/envs.py','def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):','def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}, fixed_game_files=None, worker_seeds=None):')
patch('agent_system/environments/env_package/alfworld/envs.py','        self.prev_admissible_commands = [None for _ in range(self.num_processes)]\n','        self.prev_admissible_commands = [None for _ in range(self.num_processes)]\n')
patch('agent_system/environments/env_package/alfworld/envs.py','        # Create Ray remote actors instead of processes\n','''        if fixed_game_files is not None and len(fixed_game_files) != self.num_processes:
            raise ValueError("Fixed-game count must match physical environment workers")
        if worker_seeds is not None and len(worker_seeds) != self.num_processes:
            raise ValueError("Seed count must match physical environment workers")
        # Create Ray remote actors instead of processes
''')
patch('agent_system/environments/env_package/alfworld/envs.py','            fixed_game_file = strict_game_files[i] if strict_game_files is not None else None\n','''            fixed_game_file = (fixed_game_files[i] if fixed_game_files is not None else
                               strict_game_files[i] if strict_game_files is not None else None)
''')
patch('agent_system/environments/env_package/alfworld/envs.py','                seed + (i // self.group_n),\n','                worker_seeds[i] if worker_seeds is not None else seed + (i // self.group_n),\n')
# Capture initial task identities before any actions; never mutate the old env state for replay.
patch('agent_system/multi_turn_rollout/rollout_loop.py','            refinement_context = refinement.begin_rollout(obs, envs, actor_rollout_wg, phase)\n','''            refinement_context = refinement.begin_rollout(obs, envs, actor_rollout_wg, phase)
            refinement_context.update(initial_gamefiles=current_gamefiles.tolist(),
                                      initial_anchors=list(obs['anchor']),
                                      generation_meta_info=dict(gen_batch.meta_info))
            if phase == 'train' and refinement.quality_step():
                refinement_context['pair_keys'] = [
                    f"{int(self.config.env.seed)}:{int(refinement.trainer.global_steps)}:{i//n}:{i%n}:{game}"
                    for i,game in enumerate(current_gamefiles)]
        rewrite_pair_keys = getattr(self, 'rewrite_pair_keys', None)
        if refinement is not None and phase == 'train' and refinement.quality_step():
            rewrite_pair_keys = refinement_context['pair_keys']
''')
patch('agent_system/multi_turn_rollout/rollout_loop.py','            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]\n','''            if rewrite_pair_keys is not None:
                from verl.trainer.ppo.rewrite_quality import paired_execution_seed
                batch.non_tensor_batch['rewrite_request_seed'] = np.asarray(
                    [paired_execution_seed(key,_step) for key in rewrite_pair_keys],dtype=np.int64)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
''')
patch('agent_system/multi_turn_rollout/rollout_loop.py','            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]\n','''            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if rewrite_pair_keys is not None:
                non_tensor_batch_keys_to_pop.append('rewrite_request_seed')
''')
patch('agent_system/multi_turn_rollout/rollout_loop.py','            batch_input.meta_info = gen_batch.meta_info\n','''            batch_input.meta_info = gen_batch.meta_info
            if rewrite_pair_keys is not None:
                batch_input.meta_info = {**gen_batch.meta_info,'rewrite_quality_execution':True}
''')
patch('verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py','            if is_validate and seed_mode == "per_task_turn":\n                seeds = non_tensor_batch.get("eval_request_seed")\n','''            quality_execution = prompts.meta_info.get('rewrite_quality_execution', False)
            if (is_validate and seed_mode == "per_task_turn") or quality_execution:
                seeds = non_tensor_batch.get('rewrite_request_seed' if quality_execution else 'eval_request_seed')
''')
patch('verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py','                if not do_sample or effective_kwargs["temperature"] <= 0 or effective_kwargs["n"] != 1:\n','                if effective_kwargs["n"] != 1 or (not quality_execution and (not do_sample or effective_kwargs["temperature"] <= 0)):\n')
patch('verl/workers/actor/dp_actor.py','        refinement_payload = data.meta_info.get("skill_refinement")\n','''        refinement_payload = data.meta_info.get("skill_refinement")
        rewrite_quality_payload = data.meta_info.get("rewrite_quality")
''')
patch('verl/workers/actor/dp_actor.py','                grad_norm = self._optimizer_step()\n','''                if rewrite_quality_payload is not None:
                    from verl.trainer.ppo.rewrite_quality import backward_rewrite_quality
                    append_to_dict(metrics, backward_rewrite_quality(self, rewrite_quality_payload))
                grad_norm = self._optimizer_step()
''')
# Existing task-only skill SFT remains separate from the new rewrite-context loss.
patch('verl/trainer/ppo/reskill_grpo.py','        self.pending_payload = None\n        self.pending_metrics = {}\n','''        self.pending_payload = None
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
''')
patch('verl/trainer/ppo/reskill_grpo.py','    def _prompt_ids(self, prompt):\n','''    def quality_step(self):
        return (self.quality_cfg.get('enable', False)
                and int(self.trainer.global_steps) > 0
                and int(self.trainer.global_steps) % int(self.quality_cfg.interval) == 0)

    def _prompt_ids(self, prompt):
''')
patch('verl/trainer/ppo/reskill_grpo.py',"        return t.tokenizer.batch_decode(generated.batch['responses'], skip_special_tokens=True)\n",'''        # Keep exact generated target tokens (excluding stop/padding), for rewrite supervision.
        stops = {t.tokenizer.pad_token_id,t.tokenizer.eos_token_id}
        self.last_generation_token_ids = []
        for row in generated.batch['responses'].detach().cpu().tolist():
            end = next((i for i,value in enumerate(row) if value in stops),len(row))
            self.last_generation_token_ids.append(row[:end])
        return t.tokenizer.batch_decode(generated.batch['responses'], skip_special_tokens=True)
''')
patch('verl/trainer/ppo/reskill_grpo.py',"            self.pending_payload = None\n            self.pending_metrics = {}\n",'''            self.pending_payload = None
            self.pending_quality_payload = None
            self.pending_metrics = {}
''')
patch('verl/trainer/ppo/reskill_grpo.py',"            record = {'task': task, 'initial_skill': context['skills'][group],\n", "            record = {'group_index':group, 'task': task, 'initial_skill': context['skills'][group],\n")
patch('verl/trainer/ppo/reskill_grpo.py',"            for record, text in zip(eligible, rewrites):\n                record['raw_rewrite'] = text\n",'''            token_rows = getattr(self,'last_generation_token_ids',None)
            if token_rows is None or len(token_rows) != len(rewrites):
                token_rows = [t.tokenizer.encode(text,add_special_tokens=False) for text in rewrites]
            for record, text, token_ids in zip(eligible, rewrites, token_rows):
                record['raw_rewrite'] = text
                record['rewrite_token_ids'] = token_ids
''')
patch('verl/trainer/ppo/reskill_grpo.py','        payload = sft_payload(t.tokenizer, pairs, int(t.config.data.max_prompt_length),\n','''        quality_started = time.perf_counter()
        self.pending_metrics['rewrite_quality/active_step'] = int(self.quality_step())
        if self.quality_step():
            from verl.trainer.ppo.rewrite_quality import evaluate_rewritten_skills, rewrite_quality_payload
            evaluate_rewritten_skills(self,records,context,actor)
            evaluated = [record for record in records if 'rewrite_quality' in record]
            masked = 0
            for record in evaluated:
                if not record['rewrite_quality']['q']:
                    masked += int(record['accepted'])
                    record.update(accepted=False,reason='rewrite_quality_rejected')
            # Rebuild, rather than retain earlier positive pairs: no invalid target can receive both losses.
            pairs = [(record['task'],record['refined_skill']) for record in records if record['accepted']]
            quality_payload = rewrite_quality_payload(t.tokenizer,evaluated,int(self.cfg.rewrite_max_prompt_length),
                                                       int(self.cfg.rewrite_max_new_tokens),self.template_kwargs)
            if quality_payload is not None:
                quality_payload.update(loss_coef=float(self.quality_cfg.loss_coef),beta=float(self.quality_cfg.beta),
                                       micro_batch_size=int(self.quality_cfg.micro_batch_size))
            self.pending_quality_payload = quality_payload
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
''')
patch('verl/trainer/ppo/reskill_grpo.py',"        self.pending_payload = None\n        return batch\n",'''        if self.pending_quality_payload is not None and float(self.quality_cfg.loss_coef) > 0:
            batch.meta_info['rewrite_quality'] = self.pending_quality_payload
        self.pending_payload = None
        self.pending_quality_payload = None
        return batch
''')
print('Native periodic Rewrite Quality Loss hooks installed')
