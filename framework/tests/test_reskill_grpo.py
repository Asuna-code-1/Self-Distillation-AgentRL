"""CPU checks for ReSkill-GRPO; no Ray cluster, GPU allocation or model forward."""
import json
from pathlib import Path
from contextlib import contextmanager
from uuid import uuid4
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from verl import DataProto
from verl.trainer.ppo.reskill_grpo import (
    GRPOSkillRefinement, backward_refinement_sft, condition_observation,
    parse_skill, sft_payload, sft_shard_indices, skill_prompt,
)


@contextmanager
def artifact_directory():
    # Keep audit artifacts: never remove files on this shared machine.
    path = Path(__file__).resolve().parents[1] / 'local_assets/tests' / uuid4().hex
    path.mkdir(parents=True)
    yield str(path)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(c) + 2 for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(int(i) - 2) for i in ids if int(i) > 1)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row) for row in rows]

    def apply_chat_template(self, chat, tokenize=True, **kwargs):
        text = '<user>' + chat[0]['content'] + '</user><assistant>'
        return self.encode(text) if tokenize else text

    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.encode(text)])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


def config():
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'verl/trainer/config/ppo_trainer.yaml')
    cfg.algorithm.adv_estimator = 'grpo'
    cfg.algorithm.reskill_grpo.enable = True
    cfg.actor_rollout_ref.rollout.max_model_len = 9216
    cfg.env.env_name = 'alfworld/AlfredTWEnv'
    cfg.env.rollout.n = 2
    cfg.data.max_prompt_length = 2048
    cfg.data.max_response_length = 512
    cfg.data.return_raw_chat = True
    cfg.env.max_steps = 2
    return cfg


def trainer(cfg=None):
    return SimpleNamespace(config=cfg or config(), tokenizer=Tokenizer(), processor=None,
                           traj_collector=SimpleNamespace(), global_steps=1)


class FakeActor:
    world_size = 4

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.requests = []

    def generate_sequences(self, batch):
        self.requests.append(batch)
        special = batch.meta_info.get('skill_refinement_generation', False)
        text = '<skill>Check the room before moving.</skill>' if special else 'look'
        rows = [self.tokenizer.encode(text) + [1] for _ in range(len(batch))]
        responses = torch.tensor(rows)
        inputs = torch.cat((batch.batch['input_ids'], responses), -1)
        mask = torch.cat((batch.batch['attention_mask'], torch.ones_like(responses)), -1)
        return DataProto.from_dict(tensors={
            'prompts': batch.batch['input_ids'], 'responses': responses,
            'input_ids': inputs, 'attention_mask': mask,
            'position_ids': (mask.cumsum(-1) - 1).clamp_min(0)})


class FakeEnvs:
    def __init__(self, count=2):
        self.count = count
        self.tasks = ['put the mug on the table'] * count
        self.turn = 0

    def obs(self):
        return {'text': ['Your task is to: put the mug on the table\nNow it\'s your turn to act.'] * self.count,
                'anchor': ['room'] * self.count, 'image': None}

    def reset(self, **kwargs):
        self.turn = 0
        return self.obs(), [{'extra.gamefile': 'same-game'} for _ in self.tasks]

    def step(self, actions):
        self.turn += 1
        return (self.obs(), np.array([1.] + [0.] * (self.count - 1)),
                np.ones(self.count, dtype=bool),
                [{'is_action_valid': True, 'extra.gamefile': 'same-game'} for _ in self.tasks])

    def success_evaluator(self, **kwargs):
        return {'success_rate': kwargs['episode_rewards'] > 0}


class ReSkillTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_strict_parse(self):
        self.assertEqual(parse_skill(' <skill>do checks</skill> ', 20), 'do checks')
        for text in ('plain text', '<skill></skill>', '<skill>x</skill>extra',
                     '<skill><skill>x</skill></skill>', '<skill>x</skill><skill>y</skill>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_skill(text, 20)
        with self.assertRaises(ValueError):
            parse_skill('<skill>long</skill>', 2)

    def test_prompt_placeholders_cannot_become_positive_skill_targets(self):
        self.assertEqual(parse_skill('<skill>先检查水槽，再清洗杯子。</skill>', 100), '先检查水槽，再清洗杯子。')
        c = GRPOSkillRefinement(trainer())
        rewrite_prompt = c._rewrite_prompt('put a mug on the table', 'find the mug', [])
        self.assertNotIn('<skill>your skill</skill>', skill_prompt('task'))
        self.assertNotIn('<skill>revised skill</skill>', rewrite_prompt)
        for content in ('revised skill', 'REVISED  SKILL.', 'your skill',
                        'procedural skill', 'rewritten skill', '...'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_skill(f'<skill>{content}</skill>', 100)
        with artifact_directory() as directory:
            c.root = Path(directory)
            context = {'tasks':['task'], 'skills':['old plan'],
                       'raw_skills':['<skill>old plan</skill>'], 'group_size':2}
            with patch.object(c, '_generate', return_value=['<skill>revised skill</skill>']):
                c.finish_rollout(context, [[], []], [10, 10], None, 'train')
            self.assertIsNone(c.pending_payload)
            self.assertEqual(c.pending_metrics['skill_refinement/rejected_invalid_rewrite'], 1)

    def test_validation_saves_used_skills_and_outcomes_without_rewriting(self):
        c=GRPOSkillRefinement(trainer());c.trainer.global_steps=5
        context={'tasks':['task A','task B'],'skills':['check the sink',''],
                 'raw_skills':['<skill>check the sink</skill>','malformed output'],
                 'group_size':1,'episode_skills':['check the sink',''],
                 'initial_gamefiles':['game-A','game-B']}
        episodes=[[{'active_masks':True}],[{'active_masks':True},{'active_masks':False}]]
        with artifact_directory() as directory:
            c.root=Path(directory)/'skill_refinement'
            with patch.object(c,'_generate',side_effect=AssertionError('validation must not rewrite')):
                c.finish_rollout(context,episodes,[10.,0.],None,'validate')
                c.finish_rollout(context,episodes,[10.,0.],None,'validate')
            path=Path(directory)/'validation_skills/step-5.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertEqual(rows[0]['skill'],'check the sink')
            self.assertEqual(rows[0]['gamefile'],'game-A')
            self.assertTrue(rows[0]['success'])
            self.assertFalse(rows[1]['skill_valid'])
            self.assertEqual(rows[1]['raw_skill'],'malformed output')
            self.assertIsNone(c.pending_payload)

    def test_prompt_conditioning_preserves_action_instructions(self):
        original = "Your task is to: clean mug\nNow it's your turn to act."
        output = condition_observation(original, 'wash first')
        self.assertLess(output.index('clean mug'), output.index('Episode-Level Skill'))
        self.assertLess(output.index('wash first'), output.index("Now it's your turn"))
        self.assertIn('wash first', condition_observation('observation only', 'wash first'))

    def test_sft_uses_task_only_prompt_and_masks_padding(self):
        payload = sft_payload(self.tokenizer, [('short', 'plan'), ('longer task', 'longer plan')], 1024, 128)
        width = payload['responses'].shape[1]
        for i, task in enumerate(('short', 'longer task')):
            prompt = self.tokenizer.decode(payload['input_ids'][i, :-width])
            self.assertIn(skill_prompt(task), prompt)
            self.assertNotIn('trajectories', prompt)
            mask = payload['attention_mask'][i, -width:]
            target = payload['responses'][i][mask.bool()]
            self.assertEqual(int(target[-1]), self.tokenizer.eos_token_id)
            self.assertTrue(torch.all(payload['responses'][i][~mask.bool()] == 0))
            positions = payload['position_ids'][i][payload['attention_mask'][i].bool()]
            self.assertTrue(torch.equal(positions, torch.arange(len(positions))))

    def test_over_budget_target_is_rejected_not_truncated(self):
        self.assertIsNone(sft_payload(self.tokenizer, [('task', 'plan')], 1, 128))
        self.assertIsNone(sft_payload(self.tokenizer, [('task', 'plan')], 1024, 1))

    def test_all_ranks_have_equal_forward_counts_and_padding_zero_weight(self):
        for count in (1, 3, 4, 5, 9):
            shards = [sft_shard_indices(count, rank, 4) for rank in range(4)]
            self.assertEqual(len({len(indices) for indices, valid in shards}), 1)
            seen = [int(i) for indices, valid in shards for i, weight in zip(indices, valid) if weight]
            self.assertEqual(sorted(seen), list(range(count)))

    def test_gradient_matches_global_mean_for_uneven_distributed_shards(self):
        payload = sft_payload(self.tokenizer, [('task', 'p' * n) for n in (1, 2, 3, 4, 5)], 1024, 128)
        payload.update(loss_coef=0.1, micro_batch_size=2)
        gradients = []
        for rank in range(4):
            module = torch.nn.Linear(1, 1, bias=False)
            module.weight.data.fill_(0.2)
            # Seed a GRPO gradient, proving the helper does not clear it.
            module.weight.grad = torch.full_like(module.weight, 3.)
            actor = SimpleNamespace(actor_module=module, ulysses_sequence_parallel_size=1)
            def forward(batch, temperature, calculate_entropy):
                self.assertEqual(temperature, 1.)
                return None, module.weight.flatten()[0] * batch['responses'].float()
            actor._forward_micro_batch = forward
            with patch('torch.distributed.is_initialized', return_value=True), \
                 patch('torch.distributed.get_rank', return_value=rank), \
                 patch('torch.distributed.get_world_size', return_value=4):
                backward_refinement_sft(actor, payload)
            gradients.append(module.weight.grad.item())
        mask = payload['attention_mask'][:, -payload['responses'].shape[1]:]
        expected = 3 - 0.1 * ((payload['responses'] * mask).sum(-1) / mask.sum(-1)).mean().item()
        self.assertAlmostEqual(sum(gradients) / 4, expected, places=5)

    def test_native_actor_combines_grpo_and_sft_in_one_optimizer_step(self):
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        cfg = config().actor_rollout_ref.actor
        cfg.use_kl_loss = False
        cfg.entropy_coeff = 0.
        cfg.ppo_mini_batch_size = 2
        cfg.ppo_micro_batch_size_per_gpu = 1
        cfg.ppo_epochs = 1
        cfg.use_dynamic_bsz = False
        for enable in (False, True):
            module = torch.nn.Linear(1, 1, bias=False)
            module.weight.data.zero_()
            actor = DataParallelPPOActor(cfg, module, torch.optim.Adam(module.parameters(), lr=.01))
            steps = []
            def forward(micro_batch, temperature, calculate_entropy):
                shape = micro_batch['responses'].shape
                return None, module.weight.flatten()[0].expand(shape)
            actor._forward_micro_batch = forward
            def step():
                steps.append(module.weight.grad.detach().clone())
                actor.actor_optimizer.step()
                return steps[-1].norm()
            actor._optimizer_step = step
            data = DataProto.from_dict(tensors={
                'responses': torch.ones((2, 3), dtype=torch.long),
                'input_ids': torch.ones((2, 5), dtype=torch.long),
                'attention_mask': torch.ones((2, 5), dtype=torch.long),
                'position_ids': torch.arange(5).expand(2, -1),
                'old_log_probs': torch.zeros((2, 3)),
                'advantages': torch.tensor([[1., 1., 1.], [2., 2., 2.]])},
                meta_info={'temperature': 1.})
            if enable:
                payload = sft_payload(self.tokenizer, [('task', 'plan')], 1024, 128)
                payload.update(loss_coef=.1, micro_batch_size=1)
                data.meta_info['skill_refinement'] = payload
            with patch('verl.workers.actor.dp_actor.get_torch_device',
                       return_value=SimpleNamespace(current_device=lambda: 'cpu')), \
                 patch('verl.utils.debug.performance._get_current_mem_info', return_value=(0, 0, 0, 0)):
                # Stub memory telemetry only; execute the real native update loop.
                DataParallelPPOActor.update_policy(actor, data)
            self.assertEqual(len(steps), 1)
            self.assertAlmostEqual(steps[0].item(), -1.6 if enable else -1.5, places=5)
            self.assertEqual(actor.actor_optimizer.state[module.weight]['step'].item(), 1.)

    def test_evaluation_has_no_rewrite_or_sft(self):
        controller = GRPOSkillRefinement(trainer())
        controller.pending_payload = {'sentinel': 1}
        with patch.object(controller, '_generate', side_effect=AssertionError('eval feedback used')):
            controller.finish_rollout({}, [], [], None, 'validate')
        self.assertEqual(controller.pending_payload, {'sentinel': 1})

    def test_invalid_group_alignment_fails_before_generation(self):
        controller = GRPOSkillRefinement(trainer())
        for tasks, anchors in ((['a', 'b'], ['x', 'x']), (['a', 'a'], ['x', 'y'])):
            with self.assertRaises(ValueError):
                controller.begin_rollout({'text': ['a', 'a'], 'anchor': anchors},
                                         SimpleNamespace(tasks=tasks), None, 'train')

    def test_context_and_other_loss_guards(self):
        for change in ('context', 'group', 'filter', 'distill', 'sp', 'backend'):
            cfg = config()
            if change == 'context': cfg.actor_rollout_ref.rollout.max_model_len = None
            if change == 'group': cfg.env.rollout.n = 1
            if change == 'filter': cfg.algorithm.filter_groups.enable = True
            if change == 'distill': cfg.actor_rollout_ref.actor.use_sdl_loss = True
            if change == 'sp': cfg.actor_rollout_ref.actor.ulysses_sequence_parallel_size = 2
            if change == 'backend': cfg.actor_rollout_ref.actor.strategy = 'megatron'
            with self.subTest(change=change), self.assertRaises(ValueError):
                GRPOSkillRefinement(trainer(cfg))

    def test_native_prompt_generation_preserves_metadata_budget_and_unpadding(self):
        t = trainer()
        controller = GRPOSkillRefinement(t)
        actor = FakeActor(t.tokenizer)
        outputs = controller._generate(['short prompt', 'second prompt'], actor, 384, 8192, greedy=True)
        self.assertEqual(len(outputs), 2)
        request = actor.requests[0]
        self.assertEqual(len(request), 4)
        self.assertEqual(request.batch['input_ids'].shape[1], 8192)
        self.assertEqual(request.meta_info['sampling_params'], {'temperature': 0., 'n': 1, 'max_tokens': 384})
        self.assertEqual(t.config.data.max_prompt_length, 2048)

    def test_native_rollout_training_and_validation(self):
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
        t = trainer()
        t.traj_collector = TrajectoryCollector(t.config, t.tokenizer)
        controller = GRPOSkillRefinement(t)
        with artifact_directory() as temporary:
            controller.root = Path(temporary)
            actor = FakeActor(t.tokenizer)
            def batch(count, validate):
                return DataProto.from_dict(
                    tensors={'input_ids': torch.zeros((count, 1), dtype=torch.long)},
                    non_tensors={'raw_prompt': np.asarray([[]] * count, dtype=object),
                                 'data_source': np.asarray(['text'] * count, dtype=object)},
                    meta_info={'validate': validate, 'eos_token_id': 1, 'pad_token_id': 0})
            output = t.traj_collector.multi_turn_loop(batch(1, False), actor, FakeEnvs())
            metrics = {}
            controller.prepare_actor_batch(output, metrics)
            self.assertEqual(metrics['skill_refinement/accepted'], 1)
            self.assertEqual(output.meta_info['skill_refinement']['loss_coef'], .1)
            self.assertIsNone(controller.pending_payload)
            audit = json.loads((controller.root / 'step-1.jsonl').read_text())
            self.assertEqual(len(audit['groups'][0]['trajectories']), 2)
            self.assertTrue(audit['groups'][0]['trajectories'][0]['steps'][0]['next_observation'])
            self.assertEqual(len(actor.requests), 3)  # skill, action, rewrite
            previous_audit = (controller.root / 'step-1.jsonl').read_text()
            actor.requests = []
            output = t.traj_collector.multi_turn_loop(batch(1, True), actor, FakeEnvs(1), is_train=False)
            self.assertEqual(len(actor.requests), 2)  # skill and action; no rewrite
            self.assertEqual(actor.requests[0].meta_info['sampling_params']['temperature'], 0.)
            self.assertNotIn('skill_refinement', output.meta_info)
            self.assertEqual((controller.root / 'step-1.jsonl').read_text(), previous_audit)

    def test_no_success_rejects_supervision(self):
        t = trainer()
        controller = GRPOSkillRefinement(t)
        with artifact_directory() as temporary:
            controller.root = Path(temporary)
            context = {'group_size': 2, 'tasks': ['task'], 'skills': ['old'], 'raw_skills': ['<skill>old</skill>']}
            with patch.object(controller, '_generate', return_value=['<skill>new</skill>']):
                controller.finish_rollout(context, [[], []], [0, 0], None, 'train')
            self.assertIsNone(controller.pending_payload)
            self.assertEqual(controller.pending_metrics['skill_refinement/rejected_no_successful_evidence'], 1)


if __name__ == '__main__':
    unittest.main()
