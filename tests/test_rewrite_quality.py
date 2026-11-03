"""Rewrite quality gating, objective/gradient, supervision and paired-seed checks."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from verl import DataProto
from verl.trainer.ppo.reskill_grpo import GRPOSkillRefinement,sft_payload
from verl.trainer.ppo.rewrite_quality import (
    normalized_edit_distance,paired_execution_seed,quality_signal,
    unlikelihood_from_log_probs,rewrite_quality_payload,backward_rewrite_quality,
    evaluate_rewritten_skills,
)
from test_reskill_grpo import Tokenizer,config,trainer,artifact_directory


class RewriteQualityTests(unittest.TestCase):
    def make_controller(self,step=5):
        cfg=config();cfg.algorithm.reskill_grpo.rewrite_quality.enable=True
        t=trainer(cfg);t.global_steps=step
        return GRPOSkillRefinement(t)

    def context(self):
        return {'tasks':['task'],'skills':['old plan'],'raw_skills':['<skill>old plan</skill>'],
                'group_size':2,'episode_skills':['old plan']*2}

    def run_finish(self,controller,q):
        def evaluate(ctrl,records,context,actor):
            record=records[0]
            record['rewrite_quality']=quality_signal([10,0],[10,10] if q else [0,0],
                                                    record['initial_skill'],record['refined_skill'])
            return 1
        with artifact_directory() as directory:
            controller.root=Path(directory)
            with patch.object(controller,'_generate',return_value=['<skill>new plan</skill>']), \
                 patch('verl.trainer.ppo.rewrite_quality.evaluate_rewritten_skills',side_effect=evaluate):
                controller.finish_rollout(self.context(),[[],[]],[10,0],None,'train')

    def test_success_gate_is_strict_and_improvement_is_strict(self):
        c=self.make_controller()
        self.assertEqual(evaluate_rewritten_skills(c,[{'refined_skill':'x','trajectories':[{'reward':10}]*2}],{'group_size':2},None),0)
        cfg=c.quality_cfg;cfg.old_success_threshold=.75
        record={'refined_skill':'x','trajectories':[{'reward':10}]*3+[{'reward':0}]}
        self.assertEqual(evaluate_rewritten_skills(c,[record],{'group_size':4},None),0)
        self.assertEqual(quality_signal([10,0],[10,0],'old','new')['q'],0)
        self.assertEqual(quality_signal([10,0],[0,0],'old','new')['q'],0)
        self.assertEqual(quality_signal([10,0],[10,10],'old','new')['q'],1)

    def test_formatting_whitespace_and_case_do_not_count_as_content_change(self):
        self.assertEqual(normalized_edit_distance('1. **Check** the sink.\n2. Wash the mug.',
                                                  'check the sink.\nwash the mug.'),0)
        self.assertEqual(normalized_edit_distance('```text\nCheck sink\n```','check\tsink'),0)
        self.assertGreater(normalized_edit_distance('wash the mug','do not wash the mug'),0)
        self.assertEqual(quality_signal([10,0],[10,10],'**check** sink','check\n sink')['q'],0)

    def test_schedule_and_validation_do_not_add_quality_gradients(self):
        c=self.make_controller(4)
        with artifact_directory() as directory:
            c.root=Path(directory)
            with patch.object(c,'_generate',return_value=['<skill>new plan</skill>']), \
                 patch('verl.trainer.ppo.rewrite_quality.evaluate_rewritten_skills',side_effect=AssertionError('off-period replay')):
                c.finish_rollout(self.context(),[[],[]],[10,0],None,'train')
        self.assertIsNone(c.pending_quality_payload)
        self.assertIsNotNone(c.pending_payload)
        self.assertFalse(c.quality_step())
        for step in range(161):
            c.trainer.global_steps=step
            self.assertEqual(c.quality_step(),step>0 and step%5==0)
        c.trainer.global_steps=5;self.assertTrue(c.quality_step())
        with patch.object(c,'_generate',side_effect=AssertionError('validation feedback used')):
            c.finish_rollout({},[],[],None,'validate')

    def test_negative_rewrite_is_masked_from_original_skill_sft(self):
        c=self.make_controller();self.run_finish(c,False)
        self.assertIsNone(c.pending_payload)
        self.assertIsNotNone(c.pending_quality_payload)
        self.assertEqual(c.pending_quality_payload['quality_q'].tolist(),[False])
        self.assertEqual(c.pending_metrics['rewrite_quality/masked_regular_skill_sft'],1)
        self.assertEqual(c.pending_metrics['rewrite_quality/extra_trajectories'],2)

    def test_positive_rewrite_keeps_original_supervision_and_uses_rewrite_context(self):
        c=self.make_controller();self.run_finish(c,True)
        self.assertIsNotNone(c.pending_payload)
        quality=c.pending_quality_payload
        self.assertEqual(quality['quality_q'].tolist(),[True])
        width=quality['responses'].shape[1]
        prompt=c.trainer.tokenizer.decode(quality['input_ids'][0,:-width])
        self.assertIn('Analyze the completed trajectories',prompt)
        self.assertIn('old plan',prompt)
        self.assertNotIn('Before solving the task',prompt)
        ordinary=c.pending_payload;width=ordinary['responses'].shape[1]
        original=c.trainer.tokenizer.decode(ordinary['input_ids'][0,:-width])
        self.assertIn('Before solving the task',original)
        self.assertNotIn('trajectories',original)

    def test_no_eligible_tasks_means_no_replay_or_quality_payload(self):
        c=self.make_controller()
        with artifact_directory() as directory:
            c.root=Path(directory)
            with patch.object(c,'_generate',return_value=['<skill>new plan</skill>']):
                c.finish_rollout(self.context(),[[],[]],[10,10],None,'train')
        self.assertIsNone(c.pending_quality_payload)
        self.assertEqual(c.pending_metrics['rewrite_quality/evaluated_groups'],0)
        self.assertEqual(c.pending_metrics['actor/rewrite_quality_loss'],0.0)
        self.assertEqual(c.pending_metrics['actor/rewrite_quality_weighted_loss'],0.0)
        self.assertIsNotNone(c.pending_payload)

    def test_payload_supervises_exact_generated_tokens_under_actual_prompt(self):
        tokenizer=Tokenizer()
        record={'rewrite_prompt':'actual task + old skill + OLD feedback only',
                'raw_rewrite':'<skill>new plan</skill>','rewrite_token_ids':[31,32,33],
                'rewrite_quality':{'q':0}}
        payload=rewrite_quality_payload(tokenizer,[record],1024,384)
        self.assertEqual(payload['responses'].tolist(),[[31,32,33]])
        prompt=tokenizer.decode(payload['input_ids'][0,:-3])
        self.assertIn(record['rewrite_prompt'],prompt)
        self.assertEqual(payload['quality_q'].tolist(),[False])
        self.assertIsNone(rewrite_quality_payload(tokenizer,[record],1,384))

    def test_unlikelihood_is_finite_nonnegative_and_gradient_has_correct_direction(self):
        logp=torch.tensor([-100.,-5.,-.7,-.00001,-1e-12,0.],requires_grad=True)
        loss=unlikelihood_from_log_probs(logp)
        self.assertTrue(torch.isfinite(loss).all());self.assertTrue((loss>=0).all())
        loss.sum().backward()
        self.assertTrue(torch.isfinite(logp.grad).all());self.assertTrue((logp.grad>=0).all())
        self.assertGreater(logp.grad[2].item(),0)

    def test_global_normalization_for_uneven_quality_shards(self):
        tokenizer=Tokenizer()
        records=[{'rewrite_prompt':'feedback','raw_rewrite':'<skill>plan</skill>',
                  'rewrite_quality':{'q':q}} for q in (1,0,0,1,0)]
        payload=rewrite_quality_payload(tokenizer,records,1024,384)
        payload.update(loss_coef=.1,beta=1.,micro_batch_size=1)
        gradients=[]
        for rank in range(4):
            module=torch.nn.Linear(1,1,bias=False);module.weight.data.fill_(-.7)
            module.weight.grad=torch.full_like(module.weight,3.)
            actor=SimpleNamespace(actor_module=module,ulysses_sequence_parallel_size=1)
            actor._forward_micro_batch=lambda batch,temperature,calculate_entropy:(None,module.weight.flatten()[0].expand(batch['responses'].shape))
            with patch('torch.distributed.is_initialized',return_value=True), \
                 patch('torch.distributed.get_rank',return_value=rank), \
                 patch('torch.distributed.get_world_size',return_value=4):
                backward_rewrite_quality(actor,payload)
            gradients.append(module.weight.grad.item())
        ul_derivative=math.exp(-.7)/(1-math.exp(-.7))
        expected=3.+.1*(-2+3*ul_derivative)/5
        self.assertAlmostEqual(sum(gradients)/4,expected,places=6)

    def test_paired_seeds_are_skill_independent_and_distinguish_replicas(self):
        key='seed=0:step=5:group=3:replica=0:game=A'
        self.assertEqual(paired_execution_seed(key,2),paired_execution_seed(key,2))
        self.assertNotEqual(paired_execution_seed(key,2),paired_execution_seed(key,3))
        self.assertNotEqual(paired_execution_seed(key,2),paired_execution_seed(key.replace('replica=0','replica=1'),2))

    def test_quality_gradient_is_added_before_single_native_optimizer_step(self):
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        cfg=config().actor_rollout_ref.actor
        cfg.use_kl_loss=False;cfg.entropy_coeff=0.;cfg.ppo_mini_batch_size=2
        cfg.ppo_micro_batch_size_per_gpu=1;cfg.ppo_epochs=1;cfg.use_dynamic_bsz=False
        for q in (False,True):
            module=torch.nn.Linear(1,1,bias=False);module.weight.data.fill_(-.7)
            actor=DataParallelPPOActor(cfg,module,torch.optim.Adam(module.parameters(),lr=.01))
            actor._forward_micro_batch=lambda micro_batch,temperature,calculate_entropy:(None,module.weight.flatten()[0].expand(micro_batch['responses'].shape))
            gradients=[]
            def step():
                gradients.append(module.weight.grad.item());actor.actor_optimizer.step()
                return torch.tensor(abs(gradients[-1]))
            actor._optimizer_step=step
            record={'rewrite_prompt':'original feedback','raw_rewrite':'<skill>new</skill>','rewrite_quality':{'q':int(q)}}
            payload=rewrite_quality_payload(Tokenizer(),[record],1024,384)
            payload.update(loss_coef=.1,beta=1.,micro_batch_size=1)
            data=DataProto.from_dict(tensors={
                'responses':torch.ones((2,3),dtype=torch.long),'input_ids':torch.ones((2,5),dtype=torch.long),
                'attention_mask':torch.ones((2,5),dtype=torch.long),'position_ids':torch.arange(5).expand(2,-1),
                'old_log_probs':torch.full((2,3),-.7),'advantages':torch.tensor([[1.,1.,1.],[2.,2.,2.]])},
                meta_info={'temperature':1.,'rewrite_quality':payload})
            if q:
                ordinary=sft_payload(Tokenizer(),[('task','new')],1024,384)
                ordinary.update(loss_coef=.1,micro_batch_size=1)
                data.meta_info['skill_refinement']=ordinary
            with patch('verl.workers.actor.dp_actor.get_torch_device',return_value=SimpleNamespace(current_device=lambda:'cpu')), \
                 patch('verl.utils.debug.performance._get_current_mem_info',return_value=(0,0,0,0)):
                DataParallelPPOActor.update_policy(actor,data)
            expected=-1.7 if q else -1.5+.1*math.exp(-.7)/(1-math.exp(-.7))
            self.assertEqual(len(gradients),1)
            self.assertAlmostEqual(gradients[0],expected,places=5)
            self.assertEqual(actor.actor_optimizer.state[module.weight]['step'].item(),1.)

if __name__=='__main__':unittest.main()
