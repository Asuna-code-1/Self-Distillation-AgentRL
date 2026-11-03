"""Overflow regression, clipped-objective equivalence, and safe skipped updates."""
import json,os,unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
import torch
from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.numerics import finite_loss_all_ranks
from test_reskill_grpo import config


class NumericsStabilityTests(unittest.TestCase):
    def test_clipped_objective_and_gradients_match_independent_signed_formula(self):
        torch.manual_seed(7)
        delta=torch.rand(5,10,dtype=torch.float64)*20-10
        old=torch.full_like(delta,-12.)
        current=(old+delta).requires_grad_()
        adv=torch.randn_like(delta);mask=(torch.rand_like(delta)>.3).double()
        loss,*_=compute_policy_loss(old,current,adv,mask,cliprange=.2)
        actual=torch.autograd.grad(loss,current)[0]
        reference_delta=delta.clone().requires_grad_();ratio=reference_delta.exp()
        terms=torch.where(adv>=0,-adv*ratio.clamp(max=1.2),-adv*ratio.clamp(min=.8,max=3.))
        reference=(terms*mask).sum()/(mask.sum()+1e-8)
        expected=torch.autograd.grad(reference,reference_delta)[0]
        torch.testing.assert_close(loss,reference,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)

    def test_extreme_finite_ratios_have_finite_loss_and_gradients(self):
        for dtype in [torch.float32,torch.bfloat16,torch.float64]:
            for delta in [100.,1000.,1000000.]:
                for advantage,expected in [(1.,-1.2),(-1.,3.),(0.,0.)]:
                    old=torch.tensor([[-delta-1]],dtype=dtype)
                    current=torch.tensor([[-1.]],dtype=dtype,requires_grad=True)
                    loss,*_=compute_policy_loss(old,current,torch.full_like(current,advantage),torch.ones_like(current),cliprange=.2)
                    loss.backward()
                    self.assertTrue(torch.isfinite(current.grad).all())
                    self.assertAlmostEqual(loss.item(),expected,places=5)
                    self.assertEqual(current.grad.item(),0.)

    def test_inactive_nonfinite_slots_do_not_contaminate_valid_tokens(self):
        old=torch.tensor([[-2.,float('nan'),-1001.]])
        current=torch.tensor([[-2.,float('nan'),-1.]],requires_grad=True)
        adv=torch.tensor([[1.,float('nan'),0.]])
        loss,*_=compute_policy_loss(old,current,adv,torch.tensor([[1.,0.,0.]]),cliprange=.2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertEqual(current.grad[0,1:].tolist(),[0.,0.])

    def test_rank_agreement_can_reject_another_ranks_nonfinite_loss(self):
        with patch.dict(os.environ,{'RESKILL_COORDINATED_FINITE_CHECK':'1'}), \
             patch('torch.distributed.is_initialized',return_value=True), \
             patch('torch.distributed.all_reduce',side_effect=lambda tensor,**kwargs:tensor.zero_()) as reduce:
            self.assertFalse(finite_loss_all_ranks(torch.tensor(1.)))
            reduce.assert_called_once()

    def actor(self):
        cfg=config().actor_rollout_ref.actor
        model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(.25)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        actor=DataParallelPPOActor(cfg,model,optimizer)
        actor._numerics_context={'batch':{'old_log_probs':torch.tensor([[-1001.]])}}
        return actor,model,optimizer

    def environment(self):
        directory=Path('/data/fd/donghaosen/Self-Distillation-AgentRL/logs/tests')/uuid4().hex
        return {'RESKILL_FAIL_ON_NONFINITE_GRAD':'0','RESKILL_MAX_CONSECUTIVE_NONFINITE':'3','RESKILL_NUMERICS_DIR':str(directory)}

    def test_bad_gradient_does_not_update_weights_or_adam_and_next_update_recovers(self):
        actor,model,optimizer=self.actor();before=model.weight.detach().clone();env=self.environment()
        with patch.dict(os.environ,env):
            model.weight.grad=torch.full_like(model.weight,float('nan'))
            actor._optimizer_step()
            torch.testing.assert_close(model.weight,before)
            self.assertEqual(len(optimizer.state),0)
            self.assertTrue(actor._last_optimizer_step_skipped)
            self.assertIsNone(model.weight.grad)
            self.assertEqual(len(list(Path(env['RESKILL_NUMERICS_DIR']).glob('*.pt'))),1)
            model.weight.grad=torch.ones_like(model.weight)
            actor._optimizer_step()
            self.assertFalse(actor._last_optimizer_step_skipped)
            self.assertEqual(optimizer.state[model.weight]['step'].item(),1.)
            self.assertEqual(actor._nonfinite_streak,0)

    def test_bad_forward_discards_even_previously_accumulated_finite_gradients(self):
        actor,model,optimizer=self.actor();before=model.weight.detach().clone()
        with patch.dict(os.environ,self.environment()):
            model.weight.grad=torch.ones_like(model.weight)
            actor._nonfinite_forward_loss=True
            actor._optimizer_step()
            torch.testing.assert_close(model.weight,before)
            self.assertEqual(len(optimizer.state),0)
            self.assertIsNone(model.weight.grad)

    def test_repeated_bad_gradients_keep_safety_stop(self):
        actor,model,optimizer=self.actor()
        with patch.dict(os.environ,self.environment()):
            for _ in range(2):
                model.weight.grad=torch.full_like(model.weight,float('nan'));actor._optimizer_step()
            model.weight.grad=torch.full_like(model.weight,float('nan'))
            with self.assertRaises(FloatingPointError):actor._optimizer_step()
            self.assertEqual(len(optimizer.state),0)

if __name__=='__main__':unittest.main()
