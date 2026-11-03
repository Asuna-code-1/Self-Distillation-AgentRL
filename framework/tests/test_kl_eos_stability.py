"""Regress the observed KL overflow and EOS/PAD masking failure on CPU."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.torch_functional import get_response_mask

class KLEOSStabilityTests(unittest.TestCase):
    def test_k3_matches_original_clipped_values_and_gradients(self):
        delta=torch.linspace(-100,100,20001,dtype=torch.float64)
        lp=(-200-delta).requires_grad_();ref=torch.full_like(lp,-200)
        actual=kl_penalty(lp,ref,'low_var_kl')
        actual_grad=torch.autograd.grad(actual.sum(),lp)[0]
        old_lp=lp.detach().clone().requires_grad_();d=ref-old_lp
        expected=(d.exp()-d-1).clamp(-10,10)
        expected_grad=torch.autograd.grad(expected.sum(),old_lp)[0]
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-12,atol=1e-12)

    def test_observed_step68_and_extreme_kl_have_finite_gradients(self):
        for dtype in [torch.bfloat16,torch.float32,torch.float64]:
            lp=torch.tensor([-100.75,-1e6,-1.,-12.],dtype=dtype,requires_grad=True)
            ref=torch.tensor([-11.875165939331055,-1.,-1e6,-12.],dtype=dtype)
            out=kl_penalty(lp,ref,'k3');out.sum().backward()
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.isfinite(lp.grad).all())
            torch.testing.assert_close(out,torch.tensor([10.,10.,10.,0.],dtype=out.dtype))

    def test_generated_lengths_exclude_padding_even_with_wrong_eos(self):
        ids=torch.tensor([[23,248046,248044,248044],[23,248044,31,248046],[248044]*4])
        actual=get_response_mask(ids,eos_token=248044,response_lengths=[2,4,0])
        torch.testing.assert_close(actual,torch.tensor([[1,1,0,0],[1,1,1,1],[0,0,0,0]]))
        torch.testing.assert_close(get_response_mask(ids[:1],eos_token=248046),actual[:1])
        with self.assertRaises(ValueError):get_response_mask(ids,response_lengths=[2,4,5])

    def test_real_rollout_uses_engine_lengths_instead_of_wrong_eos(self):
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
        from omegaconf import OmegaConf
        actor=object.__new__(vLLMRollout)
        actor.config=OmegaConf.create({'free_cache_engine':False,'response_length':5})
        actor.pad_token_id=248044;actor.lora_kwargs={};actor.sampling_params=SimpleNamespace(n=1)
        actor.update_sampling_params=lambda **kwargs:nullcontext()
        ids=[23,248046]
        sample=SimpleNamespace(token_ids=ids,logprobs=[{i:SimpleNamespace(logprob=-.5)} for i in ids])
        actor.inference_engine=SimpleNamespace(generate=lambda **kwargs:[SimpleNamespace(outputs=[sample])])
        request=DataProto.from_dict(tensors={
            'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.ones(1,2,dtype=torch.long),
            'position_ids':torch.tensor([[0,1]])},
            non_tensors={'raw_prompt_ids':np.array([[1,2]],dtype=object)},meta_info={'eos_token_id':248044})
        with patch('verl.utils.debug.performance._get_current_mem_info',return_value=(0,0,0,0)):
            out=actor.generate_sequences(request)
        torch.testing.assert_close(out.batch['attention_mask'][:,-5:],torch.tensor([[1,1,0,0,0]]))

    def test_native_actor_masks_invalid_reference_padding_before_kl(self):
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        from test_reskill_grpo import config
        cfg=config().actor_rollout_ref.actor
        cfg.use_kl_loss=True;cfg.kl_loss_type='low_var_kl';cfg.kl_loss_coef=.01;cfg.entropy_coeff=0.
        cfg.ppo_mini_batch_size=1;cfg.ppo_micro_batch_size_per_gpu=1;cfg.ppo_epochs=1;cfg.use_dynamic_bsz=False
        model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(-2.)
        actor=DataParallelPPOActor(cfg,model,torch.optim.Adam(model.parameters(),lr=.001))
        actor._forward_micro_batch=lambda micro_batch,**kwargs:(None,model.weight.flatten()[0].expand(1,2))
        data=DataProto.from_dict(tensors={
            'responses':torch.ones(1,2,dtype=torch.long),'input_ids':torch.ones(1,4,dtype=torch.long),
            'attention_mask':torch.tensor([[1,1,1,0]]),'position_ids':torch.arange(4).view(1,4),
            'old_log_probs':torch.full((1,2),-2.),'advantages':torch.tensor([[1.,0.]]),
            'ref_log_prob':torch.tensor([[-1.8,float('nan')]])},meta_info={'temperature':1.})
        with patch('verl.workers.actor.dp_actor.get_torch_device',return_value=SimpleNamespace(current_device=lambda:'cpu')), \
             patch('verl.utils.debug.performance._get_current_mem_info',return_value=(0,0,0,0)):
            actor.update_policy(data)
        self.assertFalse(actor._last_optimizer_step_skipped)
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertEqual(actor.actor_optimizer.state[model.weight]['step'].item(),1.)

if __name__=='__main__':unittest.main()
