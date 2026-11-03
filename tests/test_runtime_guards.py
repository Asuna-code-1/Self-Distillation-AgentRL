import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from verl.workers.actor.dp_actor import DataParallelPPOActor

class RuntimeGuards(unittest.TestCase):
    def test_nonfinite_gradient_stops_before_parameter_update(self):
        model=torch.nn.Linear(1,1,bias=False)
        actor=DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module=model;actor.actor_optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        actor.config=SimpleNamespace(grad_clip=1.)
        before=model.weight.detach().clone();model.weight.grad=torch.full_like(model.weight,float('nan'))
        with patch.dict(os.environ,{'RESKILL_FAIL_ON_NONFINITE_GRAD':'1'}),patch('torch.distributed.get_rank',return_value=0):
            with self.assertRaises(FloatingPointError):actor._optimizer_step()
        self.assertTrue(torch.equal(before,model.weight.detach()))
        self.assertIsNone(model.weight.grad)
