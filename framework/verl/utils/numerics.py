"""Optional rank-coordinated finite checks before collective actor backwards."""
import os
import torch


def finite_loss_all_ranks(*values):
    finite = torch.stack([torch.isfinite(value.detach()).all() for value in values]).all().to(torch.int32)
    if os.environ.get('RESKILL_COORDINATED_FINITE_CHECK') == '1' and torch.distributed.is_initialized():
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
    return bool(finite.item())
