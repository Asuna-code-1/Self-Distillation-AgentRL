"""Sequence boundaries for packed Qwen3.5 convolution/Gated DeltaNet layers."""
import torch


def packed_hybrid_kwargs(config, position_ids):
    model_type = getattr(config, 'model_type', '')
    if model_type not in ('qwen3_5', 'qwen3_5_text', 'qwen3_5_moe', 'qwen3_5_moe_text'):
        return {}
    # The first coordinate is the text coordinate for multimodal RoPE.
    positions = position_ids[0] if position_ids.ndim == 3 else position_ids
    if positions.ndim != 2 or positions.shape[0] != 1:
        raise ValueError('Packed hybrid attention requires one concatenated sequence and SP=1')
    flat = positions.reshape(-1)
    starts = torch.nonzero(flat == 0, as_tuple=False).flatten()
    if not len(starts) or int(starts[0]) != 0:
        raise ValueError('Every packed sequence must begin at text position zero')
    cu = torch.cat((starts, starts.new_tensor([flat.numel()]))).to(torch.int32)
    sequence_ids = ((flat == 0).to(torch.int32).cumsum(0) - 1).to(torch.int32).unsqueeze(0).contiguous()
    maximum = int((cu[1:] - cu[:-1]).max())
    return {'cu_seq_lens_q': cu, 'cu_seq_lens_k': cu,
            'max_length_q': maximum, 'max_length_k': maximum,
            'seq_idx': sequence_ids}
