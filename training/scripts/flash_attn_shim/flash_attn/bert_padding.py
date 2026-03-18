"""Pure-torch reimplementation of flash_attn.bert_padding (verl subset)."""
import torch
from einops import rearrange  # noqa: F401  (flash_attn re-exports this)


def index_first_axis(tensor, indices):
    return tensor[indices]


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """(batch, seqlen, ...) + (batch, seqlen) -> unpadded, indices, cu_seqlens, max_seqlen."""
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item())
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    out = hidden_states.reshape(-1, *hidden_states.shape[2:])[indices]
    return out, indices, cu_seqlens, max_seqlen


def pad_input(hidden_states, indices, batch, seqlen):
    dim = hidden_states.shape[-1]
    output = torch.zeros(batch * seqlen, dim,
                         device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, dim)
