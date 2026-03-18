"""Pure-torch shim for flash_attn (verl bert_padding usage only).

verl 0.8's trainer unconditionally imports flash_attn.bert_padding for
padding/unpadding helpers in `_compute_old_log_prob`. flash-attn has no
prebuilt wheel for torch 2.11, so this shim provides the exact APIs verl
uses, implemented in plain torch. flash_attn_func (CUDA kernels) is NOT
provided; attention itself runs through SDPA.
"""
