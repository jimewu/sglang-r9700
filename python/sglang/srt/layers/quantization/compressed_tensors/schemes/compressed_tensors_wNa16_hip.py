# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
R9700 (gfx1201/RDNA4) HIP W4A16 dense linear scheme.

Fused WMMA GEMM: per-row INT8 activation quantization pre-kernel + gfx1201
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` WMMA kernel, using the
`r9700_w4a16` custom op. Keeps the original INT4 packed format in memory.

Weight layout is identical to CompressedTensorsWNA16Triton:
  weight_packed      (N, K/8)      int32  8 uint4 per int32 (low nibble first)
  weight_scale       (N, K/groups) fp16    per-group weight scales
  weight_zero_point  (N/8, K/groups) int32 packed zero points (asymmetric)
"""

from typing import Callable, List, Optional

import torch

from compressed_tensors.quantization import ActivationOrdering

from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.utils import replace_parameter

__all__ = ["CompressedTensorsWNA16HIP"]

_OP = None


def _get_op():
    """Lazily import the r9700_w4a16 WMMA custom op (compiled for gfx1201)."""
    global _OP
    if _OP is None:
        import r9700_w4a16

        _OP = r9700_w4a16.mmq_q4_gemm
    return _OP


class CompressedTensorsWNA16HIP(CompressedTensorsLinearScheme):
    """R9700 gfx1201 fused WMMA W4A16 linear method.

    Uses the r9700_w4a16 custom op (per-row INT8 act-quant + gfx1201 iu8 WMMA)
    instead of on-the-fly dequantization + torch.matmul.
    """

    def __init__(self, strategy: str, num_bits: int,
                 group_size: Optional[int] = None,
                 symmetric: Optional[bool] = True,
                 actorder: Optional[ActivationOrdering] = None):
        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP
        self.num_bits = num_bits
        self.w_q_name = "weight_packed"
        self.w_s_name = "weight_scale"
        self.w_zp_name = "weight_zero_point" if not self.symmetric else None

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(self, layer, output_size, input_size,
                       output_partition_sizes, input_size_per_partition,
                       params_dtype, weight_loader, **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        gs = self.group_size if self.group_size != -1 else input_size
        row_parallel = (input_size != input_size_per_partition)
        sz = input_size // gs
        if row_parallel:
            assert input_size_per_partition % gs == 0
            sz = input_size_per_partition // gs

        w = PackedvLLMParameter(
            input_dim=1, output_dim=0, weight_loader=weight_loader,
            packed_factor=self.pack_factor, packed_dim=1,
            data=torch.empty(output_size_per_partition,
                             input_size_per_partition // self.pack_factor,
                             dtype=torch.int32))
        ws = GroupQuantScaleParameter(
            output_dim=0, input_dim=1, weight_loader=weight_loader,
            data=torch.empty(output_size_per_partition, sz, dtype=params_dtype))
        layer.register_parameter("weight_packed", w)
        layer.register_parameter("weight_scale", ws)

        if not self.symmetric:
            qz = PackedColumnParameter(
                output_dim=0, packed_dim=0, packed_factor=self.pack_factor,
                weight_loader=weight_loader,
                data=torch.zeros(output_size_per_partition // self.pack_factor,
                                 sz, dtype=torch.int32))
            wsh = BasevLLMParameter(
                data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader)
            layer.register_parameter("weight_zero_point", qz)
            layer.register_parameter("weight_shape", wsh)

        if self.has_g_idx:
            gidx = RowvLLMParameter(
                data=torch.empty(input_size_per_partition, dtype=torch.int32),
                input_dim=0, weight_loader=weight_loader)
            layer.register_parameter("weight_g_idx", gidx)

    def process_weights_after_loading(self, layer):
        """Ensure packed weights are contiguous (no dequantization at load time)."""
        for name in ["weight_packed", "weight_scale", "weight_zero_point"]:
            p = getattr(layer, name, None)
            if p is not None and not p.data.is_contiguous():
                replace_parameter(layer, name, torch.nn.Parameter(
                    p.data.contiguous(), requires_grad=False))

    def apply_weights(self, layer, x, bias=None):
        """Fused WMMA GEMM via the r9700_w4a16 custom op."""
        op = _get_op()

        w_q = getattr(layer, self.w_q_name)
        w_s = getattr(layer, self.w_s_name)
        w_zp = getattr(layer, self.w_zp_name) if self.w_zp_name else None

        M, K = x.shape
        N = w_q.shape[0]
        ng = w_s.shape[1]

        # Kernel reads fp16 activations and fp16 scales.
        x_fp16 = x if x.dtype == torch.float16 else x.half()
        if w_s.dtype != torch.float16:
            w_s_fp16 = w_s.data.half()
        else:
            w_s_fp16 = w_s.data

        w_zeros = w_zp.data if w_zp is not None else torch.empty(0, device=x.device)
        out = torch.empty((M, N), dtype=torch.float16, device=x.device)

        op(x_fp16, w_q.data, w_s_fp16, w_zeros, out, 1)

        # Output is fp16 native; cast to activation dtype (bf16 for this model).
        if x.dtype != torch.float16:
            out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out
