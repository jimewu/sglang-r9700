# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""
ROCm/HIP-compatible W4A16 dense linear scheme.

Dequantizes INT4 weights to BF16 on-the-fly per forward pass and uses
torch.matmul (rocBLAS) for the GEMM. Keeps the original INT4 packed format
in memory to save VRAM.
"""

from typing import Callable, List, Optional

import torch
import triton
import triton.language as tl

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

__all__ = ["CompressedTensorsWNA16Triton"]


@triton.jit
def _w4a16_dequant_kernel(
    b_packed_ptr, scales_ptr, zp_ptr, output_ptr,
    stride_bn, stride_bk_packed,
    stride_sn, stride_sg,
    stride_zpn, stride_zpg,
    stride_on, stride_ok,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_ZP: tl.constexpr,
    K_PACKED: tl.constexpr, NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Dequantize pack-quantized INT4 weights to BF16 (one call per layer)."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N

    for start_k in range(0, K, BLOCK_K):
        k_abs = start_k + offs_k
        k_mask = k_abs < K
        k_abs_safe = tl.where(k_abs < K, k_abs, K - 1)
        k_packed = k_abs_safe // 8
        k_bit = k_abs_safe % 8

        b_packed = tl.load(
            b_packed_ptr + offs_n[:, None] * stride_bn + k_packed[None, :] * stride_bk_packed,
            mask=(n_mask[:, None] & k_mask[None, :]), other=0)
        b_int4 = (b_packed >> (k_bit[None, :] * 4)) & 0xF

        if HAS_ZP:
            n_packed = offs_n // 8
            n_bit = offs_n % 8
            g_idx = k_abs_safe // GROUP_SIZE
            zp_packed = tl.load(
                zp_ptr + n_packed[:, None] * stride_zpn + g_idx[None, :] * stride_zpg,
                mask=(n_mask[:, None] & k_mask[None, :]), other=0)
            zp_int4 = (zp_packed >> (n_bit[:, None] * 4)) & 0xF
            b_int4 = b_int4 - zp_int4

        g_idx = k_abs_safe // GROUP_SIZE
        s = tl.load(
            scales_ptr + offs_n[:, None] * stride_sn + g_idx[None, :] * stride_sg,
            mask=(n_mask[:, None] & k_mask[None, :]), other=0.0)
        b = b_int4.to(tl.float32) * s

        out_ptrs = output_ptr + offs_n[:, None] * stride_on + k_abs[None, :] * stride_ok
        tl.store(out_ptrs, b.to(tl.bfloat16), mask=(n_mask[:, None] & k_mask[None, :]))


class CompressedTensorsWNA16Triton(CompressedTensorsLinearScheme):
    """ROCm/HIP W4A16 linear method.

    On-the-fly dequantization: each forward pass dequantizes just the
    current layer's weights and uses torch.matmul (rocBLAS) for GEMM.
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
        for name in ["weight_packed", "weight_scale"]:
            p = getattr(layer, name, None)
            if p is not None and not p.data.is_contiguous():
                replace_parameter(layer, name, torch.nn.Parameter(
                    p.data.contiguous(), requires_grad=False))
        if not self.symmetric:
            for name in ["weight_zero_point", "weight_shape"]:
                p = getattr(layer, name, None)
                if p is not None and not p.data.is_contiguous():
                    replace_parameter(layer, name, torch.nn.Parameter(
                        p.data.contiguous(), requires_grad=False))

    def apply_weights(self, layer, x, bias=None):
        """Dequantize this layer's weights on-the-fly, then matmul."""
        w_q = getattr(layer, self.w_q_name)
        w_s = getattr(layer, self.w_s_name)
        w_zp = getattr(layer, self.w_zp_name) if self.w_zp_name else None

        N, K_packed = w_q.shape
        K = K_packed * 8
        gs = self.group_size if self.group_size != -1 else K
        ng = K // gs
        has_zp = w_zp is not None

        # Allocate dequantized weight buffer
        w_deq = torch.empty((N, K), dtype=torch.bfloat16, device=w_q.device)

        def grid_fn(meta):
            return (triton.cdiv(N, meta["BLOCK_N"]),)

        _w4a16_dequant_kernel[grid_fn](
            w_q, w_s, w_zp, w_deq,
            w_q.stride(0), w_q.stride(1),
            w_s.stride(0), w_s.stride(1),
            w_zp.stride(0) if has_zp else 0,
            w_zp.stride(1) if has_zp else 0,
            w_deq.stride(0), w_deq.stride(1),
            N, K,
            BLOCK_N=64, BLOCK_K=128,
            HAS_ZP=has_zp,
            K_PACKED=K_packed, NUM_GROUPS=ng,
            GROUP_SIZE=gs,
        )

        out = torch.matmul(x, w_deq.t())
        if bias is not None:
            out = out + bias
        return out