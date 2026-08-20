#!/usr/bin/env python3
"""ONNX-export compatibility shim for MuScriptor browser exporter.

The upstream browser exporter intentionally keeps query/prefix lengths dynamic,
but PyTorch's legacy ONNX exporter requires LayerNorm.normalized_shape to be a
compile-time constant. MuScriptor-small has a fixed model width of 768, so this
shim swaps only BrowserLayer for an equivalent implementation whose normalized
shape is static. All weight loading, parity checks, FP16 conversion, INT4
quantization, manifest generation, and CLI handling remain in the v1 exporter.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import export_muscriptor_browser as base


class StaticBrowserLayer(base.BrowserLayer):
    """Numerically identical BrowserLayer with static LayerNorm width."""

    def __init__(self, layer, heads: int):
        super().__init__(layer, heads)
        self.dim = int(layer.norm1.weight.numel())
        if self.dim != 768:
            raise RuntimeError(f"MuScriptor-small width must be 768, got {self.dim}")
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        past_len: torch.Tensor,
    ):
        b, q_len, _ = x.shape
        h = F.layer_norm(x, (self.dim,), self.norm1_w, self.norm1_b, 1e-5)
        packed = torch.matmul(h, self.qkv_w).reshape(
            b, q_len, 3, self.heads, self.head_dim
        )
        q = packed[:, :, 0]
        k_new = packed[:, :, 1]
        v_new = packed[:, :, 2]

        p = past_len.reshape(()).to(torch.long)
        past_k = cache_k[:, :p]
        past_v = cache_v[:, :p]
        k_all = torch.cat([past_k, k_new], dim=1)
        v_all = torch.cat([past_v, v_new], dim=1)

        qt = q.permute(0, 2, 1, 3)
        kt = k_all.permute(0, 2, 1, 3)
        vt = v_all.permute(0, 2, 1, 3)
        scores = torch.matmul(qt, kt.transpose(-1, -2)) * self.scale

        q_pos = torch.arange(q_len, device=x.device, dtype=torch.long) + p
        k_pos = torch.arange(k_all.shape[1], device=x.device, dtype=torch.long)
        allow = k_pos.reshape(1, 1, 1, -1) <= q_pos.reshape(1, 1, -1, 1)
        scores = torch.where(allow, scores, torch.full_like(scores, -1.0e4))
        weights = torch.softmax(scores, dim=-1)
        attended = (
            torch.matmul(weights, vt)
            .permute(0, 2, 1, 3)
            .reshape(b, q_len, self.dim)
        )
        x = x + torch.matmul(attended, self.attn_out_w)

        h2 = F.layer_norm(x, (self.dim,), self.norm2_w, self.norm2_b, 1e-5)
        x = x + torch.matmul(F.gelu(torch.matmul(h2, self.ff1_w)), self.ff2_w)
        return x, k_new, v_new


# BrowserDecoder resolves BrowserLayer from the base module's globals at
# construction time, so replacing the symbol here leaves the rest untouched.
base.BrowserLayer = StaticBrowserLayer


if __name__ == "__main__":
    base.main()
