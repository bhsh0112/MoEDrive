"""
MoE-based Transformer decoder modules for Transfuser.

This file provides a minimal, self-contained implementation of:
- Top-k routed MoE feed-forward (FFN)
- Transformer decoder layer with MoE FFN (batch_first)
- Stacked transformer decoder that aggregates MoE auxiliary losses

Design goal:
Keep the same input/output shape contract as torch.nn.TransformerDecoder when batch_first=True:
    forward(tgt, memory) -> (B, T, D)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    """
    Configuration for MoE FFN routing.
    """

    num_experts: int = 4
    top_k: int = 2
    router_z_loss_coef: float = 0.0
    load_balance_coef: float = 0.0
    router_temperature: float = 1.0


class MoEFeedForward(nn.Module):
    """
    Top-k routed Mixture-of-Experts feed-forward network.

    Notes:
    - Routing is performed per token (batch, seq) independently.
    - Experts share the same architecture but have independent parameters.
    - Returns (y, aux) where aux includes router/load-balance losses and usage stats.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout: float,
        moe_cfg: MoEConfig,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if moe_cfg.num_experts <= 0:
            raise ValueError("moe_cfg.num_experts must be > 0")
        if moe_cfg.top_k <= 0 or moe_cfg.top_k > moe_cfg.num_experts:
            raise ValueError("moe_cfg.top_k must be in [1, num_experts]")

        self.moe_cfg = moe_cfg
        self.router = nn.Linear(d_model, moe_cfg.num_experts, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU() if activation == "relu" else nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ffn, d_model),
                )
                for _ in range(moe_cfg.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, T, D)
        Returns:
            y: (B, T, D)
            aux: dict of auxiliary losses/stats
        """
        b, t, d = x.shape
        moe_cfg = self.moe_cfg

        router_logits = self.router(x)  # (B, T, E)
        if moe_cfg.router_temperature != 1.0:
            router_logits = router_logits / max(moe_cfg.router_temperature, 1e-6)

        topk_vals, topk_idx = torch.topk(router_logits, k=moe_cfg.top_k, dim=-1)  # (B, T, K)
        topk_w = F.softmax(topk_vals, dim=-1, dtype=torch.float32).to(x.dtype)  # (B, T, K)

        y = torch.zeros_like(x)

        # Dispatch tokens to experts.
        # We iterate experts; for each expert, gather tokens routed to it.
        for expert_id, expert in enumerate(self.experts):
            # mask: (B, T, K) True where selected expert == expert_id
            sel = topk_idx == expert_id
            if not sel.any():
                continue

            # indices of routed tokens
            batch_idx, token_idx, kth = torch.where(sel)
            x_sel = x[batch_idx, token_idx]  # (N, D)
            y_sel = expert(x_sel)  # (N, D)
            w_sel = topk_w[batch_idx, token_idx, kth].unsqueeze(-1)  # (N, 1)
            y[batch_idx, token_idx] += y_sel * w_sel

        y = self.dropout(y)

        aux: Dict[str, torch.Tensor] = {}

        # Expert usage (counts per expert in the current batch)
        # usage_counts: (E,)
        usage_counts = torch.bincount(topk_idx.reshape(-1), minlength=moe_cfg.num_experts).to(x.dtype)
        aux["moe_usage_counts"] = usage_counts
        aux["moe_usage_fraction"] = usage_counts / usage_counts.sum().clamp_min(1.0)

        # Load balancing loss (Switch-style importance loss)
        if moe_cfg.load_balance_coef > 0:
            probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # (B, T, E)
            importance = probs.mean(dim=(0, 1))  # (E,)
            load_balance_loss = (moe_cfg.num_experts * torch.sum(importance * importance)).to(x.dtype)
            aux["moe_load_balance_loss"] = load_balance_loss * moe_cfg.load_balance_coef
        else:
            aux["moe_load_balance_loss"] = x.new_zeros(())

        # Router z-loss (stabilize large router logits)
        if moe_cfg.router_z_loss_coef > 0:
            z = torch.logsumexp(router_logits.to(torch.float32), dim=-1)  # (B, T)
            z_loss = (z * z).mean().to(x.dtype)
            aux["moe_router_z_loss"] = z_loss * moe_cfg.router_z_loss_coef
        else:
            aux["moe_router_z_loss"] = x.new_zeros(())

        aux["moe_aux_loss"] = aux["moe_load_balance_loss"] + aux["moe_router_z_loss"]
        return y, aux


class MoETransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer (batch_first) with MoE FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        moe_cfg: MoEConfig,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe_ffn = MoEFeedForward(d_model, dim_feedforward, dropout, moe_cfg)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Self-attention
        x = tgt
        sa_out = self.self_attn(
            x, x, x, key_padding_mask=tgt_key_padding_mask, need_weights=False
        )[0]
        x = self.norm1(x + self.dropout1(sa_out))

        # Cross-attention
        ca_out = self.multihead_attn(
            x, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False
        )[0]
        x = self.norm2(x + self.dropout2(ca_out))

        # MoE FFN
        ffn_out, aux = self.moe_ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x, aux


class MoETransformerDecoder(nn.Module):
    """
    Stacked MoE transformer decoder (batch_first), compatible with Transfuser usage:
        query_out = decoder(query, keyval)

    Returns:
        output: (B, Q, D)
        aux: dict with summed auxiliary losses and aggregated usage stats
    """

    def __init__(
        self,
        layer: MoETransformerDecoderLayer,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else type(layer)(  # type: ignore[misc]
            d_model=layer.norm1.normalized_shape[0],
            nhead=layer.self_attn.num_heads,
            dim_feedforward=layer.moe_ffn.experts[0][0].out_features,  # d_ffn
            dropout=layer.dropout1.p,
            moe_cfg=layer.moe_ffn.moe_cfg,
        ) for i in range(num_layers)])

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = tgt
        total_aux_loss = tgt.new_zeros(())
        usage_counts = None
        router_z_loss = tgt.new_zeros(())
        load_balance_loss = tgt.new_zeros(())

        for layer in self.layers:
            x, aux = layer(
                x,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            total_aux_loss = total_aux_loss + aux["moe_aux_loss"]
            router_z_loss = router_z_loss + aux["moe_router_z_loss"]
            load_balance_loss = load_balance_loss + aux["moe_load_balance_loss"]
            if usage_counts is None:
                usage_counts = aux["moe_usage_counts"]
            else:
                usage_counts = usage_counts + aux["moe_usage_counts"]

        if usage_counts is None:
            usage_counts = tgt.new_zeros((1,))

        aux_out: Dict[str, torch.Tensor] = {
            "moe_aux_loss": total_aux_loss,
            "moe_router_z_loss": router_z_loss,
            "moe_load_balance_loss": load_balance_loss,
            "moe_usage_counts": usage_counts,
            "moe_usage_fraction": usage_counts / usage_counts.sum().clamp_min(1.0),
        }
        return x, aux_out


