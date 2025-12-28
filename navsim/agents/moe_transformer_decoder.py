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


class MoEFullDecoderLayer(nn.Module):
    """
    A standard Transformer decoder layer (batch_first) used as an *expert*.

    This is more "aggressive" than MoETransformerDecoderLayer (which only MoE-ifies the FFN),
    because here we treat the entire layer (self-attn + cross-attn + FFN) as the expert module,
    and route *between different full layers*.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tgt
        sa_out = self.self_attn(x, x, x, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
        x = self.norm1(x + self.dropout1(sa_out))

        ca_out = self.multihead_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
        x = self.norm2(x + self.dropout2(ca_out))

        ffn = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(ffn))
        return x


class MoELayerwiseTransformerDecoder(nn.Module):
    """
    Layer-wise MoE Transformer decoder (batch_first).

    Compared to FFN-MoE:
    - Experts are *full decoder layers* (self-attn + cross-attn + FFN).
    - Routing is done at *sample-level* (per batch element), not token-level, to keep attention well-defined.
    - For each layer position, we have E expert layers and a router selecting top-k experts.

    Forward contract:
        (tgt, memory) -> (output, aux)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        moe_cfg: MoEConfig,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if moe_cfg.num_experts <= 0:
            raise ValueError("moe_cfg.num_experts must be > 0")
        if moe_cfg.top_k <= 0 or moe_cfg.top_k > moe_cfg.num_experts:
            raise ValueError("moe_cfg.top_k must be in [1, num_experts]")

        self.moe_cfg = moe_cfg
        self.num_layers = num_layers

        # Router per layer position (sample-level routing, input is pooled query state)
        self.routers = nn.ModuleList(
            [nn.Linear(d_model, moe_cfg.num_experts, bias=False) for _ in range(num_layers)]
        )

        # Expert layers per layer position
        self.experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MoEFullDecoderLayer(
                            d_model=d_model,
                            nhead=nhead,
                            dim_feedforward=dim_feedforward,
                            dropout=dropout,
                            activation=activation,
                        )
                        for _ in range(moe_cfg.num_experts)
                    ]
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = tgt
        moe_cfg = self.moe_cfg

        total_aux_loss = tgt.new_zeros(())
        router_z_loss = tgt.new_zeros(())
        load_balance_loss = tgt.new_zeros(())
        usage_counts = tgt.new_zeros((moe_cfg.num_experts,), dtype=tgt.dtype)

        for layer_idx in range(self.num_layers):
            # Sample-level router input: pool over query tokens (mean)
            pooled = x.mean(dim=1)  # (B, D)
            logits = self.routers[layer_idx](pooled)  # (B, E)
            if moe_cfg.router_temperature != 1.0:
                logits = logits / max(moe_cfg.router_temperature, 1e-6)

            topk_vals, topk_idx = torch.topk(logits, k=moe_cfg.top_k, dim=-1)  # (B, K)
            topk_w = F.softmax(topk_vals, dim=-1, dtype=torch.float32).to(x.dtype)  # (B, K)

            # Usage stats
            usage_counts = usage_counts + torch.bincount(topk_idx.reshape(-1), minlength=moe_cfg.num_experts).to(x.dtype)

            # Compute expert outputs (for selected experts only) and combine
            y = torch.zeros_like(x)
            for expert_id, expert_layer in enumerate(self.experts[layer_idx]):
                sel = topk_idx == expert_id  # (B, K)
                if not sel.any():
                    continue
                b_idx, kth = torch.where(sel)
                x_sel = x[b_idx]  # (N, Q, D)
                mem_sel = memory[b_idx]  # (N, S, D)
                out_sel = expert_layer(
                    x_sel,
                    mem_sel,
                    tgt_key_padding_mask=None if tgt_key_padding_mask is None else tgt_key_padding_mask[b_idx],
                    memory_key_padding_mask=None if memory_key_padding_mask is None else memory_key_padding_mask[b_idx],
                )  # (N, Q, D)
                w_sel = topk_w[b_idx, kth].view(-1, 1, 1)  # (N, 1, 1)
                y[b_idx] += out_sel * w_sel
            x = y

            # Aux losses (computed per layer)
            # Load balance (Switch-style) using full softmax probs over experts
            if moe_cfg.load_balance_coef > 0:
                probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # (B, E)
                importance = probs.mean(dim=0)  # (E,)
                lb = (moe_cfg.num_experts * torch.sum(importance * importance)).to(x.dtype)
                load_balance_loss = load_balance_loss + lb * moe_cfg.load_balance_coef

            if moe_cfg.router_z_loss_coef > 0:
                z = torch.logsumexp(logits.to(torch.float32), dim=-1)  # (B,)
                zl = (z * z).mean().to(x.dtype)
                router_z_loss = router_z_loss + zl * moe_cfg.router_z_loss_coef

        total_aux_loss = load_balance_loss + router_z_loss
        aux_out: Dict[str, torch.Tensor] = {
            "moe_aux_loss": total_aux_loss,
            "moe_router_z_loss": router_z_loss,
            "moe_load_balance_loss": load_balance_loss,
            "moe_usage_counts": usage_counts,
            "moe_usage_fraction": usage_counts / usage_counts.sum().clamp_min(1.0),
        }
        return x, aux_out


