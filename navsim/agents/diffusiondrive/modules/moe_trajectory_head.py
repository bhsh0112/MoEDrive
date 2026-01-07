"""
MOE-based Trajectory Head for Multi-modal Trajectory Prediction

This module replaces the diffusion-based trajectory head with a Mixture-of-Experts (MOE) approach.
Each expert generates a different trajectory mode, and the router selects the most relevant experts
based on the input features.

Key differences from diffusion approach:
1. Direct trajectory generation (no denoising process)
2. Multiple experts for multi-modal prediction
3. Router-based expert selection
4. Classification head for mode selection
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from navsim.common.enums import StateSE2Index
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.modules.blocks import (
    linear_relu_ln,
    bias_init_with_prob,
    gen_sineembed_for_position,
    GridSampleCrossBEVAttention,
)
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer


class MoETrajectoryExpert(nn.Module):
    """
    A single expert that generates one trajectory mode.
    
    Each expert consists of:
    - Cross-attention with BEV features
    - Cross-attention with agent queries
    - Cross-attention with ego query
    - Feed-forward network
    - Trajectory regression head
    """
    
    def __init__(
        self,
        num_poses: int,
        d_model: int,
        d_ffn: int,
        config: TransfuserConfig,
    ):
        super().__init__()
        self.num_poses = num_poses
        self.d_model = d_model
        
        # Cross-attention layers
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Dropout(config.tf_dropout),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        
        # Trajectory prediction head
        self.trajectory_reg_head = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Dropout(config.tf_dropout),
            nn.Linear(config.tf_d_ffn, num_poses * 3),  # (x, y, heading) for each pose
        )
        
        # Classification head for mode confidence
        self.trajectory_cls_head = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Dropout(config.tf_dropout),
            nn.Linear(config.tf_d_ffn, 1),
        )
        
        self.dropout = nn.Dropout(config.tf_dropout)
        
    def forward(
        self,
        traj_feature: torch.Tensor,
        traj_points: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        agents_query: torch.Tensor,
        ego_query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            traj_feature: (B, num_modes, num_poses, d_model) or (B, num_modes, d_model) - trajectory query features
            traj_points: (B, num_modes, num_poses, 2) - anchor points for this expert
            bev_feature: (B, C, H, W) - BEV features
            bev_spatial_shape: (H, W) - spatial shape
            agents_query: (B, num_agents, d_model) - agent queries
            ego_query: (B, 1, d_model) - ego query
            
        Returns:
            poses_reg: (B, num_modes, num_poses, 3) - predicted trajectory
            poses_cls: (B, num_modes) - mode confidence scores
        """
        bs, num_modes = traj_points.shape[:2]
        
        # Handle different input shapes for traj_feature
        if len(traj_feature.shape) == 4:
            # (B, num_modes, num_poses, d_model) -> (B, num_modes, d_model) by pooling
            traj_feature = traj_feature.mean(dim=2)  # (B, num_modes, d_model)
        elif len(traj_feature.shape) == 3:
            # (B, num_modes, d_model) - already correct
            pass
        else:
            raise ValueError(f"Unexpected traj_feature shape: {traj_feature.shape}")
        
        # GridSampleCrossBEVAttention expects:
        # - queries: (bs, num_queries, embed_dims) = (B, num_modes, d_model)
        # - traj_points: (bs, num_queries, num_points, 2) = (B, num_modes, num_poses, 2)
        # So we can pass them directly!
        
        # 1. Cross-attention with BEV features
        traj_feature = self.cross_bev_attention(
            traj_feature,  # (B, num_modes, d_model)
            traj_points,   # (B, num_modes, num_poses, 2)
            bev_feature,
            bev_spatial_shape,
        )
        traj_feature = self.norm1(traj_feature)  # (B, num_modes, d_model)
        
        # Expand traj_feature to (B, num_modes, num_poses, d_model) for subsequent processing
        traj_feature = traj_feature.unsqueeze(2).expand(bs, num_modes, self.num_poses, self.d_model)
        
        # Flatten for subsequent processing: (B*num_modes, num_poses, d_model)
        traj_feature_flat = traj_feature.view(bs * num_modes, self.num_poses, self.d_model)
        
        # 2. Cross-attention with agent queries
        # Expand agents_query for each mode
        agents_expanded = agents_query.unsqueeze(1).expand(bs, num_modes, -1, -1)
        agents_expanded = agents_expanded.reshape(bs * num_modes, -1, self.d_model)
        
        ca_out, _ = self.cross_agent_attention(
            traj_feature_flat, agents_expanded, agents_expanded
        )
        traj_feature_flat = traj_feature_flat + self.dropout(ca_out)
        traj_feature_flat = self.norm2(traj_feature_flat)
        
        # 3. Cross-attention with ego query
        ego_expanded = ego_query.unsqueeze(1).expand(bs, num_modes, -1, -1)
        ego_expanded = ego_expanded.reshape(bs * num_modes, -1, self.d_model)
        
        ego_out, _ = self.cross_ego_attention(
            traj_feature_flat, ego_expanded, ego_expanded
        )
        traj_feature_flat = traj_feature_flat + self.dropout(ego_out)
        traj_feature_flat = self.norm3(traj_feature_flat)
        
        # 4. Feed-forward network
        ffn_out = self.ffn(traj_feature_flat)
        traj_feature_flat = traj_feature_flat + self.dropout(ffn_out)
        # Note: We use norm3 for both ego attention and FFN, which is fine
        # Alternatively, we could add a separate norm4 for FFN
        
        # 5. Pool trajectory features (mean pooling over poses)
        traj_feature_pooled = traj_feature_flat.mean(dim=1)  # (B*num_modes, d_model)
        
        # 6. Predict trajectory and confidence
        poses_reg_flat = self.trajectory_reg_head(traj_feature_pooled)  # (B*num_modes, num_poses*3)
        poses_reg_flat = poses_reg_flat.view(bs * num_modes, self.num_poses, 3)
        
        # Add anchor points to regression output
        traj_points_flat = traj_points.view(bs * num_modes, self.num_poses, 2)
        poses_reg_flat[..., :2] = poses_reg_flat[..., :2] + traj_points_flat
        poses_reg_flat[..., StateSE2Index.HEADING] = poses_reg_flat[..., StateSE2Index.HEADING].tanh() * np.pi
        
        poses_cls_flat = self.trajectory_cls_head(traj_feature_pooled).squeeze(-1)  # (B*num_modes,)
        
        # Reshape back
        poses_reg = poses_reg_flat.view(bs, num_modes, self.num_poses, 3)
        poses_cls = poses_cls_flat.view(bs, num_modes)
        
        return poses_reg, poses_cls


class MoETrajectoryHead(nn.Module):
    """
    MOE-based trajectory prediction head.
    
    This replaces the diffusion-based trajectory head with a Mixture-of-Experts approach.
    Multiple experts generate different trajectory modes, and a router selects the most
    relevant experts based on the input context.
    """
    
    def __init__(
        self,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        plan_anchor_path: str,
        config: TransfuserConfig,
        num_experts: int = 4,
        top_k: int = 2,
        num_modes: int = 20,
    ):
        """
        Args:
            num_poses: number of trajectory poses to predict
            d_ffn: feed-forward network dimension
            d_model: model dimension
            plan_anchor_path: path to plan anchor file
            config: TransfuserConfig
            num_experts: number of MOE experts
            top_k: number of experts to use per sample
            num_modes: number of trajectory modes to generate
        """
        super().__init__()
        
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_modes = num_modes
        
        # Load plan anchors (used as initialization for trajectory modes)
        plan_anchor = np.load(plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )  # (num_modes, num_poses, 2)
        
        # Encode plan anchors to features
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512),
            nn.Linear(d_model, d_model),
        )
        
        # Router: selects which experts to use
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.router_temperature = 1.0
        
        # MOE Experts: each expert generates trajectories
        self.experts = nn.ModuleList([
            MoETrajectoryExpert(
                num_poses=num_poses,
                d_model=d_model,
                d_ffn=d_ffn,
                config=config,
            )
            for _ in range(num_experts)
        ])
        
        # Loss computer
        self.loss_computer = LossComputer(config)
        
        # Load balance loss coefficient
        self.load_balance_coef = 0.01
        
    def forward(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        status_encoding: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        global_img: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ego_query: (B, 1, d_model) - ego query
            agents_query: (B, num_agents, d_model) - agent queries
            bev_feature: (B, C, H, W) - BEV features
            bev_spatial_shape: (H, W) - spatial shape
            status_encoding: (B, 1, d_model) - status encoding
            targets: optional targets for training
            global_img: optional global image features
            
        Returns:
            Dictionary containing:
            - trajectory: (B, num_poses, 3) - best trajectory
            - trajectory_loss: scalar - training loss (if training)
            - trajectory_loss_dict: dict - detailed losses (if training)
            - moe_aux_loss: scalar - MOE auxiliary loss
            - moe_usage_fraction: (num_experts,) - expert usage statistics
        """
        if self.training:
            return self.forward_train(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, targets, global_img
            )
        else:
            return self.forward_test(
                ego_query, agents_query, bev_feature, bev_spatial_shape,
                status_encoding, global_img
            )
    
    def forward_train(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        status_encoding: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        global_img: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass."""
        bs = ego_query.shape[0]
        device = ego_query.device
        
        # 1. Initialize trajectory features from plan anchors
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)  # (B, num_modes, num_poses, 2)
        
        # Encode anchor points to features
        traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)  # (B, num_modes, num_poses*embed_dim)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)  # (B, num_modes, num_poses*embed_dim, d_model)
        traj_feature = traj_feature.view(bs, self.num_modes, -1, self._d_model)  # (B, num_modes, num_poses, d_model)
        # Pool over poses to get mode-level features
        traj_feature_mode = traj_feature.mean(dim=2)  # (B, num_modes, d_model)
        
        # 2. Router: select experts based on ego query
        router_input = ego_query.squeeze(1)  # (B, d_model)
        router_logits = self.router(router_input)  # (B, num_experts)
        
        if self.router_temperature != 1.0:
            router_logits = router_logits / max(self.router_temperature, 1e-6)
        
        # Select top-k experts
        topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)  # (B, top_k)
        topk_w = F.softmax(topk_vals, dim=-1)  # (B, top_k)
        
        # 3. Generate trajectories from selected experts
        all_poses_reg = []
        all_poses_cls = []
        
        for expert_id, expert in enumerate(self.experts):
            # Check which samples use this expert
            expert_mask = (topk_idx == expert_id).any(dim=1)  # (B,)
            if not expert_mask.any():
                # Create dummy outputs for this expert
                dummy_reg = torch.zeros(bs, self.num_modes, self._num_poses, 3, device=device)
                dummy_cls = torch.zeros(bs, self.num_modes, device=device)
                all_poses_reg.append(dummy_reg)
                all_poses_cls.append(dummy_cls)
                continue
            
            # Get expert weight for each sample
            expert_weights = torch.zeros(bs, device=device)
            for b_idx in range(bs):
                if expert_mask[b_idx]:
                    kth = (topk_idx[b_idx] == expert_id).nonzero(as_tuple=True)[0]
                    if len(kth) > 0:
                        expert_weights[b_idx] = topk_w[b_idx, kth[0]]
            
            # Forward through expert
            expert_reg, expert_cls = expert(
                traj_feature_mode,
                plan_anchor,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
            )
            
            # Weight expert outputs
            expert_reg = expert_reg * expert_weights.view(bs, 1, 1, 1)
            expert_cls = expert_cls * expert_weights.view(bs, 1)
            
            all_poses_reg.append(expert_reg)
            all_poses_cls.append(expert_cls)
        
        # 4. Combine expert outputs
        poses_reg = sum(all_poses_reg)  # (B, num_modes, num_poses, 3)
        poses_cls = sum(all_poses_cls)  # (B, num_modes)
        
        # 5. Compute losses
        trajectory_loss_dict = {}
        trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor)
        trajectory_loss_dict["trajectory_loss"] = trajectory_loss
        
        # 6. MOE auxiliary losses
        # Load balance loss
        probs = F.softmax(router_logits, dim=-1)  # (B, num_experts)
        importance = probs.mean(dim=0)  # (num_experts,)
        load_balance_loss = self.num_experts * torch.sum(importance * importance) * self.load_balance_coef
        
        # Expert usage statistics
        usage_counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).float()
        usage_fraction = usage_counts / usage_counts.sum().clamp_min(1.0)
        
        # 7. Select best trajectory mode
        mode_idx = poses_cls.argmax(dim=-1)  # (B,)
        mode_idx = mode_idx[..., None, None, None].expand(-1, -1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)  # (B, num_poses, 3)
        
        return {
            "trajectory": best_reg,
            "trajectory_loss": trajectory_loss,
            "trajectory_loss_dict": trajectory_loss_dict,
            "moe_aux_loss": load_balance_loss,
            "moe_load_balance_loss": load_balance_loss,
            "moe_usage_fraction": usage_fraction,
            "moe_usage_counts": usage_counts,
        }
    
    def forward_test(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: Tuple[int, int],
        status_encoding: torch.Tensor,
        global_img: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference forward pass."""
        bs = ego_query.shape[0]
        device = ego_query.device
        
        # 1. Initialize trajectory features from plan anchors
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        
        traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, self.num_modes, -1, self._d_model)
        traj_feature_mode = traj_feature.mean(dim=2)  # (B, num_modes, d_model)
        
        # 2. Router: select experts
        router_input = ego_query.squeeze(1)
        router_logits = self.router(router_input)
        
        if self.router_temperature != 1.0:
            router_logits = router_logits / max(self.router_temperature, 1e-6)
        
        topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_w = F.softmax(topk_vals, dim=-1)
        
        # 3. Generate trajectories from selected experts
        all_poses_reg = []
        all_poses_cls = []
        
        for expert_id, expert in enumerate(self.experts):
            expert_mask = (topk_idx == expert_id).any(dim=1)
            if not expert_mask.any():
                dummy_reg = torch.zeros(bs, self.num_modes, self._num_poses, 3, device=device)
                dummy_cls = torch.zeros(bs, self.num_modes, device=device)
                all_poses_reg.append(dummy_reg)
                all_poses_cls.append(dummy_cls)
                continue
            
            expert_weights = torch.zeros(bs, device=device)
            for b_idx in range(bs):
                if expert_mask[b_idx]:
                    kth = (topk_idx[b_idx] == expert_id).nonzero(as_tuple=True)[0]
                    if len(kth) > 0:
                        expert_weights[b_idx] = topk_w[b_idx, kth[0]]
            
            expert_reg, expert_cls = expert(
                traj_feature_mode,
                plan_anchor,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
            )
            
            expert_reg = expert_reg * expert_weights.view(bs, 1, 1, 1)
            expert_cls = expert_cls * expert_weights.view(bs, 1)
            
            all_poses_reg.append(expert_reg)
            all_poses_cls.append(expert_cls)
        
        # 4. Combine expert outputs
        poses_reg = sum(all_poses_reg)
        poses_cls = sum(all_poses_cls)
        
        # 5. Select best trajectory mode
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].expand(-1, -1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        
        return {
            "trajectory": best_reg,
        }

