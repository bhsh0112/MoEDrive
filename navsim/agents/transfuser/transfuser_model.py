from typing import Dict
import numpy as np
import torch
import torch.nn as nn

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from navsim.agents.moe_transformer_decoder import MoEConfig, MoELayerwiseTransformerDecoder


class TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        # Select decoder type based on config
        self._use_moe_decoder = getattr(config, "use_moe_decoder", False)
        
        if self._use_moe_decoder:
            # MoE-based decoder: route between *full decoder layers* (self-attn + cross-attn + FFN) as experts.
            # We keep the same I/O contract for downstream heads: (B, Q, D) -> (B, Q, D).
            multimodal_mode = getattr(config, "multimodal_trajectory", False)
            moe_cfg = MoEConfig(
                num_experts=getattr(config, "moe_num_experts", 4),
                top_k=getattr(config, "moe_top_k", 2),
                router_temperature=getattr(config, "moe_router_temperature", 1.0),
                router_z_loss_coef=getattr(config, "moe_router_z_loss_coef", 0.0),
                load_balance_coef=getattr(config, "moe_load_balance_coef", 0.0),
                multimodal_mode=multimodal_mode,
                trajectory_query_idx=0,  # trajectory_query is always the first query token
            )
            self._tf_decoder = MoELayerwiseTransformerDecoder(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                num_layers=config.tf_num_layers,
                moe_cfg=moe_cfg,
            )
        else:
            # Vanilla TransformerDecoder (original implementation)
            tf_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )
            self._tf_decoder = nn.TransformerDecoder(
                decoder_layer=tf_decoder_layer,
                num_layers=config.tf_num_layers,
            )
        
        # Store multimodal flag for use in forward pass
        self._multimodal_trajectory = getattr(config, "multimodal_trajectory", False)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            multimodal_mode=self._multimodal_trajectory,
            num_modes=getattr(config, "num_trajectory_modes", 20) if self._multimodal_trajectory else 1,
        )

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        
        # Handle different decoder output formats
        if self._use_moe_decoder:
            # MoE decoder returns (output, aux)
            query_out, moe_aux = self._tf_decoder(query, keyval)
        else:
            # Vanilla decoder returns only output
            query_out = self._tf_decoder(query, keyval)
            moe_aux = None

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        # Expose MoE辅助损失和路由统计信息（仅MoE场景）
        if self._use_moe_decoder and moe_aux is not None:
            output.update(
                {
                    "moe_aux_loss": moe_aux.get("moe_aux_loss"),
                    "moe_load_balance_loss": moe_aux.get("moe_load_balance_loss"),
                    "moe_router_z_loss": moe_aux.get("moe_router_z_loss"),
                    "moe_usage_fraction": moe_aux.get("moe_usage_fraction"),
                    "moe_usage_counts": moe_aux.get("moe_usage_counts"),
                }
            )

        # 多模态：优先使用专家独立输出的轨迹query
        if self._multimodal_trajectory and moe_aux is not None:
            multimodal_expert_outputs = moe_aux.get("multimodal_expert_outputs")
            if multimodal_expert_outputs is not None:
                # multimodal_expert_outputs: (B, num_experts, 1, D) -> (B, num_experts, D)
                multimodal_trajectory_queries = multimodal_expert_outputs.squeeze(2)
                trajectory_dict = self._trajectory_head(multimodal_trajectory_queries)

                trajectory_all_modes = trajectory_dict["trajectory"]  # (B, num_modes, num_poses, 3)
                trajectory_best = trajectory_dict.get("trajectory_best")
                trajectory_mode_scores = trajectory_dict.get("trajectory_mode_scores")

                # 主输出：best 轨迹，兼容旧接口
                output["trajectory"] = trajectory_best if trajectory_best is not None else trajectory_all_modes[:, 0]
                # 附加多模态输出
                output["trajectory_modes"] = trajectory_all_modes
                if trajectory_mode_scores is not None:
                    output["trajectory_mode_scores"] = trajectory_mode_scores
            else:
                # 没有多模态专家输出时，退化为单模态
                single_traj = self._trajectory_head(trajectory_query)
                output.update(single_traj)
        else:
            # 单模态
            single_traj = self._trajectory_head(trajectory_query)
            output.update(single_traj)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    """
    Trajectory prediction head.
    
    Supports both single-modal and multi-modal trajectory prediction.
    In multi-modal mode, each expert query generates a different trajectory mode.
    """

    def __init__(
        self, 
        num_poses: int, 
        d_ffn: int, 
        d_model: int,
        multimodal_mode: bool = False,
        num_modes: int = 1,
    ):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        :param multimodal_mode: if True, enable multi-modal trajectory prediction
        :param num_modes: number of trajectory modes (used only in multimodal_mode)
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._multimodal_mode = multimodal_mode
        self._num_modes = num_modes

        # Trajectory regression head: predicts (x, y, heading) for each pose
        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )
        
        # Mode classification head: predicts confidence score for each mode (only in multimodal mode)
        if self._multimodal_mode:
            self._mode_cls_head = nn.Sequential(
                nn.Linear(self._d_model, self._d_ffn),
                nn.ReLU(),
                nn.Linear(self._d_ffn, 1),  # Single scalar confidence score per mode
            )
        else:
            self._mode_cls_head = None

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        """
        Torch module forward pass.
        
        Args:
            object_queries: 
                - Single-modal: (B, 1, D) or (B, D)
                - Multi-modal: (B, num_modes, D)
        
        Returns:
            Dictionary containing:
            - trajectory: 
                - Single-modal: (B, num_poses, 3)
                - Multi-modal: (B, num_modes, num_poses, 3)
            - trajectory_mode_scores (only in multimodal_mode): (B, num_modes) - confidence scores
            - trajectory_best (only in multimodal_mode): (B, num_poses, 3) - best trajectory based on mode scores
        """
        # Handle input shape: normalize to (B, num_queries, D)
        if object_queries.dim() == 2:
            # Input is (B, D), add sequence dimension
            object_queries = object_queries.unsqueeze(1)  # (B, 1, D)
        
        batch_size = object_queries.shape[0]
        num_queries = object_queries.shape[1]
        
        # Reshape for batch processing: (B * num_queries, D)
        object_queries_flat = object_queries.view(batch_size * num_queries, self._d_model)
        
        # Predict trajectories: (B * num_queries, num_poses * 3)
        poses_flat = self._mlp(object_queries_flat)
        poses_flat = poses_flat.view(batch_size * num_queries, self._num_poses, StateSE2Index.size())
        
        # Apply heading normalization
        poses_flat[..., StateSE2Index.HEADING] = poses_flat[..., StateSE2Index.HEADING].tanh() * np.pi
        
        # Reshape back: (B, num_queries, num_poses, 3)
        poses = poses_flat.view(batch_size, num_queries, self._num_poses, StateSE2Index.size())
        
        # Handle output based on mode
        if self._multimodal_mode:
            # Multi-modal mode: output all modes
            # poses shape: (B, num_modes, num_poses, 3)
            
            # Predict mode confidence scores
            mode_scores_flat = self._mode_cls_head(object_queries_flat).squeeze(-1)  # (B * num_modes,)
            mode_scores = mode_scores_flat.view(batch_size, num_queries)  # (B, num_modes)
            
            # Select best trajectory based on mode scores
            best_mode_idx = mode_scores.argmax(dim=1)  # (B,)
            # Use advanced indexing to select best trajectory for each batch element
            batch_indices = torch.arange(batch_size, device=poses.device)
            trajectory_best = poses[batch_indices, best_mode_idx]  # (B, num_poses, 3)
            
            return {
                "trajectory": poses,  # (B, num_modes, num_poses, 3)
                "trajectory_mode_scores": mode_scores,  # (B, num_modes)
                "trajectory_best": trajectory_best,  # (B, num_poses, 3) - for backward compatibility
            }
        else:
            # Single-modal mode: output single trajectory
            # poses shape: (B, 1, num_poses, 3) -> (B, num_poses, 3)
            trajectory = poses.squeeze(1)  # (B, num_poses, 3)
            return {"trajectory": trajectory}
