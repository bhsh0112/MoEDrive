from typing import Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex


def transfuser_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """
    
    # Check if multimodal trajectory prediction is enabled
    multimodal_mode = getattr(config, "multimodal_trajectory", False)
    trajectory_modes = predictions.get("trajectory_modes")
    
    if multimodal_mode and trajectory_modes is not None:
        # Multi-modal trajectory loss
        trajectory_loss, trajectory_mode_loss = _multimodal_trajectory_loss(
            targets["trajectory"], trajectory_modes, predictions.get("trajectory_mode_scores"), config
        )
    else:
        # Single-modal trajectory loss
        trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
        trajectory_mode_loss = None
    
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    bev_semantic_loss = F.cross_entropy(
        predictions["bev_semantic_map"], targets["bev_semantic_map"].long()
    )

    moe_aux_loss = predictions.get("moe_aux_loss")
    if moe_aux_loss is None:
        moe_aux_loss = torch.zeros((), device=trajectory_loss.device, dtype=trajectory_loss.dtype)
    
    # Get trajectory mode classification loss weight (if multimodal)
    trajectory_mode_weight = getattr(config, "trajectory_mode_weight", 1.0)
    trajectory_mode_loss_value = torch.zeros((), device=trajectory_loss.device, dtype=trajectory_loss.dtype)
    if trajectory_mode_loss is not None:
        trajectory_mode_loss_value = trajectory_mode_loss

    loss = (
        config.trajectory_weight * trajectory_loss
        + trajectory_mode_weight * trajectory_mode_loss_value
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
        + config.moe_aux_loss_weight * moe_aux_loss
    )

    # Build a scalar-only loss dict for PyTorch Lightning logging.
    loss_dict: Dict[str, torch.Tensor] = {
        "loss": loss,
        "trajectory_loss": config.trajectory_weight * trajectory_loss,
        "agent_class_loss": config.agent_class_weight * agent_class_loss,
        "agent_box_loss": config.agent_box_weight * agent_box_loss,
        "bev_semantic_loss": config.bev_semantic_weight * bev_semantic_loss,
        "moe_aux_loss": config.moe_aux_loss_weight * moe_aux_loss,
    }
    
    # Add trajectory mode loss if in multimodal mode
    if trajectory_mode_loss is not None:
        loss_dict["trajectory_mode_loss"] = trajectory_mode_weight * trajectory_mode_loss_value

    # Optional MoE components (may be None if model doesn't expose them)
    moe_lb = predictions.get("moe_load_balance_loss")
    if moe_lb is not None:
        loss_dict["moe_load_balance_loss"] = config.moe_aux_loss_weight * moe_lb
    moe_z = predictions.get("moe_router_z_loss")
    if moe_z is not None:
        loss_dict["moe_router_z_loss"] = config.moe_aux_loss_weight * moe_z

    # Log expert usage as per-expert scalars to satisfy `self.log`.
    usage_frac = predictions.get("moe_usage_fraction")
    if usage_frac is not None and usage_frac.ndim == 1:
        for i in range(int(usage_frac.shape[0])):
            loss_dict[f"moe_usage_fraction_e{i}"] = usage_frac[i]

    return loss_dict


def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _multimodal_trajectory_loss(
    gt_trajectory: torch.Tensor,
    pred_trajectory_modes: torch.Tensor,
    pred_mode_scores: Optional[torch.Tensor],
    config: TransfuserConfig,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute multi-modal trajectory prediction loss using Best-of-K strategy.
    
    This function implements the "Best-of-K" approach where:
    1. For each predicted mode, compute distance to ground truth (handling heading properly)
    2. Select the mode with minimum distance (best match)
    3. Compute regression loss only for the best matching mode
    4. Compute classification loss for mode selection (encouraging correct mode prediction)
    
    Args:
        gt_trajectory: Ground truth trajectory, shape (B, num_poses, 3) where last dim is [x, y, heading]
        pred_trajectory_modes: Predicted trajectory modes, shape (B, num_modes, num_poses, 3)
        pred_mode_scores: Predicted mode confidence scores (logits), shape (B, num_modes) or None
        config: TransfuserConfig with loss weights
        
    Returns:
        Tuple containing:
        - trajectory_regression_loss: L1 loss for best matching mode, scalar tensor
        - trajectory_mode_classification_loss: Classification loss for mode selection, scalar tensor or None
    """
    batch_size, num_modes, num_poses, _ = pred_trajectory_modes.shape
    device = pred_trajectory_modes.device
    dtype = pred_trajectory_modes.dtype
    
    # Expand GT trajectory to match pred_trajectory_modes for easy comparison
    gt_trajectory_expanded = gt_trajectory.unsqueeze(1).expand(-1, num_modes, -1, -1)  # (B, num_modes, num_poses, 3)
    
    # Compute distance for each mode
    # For x, y: use L1 distance
    # For heading: use angular distance (handling circular nature of angles)
    pos_diff = torch.abs(pred_trajectory_modes[..., :2] - gt_trajectory_expanded[..., :2])  # (B, num_modes, num_poses, 2)
    pos_distance = pos_diff.sum(dim=-1)  # (B, num_modes, num_poses)
    
    # Angular distance for heading (normalized to [0, pi])
    heading_diff = pred_trajectory_modes[..., 2] - gt_trajectory_expanded[..., 2]  # (B, num_modes, num_poses)
    # Normalize to [-pi, pi]
    heading_diff = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))
    heading_distance = torch.abs(heading_diff)  # (B, num_modes, num_poses)
    
    # Combine position and heading distances with optional weights
    # Default: equal weight for position and heading
    position_weight = getattr(config, "trajectory_position_weight", 1.0)
    heading_weight = getattr(config, "trajectory_heading_weight", 1.0)
    
    combined_distance = position_weight * pos_distance + heading_weight * heading_distance  # (B, num_modes, num_poses)
    
    # Average over poses to get per-mode distance: (B, num_modes)
    per_mode_distance = combined_distance.mean(dim=2)  # (B, num_modes)
    
    # Find best matching mode (mode with minimum distance) for each sample
    best_mode_idx = per_mode_distance.argmin(dim=1)  # (B,)
    
    # Extract best trajectory for each sample: (B, num_poses, 3)
    batch_indices = torch.arange(batch_size, device=device)
    best_trajectory = pred_trajectory_modes[batch_indices, best_mode_idx]  # (B, num_poses, 3)
    
    # Compute regression loss for best matching mode
    # Position loss (L1)
    pos_loss = F.l1_loss(best_trajectory[..., :2], gt_trajectory[..., :2], reduction="mean")
    
    # Heading loss (angular distance)
    heading_diff_best = best_trajectory[..., 2] - gt_trajectory[..., 2]  # (B, num_poses)
    heading_diff_best = torch.atan2(torch.sin(heading_diff_best), torch.cos(heading_diff_best))
    heading_loss = torch.abs(heading_diff_best).mean()
    
    # Combined regression loss
    trajectory_regression_loss = position_weight * pos_loss + heading_weight * heading_loss
    
    # Compute mode classification loss
    trajectory_mode_classification_loss = None
    if pred_mode_scores is not None and pred_mode_scores.numel() > 0:
        # Use cross-entropy loss (treating mode selection as multi-class classification)
        # pred_mode_scores: (B, num_modes) - logits
        # best_mode_idx: (B,) - class indices (long tensor)
        best_mode_idx_long = best_mode_idx.long()
        trajectory_mode_classification_loss = F.cross_entropy(
            pred_mode_scores, best_mode_idx_long, reduction="mean"
        )
    
    return trajectory_regression_loss, trajectory_mode_classification_loss
