"""
Trajectory evaluation metrics for Transfuser.

This module provides common multi-modal trajectory metrics:
- MinADE: minimum Average Displacement Error over modes
- MinFDE: minimum Final Displacement Error over modes
- Miss Rate: fraction of samples where minFDE exceeds a threshold

All metrics are computed on (x, y) positions only.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def compute_minade_minfde_missrate(
    *,
    gt_trajectory: torch.Tensor,
    pred_trajectory: torch.Tensor,
    miss_threshold_m: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-modal trajectory metrics.

    Args:
        gt_trajectory: (B, T, 3) ground truth trajectory [x, y, heading].
        pred_trajectory:
            - single-modal: (B, T, 3)
            - multi-modal: (B, M, T, 3)
        miss_threshold_m: Miss threshold in meters applied to minFDE.

    Returns:
        dict:
            - minADE: scalar tensor
            - minFDE: scalar tensor
            - miss_rate: scalar tensor in [0, 1]
    """
    if gt_trajectory.ndim != 3:
        raise ValueError("gt_trajectory must be (B, T, 3)")

    b, t, _ = gt_trajectory.shape
    gt_xy = gt_trajectory[..., :2]  # (B, T, 2)

    if pred_trajectory.ndim == 3:
        # (B, T, 3) -> (B, 1, T, 3)
        pred_trajectory = pred_trajectory.unsqueeze(1)
    if pred_trajectory.ndim != 4:
        raise ValueError("pred_trajectory must be (B, T, 3) or (B, M, T, 3)")

    pred_xy = pred_trajectory[..., :2]  # (B, M, T, 2)
    if pred_xy.shape[0] != b or pred_xy.shape[2] != t:
        raise ValueError("pred_trajectory shape must match gt_trajectory on batch/time dims")

    # L2 distances in meters on xy
    # disp: (B, M, T)
    disp = torch.norm(pred_xy - gt_xy.unsqueeze(1), p=2, dim=-1)
    ade = disp.mean(dim=-1)  # (B, M)
    fde = disp[..., -1]  # (B, M)

    min_ade, _ = ade.min(dim=1)  # (B,)
    min_fde, _ = fde.min(dim=1)  # (B,)

    miss_thr = float(miss_threshold_m)
    miss_rate = (min_fde > miss_thr).to(gt_trajectory.dtype).mean()

    return {
        "minADE": min_ade.mean(),
        "minFDE": min_fde.mean(),
        "miss_rate": miss_rate,
    }


