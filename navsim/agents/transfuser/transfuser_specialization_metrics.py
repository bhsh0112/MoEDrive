"""
Expert/mode specialization metrics for Transfuser (multi-modal).

These metrics aim to quantify how different the predicted modes are.
They are NOT task metrics (like ADE/FDE), but "diversity/specialization" metrics.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def compute_mode_specialization_metrics(
    *,
    pred_trajectory_modes: torch.Tensor,
    method: str = "pairwise_l2",
) -> Dict[str, torch.Tensor]:
    """
    Compute specialization metrics from predicted multi-modal trajectories.

    Args:
        pred_trajectory_modes: (B, M, T, 3)
        method:
            - "pairwise_l2": mean pairwise L2 distance between flattened trajectories (higher is better)
            - "cosine_similarity": mean pairwise cosine similarity (lower is better)

    Returns:
        dict of scalar tensors:
            - mode_pairwise_l2 (if method == pairwise_l2)
            - mode_pairwise_cosine (if method == cosine_similarity)
    """
    if pred_trajectory_modes.ndim != 4:
        return {}
    b, m, t, d = pred_trajectory_modes.shape
    if m <= 1:
        return {}

    x = pred_trajectory_modes.reshape(b, m, t * d).to(torch.float32)  # (B, M, F)

    # Build pair mask excluding diagonal
    device = x.device
    mask = ~torch.eye(m, device=device, dtype=torch.bool)  # (M, M)

    if method == "cosine_similarity":
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-8)
        cos = torch.matmul(x_norm, x_norm.transpose(1, 2))  # (B, M, M)
        val = cos.masked_select(mask.unsqueeze(0)).view(b, m * (m - 1)).mean()
        return {"mode_pairwise_cosine": val}

    # default: pairwise_l2
    xi = x.unsqueeze(2)  # (B, M, 1, F)
    xj = x.unsqueeze(1)  # (B, 1, M, F)
    dist = torch.norm(xi - xj, p=2, dim=-1)  # (B, M, M)
    val = dist.masked_select(mask.unsqueeze(0)).view(b, m * (m - 1)).mean()
    return {"mode_pairwise_l2": val}


