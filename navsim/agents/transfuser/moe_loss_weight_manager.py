"""
Dynamic loss weight manager for MoE staged training.

This module provides a small utility to compute epoch-dependent loss weights
according to the staged training schedule.

It intentionally focuses on "loss weights" (objective balancing), not routing
parameters. Routing parameters are handled by `MoETrainingScheduler`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from navsim.agents.transfuser.moe_training_scheduler import MoETrainingScheduler


@dataclass(frozen=True)
class LossWeights:
    """A bundle of loss weights used by Transfuser."""

    moe_aux_loss_weight: float
    trajectory_position_weight: float
    trajectory_heading_weight: float
    trajectory_mode_weight: float
    expert_diversity_weight: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to a plain dict for logging / config patching."""
        return {
            "moe_aux_loss_weight": float(self.moe_aux_loss_weight),
            "trajectory_position_weight": float(self.trajectory_position_weight),
            "trajectory_heading_weight": float(self.trajectory_heading_weight),
            "trajectory_mode_weight": float(self.trajectory_mode_weight),
            "expert_diversity_weight": float(self.expert_diversity_weight),
        }


class DynamicLossWeightManager:
    """
    Dynamic loss weight manager.

    It reuses the same schedule dict as `MoETrainingScheduler` and simply extracts
    the parts relevant to loss weighting.

    Typical usage:
        mgr = DynamicLossWeightManager(num_experts=config.moe_num_experts, schedule=...)
        weights = mgr.get_weights(epoch, expert_usage_fraction=usage)
        mgr.apply_to_config(config, weights)
    """

    def __init__(
        self,
        *,
        num_experts: int,
        schedule: Optional[Dict[str, Any]] = None,
        enable_adaptive: bool = True,
    ) -> None:
        if schedule is None:
            schedule = MoETrainingScheduler.build_default_three_stage_schedule(num_experts=int(num_experts))

        self._scheduler = MoETrainingScheduler(
            num_experts=int(num_experts),
            schedule=schedule,
            enable_adaptive=bool(enable_adaptive),
        )

    def get_weights(self, *, epoch: int, expert_usage_fraction: Optional[List[float]] = None) -> LossWeights:
        """
        Compute loss weights for the current epoch.

        Args:
            epoch: current epoch (0-based).
            expert_usage_fraction: optional usage to allow schedule adaptive adjustments (if enabled).

        Returns:
            LossWeights object.
        """
        params = self._scheduler.get_current_params(epoch=int(epoch), expert_usage_fraction=expert_usage_fraction)
        return LossWeights(
            moe_aux_loss_weight=float(params.get("moe_aux_loss_weight", 0.0)),
            trajectory_position_weight=float(params.get("trajectory_position_weight", 1.0)),
            trajectory_heading_weight=float(params.get("trajectory_heading_weight", 1.0)),
            trajectory_mode_weight=float(params.get("trajectory_mode_weight", 1.0)),
            expert_diversity_weight=float(params.get("expert_diversity_weight", 0.0)),
        )

    @staticmethod
    def apply_to_config(config: Any, weights: LossWeights) -> None:
        """
        Apply weights to a config-like object (e.g., TransfuserConfig).

        This is best-effort: only sets attributes that exist.
        """
        for k, v in weights.to_dict().items():
            if hasattr(config, k):
                setattr(config, k, float(v))



