"""
MoE staged-training scheduler for Transfuser.

This module implements a small, framework-agnostic scheduler that computes
epoch-dependent MoE routing parameters and loss weights according to a
3-stage training policy (基础训练/专业化训练/精细化训练).

Design goals:
- Pure python (no torch dependency), safe to import anywhere.
- Backward compatible: if not enabled, caller can ignore it.
- Simple schedule format: a dict that can be overridden from Hydra/YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class StageInfo:
    """Training stage metadata."""

    name: str
    start_epoch: int
    end_epoch: int  # exclusive

    def contains(self, epoch: int) -> bool:
        """Return True if epoch is within [start_epoch, end_epoch)."""
        return self.start_epoch <= epoch < self.end_epoch


def _clamp_int(x: int, low: int, high: int) -> int:
    return max(low, min(high, x))


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b with t in [0, 1]."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return a + (b - a) * t


def _stage_progress(epoch: int, start: int, end: int) -> float:
    """Return progress ratio in [0, 1] for epoch inside [start, end)."""
    if end <= start:
        return 1.0
    # Use (epoch - start) / (end - start) so that start -> 0, last epoch before end -> almost 1.
    return (epoch - start) / float(end - start)


class MoETrainingScheduler:
    """
    MoE 训练策略调度器（3 阶段）。

    The scheduler returns a dict of parameters for the current epoch:
    - router_temperature
    - top_k
    - load_balance_coef
    - moe_aux_loss_weight
    - trajectory_position_weight / trajectory_heading_weight / trajectory_mode_weight
    - expert_diversity_weight (placeholder for later loss integration)

    It also supports an optional adaptive adjustment based on expert usage
    (e.g., revive dead experts, reduce over-concentration).
    """

    def __init__(
        self,
        *,
        num_experts: int,
        schedule: Dict[str, Any],
        enable_adaptive: bool = True,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        self._num_experts = int(num_experts)
        self._schedule = schedule
        self._enable_adaptive = bool(enable_adaptive)
        self._usage_history: List[Dict[str, Any]] = []

        self._stages = self._parse_stages(schedule)

    @property
    def stages(self) -> List[StageInfo]:
        """Ordered list of stage definitions."""
        return list(self._stages)

    @property
    def expert_usage_history(self) -> List[Dict[str, Any]]:
        """Epoch-level expert usage snapshots recorded via update_expert_usage()."""
        return self._usage_history

    def update_expert_usage(self, *, epoch: int, usage_fraction: List[float]) -> None:
        """
        Record expert usage fraction for later analysis/visualization.

        Args:
            epoch: current epoch.
            usage_fraction: list of length num_experts, sums to ~1.
        """
        self._usage_history.append({"epoch": int(epoch), "usage_fraction": list(usage_fraction)})

    def get_current_stage(self, epoch: int) -> StageInfo:
        """Return the stage for this epoch (clamped to first/last stage if out of range)."""
        if not self._stages:
            raise ValueError("No stages configured.")
        for s in self._stages:
            if s.contains(epoch):
                return s
        # Clamp
        if epoch < self._stages[0].start_epoch:
            return self._stages[0]
        return self._stages[-1]

    def get_current_params(
        self, *, epoch: int, expert_usage_fraction: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Compute the staged-training params for `epoch`.

        Args:
            epoch: Current epoch (0-based).
            expert_usage_fraction: Optional per-expert usage fractions to enable adaptive adjustments.

        Returns:
            A dict with dynamic parameters. Callers decide how to apply them.
        """
        stage = self.get_current_stage(epoch)
        stage_cfg = self._schedule["stages"][stage.name]
        t = _stage_progress(epoch, stage.start_epoch, stage.end_epoch)

        # Route params
        router_temperature = _lerp(
            float(stage_cfg["router_temperature"][0]),
            float(stage_cfg["router_temperature"][1]),
            t,
        )
        top_k = int(round(_lerp(float(stage_cfg["top_k"][0]), float(stage_cfg["top_k"][1]), t)))
        top_k = _clamp_int(top_k, 1, self._num_experts)

        load_balance_coef = _lerp(
            float(stage_cfg["load_balance_coef"][0]),
            float(stage_cfg["load_balance_coef"][1]),
            t,
        )
        moe_aux_loss_weight = _lerp(
            float(stage_cfg["moe_aux_loss_weight"][0]),
            float(stage_cfg["moe_aux_loss_weight"][1]),
            t,
        )

        # Loss weights
        trajectory_position_weight = _lerp(
            float(stage_cfg["trajectory_position_weight"][0]),
            float(stage_cfg["trajectory_position_weight"][1]),
            t,
        )
        trajectory_heading_weight = _lerp(
            float(stage_cfg["trajectory_heading_weight"][0]),
            float(stage_cfg["trajectory_heading_weight"][1]),
            t,
        )
        trajectory_mode_weight = _lerp(
            float(stage_cfg["trajectory_mode_weight"][0]),
            float(stage_cfg["trajectory_mode_weight"][1]),
            t,
        )

        expert_diversity_weight = _lerp(
            float(stage_cfg.get("expert_diversity_weight", (0.0, 0.0))[0]),
            float(stage_cfg.get("expert_diversity_weight", (0.0, 0.0))[1]),
            t,
        )

        params: Dict[str, Any] = {
            "stage_name": stage.name,
            "router_temperature": float(router_temperature),
            "top_k": int(top_k),
            "load_balance_coef": float(load_balance_coef),
            "moe_aux_loss_weight": float(moe_aux_loss_weight),
            "trajectory_position_weight": float(trajectory_position_weight),
            "trajectory_heading_weight": float(trajectory_heading_weight),
            "trajectory_mode_weight": float(trajectory_mode_weight),
            "expert_diversity_weight": float(expert_diversity_weight),
        }

        # Optional: dynamic Top-K based on expert usage (typically enabled for stage3).
        if expert_usage_fraction is not None:
            params = self._maybe_apply_dynamic_top_k(
                epoch=epoch, stage_name=stage.name, params=params, usage_fraction=list(expert_usage_fraction)
            )

        if self._enable_adaptive and expert_usage_fraction is not None:
            params = self._apply_adaptive_adjustment(
                epoch=epoch, params=params, usage_fraction=list(expert_usage_fraction)
            )

        return params

    @staticmethod
    def build_default_three_stage_schedule(*, num_experts: int) -> Dict[str, Any]:
        """
        Build a default 3-stage schedule matching the design doc.

        Notes:
        - Epoch boundaries follow: [0, 30), [30, 100), [100, 150)
        - top_k is automatically clamped by callers using num_experts.
        """
        # Default values follow `MOE_TRAINING_STRATEGY_DESIGN.md`.
        return {
            "stages_order": ["stage1", "stage2", "stage3"],
            "stages": {
                "stage1": {
                    "epoch_range": [0, 30],
                    "router_temperature": [2.5, 1.5],
                    "top_k": [min(num_experts, 20), min(num_experts, 15)],
                    "load_balance_coef": [1e-2, 1e-2],
                    "moe_aux_loss_weight": [0.5, 0.5],
                    "expert_diversity_weight": [0.0, 0.0],
                    "trajectory_position_weight": [1.0, 1.2],
                    "trajectory_heading_weight": [1.5, 1.8],
                    "trajectory_mode_weight": [1.5, 2.0],
                },
                "stage2": {
                    "epoch_range": [30, 100],
                    "router_temperature": [1.5, 1.0],
                    "top_k": [min(num_experts, 15), min(num_experts, 10)],
                    "load_balance_coef": [1e-2, 5e-3],
                    "moe_aux_loss_weight": [0.5, 0.3],
                    "expert_diversity_weight": [0.0, 0.1],
                    "trajectory_position_weight": [1.2, 1.0],
                    "trajectory_heading_weight": [1.8, 1.5],
                    "trajectory_mode_weight": [2.0, 2.5],
                },
                "stage3": {
                    "epoch_range": [100, 150],
                    "router_temperature": [1.0, 1.0],
                    # Here we keep [10, 10] as default; dynamic top-k is implemented later.
                    "top_k": [min(num_experts, 10), min(num_experts, 10)],
                    "load_balance_coef": [5e-3, 1e-3],
                    "moe_aux_loss_weight": [0.3, 0.1],
                    "expert_diversity_weight": [0.1, 0.1],
                    "trajectory_position_weight": [1.0, 1.0],
                    "trajectory_heading_weight": [1.5, 1.5],
                    "trajectory_mode_weight": [2.5, 2.5],
                },
            },
            "adaptive": {
                "dead_expert_threshold": 0.02,
                "overuse_threshold": 0.30,
                "min_top_k": 1,
                "max_top_k": int(num_experts),
                "load_balance_boost": 1.2,
                "temperature_boost": 1.1,
                "top_k_boost": 1,
            },
            "dynamic_top_k": {
                # Enable only in stage3 by default
                "enabled": True,
                "stages": ["stage3"],
                "active_threshold": 0.05,
                "min_k": 8,
                "max_k": 12,
            },
        }

    def _parse_stages(self, schedule: Dict[str, Any]) -> List[StageInfo]:
        if "stages" not in schedule:
            raise ValueError("schedule must contain 'stages'")
        stages_cfg = schedule["stages"]
        order = schedule.get("stages_order")
        if order is None:
            # Stable default: lexical order of keys.
            order = sorted(list(stages_cfg.keys()))

        stages: List[StageInfo] = []
        for name in order:
            if name not in stages_cfg:
                raise ValueError(f"stages_order references missing stage: {name}")
            r = stages_cfg[name].get("epoch_range")
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                raise ValueError(f"stage '{name}' must have epoch_range [start, end]")
            start, end = int(r[0]), int(r[1])
            if end <= start:
                raise ValueError(f"stage '{name}' epoch_range end must be > start")
            stages.append(StageInfo(name=str(name), start_epoch=start, end_epoch=end))

        # Ensure non-decreasing ranges
        stages_sorted = sorted(stages, key=lambda s: s.start_epoch)
        return stages_sorted

    def _apply_adaptive_adjustment(
        self, *, epoch: int, params: Dict[str, Any], usage_fraction: List[float]
    ) -> Dict[str, Any]:
        """
        Apply simple adaptive rules to avoid expert death and over-concentration.

        This implements the spirit of the design doc's pseudo-code, but keeps it minimal:
        - If some experts are "dead": increase load balance, temperature, and top_k.
        - If one expert dominates: slightly increase temperature and load balance.
        """
        adaptive = self._schedule.get("adaptive", {})
        dead_thr = float(adaptive.get("dead_expert_threshold", 0.02))
        overuse_thr = float(adaptive.get("overuse_threshold", 0.30))
        min_top_k = int(adaptive.get("min_top_k", 1))
        max_top_k = int(adaptive.get("max_top_k", self._num_experts))

        lb_boost = float(adaptive.get("load_balance_boost", 1.2))
        temp_boost = float(adaptive.get("temperature_boost", 1.1))
        topk_boost = int(adaptive.get("top_k_boost", 1))

        # Basic sanity
        if len(usage_fraction) != self._num_experts:
            return params

        dead_experts = sum(1 for u in usage_fraction if float(u) < dead_thr)
        max_usage = max(float(u) for u in usage_fraction) if usage_fraction else 0.0

        out = dict(params)
        out["adaptive_epoch"] = int(epoch)
        out["dead_experts"] = int(dead_experts)
        out["max_usage"] = float(max_usage)

        if dead_experts > 0:
            out["load_balance_coef"] = float(out["load_balance_coef"]) * lb_boost
            out["router_temperature"] = float(out["router_temperature"]) * temp_boost
            out["top_k"] = _clamp_int(int(out["top_k"]) + topk_boost, min_top_k, min(max_top_k, self._num_experts))

        if max_usage > overuse_thr:
            out["load_balance_coef"] = float(out["load_balance_coef"]) * (lb_boost ** 0.5)
            out["router_temperature"] = float(out["router_temperature"]) * (temp_boost ** 0.5)

        # Always clamp
        out["top_k"] = _clamp_int(int(out["top_k"]), 1, self._num_experts)
        out["router_temperature"] = max(float(out["router_temperature"]), 1e-6)
        out["load_balance_coef"] = max(float(out["load_balance_coef"]), 0.0)
        out["moe_aux_loss_weight"] = max(float(out["moe_aux_loss_weight"]), 0.0)
        return out

    def _maybe_apply_dynamic_top_k(
        self, *, epoch: int, stage_name: str, params: Dict[str, Any], usage_fraction: List[float]
    ) -> Dict[str, Any]:
        """
        Optionally override params["top_k"] based on expert usage.

        Strategy (from design doc):
        - active_experts = count(usage_fraction > active_threshold)
        - top_k = clamp(active_experts, min_k, max_k)

        Notes:
        - This is typically enabled for stage3 to reduce compute and keep routing stable.
        - Adaptive adjustment (dead experts / overuse) is applied *after* this step and may further adjust top_k.
        """
        cfg = self._schedule.get("dynamic_top_k", {})
        if not bool(cfg.get("enabled", False)):
            return params

        stages = cfg.get("stages", ["stage3"])
        if isinstance(stages, (list, tuple)) and stage_name not in set(stages):
            return params

        if len(usage_fraction) != self._num_experts:
            return params

        thr = float(cfg.get("active_threshold", 0.05))
        min_k = int(cfg.get("min_k", 8))
        max_k = int(cfg.get("max_k", 12))

        active_experts = sum(1 for u in usage_fraction if float(u) > thr)
        dynamic_k = _clamp_int(active_experts, min_k, max_k)
        dynamic_k = _clamp_int(dynamic_k, 1, self._num_experts)

        out = dict(params)
        out["top_k"] = int(dynamic_k)
        out["dynamic_top_k_applied"] = True
        out["dynamic_top_k_active_experts"] = int(active_experts)
        out["dynamic_top_k_threshold"] = float(thr)
        out["dynamic_top_k_min_k"] = int(min_k)
        out["dynamic_top_k_max_k"] = int(max_k)
        out["dynamic_top_k_epoch"] = int(epoch)
        return out


