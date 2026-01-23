"""
PyTorch Lightning callback to apply MoE staged-training schedule.

This callback updates, per epoch:
- MoE routing params (router_temperature, top_k, load_balance_coef) in the decoder
- Loss weights stored in TransfuserConfig (moe_aux_loss_weight, trajectory_*_weight)

It is intentionally lightweight and backward compatible:
- Only active when config.moe_staged_training_enabled is True.
- Only applies to MoE decoder runs (use_moe_decoder=True).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    import pytorch_lightning as pl  # type: ignore
except Exception:  # pragma: no cover
    # Lightweight fallback for environments without Lightning installed.
    class _Callback:  # noqa: D401
        """Fallback base class to allow importing this module without Lightning."""

        pass

    class _PL:  # noqa: D401
        """Fallback namespace matching `pytorch_lightning` API we use."""

        Callback = _Callback
        Trainer = Any
        LightningModule = Any

    pl = _PL()  # type: ignore

from navsim.agents.transfuser.moe_training_scheduler import MoETrainingScheduler
from navsim.agents.transfuser.moe_loss_weight_manager import DynamicLossWeightManager

if TYPE_CHECKING:
    # NOTE: `TransfuserConfig` depends on `nuplan`. Keep it typing-only so this callback
    # can be imported in lightweight environments (e.g., unit tests / smoke checks) where
    # `nuplan` is not installed.
    from navsim.agents.transfuser.transfuser_config import TransfuserConfig


class MoEStagedTrainingCallback(pl.Callback):
    """MoE 分阶段训练回调：按 epoch 动态更新路由参数与损失权重。"""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._last_usage_fraction: Optional[List[float]] = None
        self._current_stage_name: str = "unknown"
        self._current_stage_id: int = 0

        schedule = config.moe_staged_training_schedule
        if schedule is None:
            schedule = MoETrainingScheduler.build_default_three_stage_schedule(
                num_experts=int(getattr(config, "moe_num_experts", 1))
            )
        else:
            # Optionally override dynamic_top_k settings from config knobs (if provided).
            # This keeps YAML overrides simple without requiring users to fully specify schedule dict.
            dyn_enabled = getattr(config, "moe_dynamic_top_k_enabled", None)
            if dyn_enabled is not None:
                schedule.setdefault("dynamic_top_k", {})
                schedule["dynamic_top_k"]["enabled"] = bool(dyn_enabled)
                schedule["dynamic_top_k"].setdefault("stages", ["stage3"])
                schedule["dynamic_top_k"]["min_k"] = int(getattr(config, "moe_dynamic_top_k_min_k", 8))
                schedule["dynamic_top_k"]["max_k"] = int(getattr(config, "moe_dynamic_top_k_max_k", 12))
                schedule["dynamic_top_k"]["active_threshold"] = float(
                    getattr(config, "moe_dynamic_top_k_active_threshold", 0.05)
                )

        self._scheduler = MoETrainingScheduler(
            num_experts=int(getattr(config, "moe_num_experts", 1)),
            schedule=schedule,
            enable_adaptive=bool(getattr(config, "moe_staged_training_adaptive", True)),
        )
        # Loss weights are managed by a dedicated manager for better modularity.
        self._loss_weight_mgr = DynamicLossWeightManager(
            num_experts=int(getattr(config, "moe_num_experts", 1)),
            schedule=schedule,
            enable_adaptive=bool(getattr(config, "moe_staged_training_adaptive", True)),
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Apply staged-training params at the start of each training epoch."""
        if not bool(getattr(self._config, "moe_staged_training_enabled", False)):
            return
        if not bool(getattr(self._config, "use_moe_decoder", False)):
            return

        epoch = int(trainer.current_epoch)
        params = self._scheduler.get_current_params(
            epoch=epoch, expert_usage_fraction=self._last_usage_fraction
        )

        stage_name = str(params.get("stage_name", "unknown"))
        stage_id = 0
        if stage_name == "stage1":
            stage_id = 1
        elif stage_name == "stage2":
            stage_id = 2
        elif stage_name == "stage3":
            stage_id = 3
        self._current_stage_name = stage_name
        self._current_stage_id = int(stage_id)

        # 1) Update config (loss weights and also keep a single source-of-truth for current routing params)
        self._config.moe_router_temperature = float(params["router_temperature"])
        self._config.moe_top_k = int(params["top_k"])
        self._config.moe_load_balance_coef = float(params["load_balance_coef"])

        # Loss weights: compute via DynamicLossWeightManager and apply to config.
        weights = self._loss_weight_mgr.get_weights(epoch=epoch, expert_usage_fraction=self._last_usage_fraction)
        self._loss_weight_mgr.apply_to_config(self._config, weights)

        # 2) Update decoder routing params (runtime)
        decoder = _try_get_moe_decoder(lightning_module)
        if decoder is not None:
            if hasattr(decoder, "set_router_temperature"):
                decoder.set_router_temperature(float(params["router_temperature"]))
            if hasattr(decoder, "set_top_k"):
                decoder.set_top_k(int(params["top_k"]))
            if hasattr(decoder, "set_load_balance_coef"):
                decoder.set_load_balance_coef(float(params["load_balance_coef"]))

        try:
            lightning_module.log("train/moe_stage_id", float(stage_id), on_step=False, on_epoch=True)
            lightning_module.log("train/moe_router_temperature", float(params["router_temperature"]), on_step=False, on_epoch=True)
            lightning_module.log("train/moe_top_k", float(params["top_k"]), on_step=False, on_epoch=True)
            lightning_module.log("train/moe_load_balance_coef", float(params["load_balance_coef"]), on_step=False, on_epoch=True)
            # Log current loss weights from config (authoritative after manager apply)
            if hasattr(self._config, "moe_aux_loss_weight"):
                lightning_module.log("train/moe_aux_loss_weight", float(getattr(self._config, "moe_aux_loss_weight")), on_step=False, on_epoch=True)
            if hasattr(self._config, "expert_diversity_weight"):
                lightning_module.log("train/expert_diversity_weight", float(getattr(self._config, "expert_diversity_weight")), on_step=False, on_epoch=True)
            if params.get("dynamic_top_k_applied"):
                lightning_module.log("train/moe_top_k_active_experts", float(params.get("dynamic_top_k_active_experts", 0)), on_step=False, on_epoch=True)
                lightning_module.log("train/moe_top_k_dynamic_applied", 1.0, on_step=False, on_epoch=True)
            else:
                lightning_module.log("train/moe_top_k_dynamic_applied", 0.0, on_step=False, on_epoch=True)
        except Exception:
            # Avoid breaking training due to logger availability.
            pass

    def on_train_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """
        Collect epoch-level expert usage fractions from logged metrics.

        The loss code logs per-expert scalars as:
            train/moe_usage_fraction_e{i}
        We read them back to enable adaptive scheduling next epoch.
        """
        if not bool(getattr(self._config, "moe_staged_training_enabled", False)):
            return
        if not bool(getattr(self._config, "use_moe_decoder", False)):
            return

        num_experts = int(getattr(self._config, "moe_num_experts", 0))
        if num_experts <= 0:
            return

        usage: List[float] = []
        metrics: Dict[str, Any] = dict(trainer.callback_metrics)
        for i in range(num_experts):
            key = f"train/moe_usage_fraction_e{i}"
            v = metrics.get(key)
            if v is None:
                usage = []
                break
            try:
                usage.append(float(v.detach().cpu().item() if hasattr(v, "detach") else float(v)))
            except Exception:
                usage = []
                break

        if usage:
            s = sum(usage)
            if s > 0:
                usage = [u / s for u in usage]
            # Keep for scheduler adaptive (next epoch)
            self._last_usage_fraction = usage
            self._scheduler.update_expert_usage(epoch=int(trainer.current_epoch), usage_fraction=usage)


def _try_get_moe_decoder(lightning_module: pl.LightningModule):
    """
    Best-effort lookup for the MoE decoder instance.

    Expected object path:
        lightning_module.agent._transfuser_model._tf_decoder
    """
    agent = getattr(lightning_module, "agent", None)
    if agent is None:
        return None
    model = getattr(agent, "_transfuser_model", None)
    if model is None:
        return None
    decoder = getattr(model, "_tf_decoder", None)
    return decoder




