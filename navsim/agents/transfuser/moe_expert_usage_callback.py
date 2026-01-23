"""
MoE expert usage monitoring callback (standalone).

This callback is responsible for:
- Reading per-expert usage fractions logged by the loss (train/moe_usage_fraction_e{i})
- Logging aggregate stats: usage entropy, dead experts count, max usage
- TensorBoard visualization: usage heatmap (experts x epochs)
- Persisting usage history to CSV/JSONL for offline analysis

It intentionally does NOT update routing parameters or loss weights.
Those are handled by MoEStagedTrainingCallback / scheduler.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    import pytorch_lightning as pl  # type: ignore
except Exception:  # pragma: no cover
    class _Callback:
        """Fallback base class to allow importing this module without Lightning."""

        pass

    class _PL:
        Callback = _Callback
        Trainer = Any
        LightningModule = Any

    pl = _PL()  # type: ignore

from navsim.agents.transfuser.moe_training_scheduler import MoETrainingScheduler

if TYPE_CHECKING:
    from navsim.agents.transfuser.transfuser_config import TransfuserConfig

logger = logging.getLogger(__name__)


class MoEExpertUsageCallback(pl.Callback):
    """独立的 MoE 专家使用监控回调（日志 + 可视化 + 落盘）。"""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._usage_history: List[List[float]] = []

        schedule = getattr(config, "moe_staged_training_schedule", None)
        if schedule is None:
            schedule = MoETrainingScheduler.build_default_three_stage_schedule(
                num_experts=int(getattr(config, "moe_num_experts", 1))
            )
        self._scheduler = MoETrainingScheduler(
            num_experts=int(getattr(config, "moe_num_experts", 1)),
            schedule=schedule,
            enable_adaptive=bool(getattr(config, "moe_staged_training_adaptive", True)),
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Collect expert usage (train) and log/visualize/persist."""
        if not bool(getattr(self._config, "use_moe_decoder", False)):
            return

        num_experts = int(getattr(self._config, "moe_num_experts", 0))
        if num_experts <= 0:
            return

        usage = _read_usage_fraction_from_metrics(
            metrics=dict(getattr(trainer, "callback_metrics", {})),
            num_experts=num_experts,
        )
        if not usage:
            return

        # Normalize
        s = sum(usage)
        if s > 0:
            usage = [u / s for u in usage]
        self._usage_history.append(list(usage))

        # Aggregate stats
        dead_thr, overuse_thr = _get_usage_thresholds_from_schedule(getattr(self._scheduler, "_schedule", {}))
        dead_experts = sum(1 for u in usage if float(u) < dead_thr)
        max_usage = max(float(u) for u in usage)
        usage_entropy = _entropy(usage)

        # Warnings (once per epoch)
        epoch = int(getattr(trainer, "current_epoch", 0))
        stage_name, stage_id = _get_stage_name_id(self._scheduler, epoch)
        if dead_experts > 0:
            logger.warning(
                "[MoEExpertUsage] epoch=%d stage=%s dead_experts=%d (thr=%.3f)",
                epoch,
                stage_name,
                dead_experts,
                dead_thr,
            )
        if max_usage > overuse_thr:
            logger.warning(
                "[MoEExpertUsage] epoch=%d stage=%s max_usage=%.3f (thr=%.3f)",
                epoch,
                stage_name,
                max_usage,
                overuse_thr,
            )

        # Log scalars
        try:
            lightning_module.log("train/moe_usage_entropy", float(usage_entropy), on_step=False, on_epoch=True)
            lightning_module.log("train/moe_dead_experts", float(dead_experts), on_step=False, on_epoch=True)
            lightning_module.log("train/moe_max_usage", float(max_usage), on_step=False, on_epoch=True)
        except Exception:
            pass

        # Visualize heatmap
        if bool(getattr(self._config, "moe_usage_visualization_enabled", False)) and getattr(trainer, "is_global_zero", True):
            _maybe_log_usage_heatmap(trainer=trainer, tag="moe/usage_heatmap_train", usage_history=self._usage_history)

        # Persist CSV/JSONL
        if bool(getattr(self._config, "moe_usage_history_save_enabled", False)) and getattr(trainer, "is_global_zero", True):
            _maybe_persist_usage_history(
                trainer=trainer,
                config=self._config,
                epoch=epoch,
                stage_name=stage_name,
                stage_id=stage_id,
                params_snapshot={
                    "moe_router_temperature": float(getattr(self._config, "moe_router_temperature", 0.0)),
                    "moe_top_k": int(getattr(self._config, "moe_top_k", 0)),
                    "moe_load_balance_coef": float(getattr(self._config, "moe_load_balance_coef", 0.0)),
                },
                usage_fraction=usage,
            )


def _read_usage_fraction_from_metrics(*, metrics: Dict[str, Any], num_experts: int) -> List[float]:
    usage: List[float] = []
    for i in range(int(num_experts)):
        key = f"train/moe_usage_fraction_e{i}"
        v = metrics.get(key)
        if v is None:
            return []
        try:
            usage.append(float(v.detach().cpu().item() if hasattr(v, "detach") else float(v)))
        except Exception:
            return []
    return usage


def _entropy(p: List[float]) -> float:
    import math

    eps = 1e-12
    return float(-sum(float(x) * math.log(float(x) + eps) for x in p))


def _get_usage_thresholds_from_schedule(schedule: Any) -> tuple[float, float]:
    # Defaults aligned with design doc
    dead_thr = 0.02
    overuse_thr = 0.30
    if isinstance(schedule, dict):
        adaptive = schedule.get("adaptive", {})
        if isinstance(adaptive, dict):
            dead_thr = float(adaptive.get("dead_expert_threshold", dead_thr))
            overuse_thr = float(adaptive.get("overuse_threshold", overuse_thr))
    return dead_thr, overuse_thr


def _get_stage_name_id(scheduler: MoETrainingScheduler, epoch: int) -> tuple[str, int]:
    params = scheduler.get_current_params(epoch=int(epoch), expert_usage_fraction=None)
    name = str(params.get("stage_name", "unknown"))
    sid = 0
    if name == "stage1":
        sid = 1
    elif name == "stage2":
        sid = 2
    elif name == "stage3":
        sid = 3
    return name, sid


def _resolve_output_dir(trainer: Any) -> Optional[str]:
    logger_obj = getattr(trainer, "logger", None)
    if logger_obj is not None:
        log_dir = getattr(logger_obj, "log_dir", None)
        if isinstance(log_dir, str) and log_dir:
            return log_dir
    root = getattr(trainer, "default_root_dir", None)
    if isinstance(root, str) and root:
        return root
    return None


def _maybe_log_usage_heatmap(*, trainer: Any, tag: str, usage_history: List[List[float]]) -> None:
    logger_obj = getattr(trainer, "logger", None)
    if logger_obj is None:
        return
    exp = getattr(logger_obj, "experiment", None)
    if exp is None or not hasattr(exp, "add_image"):
        return
    if not usage_history:
        return

    num_experts = len(usage_history[0])
    if num_experts <= 0:
        return

    import torch

    h = num_experts
    w = len(usage_history)
    img = torch.zeros((h, w), dtype=torch.float32)
    for j, u in enumerate(usage_history):
        if len(u) != h:
            continue
        img[:, j] = torch.tensor(u, dtype=torch.float32)

    img_u8 = (img.clamp(0, 1) * 255.0).to(torch.uint8)
    img_rgb = img_u8.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
    exp.add_image(tag, img_rgb, global_step=int(getattr(trainer, "current_epoch", 0)))


def _maybe_persist_usage_history(
    *,
    trainer: Any,
    config: Any,
    epoch: int,
    stage_name: str,
    stage_id: int,
    params_snapshot: Dict[str, Any],
    usage_fraction: List[float],
) -> None:
    out_dir = _resolve_output_dir(trainer)
    if not out_dir:
        return

    import csv
    import json
    import os

    csv_name = str(getattr(config, "moe_usage_history_csv_name", "moe_expert_usage_history.csv"))
    jsonl_name = str(getattr(config, "moe_usage_history_jsonl_name", "moe_expert_usage_history.jsonl"))

    csv_path = os.path.join(out_dir, csv_name)
    jsonl_path = os.path.join(out_dir, jsonl_name)

    num_experts = len(usage_fraction)
    row: Dict[str, Any] = {
        "epoch": int(epoch),
        "stage_name": str(stage_name),
        "stage_id": int(stage_id),
        **{k: v for k, v in params_snapshot.items()},
    }
    for i, u in enumerate(usage_fraction):
        row[f"usage_fraction_e{i}"] = float(u)

    try:
        os.makedirs(out_dir, exist_ok=True)
        file_exists = os.path.exists(csv_path)
        fieldnames = ["epoch", "stage_name", "stage_id", "moe_router_temperature", "moe_top_k", "moe_load_balance_coef"] + [
            f"usage_fraction_e{i}" for i in range(num_experts)
        ]
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        logger.warning("[MoEUsageHistory] failed to write csv: %s", e)

    try:
        rec = dict(row)
        rec["num_experts"] = int(num_experts)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        logger.warning("[MoEUsageHistory] failed to write jsonl: %s", e)



