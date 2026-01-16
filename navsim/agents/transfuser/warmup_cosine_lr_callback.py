"""
Warmup + Cosine learning-rate schedule callback.

This callback is intentionally simple and optimizer-agnostic:
- It reads base lr from optimizer param groups on first use.
- It updates lr once per epoch (on_train_epoch_start).

Schedule:
- Warmup: linear from ~0 to base_lr over lr_warmup_epochs
- Cosine: decay to lr_min_ratio * base_lr by trainer.max_epochs
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

try:
    import pytorch_lightning as pl  # type: ignore
except Exception:  # pragma: no cover
    class _Callback:
        pass

    class _PL:
        Callback = _Callback
        Trainer = Any
        LightningModule = Any

    pl = _PL()  # type: ignore


class WarmupCosineLRSchedulerCallback(pl.Callback):
    """按 epoch 应用 warmup + cosine 的学习率调度回调。"""

    def __init__(
        self,
        *,
        warmup_epochs: int = 5,
        min_lr_ratio: float = 0.1,
        stage3_fixed_enabled: bool = False,
        stage3_start_epoch: int = 100,
    ) -> None:
        self._warmup_epochs = max(int(warmup_epochs), 0)
        self._min_lr_ratio = float(min_lr_ratio)
        self._stage3_fixed_enabled = bool(stage3_fixed_enabled)
        self._stage3_start_epoch = int(stage3_start_epoch)
        self._base_lrs: Optional[List[float]] = None

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        if not hasattr(trainer, "optimizers") or not trainer.optimizers:
            return

        opt = trainer.optimizers[0]
        if self._base_lrs is None:
            self._base_lrs = [float(pg.get("lr", 0.0)) for pg in opt.param_groups]

        max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)

        # Fallback: if max_epochs is unknown, keep lr unchanged
        if max_epochs <= 0:
            return

        lr_factors = self._compute_lr_factors(epoch=epoch, max_epochs=max_epochs)
        for pg, base_lr in zip(opt.param_groups, self._base_lrs):
            pg["lr"] = float(base_lr) * float(lr_factors)

        # Optional logging
        try:
            lightning_module.log("train/lr", float(opt.param_groups[0]["lr"]), on_step=False, on_epoch=True)
        except Exception:
            pass

    def _compute_lr_factors(self, *, epoch: int, max_epochs: int) -> float:
        """Return lr multiplier in (0, 1]."""
        min_ratio = max(0.0, min(1.0, self._min_lr_ratio))
        warmup = self._warmup_epochs

        # Scheme (2): stage3 fixed lr
        if self._stage3_fixed_enabled and epoch >= self._stage3_start_epoch:
            return min_ratio

        if warmup > 0 and epoch < warmup:
            # epoch 0 -> 1/warmup, epoch warmup-1 -> 1.0
            return float(epoch + 1) / float(warmup)

        # Cosine decay from 1.0 down to min_ratio
        # If scheme (2) is enabled, we decay only until stage3_start_epoch, then fix.
        cosine_end_epoch = max_epochs
        if self._stage3_fixed_enabled:
            cosine_end_epoch = max(warmup + 1, min(self._stage3_start_epoch, max_epochs))

        # We want progress=0 at epoch=warmup, and progress=1 at epoch=cosine_end_epoch-1.
        denom = max(1, (cosine_end_epoch - warmup - 1))
        progress = float(epoch - warmup) / float(denom)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine


