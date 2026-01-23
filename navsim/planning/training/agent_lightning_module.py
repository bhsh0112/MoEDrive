import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_metrics import compute_minade_minfde_missrate
from navsim.agents.transfuser.transfuser_specialization_metrics import compute_mode_specialization_metrics


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features, targets)
        # loss = self.agent.compute_loss(features, targets, prediction)
        # self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))

        # Validation trajectory metrics (optional; multi-modal aware)
        if logging_prefix == "val":
            cfg = getattr(self.agent, "_config", None)
            metrics_enabled = bool(getattr(cfg, "trajectory_metrics_enabled", False)) if cfg is not None else False
            if metrics_enabled and "trajectory" in targets:
                miss_thr = float(getattr(cfg, "trajectory_miss_threshold_m", 2.0)) if cfg is not None else 2.0
                pred_traj = prediction.get("trajectory_modes", prediction.get("trajectory"))
                if pred_traj is not None:
                    metrics = compute_minade_minfde_missrate(
                        gt_trajectory=targets["trajectory"],
                        pred_trajectory=pred_traj,
                        miss_threshold_m=miss_thr,
                    )
                    for mk, mv in metrics.items():
                        self.log(f"val/{mk}", mv, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))

            # Expert specialization metrics (multi-modal only)
            spec_enabled = bool(getattr(cfg, "expert_specialization_metrics_enabled", False)) if cfg is not None else False
            if spec_enabled:
                traj_modes = prediction.get("trajectory_modes")
                if traj_modes is not None:
                    method = str(getattr(cfg, "expert_specialization_method", "pairwise_l2")) if cfg is not None else "pairwise_l2"
                    spec = compute_mode_specialization_metrics(pred_trajectory_modes=traj_modes, method=method)
                    for sk, sv in spec.items():
                        self.log(f"val/{sk}", sv, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=len(batch[0]))
        return loss_dict['loss']

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
