"""
Smoke test for MoE logging integration.

This script checks that:
1) transfuser_loss (baseline) returns a scalar-only loss_dict compatible with AgentLightningModule logging
2) diffusiondrive transfuser_loss returns a scalar-only loss_dict and includes MoE metrics

It does NOT run the full Transfuser backbone/model forward (no dataset required).

Usage:
    python MOEDrive/scripts/smoke_test_moe_logging.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make repo importable when running as a plain script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _assert_scalar_dict(d: dict) -> None:
    for k, v in d.items():
        assert isinstance(k, str), f"non-str key: {k}"
        assert torch.is_tensor(v), f"{k} is not a tensor: {type(v)}"
        assert v.ndim == 0, f"{k} is not scalar, shape={tuple(v.shape)}"
        assert torch.isfinite(v).all(), f"{k} contains NaN/Inf"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ---------------------------
    # Baseline transfuser_loss
    # ---------------------------
    from navsim.agents.transfuser.transfuser_loss import transfuser_loss as base_loss
    from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex as BaseBB

    class BaseCfg:
        latent = False
        latent_rad_thresh = 0.0
        trajectory_weight = 10.0
        agent_class_weight = 10.0
        agent_box_weight = 1.0
        bev_semantic_weight = 10.0
        num_bev_classes = 7
        bev_pixel_size = 0.25
        # MoE
        moe_aux_loss_weight = 1.0

    B = 2
    N = 5  # num predicted agents
    G = 4  # num gt agents (same as N for simplicity)
    H = W = 16

    targets = {
        "trajectory": torch.randn(B, 8, 3, device=device),
        "bev_semantic_map": torch.randint(0, BaseCfg.num_bev_classes, (B, H, W), device=device),
        "agent_labels": (torch.rand(B, G, device=device) > 0.5),
        "agent_states": torch.randn(B, G, BaseBB.size(), device=device),
    }
    predictions = {
        "trajectory": torch.randn(B, 8, 3, device=device),
        "bev_semantic_map": torch.randn(B, BaseCfg.num_bev_classes, H, W, device=device),
        "agent_labels": torch.randn(B, N, device=device),
        "agent_states": torch.randn(B, N, BaseBB.size(), device=device),
        # MoE metrics (as emitted by model forward)
        "moe_aux_loss": torch.tensor(0.01, device=device),
        "moe_load_balance_loss": torch.tensor(0.008, device=device),
        "moe_router_z_loss": torch.tensor(0.002, device=device),
        "moe_usage_fraction": torch.tensor([0.25, 0.25, 0.25, 0.25], device=device),
    }

    base_loss_dict = base_loss(targets, predictions, BaseCfg)
    assert isinstance(base_loss_dict, dict), "baseline transfuser_loss should return dict"
    _assert_scalar_dict(base_loss_dict)
    assert "moe_aux_loss" in base_loss_dict, "baseline loss_dict missing moe_aux_loss"
    assert any(k.startswith("moe_usage_fraction_e") for k in base_loss_dict.keys()), "missing moe usage scalars"

    print("[OK] baseline transfuser_loss keys:", sorted(list(base_loss_dict.keys()))[:12], "... total", len(base_loss_dict))

    # ---------------------------------------
    # DiffusionDrive transfuser_loss (MOEDrive)
    # ---------------------------------------
    from navsim.agents.diffusiondrive.transfuser_loss import transfuser_loss as dd_loss
    from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex as DDBB

    class DDCfg:
        latent = False
        latent_rad_thresh = 0.0
        trajectory_weight = 12.0
        diff_loss_weight = 20.0
        agent_class_weight = 10.0
        agent_box_weight = 1.0
        bev_semantic_weight = 14.0
        num_bev_classes = 7
        # MoE
        moe_aux_loss_weight = 1.0

    dd_targets = {
        "trajectory": torch.randn(B, 8, 3, device=device),
        "bev_semantic_map": torch.randint(0, DDCfg.num_bev_classes, (B, H, W), device=device),
        "agent_labels": (torch.rand(B, G, device=device) > 0.5),
        "agent_states": torch.randn(B, G, DDBB.size(), device=device),
    }
    dd_predictions = {
        "trajectory": torch.randn(B, 8, 3, device=device),
        "bev_semantic_map": torch.randn(B, DDCfg.num_bev_classes, H, W, device=device),
        "agent_labels": torch.randn(B, N, device=device),
        "agent_states": torch.randn(B, N, DDBB.size(), device=device),
        "trajectory_loss": torch.tensor(0.5, device=device),
        "diffusion_loss": torch.tensor(0.25, device=device),
        # MoE
        "moe_aux_loss": torch.tensor(0.01, device=device),
        "moe_load_balance_loss": torch.tensor(0.008, device=device),
        "moe_router_z_loss": torch.tensor(0.002, device=device),
        "moe_usage_fraction": torch.tensor([0.26, 0.24, 0.27, 0.23], device=device),
    }

    dd_loss_dict = dd_loss(dd_targets, dd_predictions, DDCfg)
    assert isinstance(dd_loss_dict, dict), "diffusiondrive transfuser_loss should return dict"
    _assert_scalar_dict({k: v for k, v in dd_loss_dict.items() if torch.is_tensor(v) and v.ndim == 0})
    assert "moe_aux_loss" in dd_loss_dict, "diffusiondrive loss_dict missing moe_aux_loss"
    assert any(k.startswith("moe_usage_fraction_e") for k in dd_loss_dict.keys()), "missing moe usage scalars"

    print("[OK] diffusiondrive transfuser_loss keys:", sorted(list(dd_loss_dict.keys()))[:12], "... total", len(dd_loss_dict))
    print("[DONE] MoE logging smoke test passed.")


if __name__ == "__main__":
    main()



