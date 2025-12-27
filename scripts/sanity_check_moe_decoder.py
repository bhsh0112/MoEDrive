"""
Sanity check for MoETransformerDecoder forward pass.

This script does NOT require nuPlan/navsim datasets. It only checks:
- shape contract: (B, Q, D) + (B, S, D) -> (B, Q, D)
- aux fields existence and reasonable values (non-NaN)

Usage:
    python -m MOEDrive.scripts.sanity_check_moe_decoder
or:
    python MOEDrive/scripts/sanity_check_moe_decoder.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make repo importable when running as a plain script.
# This allows `import navsim...` without installing the project as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.moe_transformer_decoder import MoEConfig, MoETransformerDecoder, MoETransformerDecoderLayer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--q", type=int, default=31, help="query tokens, e.g. 1 + num_bounding_boxes")
    parser.add_argument("--s", type=int, default=65, help="memory tokens, e.g. 8*8 + 1")
    parser.add_argument("--d", type=int, default=256, help="d_model")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dff", type=int, default=1024, help="ffn hidden dim")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--lb", type=float, default=1e-2, help="load balance coef (inside MoE)")
    parser.add_argument("--z", type=float, default=1e-3, help="router z-loss coef (inside MoE)")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    moe_cfg = MoEConfig(
        num_experts=args.experts,
        top_k=args.topk,
        router_temperature=1.0,
        router_z_loss_coef=args.z,
        load_balance_coef=args.lb,
    )
    layer = MoETransformerDecoderLayer(
        d_model=args.d,
        nhead=args.heads,
        dim_feedforward=args.dff,
        dropout=args.dropout,
        moe_cfg=moe_cfg,
    )
    decoder = MoETransformerDecoder(layer, args.layers).to(device)

    tgt = torch.randn(args.batch, args.q, args.d, device=device)
    mem = torch.randn(args.batch, args.s, args.d, device=device)

    out, aux = decoder(tgt, mem)

    assert out.shape == (args.batch, args.q, args.d), f"bad out shape: {out.shape}"
    for k in ["moe_aux_loss", "moe_load_balance_loss", "moe_router_z_loss", "moe_usage_counts", "moe_usage_fraction"]:
        assert k in aux, f"missing aux key: {k}"

    # Basic numeric checks
    assert torch.isfinite(out).all(), "output contains NaN/Inf"
    for k, v in aux.items():
        assert torch.isfinite(v).all(), f"aux[{k}] contains NaN/Inf"

    print("OK: out.shape =", tuple(out.shape))
    print("moe_aux_loss =", float(aux["moe_aux_loss"].detach().cpu()))
    print("moe_load_balance_loss =", float(aux["moe_load_balance_loss"].detach().cpu()))
    print("moe_router_z_loss =", float(aux["moe_router_z_loss"].detach().cpu()))
    print("moe_usage_fraction =", aux["moe_usage_fraction"].detach().cpu().tolist())


if __name__ == "__main__":
    main()


