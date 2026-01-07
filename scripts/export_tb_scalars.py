"""
Export TensorBoard scalar summaries (from tfevents) to CSV.

This is useful when you can't open TensorBoard UI and want to inspect trends like:
  - train/moe_usage_fraction_e*
  - train/moe_aux_loss_step
  - train/loss_step

Usage examples:
  python MOEDrive/scripts/export_tb_scalars.py \
    --event_file exp/.../lightning_logs/version_0/events.out.tfevents... \
    --out_csv exp/.../tb_scalars.csv

  python MOEDrive/scripts/export_tb_scalars.py \
    --logdir exp/.../lightning_logs \
    --out_csv exp/.../tb_scalars.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _pick_default_tags() -> List[str]:
    tags: List[str] = [
        "train/loss_step",
        "train/trajectory_loss_step",
        "train/diffusion_loss_step",
        "train/agent_class_loss_step",
        "train/agent_box_loss_step",
        "train/bev_semantic_loss_step",
        "train/moe_aux_loss_step",
        "train/moe_load_balance_loss_step",
        "train/moe_router_z_loss_step",
    ]
    # e0..e31 to be safe; non-existing tags will be skipped.
    tags.extend([f"train/moe_usage_fraction_e{i}_step" for i in range(32)])
    return tags


def _load_accumulator(event_file: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import tensorboard EventAccumulator. "
            "Install tensorboard in the current env (pip/conda) and retry."
        ) from e

    acc = EventAccumulator(
        str(event_file),
        size_guidance={
            "scalars": 0,  # load all
        },
    )
    acc.Reload()
    return acc


def _find_event_file_from_logdir(logdir: Path) -> Path:
    candidates = sorted(logdir.rglob("events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No events.out.tfevents.* found under logdir={logdir}")
    # Prefer the newest
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _collect_scalars(acc, tags: Iterable[str]) -> Dict[str, List[dict]]:
    available = set(acc.Tags().get("scalars", []))
    picked = [t for t in tags if t in available]
    if not picked:
        raise RuntimeError(
            "No requested scalar tags found in this event file. "
            f"Available scalar tags: {sorted(list(available))[:50]} ..."
        )

    data: Dict[str, List[dict]] = {}
    for tag in picked:
        evs = acc.Scalars(tag)
        data[tag] = [{"wall_time": e.wall_time, "step": int(e.step), "value": float(e.value)} for e in evs]
    return data


def _write_csv(rows: List[dict], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write.")

    # stable field order
    fieldnames = ["step", "tag", "value", "wall_time"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--event_file", type=str, help="Path to a single events.out.tfevents.* file")
    g.add_argument("--logdir", type=str, help="Directory containing lightning_logs/.../events.out.tfevents.*")
    p.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    p.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help="Explicit tags to export (default exports common train/* and moe_* tags)",
    )
    args = p.parse_args(argv)

    if args.event_file:
        event_file = Path(args.event_file)
    else:
        event_file = _find_event_file_from_logdir(Path(args.logdir))

    if not event_file.exists():
        raise FileNotFoundError(str(event_file))

    tags = args.tags if args.tags is not None else _pick_default_tags()
    acc = _load_accumulator(event_file)
    scalars = _collect_scalars(acc, tags)

    rows: List[dict] = []
    for tag, evs in scalars.items():
        for e in evs:
            rows.append({"step": e["step"], "tag": tag, "value": e["value"], "wall_time": e["wall_time"]})

    rows.sort(key=lambda r: (r["step"], r["tag"]))
    out_csv = Path(args.out_csv)
    _write_csv(rows, out_csv)

    print(f"event_file={event_file}")
    print(f"out_csv={out_csv}")
    print(f"num_tags={len(scalars)} num_rows={len(rows)}")
    # quick summary for usage fraction tags
    usage_tags = [t for t in scalars.keys() if "moe_usage_fraction_e" in t]
    if usage_tags:
        # print last step values (if any)
        last_step = max(r["step"] for r in rows)
        last_usage = {r["tag"]: r["value"] for r in rows if r["step"] == last_step and r["tag"] in usage_tags}
        if last_usage:
            ordered = [last_usage.get(t) for t in sorted(last_usage.keys())]
            print(f"last_step={last_step} last_usage_fraction={ordered}")


if __name__ == "__main__":
    main()


