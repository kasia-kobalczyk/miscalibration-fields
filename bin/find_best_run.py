#!/usr/bin/env python3
"""Scan all runs for a given (dataset, model) grid and save the best run info.

Writes  outputs/<dataset>/<model>/best_run.json  with:
  - best run directory
  - target metric name, value, and mode
  - hyperparameters (from hydra overrides)
  - epoch of best checkpoint

Usage:
    python bin/find_best_run.py --output_dir outputs --dataset hellaswag --model Llama-3.1-8B
    python bin/find_best_run.py --output_dir outputs  # scans all dataset/model pairs
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml


def load_run_info(run_dir: Path) -> dict | None:
    """Extract target metric value and config from a single run directory."""
    ckpt_path = run_dir / "best_phi.pt"
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    overrides_path = run_dir / ".hydra" / "overrides.yaml"

    if not ckpt_path.exists():
        return None

    # Load checkpoint to get the metric value and epoch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", None)

    # Read config to know which metric was tracked
    target_metric = "val/corrected_bs_score"
    target_mode = "min"
    if hydra_cfg_path.exists():
        with open(hydra_cfg_path) as f:
            cfg = yaml.safe_load(f)
        trainer_cfg = cfg.get("trainer", {})
        target_metric = trainer_cfg.get("target_metric", target_metric)
        target_mode = trainer_cfg.get("target_metric_mode", target_mode)

    metric_value = ckpt.get(target_metric)
    if metric_value is None:
        return None

    # Read overrides for the hyperparameters
    overrides = {}
    if overrides_path.exists():
        with open(overrides_path) as f:
            raw = yaml.safe_load(f) or []
        for entry in raw:
            entry = entry.lstrip("+")
            if "=" in entry:
                k, v = entry.split("=", 1)
                overrides[k] = v

    return {
        "run_dir": str(run_dir),
        "target_metric": target_metric,
        "target_metric_mode": target_mode,
        "target_metric_value": float(metric_value),
        "epoch": epoch,
        "overrides": overrides,
    }


def find_best(model_dir: Path) -> dict | None:
    """Find the best run across all timestamped subdirectories."""
    if not model_dir.is_dir():
        return None

    best = None
    for run_dir in sorted(model_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        info = load_run_info(run_dir)
        if info is None:
            continue

        maximize = info["target_metric_mode"] == "max"
        if best is None:
            best = info
        elif maximize and info["target_metric_value"] > best["target_metric_value"]:
            best = info
        elif not maximize and info["target_metric_value"] < best["target_metric_value"]:
            best = info

    return best


def process(output_dir: Path, dataset: str, model: str):
    model_dir = output_dir / dataset / model
    best = find_best(model_dir)
    if best is None:
        print(f"  No valid runs found in {model_dir}")
        return

    out_path = model_dir / "best_run.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)

    print(f"  {best['target_metric']} = {best['target_metric_value']:.6f}")
    print(f"  run: {best['run_dir']}")
    print(f"  overrides: {best['overrides']}")
    print(f"  saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (subfolder of output_dir). If omitted, scan all.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (subfolder of dataset dir). If omitted, scan all.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output dir {output_dir} does not exist.")
        return

    # Collect (dataset, model) pairs to process
    pairs = []
    if args.dataset and args.model:
        pairs.append((args.dataset, args.model))
    elif args.dataset:
        ds_dir = output_dir / args.dataset
        if ds_dir.is_dir():
            pairs.extend(
                (args.dataset, m.name) for m in sorted(ds_dir.iterdir()) if m.is_dir()
            )
    else:
        for ds_dir in sorted(output_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for m_dir in sorted(ds_dir.iterdir()):
                if m_dir.is_dir():
                    pairs.append((ds_dir.name, m_dir.name))

    for dataset, model in pairs:
        print(f"[{dataset} / {model}]")
        process(output_dir, dataset, model)
        print()


if __name__ == "__main__":
    main()
