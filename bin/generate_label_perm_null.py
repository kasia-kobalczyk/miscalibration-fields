#!/usr/bin/env python3
"""Label-permutation null test for the discovery pipeline.

For one (dataset, model) pair:
  1. Load the real best run (best_phi.pt + cfg with chosen sigma, lambda_mass).
  2. Compute the real metric vector once on the test split.
  3. For b in 0..B-1: subprocess `train.py` with `+label_permutation_seed=b`
     (permutes y_train and y_val), reload the trained phi_null_b, recompute
     hat_delta_test using the *permuted* training labels as the bank, and
     compute the metric vector on (real f_test, real y_test).
  4. Write per-run CSV + summary CSV with permutation p-values.

Usage:
    python bin/generate_label_perm_null.py --dataset hhrlhf_judge --model Qwen3-8B-Base --B 20
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from calib_atlas.data.datamodule import make_loaders
from calib_atlas.models.phi_mlp import PhiMLP
from calib_atlas.utils.eval_utils import (
    build_bank,
    compute_hat_delta_from_bank,
    gather_data,
    get_smece,
)
from calib_atlas.utils.label_perm import (
    build_label_permutation_map,
    wrap_loader_with_label_permutation,
)
from calib_atlas.utils.seed import seed_everything


EPS = 0.05
NULL_SUBDIR = "null_runs"


def load_phi(model_cfg, ckpt_path: Path) -> PhiMLP:
    model = PhiMLP(model_cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def compute_metrics(hat_delta_test, f_test, y_test) -> dict:
    hd = np.asarray(hat_delta_test, dtype=float)
    f = np.asarray(f_test, dtype=float)
    y = np.asarray(y_test, dtype=float)

    over = hd < -EPS
    under = hd > EPS
    good = ~(over | under)

    def smece_or_nan(mask):
        if int(mask.sum()) < 2:
            return float("nan")
        return float(get_smece(f[mask], y[mask]))

    smece_all = float(get_smece(f, y))
    smece_over = smece_or_nan(over)
    smece_under = smece_or_nan(under)
    smece_good = smece_or_nan(good)

    valid = [v for v in (smece_over, smece_under) if not np.isnan(v)]
    if valid:
        worst = max(valid)
        worst_slice_gap = worst - smece_all
    else:
        worst_slice_gap = float("nan")

    return {
        "field_std": float(np.std(hd)),
        "field_mean_abs": float(np.mean(np.abs(hd))),
        "smece_all": smece_all,
        "smece_over": smece_over,
        "smece_under": smece_under,
        "smece_good": smece_good,
        "worst_slice_gap": worst_slice_gap,
        "over_size": float(over.mean()),
        "under_size": float(under.mean()),
        "good_size": float(good.mean()),
    }


def evaluate_phi(phi: PhiMLP, cfg, device: str, perm_seed: int | None):
    """Compute (hat_delta_test, f_test, y_test) using `phi`.

    `perm_seed=None` -> real evaluation: bank uses real train labels.
    `perm_seed=b`    -> null evaluation: bank uses train labels permuted with the
        same `np.random.default_rng(b)` stream that `train.py` saw, so r_bank
        matches what the model was trained against.
    """
    seed_everything(int(cfg.seed))  # determinism for make_loaders
    loaders = make_loaders(cfg.data, splits=["train", "val", "test"])
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    phi.to(device)
    test_pack = gather_data(phi, test_loader, cfg)

    if perm_seed is not None:
        rng = np.random.default_rng(int(perm_seed))
        train_perm_map = build_label_permutation_map(train_loader, rng)
        # Note: in train.py the val perm follows the train perm and advances
        # the rng. We don't need val_perm here, but we don't need to advance
        # the rng either since we only consume train_perm.
        train_loader = wrap_loader_with_label_permutation(
            train_loader, train_perm_map, shuffle=False
        )

    bank_z, bank_f, bank_r = build_bank(phi, train_loader, device=device)

    hat_delta_test, _ = compute_hat_delta_from_bank(
        z_query=test_pack["z"],
        f_query=test_pack["f"],
        z_bank=bank_z,
        f_bank=bank_f,
        r_bank=bank_r,
        sigma=float(cfg.loss.sigma),
        tau=float(cfg.loss.tau) if cfg.loss.tau is not None else None,
        eps=float(cfg.loss.eps),
        chunk_q=2048,
    )
    return (
        hat_delta_test.detach().cpu().numpy(),
        test_pack["f"].detach().cpu().numpy(),
        test_pack["y"].detach().cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--B", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output_dir", default=str(PROJECT_DIR / "outputs"),
        help="Root outputs/ directory.",
    )
    parser.add_argument(
        "--keep_run_dirs", action="store_true",
        help="Keep null_runs/b{NN}/ dirs after evaluation (else delete).",
    )
    args = parser.parse_args()
    if args.skip_training:
        args.keep_run_dirs = True

    output_dir = Path(args.output_dir) / args.dataset / args.model
    if not output_dir.is_dir():
        print(f"Output dir not found: {output_dir}")
        sys.exit(1)

    best_run_path = output_dir / "best_run.json"
    if not best_run_path.exists():
        print(f"best_run.json missing at {best_run_path}")
        sys.exit(1)
    with open(best_run_path) as fh:
        best_run = json.load(fh)
    real_run_dir = Path(best_run["run_dir"])
    cfg = OmegaConf.load(real_run_dir / ".hydra" / "config.yaml")
    cfg.device = args.device

    torch.use_deterministic_algorithms(False)

    # ----- Real eval -----
    print(f"\n=== Real eval: {args.dataset} / {args.model} ===")
    print(f"  sigma={float(cfg.loss.sigma)}  lambda_mass={cfg.loss.lambda_mass}")
    real_phi = load_phi(cfg.model, real_run_dir / "best_phi.pt")
    real_hd, f_test, y_test = evaluate_phi(real_phi, cfg, args.device, perm_seed=None)
    real_metrics = compute_metrics(real_hd, f_test, y_test)
    print(
        f"  real: field_std={real_metrics['field_std']:.4f}  "
        f"worst_slice_gap={real_metrics['worst_slice_gap']:.4f}  "
        f"smece_all={real_metrics['smece_all']:.4f}"
    )

    rows = [{
        "kind": "real",
        "null_id": -1,
        "sigma": float(cfg.loss.sigma),
        "lambda_mass": str(cfg.loss.lambda_mass),
        **real_metrics,
    }]

    # ----- Null loop -----
    for b in range(args.B):
        print(f"\n=== Null b={b}/{args.B} ===")
        null_run_dir = output_dir / NULL_SUBDIR / f"b{b:02d}"
        ckpt = null_run_dir / "best_phi.pt"
        if not ckpt.exists():
            print(f"  [skip] missing {ckpt} — run training first")
            continue

        null_phi = load_phi(cfg.model, null_run_dir / "best_phi.pt")
        null_hd, _f, _y = evaluate_phi(null_phi, cfg, args.device, perm_seed=b)
        null_metrics = compute_metrics(null_hd, f_test, y_test)
        print(
            f"  null b={b}: field_std={null_metrics['field_std']:.4f}  "
            f"worst_slice_gap={null_metrics['worst_slice_gap']:.4f}"
        )
        rows.append({
            "kind": "null",
            "null_id": b,
            "sigma": float(cfg.loss.sigma),
            "lambda_mass": str(cfg.loss.lambda_mass),
            **null_metrics,
        })

        if not args.keep_run_dirs:
            shutil.rmtree(null_run_dir, ignore_errors=True)

    # ----- Save outputs -----
    runs_df = pd.DataFrame(rows)
    runs_csv = output_dir / "label_perm_null_runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"\nSaved per-run metrics to {runs_csv}")

    null_df = runs_df[runs_df["kind"] == "null"]
    summary_rows = []
    for metric in ["field_std", "field_mean_abs", "worst_slice_gap"]:
        real_val = float(real_metrics[metric])
        null_vals = null_df[metric].dropna().to_numpy()
        n = len(null_vals)
        null_mean = float(null_vals.mean()) if n else float("nan")
        null_sd = float(null_vals.std(ddof=1)) if n > 1 else float("nan")
        if np.isnan(real_val) or n == 0:
            p_value = float("nan")
        else:
            p_value = (1 + int(np.sum(null_vals >= real_val))) / (n + 1)
        summary_rows.append({
            "metric": metric,
            "real": real_val,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "B_effective": n,
            "p_value": p_value,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "label_perm_null_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")
    print("\n" + summary_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
