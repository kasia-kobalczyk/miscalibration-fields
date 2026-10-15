#!/usr/bin/env python3
"""Compute the raw-embedding kernel-smoothing baseline for each model in a dataset.

For each model under ``outputs/<dataset>/`` with a ``best_run.json``, runs
Nadaraya-Watson kernel smoothing of (y - f) over the L2-normalized raw LLM
embeddings x (no learned phi). Sigma is selected per (dataset, model) via a
grid search on validation Brier of the **raw additive** correction
``(f + hat_delta).clamp(0, 1)`` (no range-aware scaling). Once sigma is fixed,
``alpha`` for the range-aware corrector is fit on train.

Outputs (per dataset):
    - ``outputs/<dataset>/raw_hat_delta_test.csv``: per-test-example
      ``(model, f, y, hat_delta)`` mirroring the schema of
      ``hat_delta_test.csv`` produced by ``generate_ece_table.py``.
    - ``outputs/<dataset>/raw_hat_delta_meta.csv``: per-model metadata
      ``(model, embedding_dim, sigma_best, val_brier_best, val_brier_grid,
        alpha_range_aware)``.

Usage:
    python bin/generate_raw_hat_delta.py --dataset hhrlhf_judge
    python bin/generate_raw_hat_delta.py --dataset mmlu_pro --models Qwen3-1.7B-Base
    python bin/generate_raw_hat_delta.py --dataset medmcqa --device cuda:0
"""

import argparse
import json
import os
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
from calib_atlas.models.baselines import RawEmbeddingKernelSmoother
from calib_atlas.utils.eval_utils import fit_range_aware_scale
from calib_atlas.utils.seed import seed_everything


SIGMA_GRID = [1.0, 0.5, 0.1, 0.05]
DATASETS_ROOT = PROJECT_DIR / "datasets"


def _read_embedding_dim(dataset: str, model: str) -> int:
    cfg_path = DATASETS_ROOT / dataset / model / "embeddings" / "config.json"
    with open(cfg_path) as fh:
        return int(json.load(fh)["embed_dim"])


def _gather_raw(loader, device: str):
    xs, fs, ys = [], [], []
    for batch in loader:
        emb = batch["emb"].to(device).float()
        if emb.dim() == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(1)
        f = batch["f"].to(device).float().flatten()
        y = batch["y"].to(device).float().flatten()
        xs.append(emb)
        fs.append(f)
        ys.append(y)
    return torch.cat(xs, 0), torch.cat(fs, 0), torch.cat(ys, 0)


def process_model(
    output_dir: Path,
    dataset: str,
    model: str,
    device: str,
    sigma_grid: list,
    max_bank_size: int,
    tau: float | None,
    seed: int,
):
    best_run_path = output_dir / dataset / model / "best_run.json"
    if not best_run_path.exists():
        print(f"  [skip] no best_run.json at {best_run_path}")
        return None

    with open(best_run_path) as fh:
        best_run = json.load(fh)
    run_dir = Path(best_run["run_dir"])
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    seed_everything(int(cfg["seed"]))

    embed_dim = _read_embedding_dim(dataset, model)

    loaders = make_loaders(cfg.data, splits=["train", "val", "test"])
    train_loader, val_loader, test_loader = (
        loaders["train"],
        loaders["val"],
        loaders["test"],
    )

    print(f"  Gathering raw embeddings on {device} (embedding_dim={embed_dim})...")
    x_tr, f_tr, y_tr = _gather_raw(train_loader, device)
    x_va, f_va, y_va = _gather_raw(val_loader, device)
    x_te, f_te, y_te = _gather_raw(test_loader, device)

    assert x_tr.shape[1] == embed_dim, (
        f"embedding_dim mismatch for {dataset}/{model}: "
        f"config.json says {embed_dim} but emb tensor has {x_tr.shape[1]}"
    )
    print(
        f"  splits: |train|={x_tr.shape[0]} |val|={x_va.shape[0]} "
        f"|test|={x_te.shape[0]}"
    )

    # Subsample bank for val/test queries to bound memory.
    n_train = x_tr.shape[0]
    n_bank = min(n_train, max_bank_size)
    g = torch.Generator(device="cpu").manual_seed(seed)
    bank_idx = torch.randperm(n_train, generator=g)[:n_bank]
    x_bank = x_tr[bank_idx].contiguous()
    f_bank = f_tr[bank_idx].contiguous()
    y_bank = y_tr[bank_idx].contiguous()
    print(f"  bank size = {n_bank} (max_bank_size={max_bank_size})")

    # --- Stage 1: sigma selection on val (raw additive Brier) ---
    val_brier_grid: dict[float, float] = {}
    for sigma in sigma_grid:
        smoother = RawEmbeddingKernelSmoother(
            sigma=sigma, tau=tau, normalize=True
        )
        smoother.fit(x_bank, f_bank, y_bank)
        hat_delta_val = smoother.predict_hat_delta(x_va, f_va)
        f_corr = (f_va + hat_delta_val).clamp(0.0, 1.0)
        brier = float(((f_corr - y_va) ** 2).mean().item())
        val_brier_grid[float(sigma)] = brier
        print(f"  sigma={sigma:>6}: val_brier(f+hat_delta)={brier:.6f}")

    sigma_best = min(val_brier_grid, key=val_brier_grid.get)
    val_brier_best = val_brier_grid[sigma_best]
    print(f"  -> sigma_best={sigma_best} (val_brier={val_brier_best:.6f})")

    # --- Stage 2: alpha fit (range-aware), sigma fixed, fit on VAL ---
    # Refit on the subsampled bank with sigma_best, then compute hat_delta_val
    # and hat_delta_test from the same bank so the two queries are produced
    # under identical conditions.
    smoother_sub = RawEmbeddingKernelSmoother(
        sigma=sigma_best, tau=tau, normalize=True
    )
    smoother_sub.fit(x_bank, f_bank, y_bank)
    hat_delta_val = smoother_sub.predict_hat_delta(x_va, f_va)
    hat_delta_test = smoother_sub.predict_hat_delta(x_te, f_te)

    alpha = float(
        fit_range_aware_scale(
            f_va.cpu().numpy(),
            y_va.cpu().numpy(),
            hat_delta_val.cpu().numpy(),
            objective="brier_score",
        )
    )
    print(f"  alpha (range-aware, brier, fit on val) = {alpha:.6f}")

    test_df = pd.DataFrame(
        {
            "model": model,
            "f": f_te.cpu().numpy(),
            "y": y_te.cpu().numpy(),
            "hat_delta": hat_delta_test.cpu().numpy(),
        }
    )
    meta_row = {
        "model": model,
        "embedding_dim": embed_dim,
        "sigma_best": sigma_best,
        "val_brier_best": val_brier_best,
        "val_brier_grid": json.dumps(val_brier_grid),
        "alpha_range_aware": alpha,
    }
    return test_df, meta_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, default=str(PROJECT_DIR / "outputs")
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model names. If omitted, all subdirs of the dataset folder.",
    )
    parser.add_argument(
        "--sigma_grid",
        nargs="+",
        type=float,
        default=SIGMA_GRID,
        help="Sigma values to grid-search over. Default matches bin/run_datasets.sh.",
    )
    parser.add_argument("--max_bank_size", type=int, default=20000)
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Optional bandwidth for the prediction-space (f) kernel. "
        "Default None matches configs/config.yaml.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Path to save per-example CSV. "
        "Defaults to <output_dir>/<dataset>/raw_hat_delta_test.csv.",
    )
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / args.dataset
    if not dataset_dir.is_dir():
        print(f"Dataset dir not found: {dataset_dir}")
        sys.exit(1)

    if args.models:
        models = args.models
    else:
        models = [d.name for d in sorted(dataset_dir.iterdir()) if d.is_dir()]

    test_dfs, meta_rows = [], []
    torch.use_deterministic_algorithms(False)
    for model in models:
        print(f"\n=== {args.dataset} / {model} ===")
        result = process_model(
            output_dir=output_dir,
            dataset=args.dataset,
            model=model,
            device=args.device,
            sigma_grid=args.sigma_grid,
            max_bank_size=args.max_bank_size,
            tau=args.tau,
            seed=args.seed,
        )
        if result is None:
            continue
        test_df, meta_row = result
        test_dfs.append(test_df)
        meta_rows.append(meta_row)

    if not args.no_save:
        if test_dfs:
            save_path = args.save_csv or str(
                dataset_dir / "raw_hat_delta_test.csv"
            )
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            pd.concat(test_dfs, ignore_index=True).to_csv(save_path, index=False)
            print(f"\nSaved per-example raw hat_delta to {save_path}")
        if meta_rows:
            meta_path = str(dataset_dir / "raw_hat_delta_meta.csv")
            pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
            print(f"Saved meta (sigma_best, alpha, ...) to {meta_path}")


if __name__ == "__main__":
    main()
