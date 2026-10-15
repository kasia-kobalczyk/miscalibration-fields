#!/usr/bin/env python3
"""Seed-stability test for the discovery pipeline.

For one (dataset, model) pair, evaluate K trained phi checkpoints
(produced by `bin/run_seed_stability.sh` with `+train_seed=k`,
`outputs/<dataset>/<model>/seed_runs/s<k>/best_phi.pt`) and report:

  - per-seed metrics (field_std, worst_slice_gap, smece slices, ...);
  - pairwise Pearson correlation between hat_delta_test fields;
  - pairwise sign agreement on the {good, over, under} labelling at eps=0.05;
  - aggregate worst_slice_gap mean ± sd across seeds.

Usage:
    # After running bin/run_seed_stability.sh ... --K 5
    python bin/generate_seed_stability.py \
        --dataset hhrlhf_judge --model Qwen3-8B-Base --K 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import pearsonr

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
from calib_atlas.utils.seed import seed_everything


EPS = 0.05
SEED_SUBDIR = "seed_runs"


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
        worst_slice_gap = max(valid) - smece_all
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


def evaluate_phi(phi: PhiMLP, cfg, device: str):
    """Compute (hat_delta_test, f_test, y_test) using `phi` with real labels.

    Re-seeds with `cfg.seed` so that `make_loaders`'s permutation produces the
    exact same train/val/test split that the seed-stability training runs saw.
    """
    seed_everything(int(cfg.seed))
    loaders = make_loaders(cfg.data, splits=["train", "val", "test"])
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    phi.to(device)
    test_pack = gather_data(phi, test_loader, cfg)
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
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output_dir", default=str(PROJECT_DIR / "outputs"),
        help="Root outputs/ directory.",
    )
    args = parser.parse_args()

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

    print(f"\n=== Seed stability: {args.dataset} / {args.model} ===")
    print(f"  sigma={float(cfg.loss.sigma)}  lambda_mass={cfg.loss.lambda_mass}  K={args.K}")

    per_seed_rows = []
    delta_by_seed = []
    f_test_ref = None
    y_test_ref = None

    for k in range(args.K):
        run_dir = output_dir / SEED_SUBDIR / f"s{k:02d}"
        ckpt = run_dir / "best_phi.pt"
        if not ckpt.exists():
            print(f"  [skip] missing {ckpt}")
            continue

        phi_k = load_phi(cfg.model, ckpt)
        hd_k, f_test, y_test = evaluate_phi(phi_k, cfg, args.device)

        if f_test_ref is None:
            f_test_ref = f_test
            y_test_ref = y_test
        else:
            # Sanity: identical test split across seeds (cfg.seed fixed)
            assert np.array_equal(f_test, f_test_ref), (
                "f_test diverged across seeds — data split is not reproducible. "
                "Did you forget to fix cfg.seed?"
            )
            assert np.array_equal(y_test, y_test_ref), (
                "y_test diverged across seeds — see above."
            )

        delta_by_seed.append(hd_k)
        m = compute_metrics(hd_k, f_test, y_test)
        print(
            f"  seed k={k}: field_std={m['field_std']:.4f}  "
            f"worst_slice_gap={m['worst_slice_gap']:.4f}  "
            f"smece_all={m['smece_all']:.4f}"
        )
        per_seed_rows.append({"seed": k, **m})

    K_eff = len(delta_by_seed)
    if K_eff < 2:
        print(f"\nNeed at least 2 trained seeds for stability metrics; got {K_eff}")
        sys.exit(1)

    delta = np.stack(delta_by_seed, axis=0)  # (K_eff, N_test)

    # Pairwise stability metrics over all (i, j) with i < j
    corrs, sign_agreements = [], []
    s_signed = [
        np.sign(np.where(np.abs(d) > EPS, d, 0)) for d in delta
    ]
    for i in range(K_eff):
        for j in range(i + 1, K_eff):
            corrs.append(float(pearsonr(delta[i], delta[j]).statistic))
            sign_agreements.append(float((s_signed[i] == s_signed[j]).mean()))

    corrs = np.asarray(corrs)
    sign_agreements = np.asarray(sign_agreements)
    gaps = np.asarray([row["worst_slice_gap"] for row in per_seed_rows])
    field_stds = np.asarray([row["field_std"] for row in per_seed_rows])

    summary = {
        "K": K_eff,
        "n_pairs": int(len(corrs)),
        "corr_mean": float(corrs.mean()),
        "corr_sd": float(corrs.std(ddof=1)) if len(corrs) > 1 else float("nan"),
        "sign_agree_mean": float(sign_agreements.mean()),
        "sign_agree_sd": (
            float(sign_agreements.std(ddof=1)) if len(sign_agreements) > 1 else float("nan")
        ),
        "gap_mean": float(np.nanmean(gaps)),
        "gap_sd": float(np.nanstd(gaps, ddof=1)) if K_eff > 1 else float("nan"),
        "field_std_mean": float(field_stds.mean()),
        "field_std_sd": float(field_stds.std(ddof=1)) if K_eff > 1 else float("nan"),
    }

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_csv = output_dir / "seed_stability_per_seed.csv"
    per_seed_df.to_csv(per_seed_csv, index=False)
    print(f"\nSaved per-seed metrics to {per_seed_csv}")

    summary_df = pd.DataFrame([summary])
    summary_csv = output_dir / "seed_stability_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")
    print("\n" + summary_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
