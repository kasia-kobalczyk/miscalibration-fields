#!/usr/bin/env python3
"""Compute the per-model ECE table for a dataset.

For each model under ``outputs/<dataset>/``, load the best kernel run, refit
isotonic regression and soft / range-aware Brier-score calibrators on the
training split, and report ECE on the test split (overall + over/under/good
subsets determined by ``|hat_delta|``).

The columns reported are:
    - corrected_f_range_aware_bs
    - corrected_f_soft_bs
    - corrected_f_iso

Usage:
    python bin/generate_ece_table.py --dataset hhrlhf_judge
    python bin/generate_ece_table.py --dataset hellaswag --models Qwen3-1.7B-Base
    python bin/generate_ece_table.py --dataset hhrlhf_judge --models Qwen3-4B-Base gemma-2-2b
    python bin/generate_ece_table.py --dataset hhrlhf_judge --save_csv outputs/hhrlhf_judge/ece_table.csv
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
from sklearn.isotonic import IsotonicRegression

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from calib_atlas.data.datamodule import make_loaders
from calib_atlas.models.phi_mlp import PhiMLP
from calib_atlas.utils.eval_utils import (
    apply_range_aware_correction,
    apply_soft_correction,
    build_bank,
    compute_hat_delta_from_bank,
    fit_range_aware_scale,
    fit_scale,
    gather_data,
    get_smece,
)
from calib_atlas.utils.seed import seed_everything
from calib_atlas.models.baselines import TemperatureScaler

F_COLS = ["f", "corrected_f_range_aware_bs", "corrected_f_iso", "corrected_f_ts"]
# Used by the bootstrap-CI path; adds the raw-embedding kernel baseline so it
# appears alongside the other calibrators when --get_ci is set.
F_COLS_CI = list(F_COLS) + ["corrected_f_range_aware_raw"]


def load_phi_model(cfg, check_point_path: str) -> PhiMLP:
    model = PhiMLP(cfg)
    check_point = torch.load(check_point_path, map_location="cpu", weights_only=False)
    model.load_state_dict(check_point["model_state_dict"])
    model.eval()
    return model


def _run_eval(run_dir: Path, device: str):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    seed_everything(cfg["seed"])
    cfg.device = device

    phi = load_phi_model(cfg.model, str(run_dir / "best_phi.pt"))
    phi.to(cfg.device)

    loaders = make_loaders(cfg.data, splits=["train", "val", "test"])
    train_loader, val_loader, test_loader = (
        loaders["train"],
        loaders["val"],
        loaders["test"],
    )

    test_pack = gather_data(phi, test_loader, cfg)
    val_pack = gather_data(phi, val_loader, cfg)

    bank_z, bank_f, bank_r = build_bank(phi, train_loader, device=cfg.device)

    tau = float(cfg.loss.tau) if cfg.loss.tau is not None else None
    hat_delta_val, _ = compute_hat_delta_from_bank(
        z_query=val_pack["z"],
        f_query=val_pack["f"],
        z_bank=bank_z,
        f_bank=bank_f,
        r_bank=bank_r,
        sigma=float(cfg.loss.sigma),
        tau=tau,
        eps=float(cfg.loss.eps),
        chunk_q=2048,
    )
    hat_delta_test, _ = compute_hat_delta_from_bank(
        z_query=test_pack["z"],
        f_query=test_pack["f"],
        z_bank=bank_z,
        f_bank=bank_f,
        r_bank=bank_r,
        sigma=float(cfg.loss.sigma),
        tau=tau,
        eps=float(cfg.loss.eps),
        chunk_q=2048,
    )

    return {
        "f_test": test_pack["f"].cpu().numpy(),
        "y_test": test_pack["y"].cpu().numpy(),
        "hat_delta_test": hat_delta_test.cpu().numpy(),
        "f_val": val_pack["f"].cpu().numpy(),
        "y_val": val_pack["y"].cpu().numpy(),
        "hat_delta_val": hat_delta_val.cpu().numpy(),
    }


def _load_raw_baseline(dataset_dir: Path, model: str):
    """Load the precomputed raw-embedding kernel baseline for ``model`` from
    ``outputs/<dataset>/raw_hat_delta_{test,meta}.csv``.

    Returns a dict ``{"f": np.ndarray, "hat_delta": np.ndarray, "alpha": float}``
    or ``None`` if either CSV is missing or has no row for ``model``.
    """
    test_csv = dataset_dir / "raw_hat_delta_test.csv"
    meta_csv = dataset_dir / "raw_hat_delta_meta.csv"
    if not test_csv.exists() or not meta_csv.exists():
        return None
    raw_test = pd.read_csv(test_csv)
    raw_meta = pd.read_csv(meta_csv).set_index("model")
    sub = raw_test[raw_test["model"] == model]
    if len(sub) == 0 or model not in raw_meta.index:
        return None
    return {
        "f": sub["f"].to_numpy(),
        "hat_delta": sub["hat_delta"].to_numpy(),
        "alpha": float(raw_meta.loc[model, "alpha_range_aware"]),
    }


def _build_test_df(pack: dict, raw_baseline: dict | None = None) -> pd.DataFrame:
    f_test, y_test = pack["f_test"], pack["y_test"]
    hat_delta_test = pack["hat_delta_test"]
    f_val, y_val = pack["f_val"], pack["y_val"]
    hat_delta_val = pack["hat_delta_val"]

    # All post-hoc calibrators are fit on the held-out validation split:
    # - hat_delta_val is computed under the same bank/conditions as
    #   hat_delta_test (no exclude_self), so the sharpness regime matches at
    #   test time;
    # - ISO / TS are fit on (f_val, y_val) for the same reason — train hat_delta
    #   would over-tune to a sharper kernel estimate that doesn't generalise.
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(f_val, y_val)
    f_test_iso = iso.predict(f_test)

    ts = TemperatureScaler()
    ts.fit(f_val, y_val)
    f_test_ts = ts.predict_proba(f_test)

    scale_soft_bs = fit_scale(
        f_val, y_val, hat_delta_val, correction="soft", objective="brier_score"
    )
    scale_range_aware_bs = fit_range_aware_scale(
        f_val, y_val, hat_delta_val, objective="brier_score"
    )

    f_test_soft_bs = apply_soft_correction(f_test, hat_delta_test, scale_soft_bs)
    f_test_range_aware_bs = apply_range_aware_correction(
        torch.as_tensor(f_test), torch.as_tensor(hat_delta_test), scale_range_aware_bs
    ).numpy()

    out = {
        "y": y_test,
        "hat_delta": hat_delta_test,
        "f": f_test,
        "corrected_f_range_aware_bs": f_test_range_aware_bs,
        "corrected_f_soft_bs": f_test_soft_bs,
        "corrected_f_iso": f_test_iso,
        "corrected_f_ts": f_test_ts,
    }

    # Raw-embedding kernel baseline (precomputed by bin/generate_raw_hat_delta.py).
    # Optional — only included when raw CSVs are present (used by --get_ci mode).
    if raw_baseline is not None:
        # Sanity check: row order must match (both come from a deterministic
        # make_loaders call seeded by cfg.seed, so f_test should be bit-identical).
        if not np.allclose(raw_baseline["f"], f_test, atol=1e-5):
            raise RuntimeError(
                "raw_hat_delta_test.csv f-values do not match the in-memory "
                "f_test. Re-run bin/generate_raw_hat_delta.py — likely the "
                "data seed/config changed since it was last generated."
            )
        f_test_range_aware_raw = apply_range_aware_correction(
            torch.from_numpy(np.array(f_test, copy=True)),
            torch.from_numpy(np.array(raw_baseline["hat_delta"], copy=True)),
            float(raw_baseline["alpha"]),
        ).numpy()
        out["corrected_f_range_aware_raw"] = f_test_range_aware_raw

    return pd.DataFrame(out)


def compute_ece_table(test_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = test_df.copy()
    df["subset_hat_delta"] = np.where(
        df["hat_delta"] < -threshold,
        "over",
        np.where(df["hat_delta"] > threshold, "under", "good"),
    )

    def _eces(d):
        return [get_smece(d[c], d["y"]) for c in F_COLS]

    ece_df = pd.DataFrame([_eces(df)], index=["All data"], columns=F_COLS)

    sub = df.groupby("subset_hat_delta")[["y", *F_COLS]].apply(_eces)
    sub_df = pd.DataFrame(sub.values.tolist(), index=sub.index, columns=F_COLS)
    ece_df = pd.concat([ece_df, sub_df], axis=0)

    counts = (df["subset_hat_delta"].value_counts() / len(df) * 100).round(1)
    ece_df = ece_df.merge(counts, left_index=True, right_index=True, how="left").fillna(
        100.0
    )
    ece_df = ece_df.rename(columns={"count": "% of data"})
    ece_df.index.name = "Subset"
    return ece_df


def compute_brier_table(test_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-slice Brier score for each F_col, mirroring compute_ece_table."""
    df = test_df.copy()
    df["subset_hat_delta"] = np.where(
        df["hat_delta"] < -threshold,
        "over",
        np.where(df["hat_delta"] > threshold, "under", "good"),
    )

    def _briers(d):
        y = np.asarray(d["y"], dtype=float)
        return [
            float(np.mean((np.asarray(d[c], dtype=float) - y) ** 2)) for c in F_COLS
        ]

    brier_df = pd.DataFrame([_briers(df)], index=["All data"], columns=F_COLS)

    sub = df.groupby("subset_hat_delta")[["y", *F_COLS]].apply(_briers)
    sub_df = pd.DataFrame(sub.values.tolist(), index=sub.index, columns=F_COLS)
    brier_df = pd.concat([brier_df, sub_df], axis=0)

    counts = (df["subset_hat_delta"].value_counts() / len(df) * 100).round(1)
    brier_df = brier_df.merge(
        counts, left_index=True, right_index=True, how="left"
    ).fillna(100.0)
    brier_df = brier_df.rename(columns={"count": "% of data"})
    brier_df.index.name = "Subset"
    return brier_df


def compute_bootstrap_ci_table(
    test_df: pd.DataFrame,
    threshold: float,
    n_boot: int,
    ci_level: float,
    seed: int,
    f_cols: list[str] | None = None,
    include_brier: bool = False,
) -> pd.DataFrame:
    """Bootstrap CIs for smECE, optionally Brier, and slice proportions.

    Returns a long-form DataFrame with columns:
        Subset, F_col, metric, mean, ci_low, ci_high, n_boot

    Two complementary bootstrap regimes are run from a shared rng:
      Regime A (within-slice) — for smECE / Brier / worst-slice gap.
        Slice membership is fixed; resample examples within each slice.
      Regime B (full-test)    — for slice proportions (pct_data).
        Resample the full test set and recompute proportions.

    Brier rows (per-slice, worst-all gap, pct_worst_brier) are emitted only
    when ``include_brier=True``.
    """
    if f_cols is None:
        f_cols = F_COLS_CI

    metrics_to_run = ["smece"] + (["brier"] if include_brier else [])

    # Slice labelling (matches compute_ece_table)
    df = test_df.copy()
    df["subset_hat_delta"] = np.where(
        df["hat_delta"] < -threshold,
        "over",
        np.where(df["hat_delta"] > threshold, "under", "good"),
    )
    slice_labels = df["subset_hat_delta"].to_numpy()
    n_test = len(df)

    f_arrays = {c: df[c].to_numpy(dtype=float) for c in f_cols if c in df.columns}
    missing = [c for c in f_cols if c not in df.columns]
    if missing:
        print(f"  [warn] f_cols missing from test_df, skipping: {missing}")
    f_cols = [c for c in f_cols if c in df.columns]
    y = df["y"].to_numpy(dtype=float)

    # Slice masks + indices (in canonical iteration order)
    slice_names = ["All data", "good", "over", "under"]
    slice_indices = {
        "All data": np.arange(n_test),
        "good": np.flatnonzero(slice_labels == "good"),
        "over": np.flatnonzero(slice_labels == "over"),
        "under": np.flatnonzero(slice_labels == "under"),
    }

    rng = np.random.default_rng(int(seed))
    alpha = 1.0 - float(ci_level)
    lo_p, hi_p = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

    # Regime A: within-slice resampling
    # boot_metrics[(slice, F_col, metric)] is an array of length n_boot.
    boot_metrics: dict[tuple, np.ndarray] = {}
    for slice_name in slice_names:
        idx_pool = slice_indices[slice_name]
        n_slice = len(idx_pool)
        if n_slice < 2:
            for c in f_cols:
                for m in metrics_to_run:
                    boot_metrics[(slice_name, c, m)] = np.full(n_boot, np.nan)
            continue

        boot_idx = rng.choice(idx_pool, size=(n_boot, n_slice), replace=True)
        for c in f_cols:
            f_c = f_arrays[c]
            sm = np.empty(n_boot, dtype=float)
            if include_brier:
                br = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                ii = boot_idx[b]
                sm[b] = float(get_smece(f_c[ii], y[ii]))
                if include_brier:
                    br[b] = float(np.mean((f_c[ii] - y[ii]) ** 2))
            boot_metrics[(slice_name, c, "smece")] = sm
            if include_brier:
                boot_metrics[(slice_name, c, "brier")] = br

    # Worst-slice gap per F_col per bootstrap (max over {good, over, under} minus All data)
    gap_metrics: dict[tuple, np.ndarray] = {}
    for c in f_cols:
        for metric in metrics_to_run:
            stk = np.stack(
                [
                    boot_metrics[("All data", c, metric)],
                    boot_metrics[("good", c, metric)],
                    boot_metrics[("over", c, metric)],
                    boot_metrics[("under", c, metric)],
                ],
                axis=0,
            )  # (3, n_boot)
            with np.errstate(invalid="ignore"):
                worst = np.nanmax(stk, axis=0)  # (n_boot,)
            all_data = boot_metrics[("All data", c, metric)]
            gap_metrics[(c, metric)] = worst - all_data

    # Regime B: full-test resampling for slice proportions
    boot_pcts: dict[str, np.ndarray] = {
        s: np.empty(n_boot, dtype=float) for s in ("good", "over", "under")
    }
    full_boot_idx = rng.choice(n_test, size=(n_boot, n_test), replace=True)
    for b in range(n_boot):
        labs_b = slice_labels[full_boot_idx[b]]
        for s in ("good", "over", "under"):
            boot_pcts[s][b] = 100.0 * float(np.mean(labs_b == s))

    # Worst-slice proportion: for each (F_col, metric, b), find the slice with
    # the highest within-slice metric in this bootstrap and record its baseline
    # proportion. Slice sizes are fixed under Regime A, so the only randomness
    # captured here is *which* slice wins the argmax across bootstraps.
    slice_keys = ("good", "over", "under")
    slice_pct_baseline = np.array(
        [100.0 * len(slice_indices[s]) / n_test for s in slice_keys]
    )
    pct_worst_metrics: dict[tuple, np.ndarray] = {}
    for c in f_cols:
        for metric in metrics_to_run:
            stk = np.stack(
                [boot_metrics[(s, c, metric)] for s in slice_keys], axis=0
            )  # (3, n_boot)
            with np.errstate(invalid="ignore"):
                all_nan = np.all(np.isnan(stk), axis=0)
                stk_safe = np.where(np.isnan(stk), -np.inf, stk)
                worst_idx = np.argmax(stk_safe, axis=0)  # (n_boot,) in {0, 1, 2}
                pcts = slice_pct_baseline[worst_idx]
                pcts = pcts.astype(float, copy=True)
                pcts[all_nan] = np.nan
            pct_worst_metrics[(c, metric)] = pcts

    # Aggregate to mean / ci_low / ci_high in long-form
    rows = []

    # Per-slice smECE / Brier (including All data)
    for slice_name in slice_names:
        for c in f_cols:
            for metric in metrics_to_run:
                vals = boot_metrics[(slice_name, c, metric)]
                if np.all(np.isnan(vals)):
                    continue
                rows.append(
                    {
                        "Subset": slice_name,
                        "F_col": c,
                        "metric": metric,
                        "mean": float(np.nanmean(vals)),
                        "ci_low": float(np.nanpercentile(vals, lo_p)),
                        "ci_high": float(np.nanpercentile(vals, hi_p)),
                        "n_boot": int(n_boot),
                    }
                )

    # Worst-slice gap rows (per F_col per metric)
    for c in f_cols:
        for metric in metrics_to_run:
            vals = gap_metrics[(c, metric)]
            if np.all(np.isnan(vals)):
                continue
            rows.append(
                {
                    "Subset": "worst-all gap",
                    "F_col": c,
                    "metric": metric,
                    "mean": float(np.nanmean(vals)),
                    "ci_low": float(np.nanpercentile(vals, lo_p)),
                    "ci_high": float(np.nanpercentile(vals, hi_p)),
                    "n_boot": int(n_boot),
                }
            )

    # Slice proportions (pct_data) — F_col NaN since proportions don't depend on calibrator
    for s in ("good", "over", "under"):
        vals = boot_pcts[s]
        rows.append(
            {
                "Subset": s,
                "F_col": None,
                "metric": "pct_data",
                "mean": float(vals.mean()),
                "ci_low": float(np.percentile(vals, lo_p)),
                "ci_high": float(np.percentile(vals, hi_p)),
                "n_boot": int(n_boot),
            }
        )

    # Proportion of test data in the worst slice (per F_col per metric).
    # Subset = "worst slice"; one row per (F_col, metric).
    for c in f_cols:
        for metric in metrics_to_run:
            vals = pct_worst_metrics.get((c, metric))
            if vals is None or np.all(np.isnan(vals)):
                continue
            rows.append(
                {
                    "Subset": "worst slice",
                    "F_col": c,
                    "metric": f"pct_worst_{metric}",
                    "mean": float(np.nanmean(vals)),
                    "ci_low": float(np.nanpercentile(vals, lo_p)),
                    "ci_high": float(np.nanpercentile(vals, hi_p)),
                    "n_boot": int(n_boot),
                }
            )

    return pd.DataFrame(rows)


def process_model(
    output_dir: Path,
    dataset: str,
    model: str,
    device: str,
    raw_baseline: dict | None = None,
):
    best_run_path = output_dir / dataset / model / "best_run.json"
    if not best_run_path.exists():
        print(f"  [skip] no best_run.json at {best_run_path}")
        return None

    with open(best_run_path) as f:
        best_run = json.load(f)
    run_dir = Path(best_run["run_dir"])

    torch.use_deterministic_algorithms(False)
    pack = _run_eval(run_dir, device=device)
    return _build_test_df(pack, raw_baseline=raw_baseline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_DIR / "outputs"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model names. If omitted, all subdirs of the dataset folder.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.15, 0.1, 0.05, 0.01],
        help="Thresholds on |hat_delta| for the over/under/good split.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Path to save the combined long-form CSV. "
        "Defaults to <output_dir>/<dataset>/ece_table.csv.",
    )
    parser.add_argument("--no_save", action="store_true", help="Disable CSV saving.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for phi inference (e.g. 'cuda:0', 'cpu', 'auto').",
    )
    parser.add_argument(
        "--get_ci",
        action="store_true",
        help="Bootstrap-CI mode. Skips writing ece_table.csv / brier_table.csv "
        "and instead writes outputs/<dataset>/results_with_ci.csv with smECE, "
        "Brier, worst-slice gaps, and slice proportions (one threshold only). "
        "Requires raw_hat_delta_test.csv and raw_hat_delta_meta.csv to exist "
        "alongside best_run.json so the raw-embedding kernel baseline is "
        "included as a column.",
    )
    parser.add_argument(
        "--ci_threshold",
        type=float,
        default=0.05,
        help="Single threshold on |hat_delta| used in --get_ci mode.",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=200,
        help="Number of bootstrap iterations (used by --get_ci).",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level (used by --get_ci).",
    )
    parser.add_argument(
        "--ci_seed", type=int, default=0, help="Bootstrap RNG seed (used by --get_ci)."
    )
    parser.add_argument(
        "--include_brier",
        action="store_true",
        help="Also compute Brier rows in --get_ci mode (per-slice, worst-all "
        "gap, and pct_worst_brier). Off by default — keeps the CI table "
        "focused on smECE.",
    )
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

    all_tables = []
    all_brier_tables = []
    all_ci_rows = []
    stats_rows = []
    hat_delta_rows = []
    for model in models:
        print(f"\n=== {args.dataset} / {model} ===")
        # In CI mode we always need the raw baseline (per the spec).
        raw_baseline = _load_raw_baseline(dataset_dir, model) if args.get_ci else None
        if args.get_ci and raw_baseline is None:
            print(
                f"  [skip] --get_ci requires raw_hat_delta_*.csv for {model} "
                f"under {dataset_dir}; not found"
            )
            continue

        test_df = process_model(
            output_dir,
            args.dataset,
            model,
            device=args.device,
            raw_baseline=raw_baseline,
        )
        if test_df is None:
            continue

        hat_delta = test_df["hat_delta"].to_numpy()
        stats = {
            "model": model,
            "hat_delta_var": float(np.var(hat_delta)),
            "hat_delta_std": float(np.std(hat_delta)),
            "hat_delta_mean": float(np.mean(hat_delta)),
            "hat_delta_abs_mean": float(np.mean(np.abs(hat_delta))),
            "n_test": int(len(hat_delta)),
        }
        stats_rows.append(stats)
        print(
            f"  hat_delta: var={stats['hat_delta_var']:.6f} "
            f"std={stats['hat_delta_std']:.6f} "
            f"mean={stats['hat_delta_mean']:.6f}"
        )

        hat_delta_rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "hat_delta": hat_delta,
                    "f": test_df["f"].to_numpy(),
                    "y": test_df["y"].to_numpy(),
                }
            )
        )

        if args.get_ci:
            print(
                f"-- bootstrap CI (T={args.ci_threshold}, B={args.n_boot}, "
                f"level={args.ci_level}, brier={args.include_brier}) --"
            )
            ci_df = compute_bootstrap_ci_table(
                test_df,
                threshold=args.ci_threshold,
                n_boot=args.n_boot,
                ci_level=args.ci_level,
                seed=args.ci_seed,
                include_brier=args.include_brier,
            )
            ci_df.insert(0, "model", model)
            ci_df.insert(1, "threshold", args.ci_threshold)
            print(ci_df.round(4).to_markdown(index=False))
            all_ci_rows.append(ci_df)
            continue  # Skip the per-threshold ECE/Brier tables in CI mode

        for threshold in args.thresholds:
            ece_df = compute_ece_table(test_df, threshold=threshold)
            print(f"\n-- threshold T = {threshold} | smECE --")
            print(ece_df.round(4).to_markdown())
            tagged = ece_df.reset_index()
            tagged.insert(0, "model", model)
            tagged.insert(1, "threshold", threshold)
            all_tables.append(tagged)

            brier_df = compute_brier_table(test_df, threshold=threshold)
            print(f"\n-- threshold T = {threshold} | Brier --")
            print(brier_df.round(4).to_markdown())
            tagged_b = brier_df.reset_index()
            tagged_b.insert(0, "model", model)
            tagged_b.insert(1, "threshold", threshold)
            all_brier_tables.append(tagged_b)

    if not args.no_save:
        if args.get_ci:
            # CI mode: only write the bootstrap-CI table; leave the existing
            # point-estimate ece/brier tables and hat_delta CSVs untouched.
            if all_ci_rows:
                ci_path = str(dataset_dir / "results_with_ci.csv")
                combined_ci = pd.concat(all_ci_rows, ignore_index=True)
                os.makedirs(os.path.dirname(ci_path) or ".", exist_ok=True)
                combined_ci.to_csv(ci_path, index=False)
                print(f"\nSaved bootstrap-CI table to {ci_path}")
        else:
            if all_tables:
                save_path = args.save_csv or str(dataset_dir / "ece_table.csv")
                combined = pd.concat(all_tables, ignore_index=True)
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                combined.to_csv(save_path, index=False)
                print(f"\nSaved combined ECE table to {save_path}")

            if all_brier_tables:
                brier_path = str(dataset_dir / "brier_table.csv")
                combined_brier = pd.concat(all_brier_tables, ignore_index=True)
                os.makedirs(os.path.dirname(brier_path) or ".", exist_ok=True)
                combined_brier.to_csv(brier_path, index=False)
                print(f"Saved combined Brier table to {brier_path}")

            if stats_rows:
                stats_path = str(dataset_dir / "hat_delta_stats.csv")
                pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
                print(f"Saved hat_delta stats to {stats_path}")

            if hat_delta_rows:
                hat_delta_path = str(dataset_dir / "hat_delta_test.csv")
                pd.concat(hat_delta_rows, ignore_index=True).to_csv(
                    hat_delta_path, index=False
                )
                print(f"Saved per-example hat_delta values to {hat_delta_path}")


if __name__ == "__main__":
    main()
