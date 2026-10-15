import hydra, torch
import os, shutil
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from calib_atlas.data.datamodule import make_loaders
from calib_atlas.models.phi_mlp import PhiMLP, TokenwisePhi
from calib_atlas.losses.kernel_calib import kernel_calib_discovery_loss
from calib_atlas.losses.baseline_calib import regression_loss
from calib_atlas.utils.seed import seed_everything
from calib_atlas.utils.wandb_utils import wandb_log, init_wandb
from calib_atlas.utils.eval_utils import (
    compute_hat_delta_from_bank,
    gather_data,
    build_bank,
    get_best_delta_log_loss,
    get_corrected_bs_score,
    best_delta_log_loss_soft,
    evaluate_kernel,
    evaluate_regression,
    get_extra_kernel_logs,
    get_extra_resreg_logs,
    get_best_corrected_bs_score,
    get_best_range_aware_bs_score,
)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    embedding_type = "tokens" if cfg.model.type == "kernel_tokenwise" else None

    train_seed = cfg.get("train_seed", None)
    if train_seed is None:
        # Original ordering: phi init before make_loaders. Preserved bit-exact
        # so existing checkpoints / commands keep reproducing.
        if cfg.model.type == "kernel_tokenwise":
            phi = TokenwisePhi(cfg.model).to(cfg.device)
        else:
            phi = PhiMLP(cfg.model).to(cfg.device)
        loaders = make_loaders(cfg.data, embedding_type=embedding_type)
    else:
        # Seed-stability mode: data split is fixed by cfg.seed, phi init and
        # minibatch shuffle controlled separately by train_seed.
        loaders = make_loaders(cfg.data, embedding_type=embedding_type)
        seed_everything(int(train_seed))
        if cfg.model.type == "kernel_tokenwise":
            phi = TokenwisePhi(cfg.model).to(cfg.device)
        else:
            phi = PhiMLP(cfg.model).to(cfg.device)
        print(
            f"[seed-stability] cfg.seed={cfg.seed} (data split)  "
            f"train_seed={train_seed} (phi init / shuffle)"
        )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    label_perm_seed = cfg.get("label_permutation_seed", None)
    if label_perm_seed is not None:
        from calib_atlas.utils.label_perm import (
            build_label_permutation_map,
            wrap_loader_with_label_permutation,
        )
        import numpy as np

        rng = np.random.default_rng(int(label_perm_seed))
        train_perm_map = build_label_permutation_map(train_loader, rng)
        val_perm_map = build_label_permutation_map(val_loader, rng)
        train_loader = wrap_loader_with_label_permutation(
            train_loader, train_perm_map, shuffle=True
        )
        val_loader = wrap_loader_with_label_permutation(
            val_loader, val_perm_map, shuffle=False
        )
        print(
            f"[label-perm] seed={label_perm_seed} | "
            f"permuted train ({len(train_perm_map)}) + val ({len(val_perm_map)}) labels"
        )

    opt = torch.optim.Adam(phi.parameters(), lr=cfg.trainer.lr)

    hydra_cfg = HydraConfig.get()
    hydra_run_dir = hydra_cfg.runtime.output_dir
    run_name = getattr(cfg, "run_name", None)
    is_sweep = HydraConfig.get().mode == RunMode.MULTIRUN
    wandb_cfg = getattr(cfg, "wandb", None)
    wandb_enabled = bool(getattr(wandb_cfg, "enabled", False)) if wandb_cfg else False
    if is_sweep:
        job_id = HydraConfig.get().job.id
        wandb_name = f"{run_name}_sweep_{job_id}"
    else:
        wandb_name = run_name
    save_dir = hydra_run_dir
    os.environ["WANDB_DIR"] = save_dir
    if wandb_enabled:
        init_wandb(cfg, run_name=wandb_name)
    target_metric = getattr(cfg.trainer, "target_metric", "val/loss_detect")
    target_mode = getattr(cfg.trainer, "target_metric_mode", "max").lower()
    maximize_metric = target_mode == "max"
    best_metric = None
    early_stop_metric = getattr(cfg.trainer, "early_stop_metric", "val/loss")
    early_stop_mode = getattr(cfg.trainer, "early_stop_metric_mode", "min").lower()
    early_stop_maximize = early_stop_mode == "max"
    save_enabled = bool(getattr(cfg.trainer, "save", False))
    best_path = None
    best_early_metric = None

    if save_enabled:
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "best_phi.pt")

    patience = getattr(cfg.trainer, "early_stop_patience", None)
    patience = int(patience) if patience is not None else None
    if patience is not None and patience <= 0:
        patience = None
    min_delta = float(getattr(cfg.trainer, "early_stop_min_delta", 0.0))
    no_improve = 0

    def fetch_metric(metrics_dict, metric_name, label):
        value = metrics_dict.get(metric_name)
        if value is None:
            available = ", ".join(sorted(metrics_dict.keys()))
            raise ValueError(
                f"Requested {label} '{metric_name}' not found in logged metrics. "
                f"Available keys this epoch: {available}"
            )
        return float(value)

    def has_improved(current, best, maximize, delta):
        if best is None:
            return True
        if maximize:
            return current > best + delta
        return current < best - delta

    global_step = 0

    for epoch in range(int(cfg.trainer.epochs)):
        # -------- Training --------
        phi.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=True)

        for batch in pbar:
            emb = batch["emb"].to(cfg.device, non_blocking=True)
            f = batch["f"].to(cfg.device, non_blocking=True)
            y = batch["y"].to(cfg.device, non_blocking=True)
            x = batch["emb"].to(cfg.device, non_blocking=True)

            if cfg.model.type == "kernel":
                z = phi(emb)
                out = kernel_calib_discovery_loss(z=z, f=f, y=y, x=x, cfg=cfg.loss)
            elif cfg.model.type == "kernel_tokenwise":
                mask = batch["attention_mask"].to(cfg.device, non_blocking=True)
                emb = emb.squeeze(1)
                mask = mask.squeeze(1)
                z, attn_weights = phi(emb, mask=mask, return_attn=True)
                out = kernel_calib_discovery_loss(z=z, f=f, y=y, x=x, cfg=cfg.loss)
                attn_entropy = (
                    -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean()
                )
                out["loss"] += cfg.loss.lambda_attn_entropy * attn_entropy
                out["attn_entropy"] = attn_entropy.item()
            elif cfg.model.type == "resreg":
                if cfg.model.get("conditional", False):
                    emb = torch.cat([emb, f.unsqueeze(1)], dim=1)
                pred_resid = phi(emb).flatten()
                out = regression_loss(
                    pred_resid=pred_resid,
                    f=f,
                    y=y,
                    cfg=cfg.loss,
                    params=phi.parameters(),
                )
            loss = out["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.trainer.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    phi.parameters(), float(cfg.trainer.grad_clip)
                )
            opt.step()

            should_log = global_step % int(cfg.trainer.log_every) == 0
            if should_log:
                train_metrics = {f"train/{k}": v.item() for k, v in out.items()}
                train_metrics["epoch"] = epoch
                if wandb_enabled:
                    wandb_log(train_metrics, step=global_step)
                pbar.set_postfix(
                    {
                        k.split("/")[-1]: f"{v:.4f}"
                        for k, v in train_metrics.items()
                        if k.startswith("train/")
                    }
                )

            global_step += 1

        # -------- Validation --------
        if cfg.model.type in ["kernel", "kernel_tokenwise"]:
            val_metrics = evaluate_kernel(
                phi, val_loader, cfg.loss, cfg.device, model_type=cfg.model.type
            )
        elif cfg.model.type == "resreg":
            val_metrics = evaluate_regression(phi, val_loader, cfg, cfg.device)

        val_logs = {f"val/{k}": v for k, v in val_metrics.items()}

        val_pack = gather_data(phi, val_loader, cfg)

        if cfg.model.type in ["kernel", "kernel_tokenwise"]:
            bank_z, bank_f, bank_r = build_bank(
                phi, train_loader, device=cfg.device, model_type=cfg.model.type
            )

            hat_delta, mass = compute_hat_delta_from_bank(
                z_query=val_pack["z"],
                f_query=val_pack["f"],
                z_bank=bank_z,
                f_bank=bank_f,
                r_bank=bank_r,
                sigma=float(cfg.loss.sigma),
                tau=float(cfg.loss.tau) if cfg.loss.tau is not None else None,
                eps=float(cfg.loss.eps),
                chunk_q=2048,
            )

            dlog_loss = get_best_delta_log_loss(
                f=val_pack["f"],
                y=val_pack["y"],
                hat_delta=hat_delta,
            )

            dlog_loss_soft = best_delta_log_loss_soft(
                f=val_pack["f"],
                y=val_pack["y"],
                hat_delta=hat_delta,
            )

            corrected_bs_score = get_corrected_bs_score(
                f=val_pack["f"],
                y=val_pack["y"],
                hat_delta=hat_delta,
            )

            best_corrected_bs_score = get_best_corrected_bs_score(
                f=val_pack["f"],
                y=val_pack["y"],
                hat_delta=hat_delta,
            )

            best_range_aware_bs_score = get_best_range_aware_bs_score(
                f=val_pack["f"],
                y=val_pack["y"],
                hat_delta=hat_delta,
            )

            val_logs["val/delta_log_loss"] = dlog_loss
            val_logs["val/delta_log_loss_soft"] = dlog_loss_soft
            val_logs["val/corrected_bs_score"] = corrected_bs_score
            val_logs["val/best_corrected_bs_score"] = best_corrected_bs_score
            val_logs["val/best_range_aware_bs_score"] = best_range_aware_bs_score

        if wandb_enabled:
            wandb_log(val_logs, step=global_step)

        # --- Oracle validation diagnostics (if eta/g exist in val) ---
        extra_logs = {}
        if "eta" in batch or "g" in batch:
            if cfg.model.type in ["kernel", "kernel_tokenwise"]:
                new_extra_logs = get_extra_kernel_logs(val_pack, hat_delta)

            elif cfg.model.type == "resreg":
                new_extra_logs = get_extra_resreg_logs(val_pack)

            for k in new_extra_logs:
                extra_logs[f"val_oracle/{k}"] = new_extra_logs[k]

        # --- Early Stopping & Model Checkpointing ---
        epoch_metrics = dict(val_logs)
        if extra_logs:
            epoch_metrics.update(extra_logs)
            if wandb_enabled:
                wandb_log(extra_logs, step=global_step)

        cached_metrics = {}

        def get_metric(name, label):
            if name not in cached_metrics:
                cached_metrics[name] = fetch_metric(epoch_metrics, name, label)
            return cached_metrics[name]

        target_value = get_metric(target_metric, "target_metric")
        print(f"Epoch {epoch} | {target_metric} = {target_value:.5f}")
        if best_metric is None:
            best_metric = target_value
            if save_enabled:
                torch.save(
                    {
                        "model_state_dict": phi.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        target_metric: target_value,
                        "epoch": epoch,
                    },
                    best_path,
                )
                print("  ↳ initial model saved")
        else:
            is_better = (
                target_value > best_metric
                if maximize_metric
                else target_value < best_metric
            )
            if is_better:
                best_metric = target_value
                if save_enabled:
                    torch.save(
                        {
                            "model_state_dict": phi.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            target_metric: target_value,
                            "epoch": epoch,
                        },
                        best_path,
                    )
                    print("  ↳ new best model saved")

        early_value = get_metric(early_stop_metric, "early_stop_metric")

        if has_improved(early_value, best_early_metric, early_stop_maximize, min_delta):
            best_early_metric = early_value
            no_improve = 0
        else:
            if patience is not None:
                no_improve += 1
                if no_improve >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

    if best_metric is None:
        raise ValueError(
            f"Target metric '{target_metric}' was never computed during training."
        )

    if wandb_enabled:
        wandb_root = os.environ.get("WANDB_DIR", save_dir)
        latest_run_path = os.path.join(wandb_root, "wandb", "latest-run")
        if os.path.exists(latest_run_path):
            try:
                if os.path.islink(latest_run_path) or not os.path.isdir(
                    latest_run_path
                ):
                    os.unlink(latest_run_path)
                else:
                    shutil.rmtree(latest_run_path)
            except OSError as exc:
                print(f"Warning: failed to clean W&B latest-run pointer: {exc}")

    if is_sweep:
        print(
            f"[Sweep job {job_id}] Best {target_metric} ({'max' if maximize_metric else 'min'}): {best_metric:.4f}"
        )
        print(f"Artifacts saved to: {save_dir}")
    else:
        print(
            f"Best {target_metric} ({'max' if maximize_metric else 'min'}): {best_metric:.4f}"
        )
        print(f"Artifacts saved to: {save_dir}")
    return float(best_metric)


if __name__ == "__main__":
    main()
