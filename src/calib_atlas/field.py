"""Load a trained run and estimate its miscalibration field.

    phi, cfg = load_run("outputs/synthetic_3_clusters/<timestamp>")
    loaders = make_run_loaders(cfg, splits=["train", "val", "test"])
    test = estimate_field(phi, cfg, bank_loader=loaders["train"], query_loader=loaders["test"])
    test["hat_delta"]  # signed field estimate per test example: > 0 under-, < 0 overconfident
"""

import os

import torch
from omegaconf import OmegaConf

from calib_atlas.data.datamodule import make_loaders
from calib_atlas.models.phi_mlp import PhiMLP
from calib_atlas.utils.eval_utils import build_bank, compute_hat_delta_from_bank, gather_data
from calib_atlas.utils.seed import seed_everything


def load_run(run_dir: str, device: str = "cpu"):
    """Return the trained representation phi and the run's config (with `device` overridden)."""
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    cfg.device = device
    phi = PhiMLP(cfg.model)
    ckpt = torch.load(os.path.join(run_dir, "best_phi.pt"), map_location=device)
    phi.load_state_dict(ckpt["model_state_dict"])
    return phi.to(device).eval(), cfg


def make_run_loaders(cfg, splits=("train", "val", "test")):
    """Rebuild the run's data loaders with the same train/val/test split used in training."""
    seed_everything(cfg.seed)
    return make_loaders(cfg.data, splits=list(splits))


@torch.no_grad()
def estimate_field(phi, cfg, bank_loader, query_loader, max_bank_size: int = 20000):
    """Kernel-smooth residuals y - f from `bank_loader` in phi-space at the points of `query_loader`.

    Returns the gathered query data (emb, f, y, z, ...) plus `hat_delta` (the field estimate)
    and `mass` (total kernel weight, i.e. how much bank data supports each estimate).
    """
    bank_z, bank_f, bank_r = build_bank(
        phi, bank_loader, device=cfg.device, max_bank_size=max_bank_size
    )
    out = gather_data(phi, query_loader, cfg)
    out["hat_delta"], out["mass"] = compute_hat_delta_from_bank(
        z_query=out["z"],
        f_query=out["f"],
        z_bank=bank_z,
        f_bank=bank_f,
        r_bank=bank_r,
        sigma=float(cfg.loss.sigma),
        tau=float(cfg.loss.tau) if cfg.loss.tau is not None else None,
        eps=float(cfg.loss.eps),
    )
    return out
