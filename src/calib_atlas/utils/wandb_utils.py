import os
import wandb
from omegaconf import OmegaConf


def init_wandb(cfg, run_name):
    if not cfg.wandb.enabled:
        return None

    return wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def wandb_log(metrics, step):
    if wandb.run is not None:
        wandb.log(metrics, step=step)
