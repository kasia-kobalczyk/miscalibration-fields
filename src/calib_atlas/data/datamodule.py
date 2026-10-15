import numpy as np
from torch.utils.data import DataLoader, Subset
from .dataset import *
from datasets import load_from_disk
import torch
import os

def collate(b):
    batch = {}
    for d in b:
        for k, v in d.items():
            batch[k] = batch.get(k, []) + [v]
    for k in batch:
        if isinstance(batch[k][0], torch.Tensor):
            batch[k] = torch.stack(batch[k])
        else:
            batch[k] = batch[k]

    return batch


def make_loaders(cfg, embedding_type=None, splits=["train", "val"]):
    if hasattr(cfg, "path"):
        path = os.path.join(DATASETS_DIR, cfg.path)  # an absolute cfg.path is used as is
        dataset = PathDataset(path)
    elif hasattr(cfg, "model_name") and hasattr(cfg, "name"):
        if cfg.name == "hhrlhf_judge":
            dataset = HHRLHFDataset(cfg.model_name, embedding_type=embedding_type)
        elif cfg.name == "arc_challenge":
            dataset = ARCDataset(
                cfg.model_name, variant="ARC-Challenge", embedding_type=embedding_type
            )
        elif cfg.name == "arc_easy":
            dataset = ARCDataset(
                cfg.model_name, variant="ARC-Easy", embedding_type=embedding_type
            )
        elif cfg.name == "mmlu":
            dataset = MMLU(
                cfg.model_name, variant="mmlu", embedding_type=embedding_type
            )
        elif cfg.name == "mmlu_pro":
            dataset = MMLU(
                cfg.model_name, variant="mmlu_pro", embedding_type=embedding_type
            )
        elif cfg.name == "medmcqa":
            dataset = MedMCQA(cfg.model_name, embedding_type=embedding_type)
        elif cfg.name == "hellaswag":
            dataset = HellaSwag(cfg.model_name, embedding_type=embedding_type)
        else:
            raise ValueError(f"Unknown dataset name {cfg.name} for HHRLHFDataset")
    N = len(dataset)
    idx = np.random.permutation(N)
    nt = int(N * cfg.test_frac)
    nv = int(N * cfg.val_frac)
    tr, va, te = idx[: N - nv - nt], idx[N - nv - nt : N - nt], idx[N - nt :]

    if "train" in splits:
        train_loader = DataLoader(
            Subset(dataset, tr),
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
    else:
        train_loader = None
    if "val" in splits:
        val_loader = DataLoader(
            Subset(dataset, va),
            batch_size=cfg.batch_size,
            collate_fn=collate,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
    else:
        val_loader = None
    if "test" in splits:
        test_loader = DataLoader(
            Subset(dataset, te),
            batch_size=cfg.batch_size,
            collate_fn=collate,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.num_workers > 0,
        )
    else:
        test_loader = None

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
