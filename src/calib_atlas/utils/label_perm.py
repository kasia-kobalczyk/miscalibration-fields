"""Label-permutation utilities for the discovery-pipeline null test.

The training and evaluation scripts both need to construct the *same*
permutation of training labels given a seed, so the helpers here are the
single source of truth.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LabelPermutedDataset(Dataset):
    """Wraps any Dataset returning a dict that includes ``id`` and ``y`` and
    overrides ``y`` via a precomputed ``id -> y_perm`` map. Items whose id
    is missing from the map are returned unchanged.
    """

    def __init__(self, base_ds: Dataset, perm_map: dict):
        self.base_ds = base_ds
        self.perm_map = perm_map

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, i):
        item = self.base_ds[i]
        key = _id_key(item["id"])
        if key in self.perm_map:
            item = dict(item)
            item["y"] = torch.as_tensor(self.perm_map[key])
        return item


def _id_key(x):
    """Coerce an id value to a hashable Python scalar.

    Handles both numeric ids (HHRLHF, MMLU, HellaSwag — int) and string ids
    (MedMCQA — UUID string). Tensor 0-d ids get unwrapped via ``.item()``.
    """
    if isinstance(x, torch.Tensor):
        return x.item()
    return x


def _to_int(x):
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def build_label_permutation_map(loader: DataLoader, rng: np.random.Generator) -> dict:
    """Iterate ``loader.dataset`` in index order, gather (id, y), permute the
    y array with ``rng``, and return a ``id -> y_perm`` dict.

    ``rng`` is advanced; calling this twice with the same generator (once for
    train, once for val) draws two independent permutations from the same
    seeded stream. This matches the user's reference pseudocode:
        rng = np.random.default_rng(seed)
        y_train_null = rng.permutation(y_train)
        y_val_null   = rng.permutation(y_val)
    """
    ids, ys = [], []
    ds = loader.dataset
    for i in range(len(ds)):
        item = ds[i]
        ids.append(_id_key(item["id"]))
        ys.append(_to_int(item["y"]))
    ys_arr = np.asarray(ys)
    ys_perm = rng.permutation(ys_arr)
    return dict(zip(ids, ys_perm.tolist()))


def wrap_loader_with_label_permutation(
    loader: DataLoader, perm_map: dict, shuffle: bool
) -> DataLoader:
    """Build a new DataLoader around a LabelPermutedDataset wrapping the
    given loader's underlying dataset, preserving batch/worker settings.
    """
    return DataLoader(
        LabelPermutedDataset(loader.dataset, perm_map),
        batch_size=loader.batch_size,
        shuffle=shuffle,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=getattr(loader, "persistent_workers", False),
    )
