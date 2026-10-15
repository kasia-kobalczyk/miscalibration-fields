import os
import numpy as np, torch
from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset
import pandas as pd

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
# Where datasets live: <repo>/datasets by default, override with $CALIB_ATLAS_DATA.
DATASETS_DIR = os.environ.get("CALIB_ATLAS_DATA", os.path.join(ROOT, "datasets"))


class PathDataset(Dataset):
    def __init__(
        self,
        path,
    ):
        self.ds = load_from_disk(path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[int(i)]
        out = {}
        for k, v in row.items():
            out[k] = torch.as_tensor(v)
        return out


class EmbeddingDataset(Dataset):
    """Base class for embedding-based datasets with probability and label columns."""

    def __init__(self, model_name, embedding_type=None):
        self.model_name = model_name
        if embedding_type is None:
            embedding_type = "last_token"
        self.embedding_type = embedding_type
        self.data_dir = self._get_data_dir()
        self.embeddings = self._load_embeddings()
        self.probas = self._load_probas()
        assert all(self.embeddings["id"] == self.probas["id"])

    def _get_data_dir(self):
        """Return the directory path where data is stored. Override in subclass."""
        raise NotImplementedError

    def _load_embeddings(self):
        """Load embeddings from disk using standard path structure."""
        if self.embedding_type == "tokens":
            path = f"{self.data_dir}/{self.model_name}/token_embeddings"
            cols = ["id", "embedding", "attention_mask", "input_ids"]
        else:
            path = f"{self.data_dir}/{self.model_name}/embeddings"
            cols = ["id", "embedding"]
        return load_from_disk(path).with_format("torch", columns=cols)

    def _load_probas(self):
        """Load probability dataframe. Override in subclass."""
        raise NotImplementedError

    def _get_prob_column(self):
        """Return the name of the probability column. Override in subclass."""
        raise NotImplementedError

    def _get_label(self, row):
        """Extract binary label from a row. Override in subclass."""
        raise NotImplementedError

    def __len__(self):
        return len(self.probas)

    def __getitem__(self, i):
        i = int(i)
        emb = self.embeddings[i]["embedding"]
        f = torch.as_tensor(self.probas.iloc[i][self._get_prob_column()])
        y = torch.as_tensor(self._get_label(self.probas.iloc[i]))
        out = {
            "emb": emb,
            "f": f,
            "y": y,
            "id": self.probas.iloc[i]["id"],
        }
        if self.embedding_type == "tokens":
            out["attention_mask"] = self.embeddings[i]["attention_mask"]
            out["input_ids"] = self.embeddings[i]["input_ids"]

        return out


class HHRLHFDataset(EmbeddingDataset):
    """HHRLHF dataset for preference learning."""

    def _get_data_dir(self):
        return f"{DATASETS_DIR}/hhrlhf_judge/"

    def _load_probas(self):
        probas_df = pd.read_json(
            f"{self.data_dir}/{self.model_name}/probas.jsonl", lines=True
        )
        df = pd.read_csv(f"{self.data_dir}/processed_data.csv")
        return probas_df.merge(
            df[["id", "correct_response", "subset"]], on="id", how="left"
        )

    def _get_prob_column(self):
        return "p_A"

    def _get_label(self, row):
        return int(row["correct_response"] == "A")


class ARCDataset(EmbeddingDataset):
    """ARC dataset for multiple-choice question answering."""

    def __init__(self, model_name, variant="ARC-Challenge", embedding_type=None):
        self.variant_arg = variant
        super().__init__(model_name, embedding_type)

    def _get_data_dir(self):
        if self.variant_arg == "ARC-Challenge":
            variant = "arc_challenge"
        elif self.variant_arg == "ARC-Easy":
            variant = "arc_easy"
        else:
            variant = self.variant_arg
        return f"{DATASETS_DIR}/{variant}/"

    def _load_probas(self):
        return pd.read_json(
            f"{self.data_dir}/{self.model_name}/probas.jsonl", lines=True
        )

    def _get_prob_column(self):
        return "p_argmax"

    def _get_label(self, row):
        return int(row["correct"])


class MMLU(EmbeddingDataset):
    """MMLU dataset for multiple-choice question answering."""

    def __init__(self, model_name, variant="mmlu", embedding_type=None):
        self.variant_arg = variant
        super().__init__(model_name, embedding_type)

    def _get_data_dir(self):
        return f"{DATASETS_DIR}/{self.variant_arg}/"

    def _load_probas(self):
        return pd.read_json(
            f"{self.data_dir}/{self.model_name}/probas.jsonl", lines=True
        )

    def _get_prob_column(self):
        return "p_argmax"

    def _get_label(self, row):
        return int(row["correct"])


class MedMCQA(EmbeddingDataset):
    """MedMCQA dataset for multiple-choice question answering."""

    def _get_data_dir(self):
        return f"{DATASETS_DIR}/medmcqa/"

    def _load_probas(self):
        return pd.read_json(
            f"{self.data_dir}/{self.model_name}/probas.jsonl", lines=True
        )

    def _get_prob_column(self):
        return "p_argmax"

    def _get_label(self, row):
        return int(row["correct"])


class HellaSwag(EmbeddingDataset):
    """HellaSwag dataset for commonsense sentence completion."""

    def _get_data_dir(self):
        return f"{DATASETS_DIR}/hellaswag/"

    def _load_probas(self):
        return pd.read_json(
            f"{self.data_dir}/{self.model_name}/probas.jsonl", lines=True
        )

    def _get_prob_column(self):
        return "p_argmax"

    def _get_label(self, row):
        return int(row["correct"])
