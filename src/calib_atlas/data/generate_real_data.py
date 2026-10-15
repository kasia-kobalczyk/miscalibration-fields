import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from calib_atlas.utils.data_utils import (
    elicit_possible_answer_probas,
    get_embedding,
    get_token_embeddings,
)

# None defers to the standard Hugging Face cache locations (HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE).
HF_HUB_DEFAULT = None
HF_DATASET_HUB_DEFAULT = None

MCQ_PROMPT_TEMPLATE = """
Question: {question}

{choices}

The correct answer is:
""".lstrip()

HELLASWAG_PROMPT_TEMPLATE = """
Context: {question}

{choices}

The correct completion is:
""".lstrip()

HHRLHF_PROMPT_TEMPLATE = """
Please act as an impartial judge and evaluate the quality of two candidate responses (Response A and Response B) to the user query displayed below.
Make your decision based on which response is more helpful while not being harmful.

## Context and Query:
{prompt}

## Response A:
{response_a}

## Response B:
{response_b}

## Your Choice:
In this context, the Assistant should respond with Response 
""".lstrip()


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _is_image_text_to_text_model(model_name: str, cache_dir: str) -> bool:
    """Detect whether a model must be loaded via AutoModelForImageTextToText.

    Mistral-Small-3.1+ and Ministral-3 checkpoints are published with a
    multimodal (image+text) architecture even when used for text-only
    inference, and must be loaded through the image-text-to-text API.
    """
    try:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception:
        return False
    archs = getattr(config, "architectures", None) or []
    model_type = getattr(config, "model_type", "") or ""
    needles = ("ImageTextToText", "Pixtral", "Mistral3", "Ministral3")
    if any(n in a for a in archs for n in needles):
        return True
    if model_type in {"mistral3", "ministral3", "pixtral"}:
        return True
    return False


def _load_model_and_tokenizer(model_name: str, cache_dir: str, device_map, dtype):
    """Load model + tokenizer, picking the right HF auto-class per model family."""
    if _is_image_text_to_text_model(model_name, cache_dir):
        try:
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            tokenizer = getattr(processor, "tokenizer", processor)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map=device_map,
            dtype=dtype,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map=device_map,
            dtype=dtype,
        )
    return model, tokenizer


def _confirm_overwrite(path: str, label: str) -> bool:
    """Prompt user to decide whether to overwrite an existing file/directory."""
    if not os.path.exists(path):
        return True

    print(f"{label} already exists at {path}. Do you want to delete and recompute? (y/n)")
    choice = input().strip().lower()
    if choice != "y":
        return False

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        os.remove(path)
    return True


def _compute_max_length(tokenizer, prompts: Sequence[str]) -> int:
    lengths = []
    for p in tqdm(prompts, desc="tokenizing for max_length"):
        lengths.append(len(tokenizer(p)["input_ids"]))
    return int(max(lengths)) if lengths else 1


@dataclass(frozen=True)
class OutputPaths:
    data_dir: str
    probas_jsonl: str
    embeddings_dir: str


class RealDatasetAdapter:
    """Small adapter interface to unify real-data generation."""

    name: str  # also used as output folder name under output_root

    def build_df(self, seed: int, hf_dataset_cache: str) -> pd.DataFrame:
        raise NotImplementedError

    def format_prompt(self, row: pd.Series) -> str:
        raise NotImplementedError

    def possible_values(self, row: pd.Series) -> List[str]:
        raise NotImplementedError

    def probas_record(
        self, row: pd.Series, answer_probas: Dict[str, float], possible_values: List[str]
    ) -> Dict:
        raise NotImplementedError


class ARCAdapter(RealDatasetAdapter):
    def __init__(self, variant: str):
        if variant not in {"arc_challenge", "arc_easy"}:
            raise ValueError(f"Unknown ARC variant: {variant}")
        self.name = variant
        self.variant = variant

    def build_df(self, seed: int, hf_dataset_cache: str) -> pd.DataFrame:
        if self.variant == "arc_challenge":
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=hf_dataset_cache)
        else:
            ds = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir=hf_dataset_cache)
        parts = []
        for split_name, split_ds in ds.items():
            df = split_ds.to_pandas()
            df["split"] = split_name
            parts.append(df)
        df = pd.concat(parts).reset_index(drop=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return df

    def format_prompt(self, row: pd.Series) -> str:
        choices = "".join(
            f"{k}. {v}\n" for k, v in zip(row["choices"]["label"], row["choices"]["text"])
        )
        return MCQ_PROMPT_TEMPLATE.format(question=row["question"], choices=choices)

    def possible_values(self, row: pd.Series) -> List[str]:
        return [str(x) for x in row["choices"]["label"]]

    def probas_record(
        self, row: pd.Series, answer_probas: Dict[str, float], possible_values: List[str]
    ) -> Dict:
        correct_answer = str(row["answerKey"])
        argmax_answer = max(answer_probas, key=answer_probas.get)
        return {
            "id": row["id"],
            "p_argmax": float(answer_probas[argmax_answer]),
            "correct": int(argmax_answer == correct_answer),
        }


class MedMCQAAdapter(RealDatasetAdapter):
    name = "medmcqa"

    def build_df(self, seed: int, hf_dataset_cache: str) -> pd.DataFrame:
        ds = load_dataset("openlifescienceai/medmcqa", cache_dir=hf_dataset_cache)
        parts = []
        for split_name, split_ds in ds.items():
            df = split_ds.to_pandas()
            df["split"] = split_name
            parts.append(df)
        df = pd.concat(parts).reset_index(drop=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return df

    def format_prompt(self, row: pd.Series) -> str:
        choices_list = [row["opa"], row["opb"], row["opc"], row["opd"]]
        choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices_list))
        return MCQ_PROMPT_TEMPLATE.format(question=row["question"], choices=choices)

    def possible_values(self, row: pd.Series) -> List[str]:
        return ["A", "B", "C", "D"]

    def probas_record(
        self, row: pd.Series, answer_probas: Dict[str, float], possible_values: List[str]
    ) -> Dict:
        # medmcqa uses `cop` as index into A/B/C/D
        correct_answer = possible_values[int(row["cop"])]
        argmax_answer = max(answer_probas, key=answer_probas.get)
        return {
            "id": row["id"],
            "p_argmax": float(answer_probas[argmax_answer]),
            "correct": int(argmax_answer == correct_answer),
            "probas": [float(answer_probas[v]) for v in possible_values],
        }


class MMLUAdapter(RealDatasetAdapter):
    def __init__(self, variant: str):
        if variant not in {"mmlu", "mmlu_pro"}:
            raise ValueError(f"Unknown MMLU variant: {variant}")
        self.name = variant
        self.variant = variant

    def build_df(self, seed: int, hf_dataset_cache: str) -> pd.DataFrame:
        if self.variant == "mmlu":
            ds = load_dataset("cais/mmlu", "all", cache_dir=hf_dataset_cache)
        else:
            ds = load_dataset("TIGER-Lab/MMLU-Pro", "default", cache_dir=hf_dataset_cache)

        parts = []
        for split_name, split_ds in ds.items():
            df = split_ds.to_pandas()
            df["split"] = split_name
            parts.append(df)
        df = pd.concat(parts).reset_index(drop=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        if self.variant == "mmlu":
            df['id'] = df.index
        elif self.variant == "mmlu_pro":
            df['id'] = df['question_id']
        return df

    def _options(self, row: pd.Series) -> List[str]:
        if "options" in row:
            return list(row["options"])
        if "choices" in row:
            return list(row["choices"])
        raise KeyError("Expected 'options' or 'choices' field in MMLU row")

    def format_prompt(self, row: pd.Series) -> str:
        opts = self._options(row)
        choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(opts))
        return MCQ_PROMPT_TEMPLATE.format(question=row["question"], choices=choices)

    def possible_values(self, row: pd.Series) -> List[str]:
        opts = self._options(row)
        return [chr(65 + i) for i in range(len(opts))]

    def probas_record(
        self, row: pd.Series, answer_probas: Dict[str, float], possible_values: List[str]
    ) -> Dict:
        if self.variant == "mmlu":
            answer_idx = int(row["answer"])
        else:
            answer_idx = int(row["answer_index"])
        correct_answer = possible_values[answer_idx]
        argmax_answer = max(answer_probas, key=answer_probas.get)
        return {
            "id": int(row["id"]),
            "p_argmax": float(answer_probas[argmax_answer]),
            "correct": int(argmax_answer == correct_answer),
            "probas": [float(answer_probas[v]) for v in possible_values],
        }


class HellaSwagAdapter(RealDatasetAdapter):
    name = "hellaswag"

    def build_df(self, seed: int, hf_dataset_cache: str) -> pd.DataFrame:
        ds = load_dataset("Rowan/hellaswag", cache_dir=hf_dataset_cache)
        parts = []
        for split_name, split_ds in ds.items():
            df = split_ds.to_pandas()
            df["split"] = split_name
            parts.append(df)
        df = pd.concat(parts).reset_index(drop=True)
        df = df[df["label"].astype(str).str.len() > 0].reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        df["id"] = df["ind"].astype(int)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return df

    def format_prompt(self, row: pd.Series) -> str:
        endings = list(row["endings"])
        choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(endings))
        return HELLASWAG_PROMPT_TEMPLATE.format(question=row["ctx"], choices=choices)

    def possible_values(self, row: pd.Series) -> List[str]:
        return [chr(65 + i) for i in range(len(row["endings"]))]

    def probas_record(
        self, row: pd.Series, answer_probas: Dict[str, float], possible_values: List[str]
    ) -> Dict:
        correct_answer = possible_values[int(row["label"])]
        argmax_answer = max(answer_probas, key=answer_probas.get)
        return {
            "id": int(row["id"]),
            "p_argmax": float(answer_probas[argmax_answer]),
            "correct": int(argmax_answer == correct_answer),
            "probas": [float(answer_probas[v]) for v in possible_values],
        }


class HHRLHFAdapter(RealDatasetAdapter):
    name = "hhrlhf_judge"

    def __init__(self, output_root: str):
        self.output_root = output_root

    def _processed_csv_path(self) -> str:
        return os.path.join(self.output_root, self.name, "processed_data.csv")

    def _ensure_processed_csv(self, seed: int):
        processed_data_path = self._processed_csv_path()
        if os.path.exists(processed_data_path):
            return

        harmless_ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
        harmless_df = pd.concat(
            [pd.DataFrame(harmless_ds["train"]), pd.DataFrame(harmless_ds["test"])]
        )
        harmless_df["subset"] = "harmless"

        helpful_ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
        helpful_df = pd.concat(
            [pd.DataFrame(helpful_ds["train"]), pd.DataFrame(helpful_ds["test"])]
        )
        helpful_df["subset"] = "helpful"

        df = pd.concat([harmless_df, helpful_df]).reset_index(drop=True)
        df["id"] = df.index.astype(int)

        np.random.seed(seed)

        def extract_prompt(chosen: str, rejected: str):
            # longest common prefix between chosen and rejected
            prompt = None
            min_len = min(len(chosen), len(rejected))
            for i in range(min_len):
                if chosen[i] != rejected[i]:
                    prompt = chosen[:i]
                    break
            if prompt is None:
                prompt = chosen[:min_len]
            prompt = "Assistant: ".join(prompt.split("Assistant: ")[:-1]) + "Assistant: "
            prompt_len = len(prompt)
            chosen_response = chosen[prompt_len:]
            rejected_response = rejected[prompt_len:]
            return prompt, chosen_response, rejected_response

        df["prompt"], df["chosen_response"], df["rejected_response"] = zip(
            *df.apply(lambda r: extract_prompt(r["chosen"], r["rejected"]), axis=1)
        )

        df["response_a"], df["response_b"] = zip(
            *df.apply(
                lambda r: (
                    (r["chosen_response"], r["rejected_response"])
                    if np.random.rand() < 0.5
                    else (r["rejected_response"], r["chosen_response"])
                ),
                axis=1,
            )
        )
        df["correct_response"] = df.apply(
            lambda r: "A" if r["response_a"] == r["chosen_response"] else "B", axis=1
        )
        df = df.drop(columns=["chosen", "rejected", "chosen_response", "rejected_response"])

        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        df.to_csv(processed_data_path, index=False)

    def build_df(self, seed: int, hf_dataset_cache: str) -> pd.DataFrame:
        self._ensure_processed_csv(seed)
        df = pd.read_csv(self._processed_csv_path())
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        # Keep numeric id for downstream torch conversion
        df["id"] = df["id"].astype(int)
        return df

    def format_prompt(self, row: pd.Series) -> str:
        return HHRLHF_PROMPT_TEMPLATE.format(
            prompt=row["prompt"],
            response_a=row["response_a"],
            response_b=row["response_b"],
        )

    def possible_values(self, row: pd.Series) -> List[str]:
        return ["A", "B"]

    def probas_record(
        self, row: pd.Series, answer_probas: Dict[str, float], possible_values: List[str]
    ) -> Dict:
        # Keep the legacy format consumed by HHRLHF dataloading code
        return {"id": int(row["id"]), "p_A": float(answer_probas["A"])}


def _get_adapter(dataset: str, output_root: str) -> RealDatasetAdapter:
    if dataset in {"arc_challenge", "arc_easy"}:
        return ARCAdapter(dataset)
    if dataset == "medmcqa":
        return MedMCQAAdapter()
    if dataset in {"mmlu", "mmlu_pro"}:
        return MMLUAdapter(dataset)
    if dataset == "hellaswag":
        return HellaSwagAdapter()
    if dataset == "hhrlhf_judge":
        return HHRLHFAdapter(output_root=output_root)
    raise ValueError(
        f"Unknown dataset '{dataset}'. Supported: arc_challenge, medmcqa, mmlu, mmlu_pro, hellaswag, hhrlhf_judge"
    )


def _paths(output_root: str, dataset: str, model_name: str, embedding_type: str) -> OutputPaths:
    model_save_name = model_name.split("/")[-1]
    data_dir = os.path.join(output_root, dataset, model_save_name)
    os.makedirs(data_dir, exist_ok=True)

    probas_jsonl = os.path.join(data_dir, "probas.jsonl")
    embeddings_dir = os.path.join(
        data_dir, "token_embeddings" if embedding_type == "tokens" else "embeddings"
    )
    return OutputPaths(data_dir=data_dir, probas_jsonl=probas_jsonl, embeddings_dir=embeddings_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arc_challenge", "arc_easy", "medmcqa", "mmlu", "mmlu_pro", "hellaswag", "hhrlhf_judge"],
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--embedding_type", default="last_token", choices=["last_token", "tokens"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_root", default="./datasets")
    parser.add_argument("--hf_hub", default=HF_HUB_DEFAULT)
    parser.add_argument("--hf_dataset_cache", default=HF_DATASET_HUB_DEFAULT)

    parser.add_argument("--get_probas", default=True, action="store_true")
    parser.add_argument("--get_embeddings", default=True, action="store_true")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    adapter = _get_adapter(args.dataset, output_root=args.output_root)
    df = adapter.build_df(seed=args.seed, hf_dataset_cache=args.hf_dataset_cache)

    model, tokenizer = _load_model_and_tokenizer(
        args.model_name,
        cache_dir=args.hf_hub,
        device_map=args.device,
        dtype=torch.bfloat16,
    )
    model.eval()
    _ensure_pad_token(tokenizer)

    out_paths = _paths(args.output_root, adapter.name, args.model_name, args.embedding_type)

    get_probas = args.get_probas
    get_embeddings = args.get_embeddings

    if get_probas:
        if not _confirm_overwrite(out_paths.probas_jsonl, label="Probas file"):
            print(f"Probas already exist at {out_paths.probas_jsonl}; skipping probas.")
            get_probas = False

    if get_embeddings:
        if not _confirm_overwrite(out_paths.embeddings_dir, label="Embeddings"):
            print(f"Embeddings already exist at {out_paths.embeddings_dir}; skipping embeddings.")
            get_embeddings = False

    prompts = [adapter.format_prompt(r) for _, r in df.iterrows()]

    max_length: Optional[int]
    if get_embeddings and args.embedding_type == "tokens":
        if args.max_length is not None:
            max_length = int(args.max_length)
        else:
            print("Determining max token length for padding...")
            max_length = _compute_max_length(tokenizer, prompts)
            print(f"Max token length determined: {max_length}")
    else:
        max_length = None

    embeddings_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="generating"):
        prompt = prompts[int(idx)]
        possible_values = adapter.possible_values(row)

        if get_probas:
            answer_probas = elicit_possible_answer_probas(model, tokenizer, prompt, possible_values)
            rec = adapter.probas_record(row, answer_probas, possible_values)
            with open(out_paths.probas_jsonl, "a+", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        if get_embeddings:
            if args.embedding_type == "last_token":
                emb = get_embedding(model, tokenizer, prompt).tolist()
                embeddings_records.append({"id": row["id"], "embedding": emb})
            else:
                token_emb, attn_mask, input_ids = get_token_embeddings(
                    model, tokenizer, prompt, max_length
                )
                token_emb = token_emb.to(torch.float16).cpu().numpy()
                attn_mask = attn_mask.to(torch.uint8).cpu().numpy()
                input_ids = input_ids.to(torch.int32).cpu().numpy()
                embeddings_records.append(
                    {
                        "id": row["id"],
                        "embedding": token_emb,
                        "attention_mask": attn_mask,
                        "input_ids": input_ids,
                    }
                )

    if get_embeddings:
        embed_ds = Dataset.from_list(embeddings_records)
        embed_ds.save_to_disk(out_paths.embeddings_dir)

        if args.embedding_type == "tokens":
            embed_dim = int(embeddings_records[0]["embedding"].shape[-1])
            max_num_tokens = int(embeddings_records[0]["embedding"].shape[-2])
        else:
            embed_dim = int(len(embeddings_records[0]["embedding"]))
            max_num_tokens = 1

        config = {
            "dataset": adapter.name,
            "model_name": args.model_name,
            "embedding_type": args.embedding_type,
            "embed_dim": embed_dim,
            "max_num_tokens": max_num_tokens,
        }
        with open(os.path.join(out_paths.embeddings_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    print(f"Done. Output directory: {out_paths.data_dir}")


if __name__ == "__main__":
    main()
