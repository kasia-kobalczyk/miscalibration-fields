import torch
import torch
from contextlib import nullcontext


def elicit_possible_answer_probas(model, tokenizer, prompt, possible_values):
    prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )

    value_texts = [str(v) for v in possible_values]
    value_inputs = tokenizer(
        value_texts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)

    batch_size = value_inputs.input_ids.size(0)
    prompt_ids = prompt_inputs.input_ids.expand(batch_size, -1)
    prompt_mask = prompt_inputs.attention_mask.expand(batch_size, -1)

    full_input_ids = torch.cat([prompt_ids, value_inputs.input_ids], dim=1)
    full_attention_mask = torch.cat([prompt_mask, value_inputs.attention_mask], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=full_input_ids, attention_mask=full_attention_mask)

    logits = outputs.logits[:, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = full_input_ids[:, 1:]
    prompt_len = prompt_ids.size(1)
    candidate_mask = torch.zeros_like(target_ids, dtype=torch.bool)
    candidate_mask[
        :, prompt_len - 1 : prompt_len - 1 + value_inputs.input_ids.size(1)
    ] = value_inputs.attention_mask.bool()

    token_logprobs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    candidate_logprobs = (token_logprobs * candidate_mask.float()).sum(dim=1)
    probs = torch.softmax(candidate_logprobs, dim=0).tolist()
    return {value: prob for value, prob in zip(value_texts, probs)}


def get_embedding(model, tokenizer, prompt):
    """
    Computes the embedding of the prompt as the embedding of the last token of the last hidden state.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.hidden_states[-1][:, -1, :].squeeze().cpu()


@torch.no_grad()
def get_token_embeddings(
    model,
    tokenizer,
    prompts,
    max_length,
    device=None,
    out_dtype=torch.float16,  # storage dtype
):
    """
    Returns padded token embeddings (last hidden layer), plus masks/ids.

    prompts: str or List[str]

    Output:
        token_embeddings: [bs, max_length, dim_model]  (out_dtype)
        attention_mask:   [bs, max_length]             (int)
        input_ids:        [bs, max_length]             (int)
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    if device is None:
        device = model.device

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Autocast to model dtype for speed; then cast output to out_dtype for storage.
    model_dtype = getattr(model, "dtype", torch.float16)
    amp_ctx = (
        torch.autocast("cuda", dtype=model_dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    with amp_ctx:
        outputs = model(**inputs, output_hidden_states=True)
        token_embeddings = outputs.hidden_states[-1]  # [bs, T, dim]

    token_embeddings = token_embeddings.to(out_dtype)  # e.g. float16 for numpy/arrow
    attention_mask = inputs["attention_mask"]
    input_ids = inputs["input_ids"]

    return token_embeddings, attention_mask, input_ids
