from typing import Any, List
from vllm import SamplingParams


def forward_for_perplexity(
    formatted_prompts: List[str],
    model: Any,
    tokenizer: Any,
) -> List[dict]:
    """
    Get logits and input_ids for a batch of formatted prompts.

    Args:
        formatted_prompts: List of formatted prompt strings (already includes assistant response)
        model: HuggingFace model instance
        tokenizer: Tokenizer instance

    Returns:
        List of dicts containing 'logits' and 'input_ids' for each prompt
    """
    import torch

    # Tokenize all prompts with padding for batching
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=False
    )

    # Move batch to model's device
    input_ids_batch = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Get logits from model for the entire batch
    with torch.no_grad():
        outputs = model(input_ids_batch, attention_mask=attention_mask)
        logits_batch = outputs.logits.cpu()  # [batch_size, seq_len, vocab_size], move to CPU

    # Process each item in the batch
    results = []
    for i in range(len(formatted_prompts)):
        # Return full padded sequences (don't trim based on attention_mask)
        # The create_padded_mask function will handle finding the unpadded sequence within padding
        logits = logits_batch[i, :, :]  # [seq_len, vocab_size]
        input_ids = input_ids_batch[i, :].cpu().tolist()

        results.append({
            'logits': logits,
            'input_ids': input_ids,
        })

    return results


async def generate_response_async(
    entry: dict,
    engine: Any,
    tokenizer: Any,
) -> str:
    """
    Generate a single response using vLLM backend asynchronously.

    Args:
        entry: Entry containing 'question' field
        engine: vLLM AsyncLLMEngine instance
        tokenizer: Tokenizer instance

    Returns:
        Generated response as a string
    """
    from src_py.utils import language_abbreviation_to_name

    question = entry['question']
    lang = entry.get('lang', 'en')

    # Map language abbreviation to full name
    language_name = language_abbreviation_to_name(lang)

    # Build language-specific instructions
    instruction = f"Please CONCISELY answer the question in {language_name} WITHOUT reasoning or explanation."

    # Combine question with instruction
    user_content = f"{question}\n\n{instruction}"

    # Build messages for chat template
    messages = [
        {
            "role": "user",
            "content": user_content
        }
    ]

    # Apply chat template to get the full formatted prompt
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Use vLLM to generate the response
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=256,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # Generate with vLLM engine
    request_id = f"mistral_response_{id(entry)}"
    results_generator = engine.generate(
        formatted_prompt,
        sampling_params,
        request_id
    )

    # Wait for completion
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise RuntimeError("vLLM generation returned no output")

    # Extract the generated text
    generated_text = final_output.outputs[0].text.strip()
    return generated_text

async def collect_preference_local_async(
    question: str,
    answer1: str,
    answer2: str,
    engine: Any,
    tokenizer: Any,
) -> tuple[float, float]:
    """
    Collect preference between two answers using Mistral backend.

    Returns:
        Tuple of (logprob_1, logprob_2) where:
        - logprob_1: log probability of token "1"
        - logprob_2: log probability of token "2"
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial judge. The user is going to provide one question and two answers. "
                'If Answer 1 is better, respond with "1". '
                'If Answer 2 is better, respond with "2". '
                "Even if the answers are identical in correctness, try your best to choose a more favorable one. "
                "IMPORTANT: You SHOULD NOT judge an answer's quality based on its language.\n"
                'Only respond with "1" or "2", without any explanation.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Answer 1: {answer1}\n"
                f"Answer 2: {answer2}\n"
                "Which answer is better? Respond with '1' for Answer 1 or '2' for Answer 2."
            ),
        },
    ]

    # Convert chat messages to prompt text
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=1.0,      # sampling at temperature 1.0
        max_tokens=100,
        stop=None,
        logprobs=10,          # get top 10 token log probabilities
    )

    # vLLM async generation (returns an async generator)
    request_id = f"mistral_preference_{id(question)}"
    final_output = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        # We only need the final result
        final_output = output

    if final_output is None:
        raise RuntimeError("vLLM generation returned no output")

    # Get the generated text for debugging
    generated_text = final_output.outputs[0].text if final_output.outputs[0].text else ""

    # Get the first token's logprobs
    if not final_output.outputs[0].logprobs or len(final_output.outputs[0].logprobs) == 0:
        raise RuntimeError(f"No logprobs returned from vLLM. Generated text: {repr(generated_text)}")

    first_token_logprobs = final_output.outputs[0].logprobs[0]

    # Get token IDs for "1" and "2"
    token_1_id = tokenizer.encode("1", add_special_tokens=False)[0]
    token_2_id = tokenizer.encode("2", add_special_tokens=False)[0]

    # Get the top tokens in the first position for debugging
    top_tokens_info = []
    for token_id, logprob_obj in sorted(first_token_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)[:5]:
        token_text = tokenizer.decode([token_id])
        top_tokens_info.append(f"ID {token_id} ({repr(token_text)}): {logprob_obj.logprob:.4f}")
    top_tokens_str = ", ".join(top_tokens_info)

    # Extract log probabilities for tokens "1" and "2"
    logprob_1 = None
    logprob_2 = None

    for token_id, logprob_obj in first_token_logprobs.items():
        if token_id == token_1_id:
            logprob_1 = logprob_obj.logprob
        elif token_id == token_2_id:
            logprob_2 = logprob_obj.logprob

    # Check if both tokens are in the top k
    if logprob_1 is None:
        raise ValueError(
            f"Token '1' (ID: {token_1_id}) not found in top-k logprobs. "
            f"Generated text: {repr(generated_text)}. "
            f"Top-5 first tokens: {top_tokens_str}"
        )
    if logprob_2 is None:
        raise ValueError(
            f"Token '2' (ID: {token_2_id}) not found in top-k logprobs. "
            f"Generated text: {repr(generated_text)}. "
            f"Top-5 first tokens: {top_tokens_str}"
        )

    return logprob_1, logprob_2
