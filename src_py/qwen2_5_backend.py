from typing import Any, List
from vllm import SamplingParams



async def generate_tool_call_async(
    engine: Any,
    tokenizer: Any,
    question: str,
    tools: List[dict],
    prompt_passing_in_english: bool
) -> str:
    """
    Generate tool calls using Qwen 2.5's native tool calling format.

    Args:
        engine: vLLM AsyncLLMEngine instance
        tokenizer: Tokenizer instance
        question: User question
        tools: List of tool definitions in Qwen 2.5 format
        prompt_passing_in_english: Whether to pass parameters in English

    Returns:
        JSON string containing the tool calls
    """
    # Build messages for Qwen 2.5's chat template
    system_message = {
        "role": "system",
        "content": (
            "You are an expert in composing functions. "
            "You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
            "If none of the functions can be used, point it out. "
            "If the given question lacks the parameters required by the function, also point it out.\n\n"
            "You should ONLY return function calls in your response. "
            "You MUST NOT include any other text, explanations, or direct answers. "
            "If you decide to invoke any function(s), you MUST use the provided tools. "
            "Do NOT attempt to answer the question directly without using the available functions."
            f"{' IMPORTANT: Pass all parameter values in English' if prompt_passing_in_english else ''}"
        )
    }

    messages = [
        system_message,
        {"role": "user", "content": question}
    ]

    # Apply chat template with tools
    # The tokenizer.apply_chat_template will format the prompt according to Qwen 2.5's conventions
    # Note: Qwen 2.5 does not support enable_thinking parameter
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Use vLLM to generate the response
    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for tool calls
        max_tokens=2048,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # Generate with vLLM engine
    request_id = f"qwen2_5_toolcall_{id(question)}"
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

async def translate_tool_question_async(
    engine: Any,
    tokenizer: Any,
    question: str
) -> str:
    """
    Translate a question to English using Qwen 2.5.

    Args:
        engine: vLLM AsyncLLMEngine instance
        tokenizer: Tokenizer instance
        question: Question to translate

    Returns:
        Translated question in English
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate the given text to English accurately. If the given text is already in English or is language agnostic, return it unchanged."
        },
        {
            "role": "user",
            "content": f"Translate the following question to English. Only output the translated question, nothing else:\n\n{question}"
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    request_id = f"qwen2_5_translate_q_{id(question)}"
    results_generator = engine.generate(
        formatted_prompt,
        sampling_params,
        request_id
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise RuntimeError("vLLM generation returned no output")

    return final_output.outputs[0].text.strip()


async def translate_tool_parameter_async(
    engine: Any,
    tokenizer: Any,
    parameter_value: str
) -> str:
    """
    Translate a parameter value to English using Qwen 2.5.

    Args:
        engine: vLLM AsyncLLMEngine instance
        tokenizer: Tokenizer instance
        parameter_value: Parameter value to translate

    Returns:
        Translated parameter value in English
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate the given text to English accurately. If the given text is already in English or is language agnostic, return it unchanged."
        },
        {
            "role": "user",
            "content": f"Translate the following text to English. Only output the translated text, nothing else:\n\n{parameter_value}"
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    request_id = f"qwen2_5_translate_a_{id(parameter_value)}"
    results_generator = engine.generate(
        formatted_prompt,
        sampling_params,
        request_id
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        raise RuntimeError("vLLM generation returned no output")

    return final_output.outputs[0].text.strip()


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
        # Get the actual sequence length (excluding padding)
        seq_len = attention_mask[i].sum().item()

        # Extract logits and input_ids for this sequence (excluding padding)
        logits = logits_batch[i, :seq_len, :]  # [seq_len, vocab_size]
        input_ids = input_ids_batch[i, :seq_len].cpu().tolist()

        results.append({
            'logits': logits,
            'input_ids': input_ids,
        })

    return results


def generate_response_batch(
    entries: List[dict],
    model: Any,
    tokenizer: Any,
) -> List[str]:
    """
    Collect generated responses for a batch of entries using HuggingFace backend.

    Args:
        entries: List of entries, each containing 'question' field
        model: HuggingFace model instance
        tokenizer: Tokenizer instance
    Returns:
        List of generated responses as strings
    """
    import torch
    from src_py.utils import language_abbreviation_to_name

    formatted_prompts = []

    for entry in entries:
        question = entry['question']
        lang = entry.get('lang', 'en')

        # Map language abbreviation to full name
        language_name = language_abbreviation_to_name(lang)

        # Build language-specific instructions (following qwen2_5_interface.py format)
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
        # Note: Qwen 2.5 does not support enable_thinking parameter
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        formatted_prompts.append(formatted_prompt)

    # Tokenize all prompts with padding for batching
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,  # Limit to reasonable length
        add_special_tokens=False
    )

    # Move batch to model's device
    input_ids_batch = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Generate responses using model's generate method
    with torch.no_grad():
        generated_ids_batch = model.generate(
            input_ids=input_ids_batch,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated responses
    responses = []
    for gen_ids in generated_ids_batch:
        response_text = tokenizer.decode(
            gen_ids,
            skip_special_tokens=True
        )
        responses.append(response_text.strip())

    return responses

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
    # Note: Qwen 2.5 does not support enable_thinking parameter
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Use vLLM to generate the response
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=256,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # Generate with vLLM engine
    request_id = f"qwen2_5_response_{id(entry)}"
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
    Collect preference between two answers using Qwen 2.5 backend.

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
    # Note: Qwen 2.5 does not support enable_thinking parameter
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=1.0,      # sampling at temperature 1.0
        max_tokens=100,         # only need one token
        stop=None,
        logprobs=10,          # get top 10 token log probabilities
    )

    # vLLM async generation (returns an async generator)
    request_id = f"qwen2_5_preference_{id(question)}"
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
