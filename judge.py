
import json
import os
import uuid

from dotenv import load_dotenv
from src_py.utils import load_config_from_file
from src_py.utils import load_json_lines_from_file
from src_py.utils import combine_entries_to_pairs
from src_py.utils import get_model_directory_safe_name

from src_py.utils import calculate_perplexity_from_logits

import argparse
import subprocess
import time
import asyncio
load_dotenv(".env")




# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run BFCL evaluation with custom configuration"
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to a Python file containing the 'config'"
)
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for local inference (default: 1)"
)
parser.add_argument(
    "--debug-limit",
    type=int,
    default=None,
    help="Limit the number of entries to process for debugging (default: None)"
)
parser.add_argument(
    "--single-gpu-memory",
    type=int,
    default=40,
    help="Memory of a single GPU in GB (default: 40)"
)

args = parser.parse_args()

# Load config from specified file
if not args.config:
    print("Error: Please specify a config file using --config argument. For example, --config config1.py")
    exit(1)

# Run maturin develop to build and install the Rust extension with file locking
import fcntl
lock_file_path = "/tmp/maturin_build_lock"
print("Acquiring build lock...")
with open(lock_file_path, "w") as lock_file:
    # Acquire exclusive lock (blocks until available)
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
        print("Building Rust extension with maturin develop...")
        # result = subprocess.run(["maturin", "develop"], check=True)
        result = subprocess.run(["maturin", "develop", "--release"], check=True)
        print("Installed Rust extension successfully.")
        time.sleep(2)  # Give some time for the build to complete
    finally:
        # Release lock
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        print("Released build lock.")

from codebase_rs import *

from src_py.vllm_backend import create_vllm_backend

print(f"Loading config from: {args.config}")
config: JudgeConfig = load_config_from_file(args.config, "config")


main_vllm_backend_created = False
main_hf_backend_created = False
assistant_api_backend_created = False
main_vllm_engine = None
main_hf_model = None
main_tokenizer = None
assistant_client = None



async def main_async():
    global main_hf_backend_created, main_hf_model, main_tokenizer
    global main_vllm_backend_created, main_vllm_engine, main_tokenizer
    global assistant_api_backend_created, assistant_client

    model_name = config.model.to_string()
    match config.experiments:
        case JudgeExperiments.Vllm(_, _):
            print("This run uses VLLM backend.")
            
            # -------------------- preference experiment -------------------- #
            print("Starting preference experiment...", flush=True)
            preference_aggregated_input_path = preference_aggregated_input_file_path(config)
            preference_aggregated_output_path = preference_aggregated_output_file_path(config)

            if os.path.exists(preference_aggregated_output_path):
                preference_dispatch_preference_results(config)
            
            preference_prepare_aggregated_input(config, debug_limit=args.debug_limit)
            combined_entries = load_json_lines_from_file(preference_aggregated_input_path)
            semaphore = asyncio.Semaphore(200)
            async def collect_single_preference_async(entry: dict) -> dict:
                """
                entry is of type TwoAnswersEntry in src/judge/generate_dataset.rs 
                """
                global main_vllm_backend_created, main_vllm_engine, main_tokenizer
                if not main_vllm_backend_created:
                    print(f"Creating VLLM backend for model {model_name} using {args.num_gpus} GPUs...", flush=True)
                    main_vllm_engine, main_tokenizer = create_vllm_backend(config.model, args.num_gpus)
                    print(f"VLLM backend created for model {model_name}", flush=True)
                    main_vllm_backend_created = True
                engine = main_vllm_engine
                tokenizer = main_tokenizer
                async with semaphore:
                    if config.model == LocalModel.Llama3_3_70B:
                        from src_py.llama3_1_backend import collect_preference_local_async
                    elif config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b, LocalModel.Qwen3_235bA22b]:
                        from src_py.qwen3_backend import collect_preference_local_async
                    elif config.model == LocalModel.UnbabelMPrometheus14B:
                        from src_py.qwen2_5_backend import collect_preference_local_async
                    elif config.model == LocalModel.Prometheus8x7bV2:
                        from src_py.mistral_backend import collect_preference_local_async
                    elif config.model == LocalModel.AyaExpanse32B:
                        from src_py.aya_expanse_backend import collect_preference_local_async
                    else:
                        raise ValueError(f"Unsupported model for preference collection: {config.model}")
                    try:
                        logprob1, logprob2 = await collect_preference_local_async(entry['question'], entry['answer1'], entry['answer2'], engine, tokenizer)
                        if logprob1 >= logprob2:
                            preferred_answer = 1
                        else:
                            preferred_answer = 2
                        logprob_signed_difference = logprob1 - logprob2
                        preference = {
                            'Ok': {
                                'preferred_answer': preferred_answer,
                                'logprob_signed_difference': logprob_signed_difference,
                                'logprob1': logprob1,
                                'logprob2': logprob2,
                            }
                        }
                    except Exception as e:
                        error_message = str(e)
                        preference = {
                            'Err': error_message
                        }
                    # The output type is PreferenceResultEntry in src/judge/result_file_model.rs
                    return {
                        "index": entry["index"],
                        "preference": preference,
                        "question": entry["question"],
                        "answer1": entry["answer1"],
                        "answer2": entry["answer2"],
                        "lang1": entry["lang1"],
                        "lang2": entry["lang2"],
                        "is_correct1": entry["is_correct1"],
                        "is_correct2": entry["is_correct2"],
                        "subject": entry["subject"],
                    }
            async def collect_all_preference_entries() -> list[dict]:
                tasks = [collect_single_preference_async(entry) for entry in combined_entries]
                with open(preference_aggregated_output_path, 'w', encoding='utf-8') as f:
                    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                        result = await coro
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')                        
                        if i % 200 == 0:
                            print(f"Preference: Written {i}/{len(combined_entries)} entries to file", flush=True)
                            f.flush()
            await collect_all_preference_entries()
            preference_dispatch_preference_results(config)
            print("Preference experiment finished.")
            # -------------------- perplexity experiment -------------------- #
            print("Starting perplexity experiment...", flush=True)
            # ---------- pass 1: collect response ---------- #
            generate_response_input_file_path = perplexity_generate_response_input_file_path(config)
            generate_response_output_file_path = perplexity_generate_response_output_file_path(config)
            if os.path.exists(generate_response_output_file_path):
                perplexity_dispatch_response_results(config)
            perplexity_prepare_response_input(config, debug_limit=args.debug_limit)
            
            
            input_entries = load_json_lines_from_file(generate_response_input_file_path)
            print(f"Total entries to generate responses: {len(input_entries)}")

            # Process entries asynchronously using vLLM
            semaphore = asyncio.Semaphore(200)

            async def collect_single_response_async(entry: dict) -> dict:
                """
                entry is of type GenerateResponseInputEntry
                """
                global main_vllm_backend_created, main_vllm_engine, main_tokenizer
                if not main_vllm_backend_created:
                    print(f"Creating VLLM backend for model {model_name} using {args.num_gpus} GPUs...", flush=True)
                    main_vllm_engine, main_tokenizer = create_vllm_backend(config.model, args.num_gpus)
                    print(f"VLLM backend created for model {model_name}", flush=True)
                    main_vllm_backend_created = True
                engine = main_vllm_engine
                tokenizer = main_tokenizer
                async with semaphore:
                    if config.model == LocalModel.Llama3_3_70B:
                        from src_py.llama3_1_backend import generate_response_async
                    elif config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b, LocalModel.Qwen3_235bA22b]:
                        from src_py.qwen3_backend import generate_response_async
                    elif config.model == LocalModel.UnbabelMPrometheus14B:
                        from src_py.qwen2_5_backend import generate_response_async
                    elif config.model == LocalModel.Prometheus8x7bV2:
                        from src_py.mistral_backend import generate_response_async
                    elif config.model == LocalModel.AyaExpanse32B:
                        from src_py.aya_expanse_backend import generate_response_async
                    else:
                        raise ValueError(f"Unsupported model for response collection: {config.model}")
                    try:
                        response = await generate_response_async(entry, engine, tokenizer)
                    except Exception as e:
                        error_message = str(e)
                        print(f"Error generating response for entry {entry['index']}: {error_message}")
                        response = f"ERROR: {error_message}"

                    return {
                        'index': entry['index'],
                        'response': response,
                        'question': entry['question'],                        
                        'lang': entry['lang'],
                        'subject': entry['subject'],
                    }

            async def collect_all_response_entries() -> list[dict]:
                tasks = [collect_single_response_async(entry) for entry in input_entries]
                with open(generate_response_output_file_path, 'w', encoding='utf-8') as f:
                    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                        result = await coro
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')                        
                        if i % 200 == 0:
                            print(f"Response: Written {i}/{len(input_entries)} entries to file", flush=True)
                            f.flush()

            await collect_all_response_entries()
            perplexity_dispatch_response_results(config)
            print(f"Completed writing all responses to {generate_response_output_file_path}")
            
            # ---------- pass 2: generate styled responses ---------- #
            generate_styled_answers_input_file_path = perplexity_generate_styled_answers_input_file_path(config)
            generate_styled_answers_output_file_path = perplexity_generate_styled_answers_output_file_path(config)
            if os.path.exists(generate_styled_answers_output_file_path):
                perplexity_dispatch_styled_answers_results(config)
            perplexity_prepare_generate_styled_answers_input(config, debug_limit=args.debug_limit)
            
            input_entries = load_json_lines_from_file(generate_styled_answers_input_file_path)
            print(f"Total entries to generate styled answers: {len(input_entries)}")

            # then write an async function to process them
            semaphore = asyncio.Semaphore(200)
            async def collect_single_styled_responses_async(
                entry: dict,
            ) -> dict:
                """
                entry is of type GenerateStyledAnswersInputEntry in src/judge/perplexity.rs 
                """
                global assistant_api_backend_created, assistant_client
                # assistant_model_name = "gpt-5"
                assistant_model = ApiModel.Gpt4_1
                assistant_model_name = assistant_model.to_string()
                if not assistant_api_backend_created:
                    print(f"Creating Assistant API client for model {assistant_model_name}...", flush=True)
                    from src_py.api_backend import create_api_backend
                    assistant_client = create_api_backend(assistant_model)
                    print(f"Assistant API client created for model {assistant_model_name}", flush=True)
                    assistant_api_backend_created = True
                client = assistant_client
                async with semaphore:
                    from src_py.gpt5_backend import generate_styled_answers_async
                    result = await generate_styled_answers_async(
                        assistant_model_name,
                        client,
                        entry['index'],
                        entry['lang'],
                        entry['question'],
                        entry['response'],
                        entry['original_answer_correct'],
                        entry['original_answer_incorrect']
                    )
                # return type is StyledAnswersEntry in src/judge/perplexity.rs
                return {
                    'index': entry['index'],                    
                    'styled_response_correct': result['styled_response_correct'],
                    'styled_response_incorrect': result['styled_response_incorrect'],
                    'response': entry['response'],
                    'original_answer_correct': entry['original_answer_correct'],
                    'original_answer_incorrect': entry['original_answer_incorrect'],
                    'question': entry['question'],
                    'lang': entry['lang'],
                    'subject': entry['subject'],
                }
            async def collect_all_styled_responses_entries() -> list[dict]:
                tasks = [collect_single_styled_responses_async(entry) for entry in input_entries]
                with open(generate_styled_answers_output_file_path, 'a', encoding='utf-8') as f:
                    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                        result = await coro
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')                        
                        if i % 200 == 0:
                            print(f"Styled Response: Written {i}/{len(input_entries)} entries to styled answers file", flush=True)
                            f.flush()
            await collect_all_styled_responses_entries()
            perplexity_dispatch_styled_answers_results(config)
            print(f"Completed writing all styled answers to {generate_styled_answers_output_file_path}")
            print("Perplexity experiment finished.")
        case JudgeExperiments.HuggingFace(_):            
            # ---------- pass 3: forward and collect perplexity ---------- #
            generate_perplexity_aggregated_input_file_path = perplexity_generate_perplexity_aggregated_input_file_path(config)
            generate_perplexity_aggregated_output_file_path = perplexity_generate_perplexity_aggregated_output_file_path(config)
            if os.path.exists(generate_perplexity_aggregated_output_file_path):
                perplexity_dispatch_generate_perplexity_results(config)
            perplexity_prepare_generate_perplexity_aggregated_input(config, debug_limit=args.debug_limit)
            
            input_entries = load_json_lines_from_file(generate_perplexity_aggregated_input_file_path)
            print(f"Total entries to calculate perplexity: {len(input_entries)}")

            # Define batch size for processing
            # Scale batch size based on GPU memory: 240 was the constant for 40GB GPUs
            # batch_size = (base_constant) * num_gpus * (gpu_memory / 40GB) / model_size_in_billions
            batch_size =  int(4 * args.num_gpus * args.single_gpu_memory / config.model.size_in_billion_parameters())
            print(f"Using batch size: {batch_size} based on model size, number of GPUs ({args.num_gpus}), and single GPU memory ({args.single_gpu_memory}GB)")
            total_processed = 0

            with open(generate_perplexity_aggregated_output_file_path, 'w') as f:
                for i in range(0, len(input_entries), batch_size):
                    batch_idx = i // batch_size + 1
                    batch_entries = input_entries[i:i+batch_size]
                    print(f"Processing batch {batch_idx}/{(len(input_entries) + batch_size - 1)//batch_size}", flush=True)

                    if not main_hf_backend_created:
                        print(f"Creating HuggingFace backend for model {model_name} using {args.num_gpus} GPUs and batch size {batch_size}...", flush=True)
                        from src_py.huggingface_backend import create_huggingface_backend
                        main_hf_model, main_tokenizer = create_huggingface_backend(config.model, batch_size)
                        print(f"HuggingFace backend created for model {model_name}", flush=True)
                        main_hf_backend_created = True
                    hf_model = main_hf_model
                    hf_tokenizer = main_tokenizer

                    # Three-part perplexity calculation
                    # Import model-specific forward function
                    if config.model == LocalModel.Llama3_3_70B:
                        from src_py.llama3_1_backend import forward_for_perplexity
                    elif config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b, LocalModel.Qwen3_235bA22b]:
                        from src_py.qwen3_backend import forward_for_perplexity
                    elif config.model == LocalModel.UnbabelMPrometheus14B:
                        from src_py.qwen2_5_backend import forward_for_perplexity
                    elif config.model == LocalModel.Prometheus8x7bV2:
                        from src_py.mistral_backend import forward_for_perplexity
                    elif config.model == LocalModel.AyaExpanse32B:
                        from src_py.aya_expanse_backend import forward_for_perplexity
                    else:
                        raise ValueError(f"Unsupported model for perplexity collection: {config.model}")

                    # Import utilities
                    from src_py.utils import (
                        language_abbreviation_to_name,
                        trim_forward_prompt_and_get_char_mask,
                        convert_char_mask_to_token_mask,
                        calculate_perplexity_from_logits_with_mask
                    )

                    # Part 1: Parse styled responses and prepare data
                    char_masks = []
                    trimmed_prompts = []

                    for entry in batch_entries:
                        question = entry['question']
                        # Use whichever styled response is present (with <answer> tags)
                        styled_response = entry['styled_response']
                        lang = entry['lang']

                        # Map language abbreviation to full name
                        language_name = language_abbreviation_to_name(lang)

                        # Build language-specific instructions
                        instruction = f"Please CONCISELY answer the question in {language_name} WITHOUT reasoning or explanation."

                        # Combine question with instruction
                        user_content = f"{question}\n\n{instruction}"

                        # Build messages for chat template (with styled response that has <answer> tags)
                        messages = [
                            {
                                "role": "user",
                                "content": user_content
                            },
                            {
                                "role": "assistant",
                                "content": styled_response  # Keep <answer> tags for now
                            }
                        ]

                        # Apply chat template to get the formatted prompt with <answer> tags
                        # Handle model-specific tokenization parameters
                        if config.model in [LocalModel.Qwen3_8B, LocalModel.Qwen3_14B, LocalModel.Qwen3_30bA3b, LocalModel.Qwen3Next80bA3b, LocalModel.Qwen3_235bA22b]:
                            # Qwen 3 models support enable_thinking parameter
                            formatted_prompt_with_tags = hf_tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False,
                                enable_thinking=False,
                            )
                        elif config.model in [LocalModel.UnbabelMPrometheus14B, LocalModel.Prometheus8x7bV2]:
                            # M-Prometheus (Qwen 2.5 base) and Prometheus-8x7b (Mistral base) don't support enable_thinking
                            formatted_prompt_with_tags = hf_tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False,
                            )
                        else:
                            # Other models (Llama, AyaExpanse, etc.)
                            formatted_prompt_with_tags = hf_tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False
                            )

                        # Now trim the <answer> tags from the complete formatted prompt and get character-level mask
                        trimmed_prompt, char_mask = trim_forward_prompt_and_get_char_mask(formatted_prompt_with_tags)

                        char_masks.append(char_mask)
                        trimmed_prompts.append(trimmed_prompt)

                    # Part 2: Forward pass through model
                    forward_results = forward_for_perplexity(trimmed_prompts, hf_model, hf_tokenizer)

                    # Part 3: Calculate perplexity for each entry and write immediately
                    for entry, forward_result, char_mask, trimmed_prompt in zip(batch_entries, forward_results, char_masks, trimmed_prompts):
                        # Convert character mask to token mask
                        input_ids, token_mask = convert_char_mask_to_token_mask(trimmed_prompt, char_mask, hf_tokenizer, debug=False)

                        # Verify input_ids match between tokenization and forward pass
                        assert input_ids == forward_result['input_ids'], \
                            f"input_ids mismatch: tokenized {len(input_ids)} tokens, forward pass has {len(forward_result['input_ids'])} tokens"

                        # Calculate perplexity using the mask
                        perplexity = calculate_perplexity_from_logits_with_mask(
                            entry['index'],
                            entry['lang'],
                            forward_result['logits'],
                            forward_result['input_ids'],
                            token_mask
                        )

                        result_entry = {
                            'index': entry['index'],
                            'perplexity': perplexity,                            
                            'styled_response': entry['styled_response'],
                            'response': entry['response'],
                            'original_answer': entry['original_answer'],
                            'question': entry['question'],
                            'is_correct': entry['is_correct'],
                            'lang': entry['lang'],
                            'subject': entry['subject'],
                        }
                        # Write result immediately
                        f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                        total_processed += 1

                    # Flush after each batch to ensure results are written
                    f.flush()
                    if batch_idx % 20 == 0:
                        print(f"Perplexity: Written {total_processed}/{len(input_entries)} entries to perplexity result file", flush=True)
            perplexity_dispatch_generate_perplexity_results(config)
            print(f"Completed writing all {total_processed} perplexity results to {generate_perplexity_aggregated_output_file_path}")

if __name__ == "__main__":
    asyncio.run(main_async())






