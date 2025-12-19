"""
Async version of fully translate to Igbo with support for DeepSeek and GPT-5.
Processes multiple translations concurrently for better performance.
"""

import argparse
import asyncio
import json
import os
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from parse_dataset import load_json_lines

# Load API keys from .env file
load_dotenv(dotenv_path=".env")


def create_translation_prompt(question_content: str) -> str:
    """
    Create a simple prompt for full translation to Igbo.
    """
    prompt = f"""Translate the following English question to Igbo. Provide a natural, fluent translation that maintains the meaning and intent of the original question.

English question:
{question_content}

Provide only the Igbo translation, without any explanations or additional text."""

    return prompt


async def translate_with_api(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    api_type: str
) -> str | None:
    """
    Use API to translate the question (supports DeepSeek and GPT-5).

    Args:
        client: AsyncOpenAI client
        prompt: The translation prompt
        model: Model name to use
        api_type: "deepseek" or "gpt5"

    Returns:
        Translated text or None if translation fails
    """
    try:
        if api_type == "gpt5":
            # GPT-5 uses responses API
            response = await client.responses.create(
                input=[
                    {"role": "developer", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                store=False
            )
            translated_text = response.output_text.strip() if hasattr(response, 'output_text') else None
        else:
            # DeepSeek uses chat completions API
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            message = response.choices[0].message
            translated_text = message.content.strip() if message.content else None

        return translated_text

    except Exception as e:
        print(f"Error calling API: {e}")
        return None


async def translate_item(
    client: AsyncOpenAI,
    dataset_line: dict,
    model: str,
    api_type: str,
    index: int,
    total: int
) -> tuple[str, dict | None]:
    """
    Translate a single item asynchronously.

    Returns:
        Tuple of (item_id, translated_line or None)
    """
    item_id = dataset_line["id"]
    question_content = dataset_line["question"][0][0]["content"]

    # Create translation prompt
    prompt = create_translation_prompt(question_content)

    # Translate
    print(f"[{index+1}/{total}] Translating {item_id} to Igbo...")
    translated_content = await translate_with_api(client, prompt, model, api_type)

    if translated_content:
        # Create translated item
        translated_line = dataset_line.copy()
        translated_line["question"] = [[{
            "role": "user",
            "content": translated_content
        }]]

        print(f"  ✓ {item_id}: {translated_content[:60]}...")
        return (item_id, translated_line)
    else:
        print(f"  ✗ {item_id}: Translation failed")
        return (item_id, None)


async def process_batch(
    client: AsyncOpenAI,
    batch: list[tuple[int, dict]],
    model: str,
    api_type: str,
    total: int
) -> list[tuple[str, dict | None]]:
    """
    Process a batch of items concurrently.
    """
    tasks = [
        translate_item(client, item, model, api_type, idx, total)
        for idx, item in batch
    ]
    return await asyncio.gather(*tasks)


def save_results(output_file: str, translated_lines: list[dict]):
    """
    Save translated lines to file, sorted by ID.
    """
    sorted_lines = sorted(
        translated_lines,
        key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for t_line in sorted_lines:
            f.write(json.dumps(t_line, ensure_ascii=False) + "\n")


async def main():
    # === Parse command line arguments ===
    parser = argparse.ArgumentParser(
        description="Async full translation to Igbo with DeepSeek or GPT-5"
    )
    parser.add_argument(
        "--api",
        choices=["deepseek", "gpt5"],
        default="gpt5",
        help="API to use (deepseek or gpt5)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: deepseek-chat for deepseek, gpt-5-nano for gpt5)"
    )
    parser.add_argument(
        "--input",
        default="dataset/BFCL_v4_multiple.json",
        help="Input dataset file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (default: auto-generated)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent translations (default: 10)"
    )

    args = parser.parse_args()

    # === Configuration ===
    api_type = args.api

    # Set default model based on API
    if args.model:
        model_name = args.model
    else:
        model_name = "deepseek-chat" if api_type == "deepseek" else "gpt-5-nano"

    # Set default output file
    if args.output:
        output_file = args.output
    else:
        output_file = f"dataset/BFCL_v4_multiple_igbo_full_{api_type}.json"

    # Get API configuration
    if api_type == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY not found in .env")
    else:  # gpt5
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in .env")

    print(f"Using {api_type.upper()} API with model: {model_name}")
    print(f"Batch size: {args.batch_size} concurrent translations")

    # Initialize async client
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # === Load input file ===
    print(f"\nLoading dataset from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        dataset = load_json_lines(f)

    # === Load existing translations (for resumption) ===
    existing_indices = set()
    translated_lines = []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            translated_lines = load_json_lines(f)
            existing_indices = {item["id"] for item in translated_lines}
        print(f"Found {len(existing_indices)} existing translations, will skip those")
    except FileNotFoundError:
        print(f"No existing translations found, starting fresh")

    # Filter out already processed items
    items_to_process = [
        (i, item) for i, item in enumerate(dataset)
        if item["id"] not in existing_indices
    ]

    if not items_to_process:
        print("\n✅ All items already translated!")
        return

    print(f"\nProcessing {len(items_to_process)} new items...")

    # === Process in batches ===
    total = len(dataset)
    batch_size = args.batch_size
    all_results = []

    for i in range(0, len(items_to_process), batch_size):
        batch = items_to_process[i:i+batch_size]
        print(f"\n--- Batch {i//batch_size + 1}/{(len(items_to_process)-1)//batch_size + 1} ---")

        results = await process_batch(client, batch, model_name, api_type, total)
        all_results.extend(results)

        # Save progress after each batch
        for item_id, translated_line in results:
            if translated_line:
                translated_lines.append(translated_line)

        save_results(output_file, translated_lines)
        print(f"Progress saved to {output_file}")

    # Check for failures
    failed = [item_id for item_id, line in all_results if line is None]
    if failed:
        print(f"\n⚠ Warning: {len(failed)} translations failed:")
        for item_id in failed[:10]:  # Show first 10
            print(f"  - {item_id}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print(f"\n✅ Translation complete! Output saved to: {output_file}")
    print(f"Total items translated: {len(translated_lines)}")
    print(f"Success rate: {len(all_results) - len(failed)}/{len(all_results)}")


if __name__ == "__main__":
    asyncio.run(main())
