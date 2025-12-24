import argparse
import asyncio
import json
import os
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src_py.utils import load_json_lines_from_file
# Load API keys from .env file
load_dotenv(dotenv_path=".env")


system_prompt = '''
You are a helpful assistant helping rephrasing user requests, while accurately preserving their meaning, including numbers and names if exist. Do not answer the requirement, just produce another one that is identical in meaning but is phrased differently. Produce ONLY the rephrased requirement, without further thoughts or explanations.
'''


async def generate_paraphrased_case(
    client: AsyncOpenAI,
    question: str,
    model: str = "gpt-5"
) -> str | None:
    """
    Generate a paraphrased version of the question using GPT-5.

    Args:
        client: AsyncOpenAI client
        question: The original question text
        model: Model name to use (default: gpt-5)

    Returns:
        Paraphrased question or None if generation fails
    """
    try:
        response = await client.responses.create(
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            model=model,
            store=False
        )
        paraphrased_text = response.output_text.strip() if hasattr(response, 'output_text') else None
        return paraphrased_text
    except Exception as e:
        print(f"Error calling GPT-5 API: {e}")
        return None


async def paraphrase_item(
    client: AsyncOpenAI,
    item: dict,
    model: str,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore
) -> tuple[str, dict | None]:
    """
    Paraphrase a single item asynchronously with semaphore control.

    Returns:
        Tuple of (item_id, paraphrased_item or None)
    """
    async with semaphore:
        item_id = item['id']
        original_question = item['question'][0][0]['content']

        print(f"[{index+1}/{total}] Paraphrasing {item_id}...")
        paraphrased_question = await generate_paraphrased_case(client, original_question, model)

        if paraphrased_question:
            paraphrased_item = item.copy()
            paraphrased_item['question'][0][0]['content'] = paraphrased_question
            print(f"  ✓ {item_id}: {paraphrased_question[:60]}...")
            return (item_id, paraphrased_item)
        else:
            print(f"  ✗ {item_id}: Paraphrasing failed")
            return (item_id, None)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate paraphrased dataset using GPT-5"
    )
    parser.add_argument(
        "--base-dir",
        default="tool/dataset",
        help="Base directory for dataset files (default: tool/dataset)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset file path (relative to base-dir or absolute)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output paraphrased dataset file path (relative to base-dir or absolute)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model name (default: gpt-5)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=200,
        help="Maximum number of concurrent paraphrasing tasks (default: 200)"
    )

    args = parser.parse_args()

    # Construct full paths
    if os.path.isabs(args.input):
        original_dataset_path = args.input
    else:
        original_dataset_path = os.path.join(args.base_dir, args.input)

    if os.path.isabs(args.output):
        paraphrased_dataset_path = args.output
    else:
        paraphrased_dataset_path = os.path.join(args.base_dir, args.output)

    print(f"Input: {original_dataset_path}")
    print(f"Output: {paraphrased_dataset_path}")
    print(f"Model: {args.model}")
    print(f"Max concurrent tasks: {args.max_concurrent}")

    # Initialize async OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

    # Load input data
    original_data = load_json_lines_from_file(original_dataset_path)

    # Load existing paraphrased data (for resumption)
    existing_indices = set()
    try:
        existing_lines = load_json_lines_from_file(paraphrased_dataset_path)
        existing_indices = {item['id'] for item in existing_lines}
        print(f"Found {len(existing_indices)} existing paraphrased items, will skip those")
    except FileNotFoundError:
        print(f"No existing paraphrased dataset found, starting fresh")
        # Create empty file
        with open(paraphrased_dataset_path, "w", encoding="utf-8") as f:
            pass

    # Filter out already processed items
    items_to_process = [
        (i, item) for i, item in enumerate(original_data)
        if item['id'] not in existing_indices
    ]

    if not items_to_process:
        print("\n✅ All items already paraphrased!")
    else:
        print(f"\nProcessing {len(items_to_process)} new items...")

        # Process with semaphore and asyncio.as_completed
        total = len(original_data)
        semaphore = asyncio.Semaphore(args.max_concurrent)
        failed = []
        completed = 0

        # Create all tasks
        tasks = [
            paraphrase_item(client, item, args.model, idx, total, semaphore)
            for idx, item in items_to_process
        ]

        # Process tasks as they complete
        for coro in asyncio.as_completed(tasks):
            item_id, paraphrased_item = await coro
            completed += 1

            if paraphrased_item:
                # Append immediately to file
                with open(paraphrased_dataset_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(paraphrased_item, ensure_ascii=False) + "\n")
            else:
                failed.append(item_id)

            if completed % 10 == 0 or completed == len(items_to_process):
                print(f"Progress: {completed}/{len(items_to_process)} completed")

        # Summary
        if failed:
            print(f"\n⚠ Warning: {len(failed)} paraphrasing tasks failed:")
            for item_id in failed[:10]:
                print(f"  - {item_id}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")

        print(f"\n✅ Paraphrasing complete!")
        print(f"Success rate: {completed - len(failed)}/{completed}")

    # Sort the output file by ID (always runs)
    print(f"\nSorting output file by ID...")
    all_lines = load_json_lines_from_file(paraphrased_dataset_path)
    sorted_lines = sorted(
        all_lines,
        key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf')
    )
    with open(paraphrased_dataset_path, "w", encoding="utf-8") as f:
        for line in sorted_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Output saved to: {paraphrased_dataset_path}")


if __name__ == "__main__":
    asyncio.run(main())