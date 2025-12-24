import argparse
import asyncio
import json
import os
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tool.parse_dataset import load_json_lines

# Load API keys from .env file
load_dotenv(dotenv_path=".env")


system_prompt = '''
You are a helpful assistant helping rephrasing user requests, while accurately preserving their meaning, including numbers and names if exist. Do not answer the requirement, just produce another one that is identical in meaning but is phrased differently. Produce ONLY the rephrased requirement, without further thoughts or explanations.
'''


def generate_paraphrased_case(question: str) -> str:
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    paraphrased_question = api_inference(ApiModel.GPT_5, input_messages)
    return paraphrased_question


def main():
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

    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        original_data = load_json_lines(f)

    paraphrased_data = []
    existing_indices = []
    try:
        with open(paraphrased_dataset_path, 'r', encoding='utf-8') as f:
            paraphrased_data = load_json_lines(f)
            existing_indices = [item['id'] for item in paraphrased_data]
    except FileNotFoundError:
        print(f"No existing paraphrased dataset found at {paraphrased_dataset_path}. A new one will be created.")

    with open(paraphrased_dataset_path, 'w', encoding='utf-8') as f:
        warning_printed = False
        for item in original_data:
            id = item['id']
            if id in existing_indices:
                if not warning_printed:
                    print(f"Warning: Skipping already processed items in {paraphrased_dataset_path}.")
                    warning_printed = True
                continue
            paraphrased_question = generate_paraphrased_case(item['question'][0][0]['content'])
            paraphrased_item = item.copy()
            paraphrased_item['question'][0][0]['content'] = paraphrased_question
            paraphrased_data.append(paraphrased_item)
            f.seek(0)
            f.truncate()
            for n in paraphrased_data:
                f.write(json.dumps(n, ensure_ascii=False) + '\n')
            f.flush()
        # sort
        paraphrased_data = sorted(paraphrased_data, key=lambda x: int(re.search(r'\d+', x["id"]).group()) if re.search(r'\d+', x["id"]) else float('inf'))
        f.seek(0)
        f.truncate()
        for n in paraphrased_data:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
        f.flush()

    print(f"Paraphrased dataset with {len(paraphrased_data)} items saved to {paraphrased_dataset_path}.")


if __name__ == "__main__":
    main()