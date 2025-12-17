import os


def generate_perplexity_dataset_mask():
    output_path = f'judge/datasets/perplexity_mask.jsonl'
    # if the file already exists, do nothing
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping generation.")
        return
    input_dataset_path = f'judge/datasets/mmmlu_normalized/en.jsonl' # use English as the benchmark
    # read all lines and json parse them
    with open(input_dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    import json
    dataset_entries = [json.loads(line) for line in lines]

    # Create gpt5 client connection
    from openai import AsyncOpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(
        api_key=api_key,
    )

    # create async runtime
    import asyncio
    async def judge_qa_pair_async(entry: dict) -> dict:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "The user will provide a multiple choice question, 4 answer choices and the correct answer. "
                    "Your task is to judge whether there is a reasonably high chance for a sufficiently intelligent being "
                    "to output the exact correct answer among the four choices, even if the choices are not given."
                    "Questions that fail to meet this requirement may include: \n"
                    "- Questions that contain 'Which of the following...' or similar phrasing. This makes it impossible to hit the correct answer without seeing the choices. For example, 'Which of the following has a red color?' A: apple. B:...\n"
                    "- Questions that are open-ended to an extent that the correct answer cannot be hit with a reasonable chance without seeing the choices. For example, Shakespeare has the motto ___. A: To thine own self be true. B:...\n"
                    "Please note that if a question has a fixed answer but with multiple possible phrasings, it is still valid. For example, What was the most important finding by the House of Lords in the Pinochet case? A: The Pinochet case confirmed that all public acts enjoy immunity. B:...\n"
                    "Your output should only contain either 'VALID' or 'INVALID', without any additional explanation."
                )
            },
            {
                "role": "user",
                "content": (f"Question: {entry['question']}\n"
                            f"A: {entry['choices'][0]}\n"
                            f"B: {entry['choices'][1]}\n"
                            f"C: {entry['choices'][2]}\n"
                            f"D: {entry['choices'][3]}\n"
                            f"Correct Answer: {entry['choices'][entry['answer']]}\n"
                            f"Does this question meet the requirement? Please only output 'VALID' or 'INVALID'.")
            }
        ]
        response = await client.chat.completions.create(
            model="gpt-5",
            input=messages,
        )
        response_str = response.output_text.strip()
        if response_str == "VALID":
            is_valid = True
        elif response_str == "INVALID":
            is_valid = False
        else:
            print(f"Unexpected response: {response_str}. Marking as INVALID.")
            is_valid = False
        print(f"Question {entry['index']}: {entry['question'][:50]}... judged as {'VALID' if is_valid else 'INVALID'}")
        return {
            "index": entry["index"],
            "valid": is_valid,
            "question": entry["question"],
            "choices": entry["choices"],
            "subject": entry["subject"],
        }
    # Run the async function for all entries in parallel
    async def process_all_entries():
        tasks = []
        for entry in dataset_entries:
            tasks.append(judge_qa_pair_async(entry))
        results = await asyncio.gather(*tasks)
        return results
    # start the runtime
    results = asyncio.run(process_all_entries())
    # dump results to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Generated perplexity dataset mask at {output_path}")

if __name__ == "__main__":
    generate_perplexity_dataset_mask()