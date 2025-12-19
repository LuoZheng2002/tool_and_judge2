import json
from typing import Set, Any, Union


def extract_string_values(obj: Any, keywords: Set[str]) -> None:
    """
    Recursively extract all string values from a compound data structure.
    For dicts, extracts both string keys and string values.
    For lists, extracts string elements and recursively processes compound elements.
    Uses a set to automatically deduplicate keywords.
    """
    if isinstance(obj, str):
        keywords.add(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            # Add string keys
            if isinstance(key, str):
                keywords.add(key)
            # Recursively process values
            extract_string_values(value, keywords)
    elif isinstance(obj, list):
        for item in obj:
            extract_string_values(item, keywords)
    # For other types (int, float, bool, None), we don't collect them


def extract_keywords_from_ground_truth(ground_truth: list) -> list:
    """
    Extract all unique string-typed parameter values from the ground truth function calls.
    Returns a list of unique keywords (duplicates removed).
    """
    all_keywords = set()

    for func_call in ground_truth:
        # Each func_call is a dict with one key (function name) and value (parameters dict)
        for func_name, params in func_call.items():
            # params is a dict where keys are parameter names and values are lists of possible values
            for param_name, possible_values in params.items():
                # possible_values is a list of potential parameter values
                extract_string_values(possible_values, all_keywords)

    return list(all_keywords)


def main():
    input_file = "tool/dataset/possible_answer/BFCL_v4_multiple.jsonl"
    output_file = "tool/dataset/BFCL_v4_multiple_keywords.jsonl"

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            record_id = data["id"]
            ground_truth = data["ground_truth"]

            # Extract all string keywords from the ground truth
            keywords = extract_keywords_from_ground_truth(ground_truth)

            # Remove empty strings if any
            keywords = [k for k in keywords if k]

            # Write output
            output_record = {
                "id": record_id,
                "keywords": keywords
            }
            outfile.write(json.dumps(output_record) + '\n')

    print(f"Successfully processed {input_file}")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
