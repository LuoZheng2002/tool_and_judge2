"""
Test script to replicate tokenizer encode-decode mismatch issue.

This script tests whether encoding text and then decoding it produces
the same text, especially with <answer></answer> tags.
"""

from transformers import AutoTokenizer



def test_encode_decode_mismatch(tokenizer_name: str, test_texts: list[str]):
    """
    Test if tokenizer encode-decode produces the same text.

    Args:
        tokenizer_name: Name of the tokenizer to load
        test_texts: List of texts to test
    """
    print(f"\n{'='*80}")
    print(f"Testing tokenizer: {tokenizer_name}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    for idx, original_text in enumerate(test_texts, 1):
        print(f"\n--- Test Case {idx} ---")
        print(f"Original text ({len(original_text)} chars):")
        print(repr(original_text))

        # Encode the text
        input_ids = tokenizer.encode(original_text, add_special_tokens=False)
        print(f"\nEncoded to {len(input_ids)} tokens")

        # Decode back
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"\nDecoded text ({len(decoded_text)} chars):")
        print(repr(decoded_text))

        # Check if they match
        if decoded_text == original_text:
            print("\n✓ MATCH: Decoded text matches original")
        else:
            print("\n✗ MISMATCH: Decoded text differs from original")
            print(f"  Original length: {len(original_text)}")
            print(f"  Decoded length: {len(decoded_text)}")

            # Find first difference
            for i, (c1, c2) in enumerate(zip(original_text, decoded_text)):
                if c1 != c2:
                    print(f"\n  First difference at position {i}:")
                    print(f"    Original: {repr(c1)}")
                    print(f"    Decoded:  {repr(c2)}")
                    print(f"\n  Context around position {i}:")
                    start = max(0, i - 50)
                    end = min(len(original_text), i + 50)
                    print(f"    Original: {repr(original_text[start:end])}")
                    print(f"    Decoded:  {repr(decoded_text[start:end])}")
                    break

            # Check if one is longer
            if len(decoded_text) > len(original_text):
                print(f"\n  Extra characters in decoded: {repr(decoded_text[len(original_text):])}")
            elif len(original_text) > len(decoded_text):
                print(f"\n  Missing characters in decoded: {repr(original_text[len(decoded_text):])}")


def test_chat_template_encode_decode(tokenizer_name: str, test_cases: list[dict]):
    """
    Test encode-decode with chat templates using BATCH encoding.
    This replicates the actual issue from judge.py where batching happens.

    Args:
        tokenizer_name: Name of the tokenizer to load
        test_cases: List of test case dicts with 'messages' and 'model_type'
    """
    print(f"\n{'='*80}")
    print(f"Testing chat template with BATCH encoding: {tokenizer_name}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {repr(tokenizer.pad_token)}")

    # Extract all messages and check model type
    all_messages = [test_case['messages'] for test_case in test_cases]
    model_type = test_cases[0].get('model_type', 'default') if test_cases else 'default'

    # Apply chat template to ALL messages at once (batch mode)
    print(f"\nApplying chat template to {len(all_messages)} conversations in BATCH...")
    if model_type == 'qwen3':
        formatted_texts = tokenizer.apply_chat_template(
            all_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    else:
        formatted_texts = tokenizer.apply_chat_template(
            all_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Display formatted texts
    for idx, formatted_text in enumerate(formatted_texts, 1):
        print(f"\n--- Test Case {idx} (after batch chat template) ---")
        print(f"Formatted text ({len(formatted_text)} chars):")
        print(repr(formatted_text[:300]))
        if len(formatted_text) > 300:
            print(f"... (truncated)")

    # BATCH ENCODE (this is where padding happens!)
    print(f"\n{'='*60}")
    print(f"BATCH ENCODING {len(formatted_texts)} texts with padding...")
    print(f"{'='*60}")

    batch_encoded = tokenizer(
        formatted_texts,
        padding=True,  # This adds padding!
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False
    )

    print(f"\nBatch encoded shape: {batch_encoded['input_ids'].shape}")
    print(f"Attention mask shape: {batch_encoded['attention_mask'].shape}")

    # Now decode each one individually and compare
    for idx, (formatted_text, input_ids) in enumerate(zip(formatted_texts, batch_encoded['input_ids']), 1):
        print(f"\n{'='*60}")
        print(f"Test Case {idx} - BATCH DECODE")
        print(f"{'='*60}")

        print(f"\nOriginal formatted text ({len(formatted_text)} chars):")
        print(repr(formatted_text[:300]))
        if len(formatted_text) > 300:
            print(f"... (truncated)")

        # Convert tensor to list
        input_ids_list = input_ids.tolist()
        print(f"\nInput IDs ({len(input_ids_list)} tokens): {input_ids_list[:20]}...")

        # Decode back (skip_special_tokens=False to see padding tokens)
        decoded_text = tokenizer.decode(input_ids_list, skip_special_tokens=False)
        print(f"\nDecoded text ({len(decoded_text)} chars):")
        print(repr(decoded_text[:300]))
        if len(decoded_text) > 300:
            print(f"... (truncated)")

        # Check if they match
        if decoded_text == formatted_text:
            print("\n✓ MATCH: Decoded text matches formatted text")
        else:
            print("\n✗ MISMATCH: Decoded text differs from formatted text")
            print(f"  Formatted length: {len(formatted_text)}")
            print(f"  Decoded length: {len(decoded_text)}")

            # Find first difference
            min_len = min(len(formatted_text), len(decoded_text))
            found_diff = False
            for i in range(min_len):
                if formatted_text[i] != decoded_text[i]:
                    print(f"\n  First difference at position {i}:")
                    print(f"    Formatted: {repr(formatted_text[i])}")
                    print(f"    Decoded:   {repr(decoded_text[i])}")
                    print(f"\n  Context around position {i}:")
                    start = max(0, i - 100)
                    end = min(len(formatted_text), i + 100)
                    print(f"    Formatted: {repr(formatted_text[start:end])}")
                    print(f"    Decoded:   {repr(decoded_text[start:end])}")
                    found_diff = True
                    break

            if not found_diff:
                # Lengths differ but all chars up to min_len match
                print(f"\n  Texts match up to position {min_len}, but lengths differ")

            # Check if one is longer
            if len(decoded_text) > len(formatted_text):
                extra = decoded_text[len(formatted_text):]
                print(f"\n  Extra {len(extra)} characters in decoded: {repr(extra[:200])}")
            elif len(formatted_text) > len(decoded_text):
                missing = formatted_text[len(decoded_text):]
                print(f"\n  Missing {len(missing)} characters in decoded: {repr(missing[:200])}")


if __name__ == "__main__":
    # Test cases with <answer></answer> tags
    simple_test_texts = [
        "Simple text without tags",
        "Text with <answer>42</answer> tags",
        "Multiple answers: <answer>A</answer> and <answer>B</answer>",
        "The answer is <answer>Tokyo is the capital of Japan.</answer> because it is.",
        "<answer>Complete answer</answer>",
        "Before <answer>middle content with 中文 characters</answer> after",
    ]

    # Chat template test cases
    chat_test_cases = [
        {
            'messages': [
                {'role': 'user', 'content': 'What is 2+2?'},
                {'role': 'assistant', 'content': 'The answer is <answer>4</answer>.'}
            ],
            'model_type': 'default'
        },
        {
            'messages': [
                {'role': 'user', 'content': 'Please CONCISELY answer the question in English WITHOUT reasoning or explanation.\n\nWhat is the capital of France?'},
                {'role': 'assistant', 'content': 'The capital of France is <answer>Paris</answer>.'}
            ],
            'model_type': 'default'
        },
        {
            'messages': [
                {'role': 'user', 'content': '请用中文简洁回答问题。\n\n中国的首都是哪里？'},
                {'role': 'assistant', 'content': '中国的首都是<answer>北京</answer>。'}
            ],
            'model_type': 'default'
        },
    ]

    # List of tokenizers to test (modify based on what's available)
    tokenizers_to_test = [
        # "meta-llama/Llama-3.3-70B-Instruct",  # Uncomment if available
        # "Qwen/Qwen2.5-14B-Instruct",  # Uncomment if available
        "prometheus-eval/prometheus-8x7b-v2.0",  # Common baseline tokenizer for testing
    ]

    print("="*80)
    print("TOKENIZER ENCODE-DECODE MISMATCH TEST")
    print("="*80)
    print("\nThis script tests whether tokenizers produce the same text after")
    print("encoding and then decoding, especially with <answer></answer> tags.")
    print("\nModify the 'tokenizers_to_test' list to test with your specific models.")

    # Run simple text tests
    for tokenizer_name in tokenizers_to_test:
        try:
            test_encode_decode_mismatch(tokenizer_name, simple_test_texts)
        except Exception as e:
            print(f"\nError testing {tokenizer_name}: {e}")

    # Run chat template tests
    for tokenizer_name in tokenizers_to_test:
        try:
            test_chat_template_encode_decode(tokenizer_name, chat_test_cases)
        except Exception as e:
            print(f"\nError testing chat template with {tokenizer_name}: {e}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
