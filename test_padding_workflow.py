"""
Test script for the complete padding-aware tokenization workflow.

This script tests the entire process:
1. Individual tokenization without padding
2. Batch tokenization with padding (mocking forward pass)
3. Creating padded masks
4. Extracting and reconstructing masked content
"""

from transformers import AutoTokenizer
from src_py.utils import (
    trim_forward_prompt_and_get_char_mask,
    convert_char_mask_to_token_mask,
    find_unpadded_sequence_in_padded,
    create_padded_mask
)


def test_padding_workflow(tokenizer_name: str):
    """
    Test the complete padding-aware workflow.

    Args:
        tokenizer_name: Name of the tokenizer to load
    """
    print("="*80)
    print(f"TESTING PADDING WORKFLOW: {tokenizer_name}")
    print("="*80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"\nSet pad_token to eos_token: {repr(tokenizer.pad_token)}")

    # Test cases with <answer></answer> tags of different lengths
    # Each test case is (text_with_tags, hardcoded_expected_answer)
    test_cases = [
        ("Question: What is 2+2? Answer: The result is <answer>4</answer>.", "4"),
        ("Question: What is the capital of France? Answer: The capital of France is <answer>Paris</answer>. It is a beautiful city.", "Paris"),
        ("Short: <answer>A</answer>", "A"),
        ("Question: Name three colors. Answer: <answer>Red, blue, and green</answer> are common colors.", "Red, blue, and green"),
        ("Multiple: <answer>First answer</answer> and then <answer>Second answer</answer> here.", "First answer Second answer"),
    ]

    print(f"\n{'='*80}")
    print("TEST CASES (with hardcoded expected answers):")
    print(f"{'='*80}")
    for idx, (text, expected) in enumerate(test_cases, 1):
        print(f"\n{idx}. {text}")
        print(f"   Expected: {repr(expected)}")

    # PART 1: Individual tokenization (no padding)
    print(f"\n{'='*80}")
    print("PART 1: INDIVIDUAL TOKENIZATION (NO PADDING)")
    print(f"{'='*80}")

    unpadded_data = []
    trimmed_texts = []

    for idx, (text_with_tags, hardcoded_expected_answer) in enumerate(test_cases, 1):
        print(f"\n--- Processing Text {idx} ---")

        # Trim <answer> tags and get character mask
        trimmed_text, char_mask = trim_forward_prompt_and_get_char_mask(text_with_tags)

        print(f"Original: {repr(text_with_tags)}")
        print(f"Trimmed:  {repr(trimmed_text)}")
        print(f"Char mask length: {len(char_mask)}, Masked chars: {sum(char_mask)}")

        # Tokenize individually WITHOUT padding
        unpadded_ids, token_mask = convert_char_mask_to_token_mask(
            trimmed_text,
            char_mask,
            tokenizer,
            debug=False
        )

        print(f"Unpadded tokens: {len(unpadded_ids)}, Masked tokens: {sum(token_mask)}")
        print(f"Unpadded IDs: {unpadded_ids}")
        print(f"Token mask:   {token_mask}")

        # Extract masked content
        masked_token_ids = [tid for tid, is_masked in zip(unpadded_ids, token_mask) if is_masked]
        extracted_answer = tokenizer.decode(masked_token_ids, skip_special_tokens=True)
        print(f"Extracted answer: {repr(extracted_answer)}")
        print(f"Expected answer:  {repr(hardcoded_expected_answer)}")

        # Verify against hardcoded expected answer
        if extracted_answer == hardcoded_expected_answer:
            print(f"✓ Match with hardcoded expected answer")
        else:
            print(f"✗ MISMATCH with hardcoded expected answer!")

        unpadded_data.append({
            'unpadded_ids': unpadded_ids,
            'token_mask': token_mask,
            'trimmed_text': trimmed_text,
            'original_text': text_with_tags,
            'expected_masked_text': hardcoded_expected_answer  # Use hardcoded value
        })
        trimmed_texts.append(trimmed_text)

    # PART 2: Batch tokenization (WITH padding) - Simulating forward pass
    print(f"\n{'='*80}")
    print("PART 2: BATCH TOKENIZATION (WITH PADDING) - SIMULATING FORWARD PASS")
    print(f"{'='*80}")

    # Batch tokenize with padding (simulating what forward_for_perplexity does)
    batch_encoded = tokenizer(
        trimmed_texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False
    )

    print(f"\nBatch shape: {batch_encoded['input_ids'].shape}")
    print(f"Max sequence length: {batch_encoded['input_ids'].shape[1]}")

    # Extract padded input_ids for each entry (simulating forward_results)
    mock_forward_results = []
    for i in range(len(trimmed_texts)):
        # Get attention mask to find actual sequence length
        attention_mask = batch_encoded['attention_mask'][i]
        seq_len = attention_mask.sum().item()

        # Extract input_ids (excluding trailing padding based on attention mask)
        # Note: We keep the full padded sequence to test padding handling
        padded_ids = batch_encoded['input_ids'][i].tolist()

        mock_forward_results.append({
            'input_ids': padded_ids,
            'seq_len': seq_len  # For verification
        })

        print(f"\nText {i+1}:")
        print(f"  Full padded length: {len(padded_ids)}")
        print(f"  Actual sequence length: {seq_len}")
        print(f"  Padded IDs: {padded_ids}")

    # PART 3: Create padded masks and extract masked content
    print(f"\n{'='*80}")
    print("PART 3: CREATE PADDED MASKS AND EXTRACT MASKED CONTENT")
    print(f"{'='*80}")

    all_success = True

    for idx, (data, forward_result) in enumerate(zip(unpadded_data, mock_forward_results), 1):
        print(f"\n{'='*60}")
        print(f"Text {idx}: {repr(data['original_text'][:60])}...")
        print(f"{'='*60}")

        unpadded_ids = data['unpadded_ids']
        unpadded_mask = data['token_mask']
        padded_ids = forward_result['input_ids']

        # Find where unpadded sequence appears in padded sequence
        try:
            offset = find_unpadded_sequence_in_padded(unpadded_ids, padded_ids)
            print(f"✓ Found unpadded sequence at offset: {offset}")
        except ValueError as e:
            print(f"✗ ERROR: {e}")
            all_success = False
            continue

        # Create padded mask
        try:
            padded_mask = create_padded_mask(unpadded_mask, unpadded_ids, padded_ids)
            print(f"✓ Created padded mask (length: {len(padded_mask)})")
            print(f"  Padded mask: {padded_mask}")
            print(f"  Total True values: {sum(padded_mask)}")
        except Exception as e:
            print(f"✗ ERROR creating padded mask: {e}")
            all_success = False
            continue

        # Verify mask alignment
        assert len(padded_mask) == len(padded_ids), \
            f"Padded mask length {len(padded_mask)} != padded_ids length {len(padded_ids)}"

        # Extract masked tokens using the padded mask
        masked_token_ids = [tid for tid, is_masked in zip(padded_ids, padded_mask) if is_masked]
        reconstructed_text = tokenizer.decode(masked_token_ids, skip_special_tokens=True)

        print(f"\nExtracted masked content:")
        print(f"  Expected: {repr(data['expected_masked_text'])}")
        print(f"  Got:      {repr(reconstructed_text)}")

        if reconstructed_text == data['expected_masked_text']:
            print(f"  ✓ MATCH!")
        else:
            print(f"  ✗ MISMATCH!")
            all_success = False

        # Verify that unpadded and padded give same masked content
        unpadded_masked_ids = [tid for tid, is_masked in zip(unpadded_ids, unpadded_mask) if is_masked]
        assert unpadded_masked_ids == masked_token_ids, \
            f"Masked token IDs don't match: {unpadded_masked_ids} != {masked_token_ids}"
        print(f"  ✓ Unpadded and padded masks produce identical token IDs")

    # SUMMARY
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if all_success:
        print("✓ ALL TESTS PASSED!")
        print("  - All unpadded sequences were found in padded sequences")
        print("  - All padded masks were created successfully")
        print("  - All masked contents were reconstructed correctly")
    else:
        print("✗ SOME TESTS FAILED!")
        print("  Please review the errors above")

    return all_success


def test_edge_cases(tokenizer_name: str):
    """
    Test edge cases like no masked content, all masked content, etc.
    """
    print(f"\n{'='*80}")
    print("TESTING EDGE CASES")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    edge_cases = [
        ("No masked content", "This text has no answer tags at all."),
        ("All masked", "<answer>Everything is masked</answer>"),
        ("Empty answer", "Empty: <answer></answer> here."),
    ]

    for case_name, text_with_tags in edge_cases:
        print(f"\n--- {case_name} ---")
        print(f"Text: {repr(text_with_tags)}")

        try:
            # Trim and get char mask
            trimmed_text, char_mask = trim_forward_prompt_and_get_char_mask(text_with_tags)

            # Tokenize individually
            unpadded_ids, token_mask = convert_char_mask_to_token_mask(
                trimmed_text,
                char_mask,
                tokenizer,
                debug=False
            )

            # Extract masked content
            masked_token_ids = [tid for tid, is_masked in zip(unpadded_ids, token_mask) if is_masked]
            masked_text = tokenizer.decode(masked_token_ids, skip_special_tokens=True)

            print(f"Masked tokens: {sum(token_mask)}/{len(token_mask)}")
            print(f"Masked content: {repr(masked_text)}")
            print(f"✓ Success")

        except Exception as e:
            print(f"✗ Error: {e}")


def test_stress_all_tokenizers():
    """
    Stress test: Test all tokenizers from config.rs to ensure they support return_offsets_mapping.
    Uses hardcoded expected answers to verify correctness.
    """
    # All local models from config.rs
    all_tokenizers = [
        "ibm-granite/granite-4.0-h-tiny",
        "ibm-granite/granite-4.0-h-small",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "Qwen/Qwen3-235B-A22B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "CohereLabs/aya-expanse-32b",
        "Unbabel/M-Prometheus-14B",
        "prometheus-eval/prometheus-8x7b-v2.0",
    ]

    # Hardcoded test case with expected answer
    test_text_with_tags = "Question: What is 2+2? Answer: <answer>4</answer>."
    expected_answer = "4"  # Hardcoded expected content

    print("="*80)
    print("STRESS TEST: ALL TOKENIZERS FROM config.rs")
    print("="*80)
    print(f"\nTesting {len(all_tokenizers)} tokenizers...")
    print(f"Test text: {repr(test_text_with_tags)}")
    print(f"Expected masked content: {repr(expected_answer)}")
    print()

    results = []

    for idx, tokenizer_name in enumerate(all_tokenizers, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(all_tokenizers)}] Testing: {tokenizer_name}")
        print(f"{'='*80}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            # Set padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"  Set pad_token to eos_token")

            # Test 1: Check if return_offsets_mapping is supported
            print(f"\n  Test 1: Checking return_offsets_mapping support...")
            try:
                test_encoded = tokenizer(
                    "test",
                    add_special_tokens=False,
                    return_tensors=None,
                    return_offsets_mapping=True
                )
                if 'offset_mapping' in test_encoded:
                    print(f"  ✓ return_offsets_mapping is supported")
                else:
                    print(f"  ✗ return_offsets_mapping not in output")
                    print(f"\n{'='*80}")
                    print(f"STRESS TEST FAILED ON FIRST ERROR")
                    print(f"{'='*80}")
                    print(f"Tokenizer: {tokenizer_name}")
                    print(f"Error: offset_mapping not in tokenizer output")
                    return False
            except Exception as e:
                print(f"  ✗ return_offsets_mapping not supported: {e}")
                print(f"\n{'='*80}")
                print(f"STRESS TEST FAILED ON FIRST ERROR")
                print(f"{'='*80}")
                print(f"Tokenizer: {tokenizer_name}")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                return False

            # Test 2: Full workflow test
            print(f"\n  Test 2: Full workflow with hardcoded expected answer...")

            # Trim and get char mask
            trimmed_text, char_mask = trim_forward_prompt_and_get_char_mask(test_text_with_tags)
            print(f"    Trimmed: {repr(trimmed_text)}")

            # Tokenize and get mask
            unpadded_ids, token_mask = convert_char_mask_to_token_mask(
                trimmed_text,
                char_mask,
                tokenizer,
                debug=False
            )

            # Extract masked content
            masked_token_ids = [tid for tid, is_masked in zip(unpadded_ids, token_mask) if is_masked]
            extracted_answer = tokenizer.decode(masked_token_ids, skip_special_tokens=True)

            print(f"    Masked tokens: {sum(token_mask)}/{len(token_mask)}")
            print(f"    Extracted: {repr(extracted_answer)}")
            print(f"    Expected:  {repr(expected_answer)}")

            # Verify against hardcoded expected answer
            if extracted_answer == expected_answer:
                print(f"  ✓ PASSED - Extracted content matches hardcoded expected answer")
                results.append((tokenizer_name, "PASSED", ""))
            else:
                print(f"  ✗ FAILED - Mismatch with hardcoded expected answer")
                print(f"\n{'='*80}")
                print(f"STRESS TEST FAILED ON FIRST ERROR")
                print(f"{'='*80}")
                print(f"Tokenizer: {tokenizer_name}")
                print(f"Expected: {repr(expected_answer)}")
                print(f"Got: {repr(extracted_answer)}")
                return False

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            print(f"\n{'='*80}")
            print(f"STRESS TEST FAILED ON FIRST ERROR")
            print(f"{'='*80}")
            print(f"Tokenizer: {tokenizer_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Print summary (only reached if all tests passed)
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY - ALL TESTS PASSED!")
    print("="*80)

    print(f"\nTotal tokenizers tested: {len(results)}")
    print(f"  ✓ All {len(results)} tokenizers PASSED")
    print(f"\nAll tokenizers support return_offsets_mapping and correctly extract masked content.")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test padding-aware tokenization workflow")
    parser.add_argument(
        "--stress-test",
        action="store_true",
        help="Test all tokenizers from config.rs"
    )
    args = parser.parse_args()

    if args.stress_test:
        # Stress test mode: test all tokenizers
        all_passed = test_stress_all_tokenizers()
        exit(0 if all_passed else 1)
    else:
        # Normal mode: test single tokenizer
        tokenizer_name = "prometheus-eval/prometheus-8x7b-v2.0"

        print("="*80)
        print("PADDING-AWARE TOKENIZATION WORKFLOW TEST")
        print("="*80)
        print(f"\nTokenizer: {tokenizer_name}")
        print("\nThis test verifies:")
        print("1. Individual tokenization produces correct masks")
        print("2. Batch tokenization adds padding correctly")
        print("3. Padded masks align with padded sequences")
        print("4. Masked content can be reconstructed from padded sequences")

        # Run main test
        success = test_padding_workflow(tokenizer_name)

        # Run edge case tests
        test_edge_cases(tokenizer_name)

        print("\n" + "="*80)
        if success:
            print("✓ TEST COMPLETE - ALL PASSED")
        else:
            print("✗ TEST COMPLETE - SOME FAILURES")
        print("="*80)
