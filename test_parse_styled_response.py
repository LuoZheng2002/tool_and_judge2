"""
Unit tests for parse_styled_response_to_mask function
"""

import sys
from transformers import AutoTokenizer


def test_parse_styled_response_to_mask():
    """Test parse_styled_response_to_mask with various cases"""
    from src_py.utils import parse_styled_response_to_mask

    # Use a simple tokenizer for testing (GPT-2 tokenizer is widely available)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Test Case 1: Simple single answer tag
    print("\n=== Test Case 1: Simple single answer ===")
    styled_response = "The answer is <answer>42</answer>."
    trimmed, mask = parse_styled_response_to_mask(styled_response, tokenizer)

    print(f"Input: {styled_response}")
    print(f"Trimmed: {trimmed}")
    print(f"Expected trimmed: 'The answer is 42.'")
    assert trimmed == "The answer is 42.", f"Trimmed mismatch: {trimmed}"

    # Tokenize to see the mask alignment
    tokens = tokenizer(trimmed, add_special_tokens=False).input_ids
    token_texts = [tokenizer.decode([t]) for t in tokens]

    print(f"Tokens: {token_texts}")
    print(f"Mask:   {mask}")
    print(f"Mask length: {len(mask)}, Tokens length: {len(tokens)}")
    assert len(mask) == len(tokens), "Mask length should match token count"

    # Check that "42" token is marked as answer
    # The exact position depends on tokenization, so we check that at least one token is marked
    assert any(mask), "At least one token should be marked as answer"
    print("✓ Test Case 1 passed")

    # Test Case 2: Multiple answer tags
    print("\n=== Test Case 2: Multiple answer tags ===")
    styled_response = "The first is <answer>true</answer> and the second is <answer>false</answer>."
    trimmed, mask = parse_styled_response_to_mask(styled_response, tokenizer)

    print(f"Input: {styled_response}")
    print(f"Trimmed: {trimmed}")
    expected_trimmed = "The first is true and the second is false."
    assert trimmed == expected_trimmed, f"Trimmed mismatch: {trimmed}"

    tokens = tokenizer(trimmed, add_special_tokens=False).input_ids
    token_texts = [tokenizer.decode([t]) for t in tokens]

    print(f"Tokens: {token_texts}")
    print(f"Mask:   {mask}")

    # Count True values in mask
    true_count = sum(mask)
    print(f"Number of answer tokens: {true_count}")
    assert true_count >= 2, "Should have at least 2 tokens marked as answers"
    print("✓ Test Case 2 passed")

    # Test Case 3: No answer tags
    print("\n=== Test Case 3: No answer tags ===")
    styled_response = "This is a plain response without any tags."
    trimmed, mask = parse_styled_response_to_mask(styled_response, tokenizer)

    print(f"Input: {styled_response}")
    print(f"Trimmed: {trimmed}")
    assert trimmed == styled_response, "Trimmed should be same as input when no tags"

    tokens = tokenizer(trimmed, add_special_tokens=False).input_ids
    token_texts = [tokenizer.decode([t]) for t in tokens]

    print(f"Tokens: {token_texts}")
    print(f"Mask:   {mask}")

    assert not any(mask), "No tokens should be marked as answer when no tags"
    print("✓ Test Case 3 passed")

    # Test Case 4: Answer at the beginning
    print("\n=== Test Case 4: Answer at the beginning ===")
    styled_response = "<answer>Yes</answer>, that is correct."
    trimmed, mask = parse_styled_response_to_mask(styled_response, tokenizer)

    print(f"Input: {styled_response}")
    print(f"Trimmed: {trimmed}")
    assert trimmed == "Yes, that is correct.", f"Trimmed mismatch: {trimmed}"

    tokens = tokenizer(trimmed, add_special_tokens=False).input_ids
    token_texts = [tokenizer.decode([t]) for t in tokens]

    print(f"Tokens: {token_texts}")
    print(f"Mask:   {mask}")

    # First token(s) should be marked
    assert mask[0], "First token should be marked as answer"
    print("✓ Test Case 4 passed")

    # Test Case 5: Answer at the end
    print("\n=== Test Case 5: Answer at the end ===")
    styled_response = "The final answer is <answer>Paris</answer>"
    trimmed, mask = parse_styled_response_to_mask(styled_response, tokenizer)

    print(f"Input: {styled_response}")
    print(f"Trimmed: {trimmed}")
    assert trimmed == "The final answer is Paris", f"Trimmed mismatch: {trimmed}"

    tokens = tokenizer(trimmed, add_special_tokens=False).input_ids
    token_texts = [tokenizer.decode([t]) for t in tokens]

    print(f"Tokens: {token_texts}")
    print(f"Mask:   {mask}")

    # Last token(s) should be marked
    assert any(mask[-2:]), "Last tokens should include answer"
    print("✓ Test Case 5 passed")

    # Test Case 6: Complex multi-word answer
    print("\n=== Test Case 6: Complex multi-word answer ===")
    styled_response = "The structure that collects urine is the <answer>urinary bladder</answer>."
    trimmed, mask = parse_styled_response_to_mask(styled_response, tokenizer)

    print(f"Input: {styled_response}")
    print(f"Trimmed: {trimmed}")
    expected = "The structure that collects urine is the urinary bladder."
    assert trimmed == expected, f"Trimmed mismatch: {trimmed}"

    tokens = tokenizer(trimmed, add_special_tokens=False).input_ids
    token_texts = [tokenizer.decode([t]) for t in tokens]

    print(f"Tokens: {token_texts}")
    print(f"Mask:   {mask}")
    print(f"Tokens marked as answer: {[token_texts[i] for i, m in enumerate(mask) if m]}")

    # Should have multiple consecutive tokens marked
    assert sum(mask) >= 2, "Multi-word answer should mark multiple tokens"
    print("✓ Test Case 6 passed")

    # Test Case 7: Empty answer tags
    print("\n=== Test Case 7: Empty answer tags ===")
    styled_response = "The answer is <answer></answer> nothing."
    trimmed, mask = parse_styled_response_to_mask(styled_response, tokenizer)

    print(f"Input: {styled_response}")
    print(f"Trimmed: {trimmed}")
    assert trimmed == "The answer is  nothing.", f"Trimmed mismatch: {trimmed}"

    tokens = tokenizer(trimmed, add_special_tokens=False).input_ids
    token_texts = [tokenizer.decode([t]) for t in tokens]

    print(f"Tokens: {token_texts}")
    print(f"Mask:   {mask}")
    print("✓ Test Case 7 passed")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    test_parse_styled_response_to_mask()