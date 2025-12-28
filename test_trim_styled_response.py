"""
Unit tests for trim_styled_response_and_get_char_mask function
"""


def test_trim_styled_response_and_get_char_mask():
    """Test trim_styled_response_and_get_char_mask with various cases"""
    from src_py.utils import trim_styled_response_and_get_char_mask

    # Test Case 1: Simple single answer tag
    print("\n=== Test Case 1: Simple single answer ===")
    styled_response = "The answer is <answer>42</answer>."
    trimmed, char_mask = trim_styled_response_and_get_char_mask(styled_response)

    print(f"Input:    '{styled_response}'")
    print(f"Trimmed:  '{trimmed}'")
    print(f"Expected: 'The answer is 42.'")
    assert trimmed == "The answer is 42.", f"Trimmed mismatch: {trimmed}"
    assert len(char_mask) == len(trimmed), "Char mask length should match trimmed length"

    # Find where "42" is in the trimmed string
    answer_start = trimmed.index("42")
    answer_end = answer_start + 2

    # Check that these positions are marked as True
    print(f"Char mask: {char_mask}")
    print(f"'42' is at positions {answer_start}-{answer_end-1}")
    assert char_mask[answer_start], "First char of answer should be marked"
    assert char_mask[answer_start + 1], "Second char of answer should be marked"
    assert not char_mask[0], "First char should not be marked"
    print("✓ Test Case 1 passed")

    # Test Case 2: Multiple answer tags
    print("\n=== Test Case 2: Multiple answer tags ===")
    styled_response = "First: <answer>true</answer>, Second: <answer>false</answer>."
    trimmed, char_mask = trim_styled_response_and_get_char_mask(styled_response)

    print(f"Input:    '{styled_response}'")
    print(f"Trimmed:  '{trimmed}'")
    expected_trimmed = "First: true, Second: false."
    assert trimmed == expected_trimmed, f"Trimmed mismatch: {trimmed}"
    assert len(char_mask) == len(trimmed), "Char mask length should match trimmed length"

    # Count True values in char_mask
    true_count = sum(char_mask)
    print(f"Char mask: {char_mask}")
    print(f"Number of masked characters: {true_count}")
    assert true_count == 4 + 5, "Should have 9 characters marked (true + false)"
    print("✓ Test Case 2 passed")

    # Test Case 3: No answer tags
    print("\n=== Test Case 3: No answer tags ===")
    styled_response = "This is a plain response."
    trimmed, char_mask = trim_styled_response_and_get_char_mask(styled_response)

    print(f"Input:    '{styled_response}'")
    print(f"Trimmed:  '{trimmed}'")
    assert trimmed == styled_response, "Trimmed should be same as input when no tags"
    assert len(char_mask) == len(trimmed), "Char mask length should match trimmed length"

    print(f"Char mask: {char_mask}")
    assert not any(char_mask), "No characters should be marked when no tags"
    print("✓ Test Case 3 passed")

    # Test Case 4: Answer at the beginning
    print("\n=== Test Case 4: Answer at the beginning ===")
    styled_response = "<answer>Yes</answer>, that is correct."
    trimmed, char_mask = trim_styled_response_and_get_char_mask(styled_response)

    print(f"Input:    '{styled_response}'")
    print(f"Trimmed:  '{trimmed}'")
    assert trimmed == "Yes, that is correct.", f"Trimmed mismatch: {trimmed}"

    print(f"Char mask: {char_mask}")
    print(f"First 3 chars masked: {char_mask[:3]}")
    assert all(char_mask[:3]), "First 3 chars (Yes) should be marked"
    assert not char_mask[3], "Char after 'Yes' should not be marked"
    print("✓ Test Case 4 passed")

    # Test Case 5: Answer at the end
    print("\n=== Test Case 5: Answer at the end ===")
    styled_response = "The answer is <answer>Paris</answer>"
    trimmed, char_mask = trim_styled_response_and_get_char_mask(styled_response)

    print(f"Input:    '{styled_response}'")
    print(f"Trimmed:  '{trimmed}'")
    assert trimmed == "The answer is Paris", f"Trimmed mismatch: {trimmed}"

    print(f"Char mask: {char_mask}")
    print(f"Last 5 chars masked: {char_mask[-5:]}")
    assert all(char_mask[-5:]), "Last 5 chars (Paris) should be marked"
    print("✓ Test Case 5 passed")

    # Test Case 6: Empty answer tags
    print("\n=== Test Case 6: Empty answer tags ===")
    styled_response = "Nothing here: <answer></answer>!"
    trimmed, char_mask = trim_styled_response_and_get_char_mask(styled_response)

    print(f"Input:    '{styled_response}'")
    print(f"Trimmed:  '{trimmed}'")
    assert trimmed == "Nothing here: !", f"Trimmed mismatch: {trimmed}"
    assert len(char_mask) == len(trimmed), "Char mask length should match trimmed length"
    print(f"Char mask: {char_mask}")
    print("✓ Test Case 6 passed")

    # Test Case 7: Adjacent answer tags
    print("\n=== Test Case 7: Adjacent answer tags ===")
    styled_response = "<answer>A</answer><answer>B</answer>"
    trimmed, char_mask = trim_styled_response_and_get_char_mask(styled_response)

    print(f"Input:    '{styled_response}'")
    print(f"Trimmed:  '{trimmed}'")
    assert trimmed == "AB", f"Trimmed mismatch: {trimmed}"
    print(f"Char mask: {char_mask}")
    assert all(char_mask), "All chars should be marked"
    print("✓ Test Case 7 passed")

    # Test Case 8: Nested-like tags (should only match outermost)
    print("\n=== Test Case 8: Answer with special characters ===")
    styled_response = "Result: <answer>x > 5</answer> is true"
    trimmed, char_mask = trim_styled_response_and_get_char_mask(styled_response)

    print(f"Input:    '{styled_response}'")
    print(f"Trimmed:  '{trimmed}'")
    assert trimmed == "Result: x > 5 is true", f"Trimmed mismatch: {trimmed}"
    print(f"Char mask: {char_mask}")

    # Find where "x > 5" is
    answer_text = "x > 5"
    answer_start = trimmed.index(answer_text)
    for i in range(len(answer_text)):
        assert char_mask[answer_start + i], f"Char at {answer_start + i} should be marked"
    print("✓ Test Case 8 passed")

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    test_trim_styled_response_and_get_char_mask()