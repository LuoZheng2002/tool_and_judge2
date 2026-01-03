# Hardcoded Answers Verification

## Confirmed: All Answers in `<answer>` Tags are Hardcoded

### Main Test (`test_padding_workflow`)

All test cases use **hardcoded expected answers** for verification:

1. **Text**: `"Question: What is 2+2? Answer: The result is <answer>4</answer>."`
   - **Hardcoded Expected**: `"4"`

2. **Text**: `"Question: What is the capital of France? Answer: The capital of France is <answer>Paris</answer>. It is a beautiful city."`
   - **Hardcoded Expected**: `"Paris"`

3. **Text**: `"Short: <answer>A</answer>"`
   - **Hardcoded Expected**: `"A"`

4. **Text**: `"Question: Name three colors. Answer: <answer>Red, blue, and green</answer> are common colors."`
   - **Hardcoded Expected**: `"Red, blue, and green"`

5. **Text**: `"Multiple: <answer>First answer</answer> and then <answer>Second answer</answer> here."`
   - **Hardcoded Expected**: `"First answer Second answer"`

### Stress Test (`test_stress_all_tokenizers`)

Uses a single hardcoded test case:

- **Text**: `"Question: What is 2+2? Answer: <answer>4</answer>."`
- **Hardcoded Expected**: `"4"`

## Implementation Details

### Code Location: `test_padding_workflow.py` lines 41-47

```python
test_cases = [
    ("Question: What is 2+2? Answer: The result is <answer>4</answer>.", "4"),
    ("Question: What is the capital of France? Answer: The capital of France is <answer>Paris</answer>. It is a beautiful city.", "Paris"),
    ("Short: <answer>A</answer>", "A"),
    ("Question: Name three colors. Answer: <answer>Red, blue, and green</answer> are common colors.", "Red, blue, and green"),
    ("Multiple: <answer>First answer</answer> and then <answer>Second answer</answer> here.", "First answer Second answer"),
]
```

### Code Location: `test_padding_workflow.py` lines 283-284

```python
test_text_with_tags = "Question: What is 2+2? Answer: <answer>4</answer>."
expected_answer = "4"  # Hardcoded expected content
```

## Verification Process

1. **Extract** content from `<answer>` tags using the workflow
2. **Compare** extracted content against hardcoded expected string
3. **Report** PASS only if exact match
4. **Fail immediately** if mismatch (in stress test mode)

## Test Results

✅ All tests pass with hardcoded expected answers
✅ No dynamic extraction is used for verification
✅ Every answer inside `<answer></answer>` tags is verified against a hardcoded string literal
