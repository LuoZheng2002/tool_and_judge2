# Padding Issue Fix Summary

## Problem
The warning in `utils.py:214-224` was firing frequently because of padding mismatches during batching. When tokenizing a batch of prompts with different lengths, padding tokens are added, causing the decoded text to differ from the original trimmed prompt.

## Root Cause
The original workflow:
1. Applied chat template to each prompt individually
2. Trimmed `<answer>` tags to get character masks
3. Batch tokenized the trimmed prompts (adding padding)
4. Tried to match decoded text with original trimmed prompt
5. **Mismatch occurred** because padding was added during batch tokenization

## Solution
Revised the perplexity experiment workflow to handle padding properly:

### New Workflow

#### Part 1: Individual Tokenization (No Padding)
For each entry:
1. Apply chat template to get formatted prompt with `<answer>` tags
2. Trim `<answer>` tags to get `trimmed_prompt` and `char_mask`
3. **Tokenize INDIVIDUALLY** (no batching, no padding) using `convert_char_mask_to_token_mask`
   - Returns `(unpadded_ids, token_mask)`
4. Store `(unpadded_ids, token_mask)` for later use

#### Part 2: Batch Forward Pass (With Padding)
- Call `forward_for_perplexity` with batch of `trimmed_prompts`
- This does batch tokenization **with padding**
- Returns `forward_results` containing `(logits, input_ids)` for each entry

#### Part 3: Align Masks and Calculate Perplexity
For each entry:
1. Get `padded_ids` from forward result
2. Call `create_padded_mask(unpadded_mask, unpadded_ids, padded_ids)` to:
   - Find where `unpadded_ids` appears in `padded_ids` (as subsequence)
   - Generate padded mask with `False` for padding positions
3. Calculate perplexity using `padded_mask` and `padded_ids`

## Code Changes

### Modified Files

#### `src_py/utils.py`
1. **Modified `convert_char_mask_to_token_mask`**:
   - Now tokenizes internally WITHOUT padding
   - Returns `(input_ids, token_mask)` tuple instead of just `token_mask`
   - Removed the `input_ids` parameter (no longer needed for comparison)

2. **Added `find_unpadded_sequence_in_padded`**:
   - Searches for unpadded token sequence within padded sequence
   - Returns the starting offset

3. **Added `create_padded_mask`**:
   - Creates a padded version of the mask aligned with padded token IDs
   - Adds `False` for all padding positions

#### `judge.py`
- Revised the perplexity experiment section (lines 382-498)
- Implements the new 3-part workflow described above
- Stores `(unpadded_ids, token_mask)` before batch processing
- Uses `create_padded_mask` to align masks after forward pass

## Benefits
1. **Eliminates padding mismatch warnings**: No more decode/encode mismatches
2. **Correct mask alignment**: Masks properly align with padded sequences
3. **Cleaner separation**: Individual tokenization vs batch processing
4. **Better error handling**: Catches alignment issues and falls back gracefully

## Testing
Use `test_tokenizer.py` to verify:
- Batch encoding with padding works correctly
- Masks align properly with padded sequences
- No warnings are generated during normal operation
