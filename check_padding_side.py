#!/usr/bin/env python3
"""
Script to check the default padding side of a tokenizer.

Usage:
    python check_padding_side.py <model_name>

Example:
    python check_padding_side.py meta-llama/Llama-3.3-70B-Instruct
    python check_padding_side.py prometheus-eval/prometheus-8x7b-v2.0
"""

import argparse
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Check the default padding side of a tokenizer"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name or path of the model/tokenizer to check"
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    print("=" * 80)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        print(f"\nModel: {args.model_name}")
        print(f"Tokenizer class: {type(tokenizer).__name__}")
        print(f"\nPadding configuration:")
        print(f"  padding_side: {tokenizer.padding_side}")
        print(f"  pad_token: {repr(tokenizer.pad_token)}")
        print(f"  pad_token_id: {tokenizer.pad_token_id}")

        if tokenizer.pad_token is None:
            print(f"\n⚠ Warning: pad_token is None!")
            print(f"  eos_token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
            print(f"  Suggestion: Set tokenizer.pad_token = tokenizer.eos_token")

        print(f"\nOther relevant tokens:")
        print(f"  bos_token: {repr(tokenizer.bos_token)} (id: {tokenizer.bos_token_id})")
        print(f"  eos_token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")

        # Try to get model_max_length
        if hasattr(tokenizer, 'model_max_length'):
            print(f"\nModel max length: {tokenizer.model_max_length}")

    except Exception as e:
        print(f"\n❌ Error loading tokenizer: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
