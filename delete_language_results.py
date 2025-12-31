#!/usr/bin/env python3
"""
Script to delete all result files for a specific language across all models.

Usage:
    python3 delete_language_results.py <language>
    
Examples:
    python3 delete_language_results.py igbo
    python3 delete_language_results.py zh
    python3 delete_language_results.py hi
    
Options:
    --dry-run    Show what would be deleted without actually deleting
"""

import os
import sys
import argparse
from pathlib import Path


def delete_language_results(language: str, dry_run: bool = False) -> int:
    """
    Delete all result files containing the specified language tag.
    
    Args:
        language: Language tag (e.g., 'igbo', 'zh', 'hi', 'en')
        dry_run: If True, only print what would be deleted
        
    Returns:
        Number of files deleted
    """
    result_base = Path("tool/result")
    
    if not result_base.exists():
        print(f"Error: {result_base} does not exist")
        return 0
    
    # Pattern to match: files containing the language in their name
    # Format: ["BFCL_v4_multiple","<lang>",...]
    search_pattern = f'*"{language}"*'
    
    deleted_count = 0
    
    # Walk through all model directories
    for model_dir in result_base.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name.endswith('.jsonl') or model_dir.name.endswith('.lock'):
            continue
            
        # Check each subdirectory (pre_translate, generate_raw, etc.)
        for subdir in model_dir.iterdir():
            if not subdir.is_dir():
                continue
                
            # Find matching files
            for file_path in subdir.glob(search_pattern):
                if file_path.is_file():
                    if dry_run:
                        print(f"[DRY RUN] Would delete: {file_path}")
                    else:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    deleted_count += 1
    
    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description="Delete all result files for a specific language"
    )
    parser.add_argument(
        "language",
        help="Language tag to delete (e.g., igbo, zh, hi, en)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    language = args.language.lower()
    
    print(f"{'='*60}")
    if args.dry_run:
        print(f"DRY RUN: Finding files for language '{language}'")
    else:
        print(f"Deleting all result files for language: '{language}'")
    print(f"{'='*60}\n")
    
    # Confirm if not dry run
    if not args.dry_run:
        confirm = input(f"Are you sure you want to delete all '{language}' results? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return
    
    count = delete_language_results(language, dry_run=args.dry_run)
    
    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"Would delete {count} files")
    else:
        print(f"Deleted {count} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

