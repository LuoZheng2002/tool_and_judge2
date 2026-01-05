#!/usr/bin/env python3
"""
Generate scatter plots comparing preference logprob_signed_difference vs log perplexity difference.

For each model and pair of languages, creates 4 plots corresponding to:
- lang1 correct, lang2 correct
- lang1 correct, lang2 incorrect
- lang1 incorrect, lang2 correct
- lang1 incorrect, lang2 incorrect

Each point plots:
- X-axis: log10 perplexity difference (log10(perplexity_lang1) - log10(perplexity_lang2))
- Y-axis: preference logprob_signed_difference
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Add src_py to path to import utils
sys.path.insert(0, str(Path(__file__).parent / 'src_py'))
from utils import language_abbreviation_to_name


def load_jsonl(file_path: str) -> List[dict]:
    """Load JSON lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                    print(f"Line: {line}")
                    raise e

    return data


def get_perplexity_value(entry: dict) -> Optional[float]:
    """Extract perplexity value from Result type."""
    # perplexity = entry.get('perplexity')
    # if isinstance(perplexity, dict):
    #     if 'Ok' in perplexity:
    #         return perplexity['Ok']
    # return None
    perplexity = entry['perplexity']
    return perplexity


def get_preference_value(entry: dict) -> Optional[float]:
    """Extract logprob_signed_difference from Result type."""
    preference = entry.get('preference')
    if isinstance(preference, dict):
        if 'Ok' in preference:
            return preference['Ok'].get('logprob_signed_difference')
    return None


def load_perplexity_data(model_name: str, lang: str) -> Dict[int, Dict[bool, float]]:
    """
    Load perplexity data for a given model and language.

    Returns:
        Dict mapping index -> {is_correct: perplexity_value}
    """
    base_path = Path(f"judge/result/{model_name}/perplexity")

    perplexity_map = {}

    # Load correct answers
    correct_file = base_path / f"{lang}_correct.jsonl"
    if correct_file.exists():
        correct_data = load_jsonl(str(correct_file))
        for entry in correct_data:
            idx = entry['index']
            perplexity = get_perplexity_value(entry)
            if perplexity is not None:
                if idx not in perplexity_map:
                    perplexity_map[idx] = {}
                perplexity_map[idx][True] = perplexity

    # Load incorrect answers
    incorrect_file = base_path / f"{lang}_incorrect.jsonl"
    if incorrect_file.exists():
        incorrect_data = load_jsonl(str(incorrect_file))
        for entry in incorrect_data:
            idx = entry['index']
            perplexity = get_perplexity_value(entry)
            if perplexity is not None:
                if idx not in perplexity_map:
                    perplexity_map[idx] = {}
                perplexity_map[idx][False] = perplexity

    return perplexity_map


def load_preference_data(model_name: str, lang1: str, lang2: str) -> Dict[Tuple[bool, bool], List[dict]]:
    """
    Load preference data for a given model and language pair.

    Returns:
        Dict mapping (is_correct1, is_correct2) -> list of preference entries
    """
    base_path = Path(f"judge/result/{model_name}/preference")

    preference_map = {
        (True, True): [],
        (True, False): [],
        (False, True): [],
        (False, False): []
    }

    # Map of files to correctness tuples
    files_map = [
        (f"{lang1}_correct_{lang2}_correct.jsonl", (True, True)),
        (f"{lang1}_correct_{lang2}_incorrect.jsonl", (True, False)),
        (f"{lang1}_incorrect_{lang2}_correct.jsonl", (False, True)),
        (f"{lang1}_incorrect_{lang2}_incorrect.jsonl", (False, False)),
    ]

    for filename, key in files_map:
        file_path = base_path / filename
        if file_path.exists():
            data = load_jsonl(str(file_path))
            preference_map[key] = data

    return preference_map


def filter_by_axis_range(
    perplexity_diffs: List[float],
    preference_values: List[float],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None
) -> Tuple[List[float], List[float], int]:
    """
    Filter points based on specified axis ranges.

    Args:
        perplexity_diffs: List of perplexity difference values (x-axis)
        preference_values: List of preference values (y-axis)
        x_range: Optional tuple of (x_min, x_max) for x-axis range
        y_range: Optional tuple of (y_min, y_max) for y-axis range

    Returns:
        Tuple of (filtered_perplexity_diffs, filtered_preference_values, num_filtered_out)
    """
    if len(perplexity_diffs) == 0:
        return [], [], 0

    perp_arr = np.array(perplexity_diffs)
    pref_arr = np.array(preference_values)

    # Create mask for all points (start with all True)
    mask = np.ones(len(perp_arr), dtype=bool)

    # Apply x-axis range filter if specified
    if x_range is not None:
        x_min, x_max = x_range
        mask &= (perp_arr >= x_min) & (perp_arr <= x_max)

    # Apply y-axis range filter if specified
    if y_range is not None:
        y_min, y_max = y_range
        mask &= (pref_arr >= y_min) & (pref_arr <= y_max)

    # Filter the data
    filtered_perp = perp_arr[mask].tolist()
    filtered_pref = pref_arr[mask].tolist()

    num_removed = len(perplexity_diffs) - len(filtered_perp)

    return filtered_perp, filtered_pref, num_removed


def create_scatter_plot(
    perplexity_diffs: List[float],
    preference_values: List[float],
    lang1: str,
    lang2: str,
    is_correct1: bool,
    is_correct2: bool,
    output_path: str,
    num_filtered_out: int = 0,
    total_points: int = None,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    test_order: bool = False
):
    """Create and save a scatter plot."""
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    plt.scatter(perplexity_diffs, preference_values, alpha=0.3, s=10, edgecolors='none')

    # Add reference lines
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, linewidth=1)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3, linewidth=1)

    # Set axis limits if ranges are specified
    if x_range is not None:
        plt.xlim(x_range[0], x_range[1])
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])

    # Get language names
    lang1_name = language_abbreviation_to_name(lang1)
    lang2_name = language_abbreviation_to_name(lang2)

    # Labels and title
    if test_order:
        plt.xlabel(f'Log10(Perplexity Difference) ({lang1_name} - {lang2_name})', fontsize=20)
    else:
        plt.xlabel(f'Log10(Perplexity) Difference ({lang1_name} - {lang2_name})', fontsize=20)
    plt.ylabel(f'Preference Log-Prob Difference ({lang1_name} - {lang2_name})', fontsize=20)
    correct1_str = "correct" if is_correct1 else "incorrect"
    correct2_str = "correct" if is_correct2 else "incorrect"

    if total_points is not None and num_filtered_out > 0:
        title = f'Preference Log-Prob Difference vs. Perplexity Difference\n{lang1_name} {correct1_str} answer vs. {lang2_name} {correct2_str} answer\n(n={len(perplexity_diffs)} after filtering, {num_filtered_out} points filtered out from {total_points})'
    else:
        title = f'Preference Log-Prob Difference vs. Perplexity Difference\n{lang1_name} {correct1_str} answer vs. {lang2_name} {correct2_str} answer\n(n={len(perplexity_diffs)})'
    plt.title(title, fontsize=20)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Increase tick label font size
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Add correlation coefficient
    if len(perplexity_diffs) > 0:
        corr = np.corrcoef(perplexity_diffs, preference_values)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate scatter plots of preference vs perplexity for a model'
    )
    parser.add_argument('model_name', type=str,
                       help='Model name (e.g., Qwen-Qwen3-8B)')
    parser.add_argument('lang1', type=str,
                       help='First language code (e.g., en)')
    parser.add_argument('lang2', type=str,
                       help='Second language code (e.g., zh_cn)')
    parser.add_argument('--output-dir', type=str, default='judge/plots/scatter',
                       help='Output directory for plots (default: judge/plots)')
    parser.add_argument('--x-min', type=float, default=None,
                       help='Minimum value for x-axis (log10 perplexity difference)')
    parser.add_argument('--x-max', type=float, default=None,
                       help='Maximum value for x-axis (log10 perplexity difference)')
    parser.add_argument('--y-min', type=float, default=None,
                       help='Minimum value for y-axis (preference values)')
    parser.add_argument('--y-max', type=float, default=None,
                       help='Maximum value for y-axis (preference values)')
    parser.add_argument('--test-order', action='store_true',
                       help='Use log10(perplexity1 - perplexity2) instead of log10(perplexity1) - log10(perplexity2)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare axis ranges
    x_range = None
    y_range = None
    if args.x_min is not None or args.x_max is not None:
        x_min = args.x_min if args.x_min is not None else -np.inf
        x_max = args.x_max if args.x_max is not None else np.inf
        x_range = (x_min, x_max)
    if args.y_min is not None or args.y_max is not None:
        y_min = args.y_min if args.y_min is not None else -np.inf
        y_max = args.y_max if args.y_max is not None else np.inf
        y_range = (y_min, y_max)

    print(f"Loading data for model: {args.model_name}")
    print(f"Languages: {args.lang1} vs {args.lang2}")
    if x_range is not None:
        print(f"X-axis range: [{x_range[0]}, {x_range[1]}]")
    if y_range is not None:
        print(f"Y-axis range: [{y_range[0]}, {y_range[1]}]")

    # Load perplexity data for both languages
    print(f"Loading perplexity data for {args.lang1}...")
    perplexity_lang1 = load_perplexity_data(args.model_name, args.lang1)
    print(f"  Found {len(perplexity_lang1)} indices with perplexity data")

    print(f"Loading perplexity data for {args.lang2}...")
    perplexity_lang2 = load_perplexity_data(args.model_name, args.lang2)
    print(f"  Found {len(perplexity_lang2)} indices with perplexity data")

    # Load preference data
    print(f"Loading preference data...")
    preference_data = load_preference_data(args.model_name, args.lang1, args.lang2)

    # Process each category
    for (is_correct1, is_correct2), entries in preference_data.items():
        print(f"\nProcessing {args.lang1} {'correct' if is_correct1 else 'incorrect'} vs "
              f"{args.lang2} {'correct' if is_correct2 else 'incorrect'}...")
        print(f"  Total preference entries: {len(entries)}")

        perplexity_diffs = []
        preference_values = []

        for entry in entries:
            idx = entry['index']

            # Get preference value
            pref_val = get_preference_value(entry)
            if pref_val is None:
                continue

            # Get perplexity values for both languages with correct/incorrect answers
            if idx not in perplexity_lang1 or idx not in perplexity_lang2:
                continue

            if is_correct1 not in perplexity_lang1[idx] or is_correct2 not in perplexity_lang2[idx]:
                continue

            perp1 = perplexity_lang1[idx][is_correct1]
            perp2 = perplexity_lang2[idx][is_correct2]

            # Calculate log10 perplexity difference
            if args.test_order:
                # Use log10(perp1 - perp2) when test_order is enabled
                # Skip if the difference is non-positive (can't take log)
                if perp1 - perp2 <= 0:
                    continue
                perp_diff = np.log10(perp1 - perp2)
            else:
                # Use log10(perp1) - log10(perp2) by default
                # Skip if either perplexity is non-positive (can't take log)
                if perp1 <= 0 or perp2 <= 0:
                    continue
                perp_diff = np.log10(perp1) - np.log10(perp2)

            perplexity_diffs.append(perp_diff)
            preference_values.append(pref_val)

        print(f"  Valid data points: {len(perplexity_diffs)}")

        if len(perplexity_diffs) > 0:
            # Apply axis range filtering if specified
            num_filtered_out = 0
            total_points = len(perplexity_diffs)

            if x_range is not None or y_range is not None:
                original_count = len(perplexity_diffs)
                perplexity_diffs, preference_values, num_filtered_out = filter_by_axis_range(
                    perplexity_diffs,
                    preference_values,
                    x_range=x_range,
                    y_range=y_range
                )
                print(f"  After range filtering: {len(perplexity_diffs)} points "
                      f"({num_filtered_out} points filtered out, "
                      f"{100 * num_filtered_out / original_count:.1f}%)")

            # Create output filename
            correct1_str = "correct" if is_correct1 else "incorrect"
            correct2_str = "correct" if is_correct2 else "incorrect"
            filename_suffix = ""
            if x_range is not None or y_range is not None:
                filename_suffix += "_filtered"
            if args.test_order:
                filename_suffix += "_test_order"
            output_filename = f"{args.model_name}_{args.lang1}_{correct1_str}_vs_{args.lang2}_{correct2_str}{filename_suffix}.pdf"
            output_path = output_dir / output_filename

            # Create plot
            create_scatter_plot(
                perplexity_diffs,
                preference_values,
                args.lang1,
                args.lang2,
                is_correct1,
                is_correct2,
                str(output_path),
                num_filtered_out=num_filtered_out,
                total_points=total_points if (x_range is not None or y_range is not None) else None,
                x_range=x_range,
                y_range=y_range,
                test_order=args.test_order
            )
        else:
            print(f"  Warning: No valid data points found for this category")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
