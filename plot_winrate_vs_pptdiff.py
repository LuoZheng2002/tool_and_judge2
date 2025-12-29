#!/usr/bin/env python3
"""
Generate win rate vs perplexity difference plots for a model and language pair.

For each model and pair of languages, creates 4 plots corresponding to:
- lang1 correct, lang2 correct
- lang1 correct, lang2 incorrect
- lang1 incorrect, lang2 correct
- lang1 incorrect, lang2 incorrect

Each plot shows:
- X-axis: log10 perplexity difference (log10(perplexity_lang1) - log10(perplexity_lang2))
- Y-axis: win rate (proportion where preferred_answer == 1)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(file_path: str) -> List[dict]:
    """Load JSON lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_perplexity_value(entry: dict) -> Optional[float]:
    """Extract perplexity value from Result type."""
    perplexity = entry['perplexity']
    return perplexity


def get_preference_value(entry: dict) -> Optional[int]:
    """Extract preferred_answer from Result type."""
    preference = entry.get('preference')
    if isinstance(preference, dict):
        if 'Ok' in preference:
            return preference['Ok'].get('preferred_answer')
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


def create_winrate_plot(
    perplexity_diffs: List[float],
    preferences: List[int],
    lang1: str,
    lang2: str,
    is_correct1: bool,
    is_correct2: bool,
    output_path: str,
    num_bins: int = 12
):
    """Create and save a win rate vs perplexity difference plot."""
    if len(perplexity_diffs) == 0:
        print(f"  Warning: No data points to plot")
        return

    # Create bins
    min_val, max_val = min(perplexity_diffs), max(perplexity_diffs)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate win rate for each bin
    x_vals, win_rates, bin_counts = [], [], []
    for i in range(len(bins) - 1):
        start, end = bins[i], bins[i + 1]
        # Get all samples in this bin
        indices = [j for j, diff in enumerate(perplexity_diffs) if start <= diff < end]
        if not indices:
            continue

        # Count wins (preference == 1)
        wins = sum(1 for j in indices if preferences[j] == 1)
        total = len(indices)
        rate = wins / total

        x_vals.append(bin_centers[i])
        win_rates.append(rate)
        bin_counts.append(total)

    if len(x_vals) == 0:
        print(f"  Warning: No valid bins with data")
        return

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, win_rates, marker='o', linewidth=2, markersize=8, color='red', label='Win rate')

    # Add horizontal reference line at 0.5
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Labels and title
    plt.xlabel(f'Difference in Log10(Perplexity) ({lang1} - {lang2})', fontsize=12)
    plt.ylabel('Winning rate (preference == 1)', fontsize=12)

    correct1_str = "correct" if is_correct1 else "incorrect"
    correct2_str = "correct" if is_correct2 else "incorrect"
    title = f'{lang1} {correct1_str} vs {lang2} {correct2_str}\n(n={len(perplexity_diffs)} points, {num_bins} bins)'
    plt.title(title, fontsize=14)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.legend()

    # Add annotation showing bin sizes
    annotation_text = f'Bin sizes: min={min(bin_counts)}, max={max(bin_counts)}, avg={np.mean(bin_counts):.1f}'
    plt.text(0.05, 0.05, annotation_text,
             transform=plt.gca().transAxes,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate win rate vs perplexity difference plots for a model'
    )
    parser.add_argument('model_name', type=str,
                       help='Model name (e.g., meta-llama-Llama-3.3-70B-Instruct)')
    parser.add_argument('lang1', type=str,
                       help='First language code (e.g., en)')
    parser.add_argument('lang2', type=str,
                       help='Second language code (e.g., zh_cn)')
    parser.add_argument('--output-dir', type=str, default='judge/plots',
                       help='Output directory for plots (default: judge/plots)')
    parser.add_argument('--num-bins', type=int, default=12,
                       help='Number of bins for win rate calculation (default: 12)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for model: {args.model_name}")
    print(f"Languages: {args.lang1} vs {args.lang2}")
    print(f"Number of bins: {args.num_bins}")

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
        preferences = []
        error_count = 0

        for entry in entries:
            idx = entry['index']

            # Get preference value
            pref_val = get_preference_value(entry)
            if pref_val is None:
                error_count += 1
                continue

            # Get perplexity values for both languages with correct/incorrect answers
            if idx not in perplexity_lang1 or idx not in perplexity_lang2:
                error_count += 1
                continue

            if is_correct1 not in perplexity_lang1[idx] or is_correct2 not in perplexity_lang2[idx]:
                error_count += 1
                continue

            perp1 = perplexity_lang1[idx][is_correct1]
            perp2 = perplexity_lang2[idx][is_correct2]

            # Calculate log10 perplexity difference
            # Skip if either perplexity is non-positive (can't take log)
            if perp1 <= 0 or perp2 <= 0:
                error_count += 1
                continue

            perp_diff = math.log10(perp1) - math.log10(perp2)

            perplexity_diffs.append(perp_diff)
            preferences.append(pref_val)

        print(f"  Valid data points: {len(perplexity_diffs)}")
        if error_count > 0:
            print(f"  Skipped {error_count} entries due to missing/invalid data")

        if len(perplexity_diffs) > 0:
            # Create output filename
            correct1_str = "correct" if is_correct1 else "incorrect"
            correct2_str = "correct" if is_correct2 else "incorrect"
            output_filename = f"{args.model_name}_{args.lang1}_{correct1_str}_vs_{args.lang2}_{correct2_str}_winrate.png"
            output_path = output_dir / output_filename

            # Create plot
            create_winrate_plot(
                perplexity_diffs,
                preferences,
                args.lang1,
                args.lang2,
                is_correct1,
                is_correct2,
                str(output_path),
                num_bins=args.num_bins
            )
        else:
            print(f"  Warning: No valid data points found for this category")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
