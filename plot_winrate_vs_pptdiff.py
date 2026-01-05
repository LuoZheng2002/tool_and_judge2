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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add src_py to path to import utils
sys.path.insert(0, str(Path(__file__).parent / 'src_py'))
from utils import language_abbreviation_to_name


def wilson_confidence_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successes (wins)
        total: Total number of trials
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 1.0)

    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


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
    num_bins: int = 12,
    test_order: bool = False,
    confidence_level: float = 0.95
):
    """Create and save a win rate vs perplexity difference plot."""
    if len(perplexity_diffs) == 0:
        print(f"  Warning: No data points to plot")
        return

    # Sort data by perplexity difference
    sorted_indices = np.argsort(perplexity_diffs)
    sorted_diffs = [perplexity_diffs[i] for i in sorted_indices]
    sorted_prefs = [preferences[i] for i in sorted_indices]

    # Create percentile-based bins
    n_samples = len(sorted_diffs)
    samples_per_bin = n_samples / num_bins

    # Calculate win rate for each percentile bin
    x_vals, win_rates, bin_counts = [], [], []
    lower_bounds, upper_bounds = [], []
    for i in range(num_bins):
        # Calculate start and end indices for this percentile bin
        start_idx = int(i * samples_per_bin)
        end_idx = int((i + 1) * samples_per_bin) if i < num_bins - 1 else n_samples

        if start_idx >= end_idx:
            continue

        # Get samples in this bin
        bin_diffs = sorted_diffs[start_idx:end_idx]
        bin_prefs = sorted_prefs[start_idx:end_idx]

        # Count wins (preference == 1)
        wins = sum(1 for p in bin_prefs if p == 1)
        total = len(bin_prefs)
        rate = wins / total

        # Calculate confidence interval
        lower, upper = wilson_confidence_interval(wins, total, confidence=confidence_level)

        # Use the median perplexity difference as the x-value for this bin
        x_val = np.median(bin_diffs)

        x_vals.append(x_val)
        win_rates.append(rate)
        bin_counts.append(total)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    if len(x_vals) == 0:
        print(f"  Warning: No valid bins with data")
        return

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot win rate with confidence interval
    plt.plot(x_vals, win_rates, marker='o', linewidth=2, markersize=8, color='red', label='Win rate', zorder=3)

    # Add shaded confidence interval
    ci_label = f'{int(confidence_level * 100)}% CI'
    plt.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.2, color='red', label=ci_label)

    # Add horizontal reference line at 0.5
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Get language names
    lang1_name = language_abbreviation_to_name(lang1)
    lang2_name = language_abbreviation_to_name(lang2)

    # Labels and title
    if test_order:
        plt.xlabel(f'Log10(Perplexity Difference) ({lang1_name} - {lang2_name})', fontsize=20)
    else:
        plt.xlabel(f'Log10(Perplexity) Difference ({lang1_name} - {lang2_name})', fontsize=20)
    plt.ylabel(f'Win Rate ({lang1_name} preferred)', fontsize=20)
    correct1_str = "correct" if is_correct1 else "incorrect"
    correct2_str = "correct" if is_correct2 else "incorrect"
    title = f'Win Rate vs. Perplexity Difference\n{lang1_name} {correct1_str} answer vs. {lang2_name} {correct2_str} answer\n(n={len(perplexity_diffs)} points, {num_bins} bins)'
    plt.title(title, fontsize=20)

    # Set y-axis limits based on confidence interval bounds with margin
    if len(lower_bounds) > 0 and len(upper_bounds) > 0:
        y_min = min(lower_bounds)
        y_max = max(upper_bounds)
        y_range = y_max - y_min
        margin = 0.1 * y_range if y_range > 0 else 0.1  # 10% margin, or 0.1 if all values are same
        plt.ylim(y_min - margin, y_max + margin)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Increase tick label font size
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Add legend
    plt.legend()

    # Add annotation showing bin sizes
    annotation_text = f'Bin sizes: min={min(bin_counts)}, max={max(bin_counts)}, avg={np.mean(bin_counts):.1f}'
    plt.text(0.05, 0.05, annotation_text,
             transform=plt.gca().transAxes,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=20)

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
    parser.add_argument('--output-dir', type=str, default='judge/plots/line',
                       help='Output directory for plots (default: judge/plots)')
    parser.add_argument('--num-bins', type=int, default=12,
                       help='Number of bins for win rate calculation (default: 12)')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                       help='Confidence level for confidence intervals (default: 0.95)')
    parser.add_argument('--test-order', action='store_true',
                       help='Use log10(perplexity1 - perplexity2) instead of log10(perplexity1) - log10(perplexity2)')

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
            if args.test_order:
                # Use log10(perp1 - perp2) when test_order is enabled
                # Skip if the difference is non-positive (can't take log)
                if perp1 - perp2 <= 0:
                    error_count += 1
                    continue
                perp_diff = math.log10(perp1 - perp2)
            else:
                # Use log10(perp1) - log10(perp2) by default
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
            suffix = "_test_order" if args.test_order else ""
            output_filename = f"{args.model_name}_{args.lang1}_{correct1_str}_vs_{args.lang2}_{correct2_str}_winrate{suffix}.pdf"
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
                num_bins=args.num_bins,
                test_order=args.test_order,
                confidence_level=args.confidence_level
            )
        else:
            print(f"  Warning: No valid data points found for this category")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
