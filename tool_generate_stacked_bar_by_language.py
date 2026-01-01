import matplotlib
matplotlib.use("Agg")  # HPC-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from typing import Dict, List

from tool_stacked_bar_common import (
    translate_modes,
    noise_modes,
    error_categories,
    category_colors,
    pascal_to_readable,
    language_tag_map,
    parse_filename_tags,
    map_noise_tag_to_mode,
    map_tags_to_translate_mode,
    category_map,
)


def load_multi_model_multi_language_statistics(model_names: List[str], result_dir: str,
                                                 languages: List[str], translate_modes_dict: Dict[str, str]) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
    """
    Load error statistics for multiple models and languages.

    Args:
        model_names: List of model directory names (e.g., ["gpt-5", "gpt-5-mini"])
        result_dir: Directory containing the result files
        languages: List of language names (e.g., ["English", "Chinese", "Hindi", "Igbo"])
        translate_modes_dict: Dictionary mapping language to translate mode (e.g., {"English": "NT", "Chinese": "FT"})

    Returns:
        Nested dictionary: dict[model][language][noise_mode][category] = count
    """
    # Initialize data structure
    data_dict = {}
    for model_name in model_names:
        data_dict[model_name] = {}
        for language in languages:
            data_dict[model_name][language] = {}
            for nm in noise_modes:
                data_dict[model_name][language][nm] = {cat: 0 for cat in error_categories}

    # Load data for each model
    for model_name in model_names:
        print(f"\nLoading data for model: {model_name}")
        for language in languages:
            translate_mode = translate_modes_dict.get(language)
            if translate_mode is None:
                raise ValueError(f"No translate mode specified for language '{language}'")

            # Load single language statistics
            lang_data = load_single_language_statistics(model_name, result_dir, language, translate_mode)
            data_dict[model_name][language] = lang_data

    return data_dict


def load_single_language_statistics(model_name: str, result_dir: str, language: str,
                                      translate_mode: str) -> Dict[str, Dict[str, int]]:
    """
    Load error statistics for a single language for a specific model and translate mode.

    Args:
        model_name: The model directory name (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
        result_dir: Directory containing the result files
        language: Language name (e.g., "Chinese", "Hindi", "Igbo", "English")
        translate_mode: The translate mode to filter by (e.g., "FT", "PT", "NT")

    Returns:
        Nested dictionary: dict[noise_mode][category] = count
    """
    # Get the language tag for filtering
    language_tag = language_tag_map.get(language)
    if language_tag is None:
        raise ValueError(f"Unknown language '{language}'. Valid options: {', '.join(language_tag_map.keys())}")

    # Initialize data structure
    data_dict = {}
    for nm in noise_modes:
        data_dict[nm] = {cat: 0 for cat in error_categories}

    # Path to model's statistics directory
    model_categorize_dir = Path(result_dir) / model_name / "statistics"

    if not model_categorize_dir.exists():
        raise ValueError(f"Model directory '{model_categorize_dir}' does not exist")

    # Parse statistics files
    for score_file in model_categorize_dir.glob("*.json"):
        try:
            with open(score_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                category_counts = data.get("category_counts")

                if category_counts is None:
                    continue

                # Extract translate and noise modes from filename
                filename = score_file.stem  # Remove .json extension

                try:
                    (dataset_name, file_language_tag, translate_level_tag, pre_translate_tag,
                     noise_tag, prompt_translate_tag, post_translate_tag) = parse_filename_tags(filename)
                except ValueError as e:
                    continue

                # Skip files that don't match the selected language
                if file_language_tag != language_tag:
                    continue

                # Map tags to modes
                try:
                    noise_mode = map_noise_tag_to_mode(noise_tag)
                    file_translate_mode = map_tags_to_translate_mode(
                        file_language_tag, translate_level_tag,
                        pre_translate_tag, prompt_translate_tag, post_translate_tag
                    )
                except ValueError as e:
                    continue

                # Skip if not the desired translate mode
                if file_translate_mode != translate_mode:
                    continue

                # Store the category counts
                for stats_category, count in category_counts.items():
                    pascal_category = category_map.get(stats_category)
                    if pascal_category and pascal_category in error_categories:
                        data_dict[noise_mode][pascal_category] = count

                print(f"  Loaded {score_file.name}: {language} + {noise_mode}")

        except Exception as e:
            print(f"  Error reading {score_file.name}: {e}")

    return data_dict


def generate_stacked_bar_chart_multi_model(model_names: List[str], output_dir: str,
                                             result_dir: str, translate_mode: str,
                                             noise_mode: str = "NO_NOISE",
                                             max_height: float = None) -> None:
    """
    Generate a stacked bar chart comparing multiple models across languages.
    Horizontal axis shows model x language combinations grouped by model.
    Uses English (NT), Chinese, Hindi, and Igbo for the specified translate mode.

    Args:
        model_names: List of model directory names (e.g., ["gpt-5", "gpt-5-mini"])
        output_dir: Directory to save the chart image
        result_dir: Directory containing the result files (default: "tool/result")
        translate_mode: The translate mode for non-English languages (e.g., "FT", "PT", "PRE", "POST")
        noise_mode: The noise mode to use (default: "NO_NOISE")
        max_height: Maximum height of the vertical axis (default: None, auto-calculated from data)
    """

    # Languages: English uses NT, others use the specified translate_mode
    languages = ["English", "Chinese", "Hindi", "Igbo"]
    translate_modes_dict = {
        "English": "NT",
        "Chinese": translate_mode,
        "Hindi": translate_mode,
        "Igbo": translate_mode
    }

    # Load data for all models and languages
    try:
        data_dict = load_multi_model_multi_language_statistics(
            model_names, result_dir, languages, translate_modes_dict
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Prepare data for plotting - grouped by model, with languages as inner categories
    bar_labels = []
    bar_data = []
    bar_positions = []
    bar_widths = []
    pos = 0
    bar_spacing = 0.6  # Spacing between bars within a model group
    group_spacing = 0.3  # Extra space between model groups

    # Language abbreviations for x-axis labels
    language_abbrev = {
        "English": "EN",
        "Chinese": "ZH",
        "Hindi": "HI",
        "Igbo": "IG"
    }

    for model_name in model_names:
        for language in languages:
            bar_labels.append(language_abbrev[language])
            category_counts = data_dict[model_name][language][noise_mode]
            bar_data.append(category_counts)
            bar_positions.append(pos)
            bar_widths.append(0.4)  # Standard bar width
            pos += bar_spacing
        pos += group_spacing  # Add extra space after each model group

    title = f"Tool Calling Errors Across Models - {translate_mode} × {noise_mode}"
    output_name = f"stacked_bar_by_language_multi_model_{translate_mode}_{noise_mode}.pdf"

    # Create DataFrame for easier plotting
    df_data = []
    for counts in bar_data:
        df_data.append([counts[cat] for cat in error_categories])

    df = pd.DataFrame(df_data, index=bar_labels, columns=error_categories)

    # Check if we have any data
    if df.sum().sum() == 0:
        print(f"Error: No error data found for the specified models and configuration")
        return

    # Print summary
    print(f"\nError distribution for models {', '.join(model_names)} - {translate_mode}:")
    print(df)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    # Convert counts to rates by dividing by 200
    df_rate = df / 200.0

    # Create stacked bars with custom positions
    x_positions = bar_positions
    bottom = np.zeros(len(bar_positions))

    for category in error_categories:
        values = df_rate[category].values
        # Convert category name to readable format for legend
        readable_label = pascal_to_readable(category)
        # Use custom bar widths for each bar
        ax.bar(x_positions, values, label=readable_label, bottom=bottom,
               color=category_colors[category], edgecolor='white', linewidth=0.5, width=bar_widths)
        bottom += values

    # Calculate totals for each bar (as rates)
    totals = df_rate.sum(axis=1).values

    # Calculate y-axis max height (ceiling to nearest 0.1)
    if max_height is None:
        data_max = totals.max()
        max_height = np.ceil(data_max * 10) / 10  # Round up to nearest 0.1
        if max_height == data_max:  # If already at boundary, add 0.1
            max_height += 0.1

    # Add total numbers on top of each bar
    for i, total in enumerate(totals):
        if total > 0:  # Only annotate if there's data
            ax.text(x_positions[i], total, f'{total:.3f}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Set y-axis range to the calculated max_height
    ax.set_ylim(0, max_height)

    # Customize plot
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=50)

    # Place legend at the top, below the title and above the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=10, ncol=4, frameon=True)

    # Handle x-axis labels and ticks
    # Set x-tick positions and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, rotation=0, ha='center', fontsize=9)

    # Add model group labels as a second row below language labels
    pos_tracker = 0
    for model_name in model_names:
        # Calculate center of the model's language bars
        group_center = pos_tracker + (len(languages) - 1) * bar_spacing / 2
        ax.text(group_center, -max_height * 0.08, model_name,
               ha='center', va='top', fontsize=11, fontweight='bold', rotation=5)
        pos_tracker += len(languages) * bar_spacing + group_spacing

    # Add "Model" label below the group names
    overall_center = (x_positions[0] + x_positions[-1]) / 2
    ax.text(overall_center, -max_height * 0.14, 'Model',
           ha='center', va='top', fontsize=12)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved stacked bar chart to {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate stacked bar charts comparing multiple models across languages. "
                    "Shows English (NT), Chinese, Hindi, and Igbo for the specified translate mode. "
                    "Languages are inner categories, models are outer categories."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model names to compare (e.g., gpt-5 gpt-5-mini gpt-5-nano)"
    )
    parser.add_argument(
        "translate_mode",
        choices=[tm for tm in translate_modes if tm != "NT"],
        help="Translate mode for non-English languages (e.g., FT, PT, PRE, POST). English always uses NT."
    )
    parser.add_argument(
        "--noise-mode",
        default="NO_NOISE",
        choices=noise_modes,
        help="Noise mode to use (default: NO_NOISE)"
    )
    parser.add_argument(
        "--output-dir",
        default="tool/plots/stacked_bars_by_language",
        help="Directory to save chart images (default: tool/plots/stacked_bars_by_language)"
    )
    parser.add_argument(
        "--result-dir",
        default="tool/result",
        help="Directory containing the result files (default: tool/result)"
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=None,
        help="Maximum height of the vertical axis (default: auto-calculated from data, rounded up to nearest 0.1)"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Generating stacked bar chart for models: {', '.join(args.models)}")
    print(f"Languages: English (NT), Chinese ({args.translate_mode}), Hindi ({args.translate_mode}), Igbo ({args.translate_mode})")
    print(f"Noise Mode: {args.noise_mode}")
    print(f"{'='*60}")

    # Generate combined chart with all models and languages
    generate_stacked_bar_chart_multi_model(
        args.models,
        args.output_dir,
        args.result_dir,
        args.translate_mode,
        noise_mode=args.noise_mode,
        max_height=args.max_height
    )
