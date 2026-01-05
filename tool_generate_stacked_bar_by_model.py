import matplotlib
matplotlib.use("Agg")  # HPC-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tool_stacked_bar_common import (
    translate_modes,
    noise_modes,
    error_categories,
    category_colors,
    pascal_to_readable,
    load_multi_model_statistics,
)


def generate_stacked_bar_chart_by_model(model_names: list, output_dir: str, result_dir: str,
                                         family_name: str,
                                         translate_mode: str,
                                         max_height: float = None) -> None:
    """
    Generate a stacked bar chart comparing models showing error type distributions.
    Horizontal axis shows model x language combinations grouped by model.

    Args:
        model_names: List of model directory names (e.g., ["gpt-5", "gpt-5-mini", "gpt-5-nano"])
        output_dir: Directory to save the chart image
        result_dir: Directory containing the result files (default: "tool/result")
        family_name: The model family name (e.g., "GPT-5", "Qwen3")
        translate_mode: The translate mode to filter by (e.g., "FT", "PT", "PRE", "POST")
        max_height: Maximum height of the vertical axis (default: None, auto-calculated from data)
    """

    # Always use NO_NOISE mode
    noise_mode = "NO_NOISE"

    # Load data using common module - load data for all languages
    languages = ["English", "Chinese", "Hindi", "Igbo"]

    # Load data for each language
    data_dict = {}
    for model_name in model_names:
        data_dict[model_name] = {}
        for lang in languages:
            try:
                # For English, use NT (Not Translated) mode; for others, use the specified translate_mode
                lang_translate_mode = "NT" if lang == "English" else translate_mode
                lang_data = load_multi_model_statistics([model_name], result_dir, lang, lang_translate_mode)
                data_dict[model_name][lang] = lang_data[model_name][noise_mode]
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load data for {model_name}, {lang}: {e}")
                # Initialize with zeros if data not found
                data_dict[model_name][lang] = {cat: 0 for cat in error_categories}

    # Prepare data for plotting - show all model x language combinations grouped by model
    bar_labels = []
    bar_data = []
    bar_positions = []
    pos = 0
    bar_spacing = 0.6  # Spacing between bars within a group
    group_spacing = 0.3  # Extra space between model groups

    # Map language to short abbreviations
    language_abbrev = {
        "English": "EN",
        "Chinese": "ZH",
        "Hindi": "HI",
        "Igbo": "IG"
    }

    for model_name in model_names:
        for lang in languages:
            bar_labels.append(language_abbrev[lang])  # Use abbreviated language name
            category_counts = data_dict[model_name][lang]
            bar_data.append(category_counts)
            bar_positions.append(pos)
            pos += bar_spacing  # Use smaller spacing between bars
        pos += group_spacing  # Add extra space after each model group

    # Generate title with new format
    title = f"Tool calling error rate among {family_name} series models - {translate_mode} - no semantic noise added"
    output_name = f"stacked_bar_by_model_{family_name}_{translate_mode}_NO_NOISE.pdf"

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
    print(f"\nError distribution for {family_name} - {translate_mode} - NO_NOISE:")
    print(df)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 10))

    # Convert counts to rates by dividing by 200
    df_rate = df / 200.0

    # Create stacked bars with custom positions
    x_positions = bar_positions
    bottom = np.zeros(len(bar_positions))

    for category in error_categories:
        values = df_rate[category].values
        # Convert category name to readable format for legend
        readable_label = pascal_to_readable(category)
        # Use half width (0.4) for combined view
        bar_width = 0.4
        ax.bar(x_positions, values, label=readable_label, bottom=bottom,
               color=category_colors[category], edgecolor='white', linewidth=0.5, width=bar_width)
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
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Set y-axis range to the calculated max_height
    ax.set_ylim(0, max_height)

    # Customize plot
    ax.set_ylabel('Error Rate', fontsize=15, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=120, wrap=True)

    # Place legend outside and above the plot, below the title
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.48), fontsize=13, ncol=2, frameon=True)

    # Handle x-axis labels and ticks
    # Set x-tick positions and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, rotation=0, ha='center', fontsize=12, fontweight='bold')

    # Set y-tick label size
    ax.tick_params(axis='y', labelsize=13)

    # Add model group labels as a second row below language labels
    # Position them closer to the axis (smaller negative offset)
    bar_spacing = 0.6
    group_spacing = 0.3
    for i, model_name in enumerate(model_names):
        # Calculate the center position of each model group
        group_center = i * (len(languages) * bar_spacing + group_spacing) + (len(languages) - 1) * bar_spacing / 2
        ax.text(group_center, -max_height * 0.08, model_name,
               ha='center', va='top', fontsize=11, fontweight='bold', rotation=4)

    # Add "Model and Configuration" label below the group names (larger negative offset)
    # Calculate the center of all bars for proper centering
    overall_center = (x_positions[0] + x_positions[-1]) / 2
    ax.text(overall_center, -max_height * 0.20, 'Model and Configuration',
           ha='center', va='top', fontsize=15, fontweight='bold')

    plt.tight_layout()

    # Adjust bottom margin to accommodate labels below the plot
    plt.subplots_adjust(bottom=0.15)

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
        description="Generate stacked bar charts comparing models showing error type distributions (NO_NOISE only)."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model names to compare (e.g., gpt-5 gpt-5-mini gpt-5-nano)"
    )
    parser.add_argument(
        "translate_mode",
        choices=translate_modes,
        help="Translate mode to filter by (e.g., FT, PT, PRE, POST)"
    )
    parser.add_argument(
        "--family-name",
        required=True,
        help="Model family name (e.g., GPT-5, Qwen3)"
    )
    parser.add_argument(
        "--output-dir",
        default="tool/plots/stacked_bars_by_model",
        help="Directory to save chart images (default: tool/plots/stacked_bars_by_model)"
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
    print(f"Generating stacked bar charts comparing models")
    print(f"Family: {args.family_name}, Translate Mode: {args.translate_mode}, Noise Mode: NO_NOISE")
    print(f"Models: {', '.join(args.models)}")
    print(f"{'='*60}")

    # Generate single combined chart with all model x language combinations
    generate_stacked_bar_chart_by_model(
        args.models,
        args.output_dir,
        args.result_dir,
        args.family_name,
        args.translate_mode,
        max_height=args.max_height
    )
