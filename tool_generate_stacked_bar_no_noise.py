import matplotlib
matplotlib.use("Agg")  # HPC-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tool_stacked_bar_common import (
    error_categories,
    category_colors,
    pascal_to_readable,
    load_model_statistics,
)


def generate_stacked_bar_chart_no_noise(model_name: str, output_dir: str, result_dir: str,
                                        language: str,
                                        max_height: float = None,
                                        output_format: str = "png") -> None:
    """
    Generate a stacked bar chart for a given model showing error type distributions.
    Shows only 3 bars: NT, PAR, and FT, all in NO_NOISE mode.

    Args:
        model_name: The model directory name (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
        output_dir: Directory to save the chart image
        result_dir: Directory containing the result files (default: "tool/result")
        language: The language name for the plot title (e.g., "Chinese", "Hindi", "Igbo")
        max_height: Maximum height of the vertical axis (default: None, auto-calculated from data)
        output_format: Output format, "png" or "pdf" (default: "png")
    """

    # Load data using common module
    try:
        data_dict = load_model_statistics(model_name, result_dir, language)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Prepare data for plotting - only NT, PAR, FT in NO_NOISE mode
    translate_modes = ["NT", "PAR", "FT"]
    bar_labels = translate_modes
    bar_data = []

    for tm in translate_modes:
        category_counts = data_dict[tm]["NO_NOISE"]
        bar_data.append(category_counts)

    title = f"{model_name}'s performance on {language} datasets of different\ntranslation levels (no semantic perturbation added)"
    output_name = f"stacked_bar_no_noise_{model_name}_{language}.{output_format}"

    # Create DataFrame for easier plotting
    df_data = []
    for counts in bar_data:
        df_data.append([counts[cat] for cat in error_categories])

    df = pd.DataFrame(df_data, index=bar_labels, columns=error_categories)

    # Check if we have any data
    if df.sum().sum() == 0:
        print(f"Error: No error data found for model '{model_name}'")
        return

    # Print summary
    print(f"\nError distribution for model '{model_name}' (NO_NOISE only):")
    print(df)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 10))

    # Convert counts to rates by dividing by 200
    df_rate = df / 200.0

    # Create stacked bars
    x_positions = np.arange(len(bar_labels))
    bar_width = 0.6
    bottom = np.zeros(len(bar_labels))

    for category in error_categories:
        values = df_rate[category].values
        # Convert category name to readable format for legend
        readable_label = pascal_to_readable(category)
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
                   ha='center', va='bottom', fontsize=15, fontweight='bold')

    # Set y-axis range to the calculated max_height
    ax.set_ylim(0, max_height)

    # Customize plot
    ax.set_ylabel('Error Rate', fontsize=15, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=130, wrap=True)

    # Place legend at the bottom with full width
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.38), fontsize=13, ncol=2, frameon=True)

    # Set x-tick positions and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, fontsize=15, fontweight='bold')

    # Set y-tick label size
    ax.tick_params(axis='y', labelsize=13)

    # Add "Translation Level" label below the bar labels
    ax.set_xlabel('Translation Level', fontsize=15, fontweight='bold')

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, format=output_format, dpi=300, bbox_inches='tight')
    print(f"\nSaved stacked bar chart to {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate stacked bar charts showing error type distributions for specified models. "
                    "Shows only 3 bars: NT, PAR, and FT, all in NO_NOISE mode. "
                    "Automatically generates plots for Chinese, Hindi, and Igbo."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model names to process (e.g., gpt-5 gpt-5-mini gpt-5-nano)"
    )
    parser.add_argument(
        "--output-dir",
        default="tool/plots/stacked_bars_no_noise",
        help="Directory to save chart images (default: tool/plots/stacked_bars_no_noise)"
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
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["png", "pdf"],
        help="Output format (default: pdf)"
    )

    args = parser.parse_args()

    # Languages to process
    languages = ["Chinese", "Hindi", "Igbo"]

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Generating stacked bar charts (NO_NOISE only) for {model}")
        print(f"{'='*60}")

        for language in languages:
            print(f"\nProcessing language: {language}")
            generate_stacked_bar_chart_no_noise(
                model,
                args.output_dir,
                args.result_dir,
                language,
                max_height=args.max_height,
                output_format=args.format
            )
