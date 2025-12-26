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
                                         language: str,
                                         translate_mode: str,
                                         max_height: float = 0.5,
                                         show_all_combined: bool = False) -> None:
    """
    Generate a stacked bar chart comparing models showing error type distributions.
    Horizontal axis shows model x noise mode combinations.

    Args:
        model_names: List of model directory names (e.g., ["gpt-5", "gpt-5-mini", "gpt-5-nano"])
        output_dir: Directory to save the chart image
        result_dir: Directory containing the result files (default: "tool/result")
        language: The language name for the plot title (e.g., "Chinese", "Hindi", "Igbo")
        translate_mode: The translate mode to filter by (e.g., "FT", "PT", "PRE", "POST")
        max_height: Maximum height of the vertical axis (default: 0.5, range: 0.0-1.0)
        show_all_combined: If True, show all model x noise mode combinations in a single plot with grouping
    """

    # Load data using common module
    try:
        data_dict = load_multi_model_statistics(model_names, result_dir, language, translate_mode)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Prepare data for plotting
    if show_all_combined:
        # Show all model x noise mode combinations grouped by model
        bar_labels = []
        bar_data = []
        bar_positions = []
        pos = 0
        bar_spacing = 0.6  # Spacing between bars within a group
        group_spacing = 0.3  # Extra space between model groups

        # Map noise modes to short abbreviations
        noise_mode_abbrev = {
            "NO_NOISE": "NO",
            "PARAPHRASE": "PARA",
            "SYNONYM": "SYNO"
        }

        for model_name in model_names:
            for nm in noise_modes:
                bar_labels.append(noise_mode_abbrev[nm])  # Use abbreviated noise mode name
                category_counts = data_dict[model_name][nm]
                bar_data.append(category_counts)
                bar_positions.append(pos)
                pos += bar_spacing  # Use smaller spacing between bars
            pos += group_spacing  # Add extra space after each model group

        title = f"Tool Calling Errors Under {language} Queries - {translate_mode}"
        output_name = f"stacked_bar_by_model_{language}_{translate_mode}_all_combined.png"
    else:
        # Default: show all combinations without special grouping
        bar_labels = []
        bar_data = []
        bar_positions = None
        for model_name in model_names:
            for nm in noise_modes:
                bar_labels.append(f"{model_name}_{nm}")
                category_counts = data_dict[model_name][nm]
                bar_data.append(category_counts)
        title = f"Tool Calling Error Rate Under {language} Queries - {translate_mode}"
        output_name = f"stacked_bar_by_model_{language}_{translate_mode}_all.png"

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
    print(f"\nError distribution for {language} - {translate_mode}:")
    print(df)

    # Plot stacked bar chart (wider if showing all combined)
    if show_all_combined:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Convert counts to rates by dividing by 200
    df_rate = df / 200.0

    # Create stacked bars with custom positions if specified
    if bar_positions is not None:
        x_positions = bar_positions
        bottom = np.zeros(len(bar_positions))
    else:
        x_positions = range(len(bar_labels))
        bottom = np.zeros(len(bar_labels))

    for category in error_categories:
        values = df_rate[category].values
        # Convert category name to readable format for legend
        readable_label = pascal_to_readable(category)
        # Use half width (0.4) for combined view, standard width (0.8) otherwise
        bar_width = 0.4 if show_all_combined else 0.8
        ax.bar(x_positions, values, label=readable_label, bottom=bottom,
               color=category_colors[category], edgecolor='white', linewidth=0.5, width=bar_width)
        bottom += values

    # Calculate totals for each bar (as rates)
    totals = df_rate.sum(axis=1).values

    # Add total numbers on top of each bar
    for i, total in enumerate(totals):
        if total > 0:  # Only annotate if there's data
            x_pos = x_positions[i] if bar_positions is not None else i
            ax.text(x_pos, total, f'{total:.3f}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Set y-axis range to the specified max_height
    ax.set_ylim(0, max_height)

    # Customize plot
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Place legend outside plot area on the right with smaller font
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

    # Handle x-axis labels and ticks
    if show_all_combined:
        # Set x-tick positions and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)

        # Add model group labels as a second row below noise mode labels
        # Position them closer to the axis (smaller negative offset)
        bar_spacing = 0.6
        group_spacing = 0.3
        for i, model_name in enumerate(model_names):
            # Calculate the center position of each model group
            group_center = i * (len(noise_modes) * bar_spacing + group_spacing) + (len(noise_modes) - 1) * bar_spacing / 2
            ax.text(group_center, -max_height * 0.08, model_name,
                   ha='center', va='top', fontsize=10, fontweight='bold')

        # Add "Model and Configuration" label below the group names (larger negative offset)
        # Calculate the center of all bars for proper centering
        overall_center = (x_positions[0] + x_positions[-1]) / 2
        ax.text(overall_center, -max_height * 0.12, 'Model and Configuration',
               ha='center', va='top', fontsize=12)
    elif len(bar_labels) > 10:
        ax.set_xlabel('Model and Configuration', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    else:
        ax.set_xlabel('Model and Configuration', fontsize=12)
        plt.xticks(rotation=0)

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
        description="Generate stacked bar charts comparing models showing error type distributions."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model names to compare (e.g., gpt-5 gpt-5-mini gpt-5-nano)"
    )
    parser.add_argument(
        "language",
        help="Language name for the plot title (e.g., Chinese, Hindi, Igbo)"
    )
    parser.add_argument(
        "translate_mode",
        choices=translate_modes,
        help="Translate mode to filter by (e.g., FT, PT, PRE, POST)"
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
        default=0.5,
        help="Maximum height of the vertical axis (default: 0.5, range: 0.0-1.0)"
    )
    parser.add_argument(
        "--all-combined",
        action="store_true",
        help="Generate a single plot with all model x noise mode combinations grouped by model"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Generating stacked bar charts comparing models")
    print(f"Language: {args.language}, Translate Mode: {args.translate_mode}")
    print(f"Models: {', '.join(args.models)}")
    print(f"{'='*60}")

    if args.all_combined:
        # Generate single combined chart with all model x noise mode combinations
        generate_stacked_bar_chart_by_model(
            args.models,
            args.output_dir,
            args.result_dir,
            args.language,
            args.translate_mode,
            max_height=args.max_height,
            show_all_combined=True
        )
    else:
        # Generate chart with all combinations (default behavior)
        generate_stacked_bar_chart_by_model(
            args.models,
            args.output_dir,
            args.result_dir,
            args.language,
            args.translate_mode,
            max_height=args.max_height,
            show_all_combined=False
        )
