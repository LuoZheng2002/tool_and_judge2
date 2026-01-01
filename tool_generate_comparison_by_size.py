#!/usr/bin/env python3
"""
Generate comparison plots showing how model size affects tool calling performance.
Compares models within the same family (GPT-5, Qwen, Llama, Granite) across sizes.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path
from tool_stacked_bar_common import (
    translate_modes, noise_modes, error_categories,
    load_model_statistics
)


# Model families with their sizes (ordered small to large)
MODEL_FAMILIES = {
    "GPT-5": [
        ("gpt-5-nano", "Nano"),
        ("gpt-5-mini", "Mini"),
        ("gpt-5", "Full"),
    ],
    "Qwen3": [
        ("Qwen-Qwen3-8B", "8B"),
        ("Qwen-Qwen3-14B", "14B"),
        ("Qwen-Qwen3-30B-A3B", "30B-A3B"),
        ("Qwen-Qwen3-32B", "32B"),
        ("Qwen-Qwen3-Next-80B-A3B-Instruct", "80B-A3B"),
    ],
    "Llama-3.1": [
        ("meta-llama-Llama-3.1-8B-Instruct", "8B"),
        ("meta-llama-Llama-3.1-70B-Instruct", "70B"),
    ],
    "Granite-4.0": [
        ("ibm-granite-granite-4.0-h-tiny", "Tiny"),
        ("ibm-granite-granite-4.0-h-small", "Small"),
    ],
}


def generate_size_comparison(family_name: str, models: list, language: str, 
                             result_dir: str, output_dir: str, output_format: str = "pdf"):
    """
    Generate a comparison plot for models of different sizes in the same family.
    Shows accuracy (1 - error rate) for each translate mode, comparing model sizes.
    """
    # Collect data for each model
    model_data = {}
    
    for model_dir, model_label in models:
        try:
            stats = load_model_statistics(model_dir, result_dir, language)
            model_data[model_label] = stats
        except ValueError as e:
            print(f"    Skipping {model_label}: {e}")
            continue
    
    if len(model_data) < 2:
        print(f"    Not enough models with data for {family_name} - {language}")
        return
    
    # Calculate accuracy for each translate mode (average across noise modes)
    # Accuracy = 1 - (sum of error rates / 200)
    
    accuracies = {label: [] for label in model_data.keys()}
    
    for trans_mode in translate_modes:
        for model_label, stats in model_data.items():
            mode_accuracies = []
            for noise_mode in noise_modes:
                cat_counts = stats.get(trans_mode, {}).get(noise_mode, {})
                total_errors = sum(cat_counts.get(cat, 0) for cat in error_categories)
                if total_errors > 0 or any(cat_counts.values()):
                    accuracy = 1 - (total_errors / 200.0)
                    mode_accuracies.append(accuracy)
            
            if mode_accuracies:
                accuracies[model_label].append(np.mean(mode_accuracies))
            else:
                accuracies[model_label].append(np.nan)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(translate_modes))
    width = 0.8 / len(model_data)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_data)))
    
    for i, (model_label, accs) in enumerate(accuracies.items()):
        offset = (i - len(model_data)/2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=model_label, color=colors[i])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            if not np.isnan(acc):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Translate Mode')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{family_name} Model Size Comparison - {language}')
    ax.set_xticks(x)
    ax.set_xticklabels(translate_modes)
    ax.legend(title='Model Size', loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"comparison_size_{family_name}_{language}.{output_format}")
    plt.savefig(output_path, format=output_format, dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model size comparison plots for each model family and language."
    )
    parser.add_argument(
        "--output-dir",
        default="tool/plots/comparisons",
        help="Directory to save comparison plots"
    )
    parser.add_argument(
        "--result-dir",
        default="tool/result",
        help="Directory containing the result files"
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["png", "pdf"],
        help="Output format (default: pdf)"
    )
    
    args = parser.parse_args()
    
    languages = ["Chinese", "Hindi", "Igbo"]
    
    for family_name, models in MODEL_FAMILIES.items():
        print(f"\n{'='*60}")
        print(f"Generating size comparison for {family_name}")
        print(f"{'='*60}")
        
        for language in languages:
            print(f"\n  Processing: {language}")
            generate_size_comparison(
                family_name, models, language,
                args.result_dir, args.output_dir, args.format
            )
