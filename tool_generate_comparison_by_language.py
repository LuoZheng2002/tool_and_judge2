#!/usr/bin/env python3
"""
Generate comparison plots showing how language affects tool calling performance.
Compares Chinese, Hindi, and Igbo for each model.
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


# All models to compare
ALL_MODELS = [
    ("gpt-5", "GPT-5"),
    ("gpt-5-mini", "GPT-5 Mini"),
    ("gpt-5-nano", "GPT-5 Nano"),
    ("deepseek-chat", "DeepSeek V3.2"),
    ("Qwen-Qwen3-8B", "Qwen3-8B"),
    ("Qwen-Qwen3-14B", "Qwen3-14B"),
    ("Qwen-Qwen3-30B-A3B", "Qwen3-30B-A3B"),
    ("Qwen-Qwen3-32B", "Qwen3-32B"),
    ("Qwen-Qwen3-Next-80B-A3B-Instruct", "Qwen3-80B-A3B"),
    ("meta-llama-Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("meta-llama-Llama-3.1-70B-Instruct", "Llama-3.1-70B"),
    ("ibm-granite-granite-4.0-h-tiny", "Granite-Tiny"),
    ("ibm-granite-granite-4.0-h-small", "Granite-Small"),
]


def generate_language_comparison(model_dir: str, model_label: str,
                                  result_dir: str, output_dir: str, output_format: str = "pdf"):
    """
    Generate a comparison plot for a single model across languages.
    Shows accuracy for each translate mode, comparing languages.
    """
    languages = ["Chinese", "Hindi", "Igbo"]
    language_colors = {"Chinese": "#E74C3C", "Hindi": "#3498DB", "Igbo": "#2ECC71"}
    
    # Collect data for each language
    lang_data = {}
    
    for language in languages:
        try:
            stats = load_model_statistics(model_dir, result_dir, language)
            lang_data[language] = stats
        except ValueError as e:
            print(f"    Skipping {language}: {e}")
            continue
    
    if len(lang_data) < 2:
        print(f"    Not enough languages with data for {model_label}")
        return
    
    # Calculate accuracy for each translate mode (average across noise modes)
    accuracies = {lang: [] for lang in lang_data.keys()}
    
    for trans_mode in translate_modes:
        for language, stats in lang_data.items():
            mode_accuracies = []
            for noise_mode in noise_modes:
                cat_counts = stats.get(trans_mode, {}).get(noise_mode, {})
                total_errors = sum(cat_counts.get(cat, 0) for cat in error_categories)
                if total_errors > 0 or any(cat_counts.values()):
                    accuracy = 1 - (total_errors / 200.0)
                    mode_accuracies.append(accuracy)
            
            if mode_accuracies:
                accuracies[language].append(np.mean(mode_accuracies))
            else:
                accuracies[language].append(np.nan)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(translate_modes))
    width = 0.8 / len(lang_data)
    
    for i, (language, accs) in enumerate(accuracies.items()):
        offset = (i - len(lang_data)/2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=language, color=language_colors.get(language, f'C{i}'))
        
        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            if not np.isnan(acc):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Translate Mode')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_label} - Language Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(translate_modes)
    ax.legend(title='Language', loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_dir.replace("/", "-")
    output_path = os.path.join(output_dir, f"comparison_language_{safe_name}.{output_format}")
    plt.savefig(output_path, format=output_format, dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate language comparison plots for each model."
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
    
    print("Generating language comparison plots for all models...")
    
    for model_dir, model_label in ALL_MODELS:
        print(f"\n  Processing: {model_label}")
        generate_language_comparison(
            model_dir, model_label,
            args.result_dir, args.output_dir, args.format
        )
