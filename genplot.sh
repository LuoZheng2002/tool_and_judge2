#!/bin/bash
cd /home/pranav/translate/tool_and_judge2

source .venv/bin/activate

MODELS=(
  "Qwen-Qwen3-14B"
  "Qwen-Qwen3-30B-A3B"
  "Qwen-Qwen3-32B"
  "Qwen-Qwen3-8B"
  "Qwen-Qwen3-Next-80B-A3B-Instruct"
  "deepseek-chat"
  "gpt-5"
  "gpt-5-mini"
  "gpt-5-nano"
  "ibm-granite-granite-4.0-h-small"
  "ibm-granite-granite-4.0-h-tiny"
  "meta-llama-Llama-3.1-8B-Instruct"
  "meta-llama-Llama-3.1-70B-Instruct"
)

LANGUAGES=("Chinese" "Hindi" "Igbo")
FORMAT="pdf"  # Output format: png or pdf

mkdir -p tool/plots/heatmaps tool/plots/stacked_bars

for model in "${MODELS[@]}"; do
  echo "=== Generating plots for $model ==="
  
  # Generate heatmaps for each language
  for lang in "${LANGUAGES[@]}"; do
    echo "  Heatmap: $lang"
    python3 tool_generate_heatmap.py "$model" "$lang" --result-dir tool/result --output-dir tool/plots/heatmaps --format "$FORMAT"
  done
  
  # Generate stacked bars (script handles all languages internally)
  echo "  Stacked bars: all languages"
  python3 tool_generate_stacked_bar.py "$model" --result-dir tool/result --output-dir tool/plots/stacked_bars --format "$FORMAT"
done

echo "Done! Generated 78 plots (39 heatmaps + 39 stacked bars) in $FORMAT format"