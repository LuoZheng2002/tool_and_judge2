#!/bin/bash
# Setup and run script for tool_and_judge2
# Works on both RunPod and local WSL

set -e

# Detect environment
if [ -d "/workspace" ]; then
    ENV_TYPE="runpod"
    WORKSPACE="/workspace/tool_and_judge2"
else
    ENV_TYPE="local"
    WORKSPACE="$HOME/translate/tool_and_judge2"
fi

echo "=============================================="
echo "  Environment: $ENV_TYPE"
echo "  Workspace: $WORKSPACE"
echo "=============================================="

cd "$WORKSPACE"

# ============================================
# Step 1: Rust setup
# ============================================
if ! command -v rustc &> /dev/null; then
    echo "[1/4] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
echo "[1/4] Rust ready: $(rustc --version)"

# ============================================
# Step 2: Python venv setup
# ============================================
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "[2/4] Python venv activated: $(python3 --version)"

# ============================================
# Step 3: Install dependencies
# ============================================
echo "[3/4] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet maturin vllm transformers openai huggingface_hub torch python-dotenv

# ============================================
# Step 4: Build Rust extension
# ============================================
echo "[4/4] Building Rust extension..."
# Fix Cargo.toml edition if needed
if grep -q 'edition = "2024"' Cargo.toml; then
    sed -i 's/edition = "2024"/edition = "2021"/' Cargo.toml
fi
maturin develop --release

# ============================================
# Step 5: Environment variables
# ============================================
if [ "$ENV_TYPE" = "runpod" ]; then
    export HF_HOME="/workspace/huggingface_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
mkdir -p "$HF_HOME"

# Load .env if exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
fi

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "Environment variables needed (set in .env or export):"
echo "  - HF_TOKEN (for HuggingFace gated models)"
echo "  - OPENAI_API_KEY (for GPT-5 models)"
echo "  - DEEPSEEK_API_KEY (for DeepSeek)"
echo ""
echo "Run commands:"
echo ""
echo "  # Local models (need GPU)"
echo "  python tool.py --config tool_config_llama8b.py --num-gpus 1"
echo "  python tool.py --config tool_config_slurm1.py --num-gpus 1      # Qwen3 8B"
echo ""
echo "  # API models (no GPU needed)"
echo "  python tool.py --config tool_config1.py --num-gpus 1            # GPT-5"
echo "  python tool.py --config tool_config4_deepseek.py --num-gpus 1   # DeepSeek"
echo ""

# If a config was passed as argument, run it
if [ -n "$1" ]; then
    NUM_GPUS="${2:-1}"
    echo "Running: python tool.py --config $1 --num-gpus $NUM_GPUS"
    python tool.py --config "$1" --num-gpus "$NUM_GPUS"
fi

