#!/bin/bash
# RunPod Setup Script for tool_and_judge2
# Usage: ./runpod_setup.sh [config_file] [num_gpus]
# Example: ./runpod_setup.sh tool_config_slurm8.py 1

set -e  # Exit on error

echo "=============================================="
echo "  RunPod Setup Script for tool_and_judge2"
echo "=============================================="

# Navigate to workspace
cd /workspace/tool_and_judge2

# ============================================
# Step 1: Install Rust if not present
# ============================================
if ! command -v rustc &> /dev/null; then
    echo "[1/5] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "[1/5] Rust already installed"
    source $HOME/.cargo/env 2>/dev/null || true
fi

# ============================================
# Step 2: Set up Python virtual environment
# ============================================
echo "[2/5] Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install dependencies if maturin not found
if ! command -v maturin &> /dev/null; then
    echo "Installing Python dependencies..."
    pip install --upgrade pip
    pip install maturin vllm transformers accelerate sentencepiece protobuf openai python-dotenv
fi

# ============================================
# Step 3: Set environment variables
# ============================================
echo "[3/5] Setting environment variables..."

# HuggingFace token
export HF_HOME="/workspace/huggingface_cache"
mkdir -p $HF_HOME

# OpenAI API key (for Pass 6 categorization)

# Load from .env if exists
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

echo "  HF_TOKEN: ${HF_TOKEN:0:10}..."
echo "  HF_HOME: $HF_HOME"
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."

# ============================================
# Step 4: Build Rust extension
# ============================================
echo "[4/5] Building Rust extension..."
if ! python -c "import codebase_rs" 2>/dev/null; then
    maturin develop --release
else
    echo "  Rust extension already built"
fi

# Verify
python -c "from codebase_rs import *; print('  ✓ Rust extension loaded successfully')"

# ============================================
# Step 5: Run config if specified
# ============================================
CONFIG=${1:-""}
NUM_GPUS=${2:-1}

echo "[5/5] Setup complete!"
echo ""

if [ -n "$CONFIG" ]; then
    echo "=============================================="
    echo "  Running: $CONFIG with $NUM_GPUS GPU(s)"
    echo "=============================================="
    python tool.py --config $CONFIG --num-gpus $NUM_GPUS
else
    echo "Available configs:"
    echo ""
    echo "  # Single GPU (A100)"
    echo "  ./runpod_setup.sh tool_config_llama8b.py 1      # Llama 3.1 8B"
    echo "  ./runpod_setup.sh tool_config_slurm1.py 1       # Qwen3 8B"
    echo "  ./runpod_setup.sh tool_config_slurm4.py 1       # Granite small"
    echo "  ./runpod_setup.sh tool_config_slurm8.py 1       # Granite tiny"
    echo ""
    echo "  # 4 GPUs (4x A100 80GB)"
    echo "  ./runpod_setup.sh tool_config_slurm3.py 4       # Qwen3 30B-A3B"
    echo "  ./runpod_setup.sh tool_config_slurm5.py 4       # Qwen3 32B"
    echo ""
    echo "  # 8 GPUs (8x H200 or 8x A100)"
    echo "  ./runpod_setup.sh tool_config_llama70b.py 8     # Llama 3.1 70B"
    echo "  ./runpod_setup.sh tool_config_slurm6.py 8       # Qwen3 Next-80B-A3B"
    echo ""
    echo "  # API models (no GPU needed)"
    echo "  ./runpod_setup.sh tool_config1.py 1             # GPT-5"
    echo ""
    echo "Run with: ./runpod_setup.sh <config_file> <num_gpus>"
fi

