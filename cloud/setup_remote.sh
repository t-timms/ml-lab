#!/bin/bash
# One-shot remote environment setup for cloud GPU instances.
# Run this once after provisioning a new instance.
#
# Usage: ssh user@host 'bash -s' < cloud/setup_remote.sh

set -euo pipefail

echo "=== ML Lab Remote Setup ==="

# System packages
sudo apt-get update -q
sudo apt-get install -y -q python3.11 python3.11-venv python3-pip git curl

# Install uv
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install conda (for mlenv compatibility)
if ! command -v conda &> /dev/null; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
fi

# Create mlenv if it doesn't exist
if ! conda env list | grep -q mlenv; then
    conda create -n mlenv python=3.11 -y
fi

# Install PyTorch in mlenv
conda run -n mlenv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
echo "=== GPU Check ==="
nvidia-smi
conda run -n mlenv python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo "=== Setup Complete ==="
