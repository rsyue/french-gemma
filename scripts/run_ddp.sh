#!/usr/bin/env bash

# Multi-GPU pretraining launcher script using torchrun (DDP).
# This script illustrates how to run pretraining with 2 and 4 GPUs.

set -euo pipefail

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# Define defaults
CONFIG="configs/nvidia_config.yaml"
GPUS=2

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -c, --config PATH   Path to training configuration file (default: $CONFIG)"
    echo "  -g, --gpus NUM      Number of GPUs to use (2 or 4, default: $GPUS)"
    echo "  -h, --help          Show this help message"
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            print_usage
            exit 1
            ;;
    esac
done

# Validate GPU count choice
if [ "$GPUS" -lt 1 ]; then
    echo "Error: GPU count must be at least 1."
    exit 1
fi

echo "=========================================================="
echo "Launching French Gemma 3 pretraining under torchrun..."
echo "Config file: $CONFIG"
echo "Target GPUs: $GPUS"
echo "=========================================================="

# Run DDP pretraining
torchrun --nproc_per_node="$GPUS" -m train.pretrain --config "$CONFIG"
