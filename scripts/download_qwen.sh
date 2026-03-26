#!/bin/bash
# Download Qwen2.5-0.5B weights via ModelScope (魔搭社区) or hf-mirror.
# Usage: bash scripts/download_qwen.sh [modelscope|hfmirror]
# Default: modelscope

set -e
cd "$(dirname "$0")/.."
mkdir -p pretrained/Qwen2.5-0.5B

METHOD="${1:-modelscope}"

if [ "$METHOD" = "hfmirror" ]; then
    echo "=== Downloading via hf-mirror.com ==="
    HF_ENDPOINT=https://hf-mirror.com \
    huggingface-cli download \
        Qwen/Qwen2.5-0.5B \
        --local-dir pretrained/Qwen2.5-0.5B \
        --local-dir-use-symlinks False
else
    echo "=== Downloading via ModelScope (魔搭社区) ==="
    HF_ENDPOINT=https://hf-mirror.com \
    huggingface-cli download \
        Qwen/Qwen2.5-0.5B \
        --local-dir pretrained/Qwen2.5-0.5B \
        --local-dir-use-symlinks False
fi

echo ""
echo "Done. Weights saved to: pretrained/Qwen2.5-0.5B"
ls pretrained/Qwen2.5-0.5B/
