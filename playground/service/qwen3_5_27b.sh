#!/bin/bash
# Serve Qwen3.5-27B locally with OpenAI-compatible API
# Usage: bash playground/service/qwen3_5_27b.sh [sglang|vllm]
#
# After launch, set in your env or .env:
#   OPENAI_BASE_URL=http://localhost:8000/v1
#   OPENAI_API_KEY=dummy

set -euo pipefail

MODEL="Qwen/Qwen3.5-27B"
HOST="0.0.0.0"
PORT=8000
TP=4                            # tensor parallelism across 4 GPUs

BACKEND="${1:-vllm}"  # default to vLLM if not specified

if [ "$BACKEND" = "sglang" ]; then
    echo "=== Starting SGLang server ==="
    # Install if needed: pip install "sglang[all]>=0.4"
    python3 -m sglang.launch_server \
        --model-path "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --tp "$TP" \
        --dtype auto \
        --trust-remote-code \
        --chat-template qwen3 \
        --mem-fraction-static 0.88 \
        --chunked-prefill-size 8192

elif [ "$BACKEND" = "vllm" ]; then
    echo "=== Starting vLLM server ==="
    # Install if needed: `uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly`
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --dtype auto \
        --language-model-only \
        --reasoning-parser qwen3 \
        --enable-prefix-caching \
        --trust-remote-code \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --served-model-name "qwen3.5-27b"

else
    echo "Unknown backend: $BACKEND"
    echo "Usage: $0 [sglang|vllm]"
    exit 1
fi
