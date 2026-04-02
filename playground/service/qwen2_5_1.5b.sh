#!/bin/bash
# Serve Qwen3.5-27B locally with OpenAI-compatible API
# Usage: bash playground/service/qwen3_5_27b.sh [sglang|vllm]
#
# After launch, set in your env or .env:
#   OPENAI_BASE_URL=http://localhost:3333/v1
#   OPENAI_API_KEY=dummy

set -euo pipefail

MODEL="/data/Qwen/Qwen2.5-1.5B-Instruct"
HOST="0.0.0.0"
PORT=3333
TP=4                            # tensor parallelism across 4 GPUs

BACKEND="${1:-vllm}"  # default to vLLM if not specified

if [ "$BACKEND" = "vllm" ]; then
    echo "=== Starting vLLM server ==="
    # Install if needed: `uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly`
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --dtype auto \
        --language-model-only \
        --enable-prefix-caching \
        --trust-remote-code \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --served-model-name "qwen2.5-1.5b-instruct"

else
    echo "Unknown backend: $BACKEND"
    exit 1
fi
