#!/bin/bash
# Serve Qwen3.5-122B-A10B locally with OpenAI-compatible API
# Usage: bash playground/service/qwen3_5_122b_a10b.sh
#
# After launch, set in your env or .env:
#   OPENAI_BASE_URL=http://localhost:8000/v1
#   OPENAI_API_KEY=dummy

set -euo pipefail

MODEL="Qwen/Qwen3.5-122B-A10B"
HOST="0.0.0.0"
PORT=8000
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
        --enable-expert-parallel \
        --enable-chunked-prefill \
        --language-model-only \
        --reasoning-parser qwen3 \
        --enable-prefix-caching \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
        --served-model-name "qwen3.5-122b-a10b"

else
    echo "Unknown backend: $BACKEND"
    exit 1
fi
