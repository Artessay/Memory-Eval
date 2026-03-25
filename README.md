# Memory-Eval

An evaluation benchmark for **Lifelong Understanding** tasks.

## Overview

Memory-Eval provides a unified framework for evaluating large language models (LLMs) on tasks that require sustained knowledge, temporal reasoning, and long-horizon understanding.

### Supported Tasks

| Task | Description | Dataset |
|------|-------------|---------|
| `mm_lifelong` | Multimodal lifelong understanding (temporal/episodic reasoning) | [CG-Bench/MM-Lifelong](https://huggingface.co/datasets/CG-Bench/MM-Lifelong) |
| `healthbench` | Medical conversation evaluation with physician rubrics | [openai/healthbench](https://huggingface.co/datasets/openai/healthbench) |

### Supported Model Backends

| Backend | Description |
|---------|-------------|
| `openai` | OpenAI and any OpenAI-compatible API (Google, Anthropic, Together AI, xAI, etc.) |
| `azure` | Azure OpenAI via API key or Microsoft Entra ID (`DefaultAzureCredential`) |
| `hf` | HuggingFace Transformers (local models) |

## Installation

Create and activate a conda environment:

```bash
conda create -n memory-eval python=3.13 -y
conda activate memory-eval
```

Install packages using `uv`:

```bash
export UV_PROJECT_ENVIRONMENT="$(python -c 'import sys; print(sys.prefix)')"
uv sync
```

Install all optional dependencies:

```bash
uv sync --extra all
```

Install Azure support only:

```bash
uv sync --extra azure
```

## Quick Start

### List available tasks and models

```bash
memory-eval list
memory-eval list --models
```

### Run generation

```bash
# Generate predictions for MM-Lifelong
memory-eval run --task mm_lifelong --model-backend openai --model-name gpt-4o

# Generate predictions using a named model config
memory-eval run --task healthbench --model-config gpt-4o --subset hard

# Generate predictions for HealthBench (hard subset)
memory-eval run --task healthbench --model-backend openai --model-name gpt-4o --subset hard

# Evaluate on HealthBench standard subset
memory-eval run --task healthbench --model-backend openai --model-name gpt-4o --subset standard

# Evaluate multiple tasks at once
memory-eval run --task mm_lifelong --task healthbench --model-backend openai --model-name gpt-4o

# Limit to 10 samples for quick testing
memory-eval run --task mm_lifelong --model-backend openai --model-name gpt-4o --limit 10

# Use a custom API base URL (e.g., Gemini via OpenAI-compatible API)
memory-eval run \
  --task healthbench \
  --model-backend openai \
  --model-name gemini-2.0-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --api-key-env GOOGLE_API_KEY \
  --subset hard

# Use Azure OpenAI with an API key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=...
memory-eval run \
    --task healthbench \
    --model-backend azure \
    --model-name gpt-4o \
    --subset hard

# Use Azure OpenAI with Microsoft Entra ID / DefaultAzureCredential
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-08-01-preview
memory-eval run \
    --task healthbench \
    --model-backend azure \
    --model-name gpt-4o \
    --subset hard
```

`run` now saves raw predictions plus the serialized sample payload needed for deferred evaluation. The output path is deterministic and based on the task, model backend, model name, and key generation parameters, for example:

```text
results/healthbench/openai/gpt-4o/subset-hard__max_tokens-512__temperature-0.0__limit-all.json
```

Re-running the same experiment updates the same result file.

### Evaluate saved results

```bash
# Evaluate a saved MM-Lifelong result with task-native metrics
memory-eval evaluate \
    --result-file results/mm_lifelong/openai/gpt-4o/max_tokens-4096__temperature-0.0__limit-all.json

# Or let evaluate locate the generation result from the same experiment parameters
memory-eval evaluate \
    --task healthbench \
    --model-config azure-gpt-4o \
    --subset standard \
    --limit 10 \
    --grader-config azure-gpt-4o

# Evaluate a saved HealthBench result with a separate grader model
memory-eval evaluate \
    --result-file results/healthbench/openai/gpt-4o/subset-hard__max_tokens-512__temperature-0.0__limit-all.json \
    --grader-config gpt-4o

# Equivalent explicit grader configuration
memory-eval evaluate \
    --result-file results/healthbench/openai/gpt-4o/subset-hard__max_tokens-512__temperature-0.0__limit-all.json \
    --grader-backend openai \
    --grader-model gpt-4o

# Evaluate a saved HealthBench result with Azure OpenAI as the grader
memory-eval evaluate \
    --result-file results/healthbench/openai/gpt-4o/subset-hard__max_tokens-512__temperature-0.0__limit-all.json \
    --grader-backend azure \
    --grader-model gpt-4o
```

For tasks such as `healthbench`, deferred evaluation lets you use a dedicated grader model after generation finishes, instead of coupling grading to the original `run` command.

Both `--model-config` and `--grader-config` reuse entries from [configs/models/default.yaml](configs/models/default.yaml), so provider-specific fields such as `base_url`, `api_key_env`, `endpoint_env`, and `api_version` only need to be defined once.

When `--output-file` is omitted, `evaluate` now writes to a separate file under `results/evaluated/...` and appends a grader marker such as `__graded-by-azure-gpt-4o.json`, leaving the original generation result untouched.

### Validate task configuration

```bash
memory-eval validate --task mm_lifelong
memory-eval validate --task healthbench
```

## Retry Behavior

The `openai` and `azure` backends automatically retry transient API failures with exponential backoff.
By default they retry up to 8 times for:

- HTTP `429` rate-limit responses
- HTTP `5xx` server-side failures
- OpenAI SDK connection and timeout errors

Non-retryable client errors such as HTTP `400` are raised immediately.

You can tune the retry behavior with environment variables:

```bash
export MEMORY_EVAL_MAX_RETRIES=8
export MEMORY_EVAL_RETRY_BASE_DELAY=1.0
export MEMORY_EVAL_RETRY_MAX_DELAY=30.0
export MEMORY_EVAL_RETRY_JITTER=0.25
```

These settings also work when instantiating a model backend directly by passing
`max_retries`, `retry_base_delay`, `retry_max_delay`, or `retry_jitter`.

## Environment Variables

Set API keys for the providers you want to use:

```bash
export OPENAI_API_KEY=...       # OpenAI models
export AZURE_OPENAI_ENDPOINT=... # Azure OpenAI endpoint
export AZURE_OPENAI_API_KEY=...  # Azure OpenAI API key (optional if using Entra ID)
export AZURE_OPENAI_API_VERSION=... # Required for Azure OpenAI; use a version supported by your resource
export GOOGLE_API_KEY=...       # Gemini models
export ANTHROPIC_API_KEY=...    # Claude models
export TOGETHER_API_KEY=...     # Together AI models
export XAI_API_KEY=...          # Grok models
```

For local development, you can also create a `.env` file in the project root. Memory-Eval loads `.env`
automatically, but does not override variables already provided by the shell, CI, or deployment platform.

```bash
cp .env.example .env
```

Example `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
GOOGLE_API_KEY=your-google-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
TOGETHER_API_KEY=your-together-api-key
XAI_API_KEY=your-xai-api-key
```

Recommended usage:

- Local development: use `.env`
- CI/CD and production: inject secrets as environment variables from your platform secret manager
- When both exist, injected environment variables win over `.env`

## Adding a New Task

1. Create a new directory: `memory_eval/tasks/my_task/`
2. Implement a class that extends `BaseTask` and decorate with `@TaskRegistry.register("my_task")`
3. Add the packaged task to `register_builtin_tasks()` in `memory_eval/tasks/__init__.py`
4. Implement `load_dataset()`, `build_messages()`, and `evaluate()` methods

```python
from memory_eval.tasks.base import BaseTask
from memory_eval.tasks.registry import TaskRegistry

@TaskRegistry.register("my_task")
class MyTask(BaseTask):
    description = "My custom task description"
    dataset_name = "my-hf-dataset"

    def load_dataset(self):
        from datasets import load_dataset
        ds = load_dataset(self.dataset_name, split="test")
        return [dict(item) for item in ds]

    def build_messages(self, sample):
        return [{"role": "user", "content": sample["question"]}]

    def evaluate(self, samples, predictions):
        from memory_eval.metrics.accuracy import compute_exact_match
        score = sum(compute_exact_match(p, s["answer"]) for p, s in zip(predictions, samples))
        return {"accuracy": score / len(samples)}
```

## Adding a New Model Backend

1. Implement a class that extends `BaseModel` and decorate with `@ModelRegistry.register("my_backend")`
2. Add the packaged backend to `register_builtin_models()` in `memory_eval/models/__init__.py`
3. Implement the `generate()` method

```python
from memory_eval.models.base import BaseModel
from memory_eval.models.registry import ModelRegistry

@ModelRegistry.register("my_backend")
class MyBackend(BaseModel):
    def generate(self, messages, max_tokens=4096, temperature=0.0, **kwargs):
        # Your implementation here
        return "response"
```

## Project Structure

```
memory_eval/
├── __init__.py
├── __main__.py
├── cli.py              # CLI interface
├── evaluator.py        # Main evaluation orchestrator
├── models/
│   ├── base.py         # Abstract model base class
│   ├── registry.py     # Model backend registry
│   ├── azure_model.py  # Azure OpenAI backend
│   ├── openai_model.py # OpenAI / OpenAI-compatible API backend
│   └── hf_model.py     # HuggingFace Transformers backend
├── tasks/
│   ├── base.py         # Abstract task base class
│   ├── registry.py     # Task registry
│   ├── mm_lifelong/    # MM-Lifelong task
│   └── healthbench/    # HealthBench task
├── metrics/
│   ├── accuracy.py     # Exact match and multiple-choice accuracy
│   └── rubric.py       # Rubric-based scoring
└── utils/
    └── io.py           # JSON/JSONL I/O utilities
configs/
├── models/
│   └── default.yaml    # Model configurations
└── tasks/
    ├── mm_lifelong.yaml
    └── healthbench.yaml
tests/
├── test_metrics.py
├── test_tasks.py
└── test_models.py
```

## Design Principles

- **Reproducible**: Deterministic evaluation with fixed seeds and temperature=0
- **Extensible**: Plugin-style task and model registries for easy extension
- **Unified**: Single pipeline for text-only and multimodal tasks
- **Compatible**: OpenAI-compatible API support for any LLM provider

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
