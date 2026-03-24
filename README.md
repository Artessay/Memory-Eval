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
| `hf` | HuggingFace Transformers (local models) |

## Installation

### Install via PyPI

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

## Quick Start

### List available tasks and models

```bash
memory-eval list
memory-eval list --models
```

### Run evaluation

```bash
# Evaluate GPT-4o on MM-Lifelong
memory-eval run --task mm_lifelong --model-backend openai --model-name gpt-4o

# Evaluate on HealthBench (hard subset)
memory-eval run --task healthbench --model-backend openai --model-name gpt-4o --subset hard

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
```

### Validate task configuration

```bash
memory-eval validate --task mm_lifelong
memory-eval validate --task healthbench
```

## Environment Variables

Set API keys for the providers you want to use:

```bash
export OPENAI_API_KEY=...       # OpenAI models
export GOOGLE_API_KEY=...       # Gemini models
export ANTHROPIC_API_KEY=...    # Claude models
export TOGETHER_API_KEY=...     # Together AI models
export XAI_API_KEY=...          # Grok models
```

## Adding a New Task

1. Create a new directory: `memory_eval/tasks/my_task/`
2. Implement a class that extends `BaseTask` and decorate with `@TaskRegistry.register("my_task")`
3. Implement `load_dataset()`, `build_messages()`, and `evaluate()` methods

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
2. Implement the `generate()` method

```python
from memory_eval.models.base import BaseModel
from memory_eval.models.registry import ModelRegistry

@ModelRegistry.register("my_backend")
class MyBackend(BaseModel):
    def generate(self, messages, max_tokens=512, temperature=0.0, **kwargs):
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
