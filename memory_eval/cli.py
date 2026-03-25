"""Command-line interface for Memory-Eval."""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
import yaml
from rich.console import Console
from rich.table import Table

from memory_eval.models import register_builtin_models
from memory_eval.tasks import register_builtin_tasks
from memory_eval.evaluator import build_evaluated_result_path, build_result_path
from memory_eval.utils.env import load_project_env

console = Console()
DEFAULT_MODEL_CONFIG_PATH = Path("configs/models/default.yaml")


@lru_cache(maxsize=8)
def _load_model_configs(config_path: str) -> Dict[str, Dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        raise click.BadParameter(f"Model config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise click.BadParameter(f"Model config file must contain a mapping: {path}")

    normalized: Dict[str, Dict[str, Any]] = {}
    for alias, config in data.items():
        if not isinstance(config, dict):
            raise click.BadParameter(f"Model config '{alias}' must be a mapping.")
        normalized[str(alias)] = dict(config)
    return normalized


def _resolve_model_spec(
    *,
    model_config: Optional[str],
    model_backend: Optional[str],
    model_name: Optional[str],
    base_url: Optional[str],
    api_key_env: Optional[str],
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH,
) -> Tuple[str, str, Dict[str, Any]]:
    config_values: Dict[str, Any] = {}
    if model_config:
        configs = _load_model_configs(str(config_path.resolve()))
        if model_config not in configs:
            available = ", ".join(sorted(configs))
            raise click.BadParameter(
                f"Unknown model config '{model_config}'. Available configs: {available}"
            )
        config_values = dict(configs[model_config])

    resolved_backend = model_backend or config_values.pop("backend", None)
    resolved_model_name = model_name or config_values.pop("model", None)

    if not resolved_backend or not resolved_model_name:
        raise click.BadParameter(
            "Provide either --model-config or both --model-backend and --model-name."
        )

    model_kwargs = {
        key: value
        for key, value in config_values.items()
        if key not in {"input_cost_per_m", "output_cost_per_m"}
    }

    if base_url:
        if resolved_backend == "azure":
            model_kwargs["azure_endpoint"] = base_url
        else:
            model_kwargs["base_url"] = base_url
    if api_key_env:
        model_kwargs["api_key_env"] = api_key_env

    return resolved_backend, resolved_model_name, model_kwargs


def _build_model(
    model_backend: str,
    model_name: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
):
    from memory_eval.models.registry import ModelRegistry

    model_cls = ModelRegistry.get(model_backend)
    resolved_kwargs: Dict[str, Any] = dict(model_kwargs or {})
    if base_url:
        if model_backend == "azure":
            resolved_kwargs["azure_endpoint"] = base_url
        else:
            resolved_kwargs["base_url"] = base_url
    if api_key_env:
        resolved_kwargs["api_key_env"] = api_key_env
    return model_cls(model_name=model_name, **resolved_kwargs)


def _build_task_config(task_name: str, subset: Optional[str] = None) -> Dict[str, Any]:
    task_config: Dict[str, Any] = {}
    if task_name == "healthbench":
        task_config["subset"] = subset or "standard"
    return task_config


def _build_evaluation_name(task_name: str, grader_backend: Optional[str], grader_model: Optional[str]) -> str:
    if grader_backend and grader_model:
        return f"grader__{task_name}__{grader_backend}__{grader_model}"
    return f"task__{task_name}"


def _resolve_result_file(
    *,
    result_file: Optional[Path],
    task_name: Optional[str],
    subset: Optional[str],
    model_config: Optional[str],
    model_backend: Optional[str],
    model_name: Optional[str],
    base_url: Optional[str],
    api_key_env: Optional[str],
    output_dir: str,
    max_tokens: int,
    temperature: float,
    limit: Optional[int],
) -> Tuple[Path, Optional[str], Optional[str]]:
    if result_file is not None:
        return result_file, model_backend, model_name

    if not task_name:
        raise click.BadParameter("Provide --result-file, or pass --task together with the run-time parameters used to create the result.")

    resolved_backend, resolved_model_name, _ = _resolve_model_spec(
        model_config=model_config,
        model_backend=model_backend,
        model_name=model_name,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    resolved_task_config = _build_task_config(task_name, subset=subset)
    candidate_path = Path(
        build_result_path(
            output_dir=output_dir,
            task_name=task_name,
            model_backend=resolved_backend,
            model_name=resolved_model_name,
            task_config=resolved_task_config,
            generation_config={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "limit": limit,
            },
        )
    )
    if not candidate_path.exists():
        raise click.BadParameter(
            f"Could not find result file at {candidate_path}. Check --task/--model-config/--model-backend/--model-name/--subset/--max-tokens/--temperature/--limit/--output-dir."
        )
    return candidate_path, resolved_backend, resolved_model_name


def _resolve_evaluate_output_file(
    *,
    output_file: Optional[Path],
    result_file: Path,
    grader_backend: Optional[str],
    grader_model: Optional[str],
    evaluated_output_dir: str,
) -> Path:
    if output_file is not None:
        return output_file
    return Path(
        build_evaluated_result_path(
            result_path=str(result_file),
            grader_backend=grader_backend,
            grader_model=grader_model,
            output_dir=evaluated_output_dir,
        )
    )


def _render_metrics_table(title: str, task_name: str, metrics: Dict[str, Any], num_samples: int) -> None:
    table = Table(title=title, show_header=True)
    table.add_column("Task", style="cyan")
    table.add_column("Metric", style="yellow")
    table.add_column("Value", style="green")
    table.add_column("# Samples")

    if metrics:
        for metric_name, value in metrics.items():
            table.add_row(
                task_name,
                metric_name,
                f"{value:.4f}" if isinstance(value, float) else str(value),
                str(num_samples),
            )
    else:
        table.add_row(task_name, "status", "predictions_saved", str(num_samples))

    console.print(table)


@click.group()
@click.version_option(package_name="memory-eval")
def main():
    """Memory-Eval: Evaluation benchmark for Lifelong Understanding tasks."""
    load_project_env()


@main.command("list")
@click.option("--tasks", "show_tasks", is_flag=True, default=False, help="List available tasks.")
@click.option("--models", "show_models", is_flag=True, default=False, help="List available model backends.")
def list_cmd(show_tasks: bool, show_models: bool):
    """List available tasks and/or model backends."""
    from memory_eval.tasks.registry import TaskRegistry
    from memory_eval.models.registry import ModelRegistry

    register_builtin_tasks()
    register_builtin_models()

    if show_tasks or not show_models:
        table = Table(title="Available Tasks", show_header=True)
        table.add_column("Task Name", style="cyan")
        table.add_column("Description")
        for name in TaskRegistry.list_tasks():
            task_cls = TaskRegistry.get(name)
            table.add_row(name, task_cls.description or "")
        console.print(table)

    if show_models:
        table = Table(title="Available Model Backends", show_header=True)
        table.add_column("Backend Name", style="green")
        for name in ModelRegistry.list_backends():
            table.add_row(name)
        console.print(table)


@main.command("run")
@click.option("--task", "task_names", multiple=True, required=True, help="Task(s) to evaluate.")
@click.option("--model-config", default=None, help="Model alias from configs/models/default.yaml.")
@click.option("--model-backend", default=None, help="Model backend type.")
@click.option("--model-name", default=None, help="Model name (e.g. gpt-4o).")
@click.option("--base-url", default=None, help="API base URL (for OpenAI-compatible APIs).")
@click.option("--api-key-env", default=None, help="Env var name containing the API key.")
@click.option("--output-dir", default="results", show_default=True, help="Directory for results.")
@click.option("--max-tokens", default=4096, show_default=True, help="Max tokens per response.")
@click.option("--temperature", default=0.0, show_default=True, help="Sampling temperature.")
@click.option("--limit", default=None, type=int, help="Limit number of samples (for testing).")
@click.option("--subset", default="standard", show_default=True, help="Dataset subset.")
def run_cmd(
    task_names,
    model_config,
    model_backend,
    model_name,
    base_url,
    api_key_env,
    output_dir,
    max_tokens,
    temperature,
    limit,
    subset,
):
    """Run evaluation on one or more tasks."""
    from memory_eval.tasks.registry import TaskRegistry
    from memory_eval.evaluator import Evaluator

    register_builtin_tasks()
    register_builtin_models()

    model_backend, model_name, model_kwargs = _resolve_model_spec(
        model_config=model_config,
        model_backend=model_backend,
        model_name=model_name,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    model = _build_model(
        model_backend=model_backend,
        model_name=model_name,
        model_kwargs=model_kwargs,
    )

    # Build tasks
    tasks = []
    for name in task_names:
        task_cls = TaskRegistry.get(name)
        task_config = _build_task_config(name, subset=subset)
        tasks.append(task_cls(config=task_config))

    evaluator = Evaluator(model=model, output_dir=output_dir, model_backend=model_backend)

    console.print(f"[bold green]Running {len(tasks)} task(s) with model '{model_name}'[/bold green]")

    results = evaluator.run_tasks(tasks, max_tokens=max_tokens, temperature=temperature, limit=limit)

    for task_name, result in results.items():
        _render_metrics_table("Run Results", task_name, result.metrics, result.num_samples)
        console.print(f"Saved predictions to: [bold]{result.metadata['result_path']}[/bold]")


@main.command("evaluate")
@click.option("--result-file", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Saved result JSON to evaluate.")
@click.option("--task", "task_name", default=None, help="Override the task name stored in the result file.")
@click.option("--subset", default=None, help="Override the dataset subset for tasks that use subsets.")
@click.option("--model-config", default=None, help="Generation model alias from configs/models/default.yaml.")
@click.option("--model-backend", default=None, help="Generation model backend type.")
@click.option("--model-name", default=None, help="Generation model name.")
@click.option("--base-url", default=None, help="Generation model API base URL.")
@click.option("--api-key-env", default=None, help="Generation model API key env var.")
@click.option("--output-dir", default="results", show_default=True, help="Directory containing generation results.")
@click.option("--evaluated-output-dir", default="results/evaluated", show_default=True, help="Directory for evaluated result files.")
@click.option("--max-tokens", default=4096, show_default=True, help="Generation max tokens used when the result file was created.")
@click.option("--temperature", default=0.0, show_default=True, help="Generation temperature used when the result file was created.")
@click.option("--limit", default=None, type=int, help="Generation sample limit used when the result file was created.")
@click.option("--grader-config", default=None, help="Grader model alias from configs/models/default.yaml.")
@click.option("--grader-backend", default=None, help="Model backend type for rubric grading.")
@click.option("--grader-model", default=None, help="Model name for rubric grading.")
@click.option("--grader-base-url", default=None, help="API base URL for the grader model.")
@click.option("--grader-api-key-env", default=None, help="Env var containing the grader API key.")
@click.option("--output-file", default=None, type=click.Path(dir_okay=False, path_type=Path), help="Optional path for the evaluated result JSON.")
def evaluate_cmd(
    result_file: Optional[Path],
    task_name: Optional[str],
    subset: Optional[str],
    model_config: Optional[str],
    model_backend: Optional[str],
    model_name: Optional[str],
    base_url: Optional[str],
    api_key_env: Optional[str],
    output_dir: str,
    evaluated_output_dir: str,
    max_tokens: int,
    temperature: float,
    limit: Optional[int],
    grader_config: Optional[str],
    grader_backend: Optional[str],
    grader_model: Optional[str],
    grader_base_url: Optional[str],
    grader_api_key_env: Optional[str],
    output_file: Optional[Path],
):
    """Evaluate a previously saved result file."""
    from memory_eval.evaluator import Evaluator
    from memory_eval.tasks.registry import TaskRegistry
    from memory_eval.utils.io import load_json

    register_builtin_tasks()
    register_builtin_models()

    resolved_result_file, resolved_model_backend, resolved_model_name = _resolve_result_file(
        result_file=result_file,
        task_name=task_name,
        subset=subset,
        model_config=model_config,
        model_backend=model_backend,
        model_name=model_name,
        base_url=base_url,
        api_key_env=api_key_env,
        output_dir=output_dir,
        max_tokens=max_tokens,
        temperature=temperature,
        limit=limit,
    )

    result_data = load_json(str(resolved_result_file))
    resolved_task_name = task_name or result_data.get("task_name")
    if not resolved_task_name:
        raise click.BadParameter("Task name was not found in the result file; pass --task.")

    task_cls = TaskRegistry.get(resolved_task_name)
    task_config = dict(result_data.get("task_config") or {})
    task_config.update(_build_task_config(resolved_task_name, subset=subset))
    task = task_cls(config=task_config)

    grader_info: Dict[str, Any] = {}
    if grader_config or grader_backend or grader_model or grader_base_url or grader_api_key_env:
        grader_backend, grader_model, grader_kwargs = _resolve_model_spec(
            model_config=grader_config,
            model_backend=grader_backend,
            model_name=grader_model,
            base_url=grader_base_url,
            api_key_env=grader_api_key_env,
        )
        grader = _build_model(
            model_backend=grader_backend,
            model_name=grader_model,
            model_kwargs=grader_kwargs,
        )
        grader_info = {
            "config": grader_config,
            "backend": grader_backend,
            "model_name": grader_model,
        }
        if hasattr(task, "set_grader_model"):
            task.set_grader_model(grader)

    resolved_output_file = _resolve_evaluate_output_file(
        output_file=output_file,
        result_file=resolved_result_file,
        grader_backend=grader_backend,
        grader_model=grader_model,
        evaluated_output_dir=evaluated_output_dir,
    )

    evaluator = Evaluator(output_dir=str(resolved_result_file.parent))
    evaluation_name = _build_evaluation_name(resolved_task_name, grader_backend, grader_model)
    updated = evaluator.evaluate_result(
        result_path=str(resolved_result_file),
        task=task,
        evaluation_name=evaluation_name,
        grader_info=grader_info,
        output_path=str(resolved_output_file),
    )

    _render_metrics_table(
        "Evaluation Results",
        resolved_task_name,
        updated.get("metrics", {}),
        updated.get("num_samples", len(updated.get("predictions", []))),
    )
    console.print(
        f"Loaded generation result from: [bold]{resolved_result_file}[/bold]"
    )
    console.print(
        f"Saved evaluated results to: [bold]{resolved_output_file}[/bold]"
    )


@main.command("validate")
@click.option("--task", "task_name", required=True, help="Task to validate.")
def validate_cmd(task_name: str):
    """Validate a task configuration by loading a few samples."""
    from memory_eval.tasks.registry import TaskRegistry

    register_builtin_tasks()

    task_cls = TaskRegistry.get(task_name)
    task = task_cls()

    console.print(f"Validating task: [cyan]{task_name}[/cyan]")
    console.print(f"Description: {task_cls.description}")
    console.print(f"Dataset: {task_cls.dataset_name}")

    try:
        samples = task.load_dataset()
        sample = samples[0] if samples else {}
        messages = task.build_messages(sample)

        console.print(f"[green]✓[/green] Loaded {len(samples)} samples")
        console.print(f"[green]✓[/green] Built messages: {len(messages)} turns")
        console.print(f"\nFirst sample keys: {list(sample.keys())}")
    except Exception as exc:
        console.print(f"[red]✗ Validation failed:[/red] {exc}")
        sys.exit(1)
