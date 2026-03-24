"""Command-line interface for Memory-Eval."""

import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="memory-eval")
def main():
    """Memory-Eval: Evaluation benchmark for Lifelong Understanding tasks."""


@main.command("list")
@click.option("--tasks", "show_tasks", is_flag=True, default=True, help="List available tasks.")
@click.option("--models", "show_models", is_flag=True, default=False, help="List available model backends.")
def list_cmd(show_tasks: bool, show_models: bool):
    """List available tasks and/or model backends."""
    from memory_eval.tasks.registry import TaskRegistry
    from memory_eval.models.registry import ModelRegistry
    import memory_eval.tasks.mm_lifelong  # noqa: F401 - register tasks
    import memory_eval.tasks.healthbench  # noqa: F401 - register tasks
    import memory_eval.models.openai_model  # noqa: F401 - register backends
    import memory_eval.models.hf_model  # noqa: F401 - register backends

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
@click.option("--model-backend", default="openai", show_default=True, help="Model backend type.")
@click.option("--model-name", required=True, help="Model name (e.g. gpt-4o).")
@click.option("--base-url", default=None, help="API base URL (for OpenAI-compatible APIs).")
@click.option("--api-key-env", default=None, help="Env var name containing the API key.")
@click.option("--output-dir", default="results", show_default=True, help="Directory for results.")
@click.option("--max-tokens", default=512, show_default=True, help="Max tokens per response.")
@click.option("--temperature", default=0.0, show_default=True, help="Sampling temperature.")
@click.option("--limit", default=None, type=int, help="Limit number of samples (for testing).")
@click.option("--subset", default="hard", show_default=True, help="Dataset subset (for HealthBench: hard/consensus).")
def run_cmd(
    task_names,
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
    from memory_eval.models.registry import ModelRegistry
    from memory_eval.evaluator import Evaluator
    import memory_eval.tasks.mm_lifelong  # noqa: F401
    import memory_eval.tasks.healthbench  # noqa: F401
    import memory_eval.models.openai_model  # noqa: F401
    import memory_eval.models.hf_model  # noqa: F401

    # Build model
    model_cls = ModelRegistry.get(model_backend)
    model_kwargs = {}
    if base_url:
        model_kwargs["base_url"] = base_url
    if api_key_env:
        model_kwargs["api_key_env"] = api_key_env
    model = model_cls(model_name=model_name, **model_kwargs)

    # Build tasks
    tasks = []
    for name in task_names:
        task_cls = TaskRegistry.get(name)
        task_config = {}
        if name == "healthbench":
            task_config["subset"] = subset
        tasks.append(task_cls(config=task_config))

    evaluator = Evaluator(model=model, output_dir=output_dir)

    console.print(f"[bold green]Running {len(tasks)} task(s) with model '{model_name}'[/bold green]")

    results = evaluator.run_tasks(tasks, max_tokens=max_tokens, temperature=temperature, limit=limit)

    # Print summary table
    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Task", style="cyan")
    table.add_column("Metric", style="yellow")
    table.add_column("Value", style="green")
    table.add_column("# Samples")

    for task_name, result in results.items():
        for metric_name, value in result.metrics.items():
            table.add_row(
                task_name,
                metric_name,
                f"{value:.4f}" if isinstance(value, float) else str(value),
                str(result.num_samples),
            )

    console.print(table)
    console.print(f"\nResults saved to: [bold]{output_dir}[/bold]")


@main.command("validate")
@click.option("--task", "task_name", required=True, help="Task to validate.")
def validate_cmd(task_name: str):
    """Validate a task configuration by loading a few samples."""
    from memory_eval.tasks.registry import TaskRegistry
    import memory_eval.tasks.mm_lifelong  # noqa: F401
    import memory_eval.tasks.healthbench  # noqa: F401

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
