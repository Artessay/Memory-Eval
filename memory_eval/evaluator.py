"""Main evaluator orchestrating model evaluation across tasks."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_eval.models.base import BaseModel
from memory_eval.tasks.base import BaseTask, TaskResult
from memory_eval.utils.io import ensure_dir, load_json, save_json


def _sanitize_path_component(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    sanitized = text.replace(os.sep, "-")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", sanitized)
    sanitized = re.sub(r"-+", "-", sanitized).strip("-.")
    return sanitized or "unknown"


def build_result_path(
    *,
    output_dir: str,
    task_name: str,
    model_backend: Optional[str],
    model_name: Optional[str],
    task_config: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> str:
    task_config = task_config or {}
    generation_config = generation_config or {}

    path_parts = [
        output_dir,
        _sanitize_path_component(task_name),
        _sanitize_path_component(model_backend or "unknown-backend"),
        _sanitize_path_component(model_name or "unknown-model"),
    ]

    filename_parts = []
    for key in sorted(task_config):
        value = task_config[key]
        if value in (None, "", [], {}, ()) or str(key).endswith("_dir"):
            continue
        filename_parts.append(f"{_sanitize_path_component(key)}-{_sanitize_path_component(value)}")

    filename_parts.append(
        f"max_tokens-{_sanitize_path_component(generation_config.get('max_tokens', 4096))}"
    )
    filename_parts.append(
        f"temperature-{_sanitize_path_component(generation_config.get('temperature', 0.0))}"
    )
    limit_value = generation_config.get("limit")
    filename_parts.append(
        f"limit-{_sanitize_path_component('all' if limit_value is None else limit_value)}"
    )

    filename = "__".join(filename_parts) + ".json"
    return os.path.join(*path_parts, filename)


def build_evaluated_result_path(
    *,
    result_path: str,
    grader_backend: Optional[str],
    grader_model: Optional[str],
    output_dir: str = "results/evaluated",
) -> str:
    normalized_result_path = os.path.normpath(result_path)
    result_basename = os.path.splitext(os.path.basename(normalized_result_path))[0]

    marker_parts = ["graded"]
    if grader_backend:
        marker_parts.append(_sanitize_path_component(grader_backend))
    if grader_model:
        marker_parts.append(_sanitize_path_component(grader_model))

    grade_suffix = marker_parts[0]
    if len(marker_parts) > 1:
        grade_suffix = f"{marker_parts[0]}-by-{'-'.join(marker_parts[1:])}"

    graded_filename = f"{result_basename}__{grade_suffix}.json"

    path_parts = Path(normalized_result_path).parts
    if "results" in path_parts:
        results_index = path_parts.index("results")
        relative_parts = path_parts[results_index + 1 : -1]
    else:
        relative_parts = path_parts[:-1]

    return os.path.join(output_dir, *relative_parts, graded_filename)


class Evaluator:
    """Orchestrates evaluation of one or more tasks with a given model."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        output_dir: str = "results",
        model_backend: Optional[str] = None,
    ):
        self.model = model
        self.output_dir = output_dir
        self.model_backend = model_backend

    def run_task(
        self,
        task: BaseTask,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        limit: Optional[int] = None,
    ) -> TaskResult:
        """Run a single task and save results."""
        if self.model is None:
            raise ValueError("A generation model is required to run tasks.")

        result = task.run(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            limit=limit,
            compute_metrics=False,
        )
        result.metadata.update(
            {
                "task_config": dict(task.config),
                "generation_config": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "limit": limit,
                },
                "model": {
                    "backend": self.model_backend,
                    "model_name": self.model.model_name,
                },
            }
        )
        result.metadata["result_path"] = self._save_result(result)
        return result

    def run_tasks(
        self,
        tasks: List[BaseTask],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        limit: Optional[int] = None,
    ) -> Dict[str, TaskResult]:
        """Run multiple tasks and return a dict of results keyed by task name."""
        results = {}
        for task in tasks:
            result = self.run_task(task, max_tokens=max_tokens, temperature=temperature, limit=limit)
            results[task.task_name] = result
        return results

    def _build_result_path(self, result: TaskResult) -> str:
        model_info = result.metadata.get("model", {})
        return build_result_path(
            output_dir=self.output_dir,
            task_name=result.task_name,
            model_backend=model_info.get("backend"),
            model_name=model_info.get("model_name"),
            task_config=result.metadata.get("task_config", {}),
            generation_config=result.metadata.get("generation_config", {}),
        )

    def _save_result(self, result: TaskResult) -> str:
        """Save a TaskResult to JSON and return the output path."""
        path = self._build_result_path(result)
        ensure_dir(os.path.dirname(path))
        extra_metadata = {
            key: value
            for key, value in result.metadata.items()
            if key not in {"task_config", "generation_config", "model", "result_path"}
        }
        save_json(
            {
                "schema_version": 2,
                "task_name": result.task_name,
                "task_config": result.metadata.get("task_config", {}),
                "model": result.metadata.get("model", {}),
                "generation_config": result.metadata.get("generation_config", {}),
                "metrics": result.metrics,
                "num_samples": result.num_samples,
                "predictions": result.predictions,
                "references": result.references,
                "evaluation_samples": result.evaluation_samples,
                "evaluations": {},
                "metadata": extra_metadata,
            },
            path,
        )
        return path

    def evaluate_result(
        self,
        result_path: str,
        task: BaseTask,
        evaluation_name: str = "default",
        grader_info: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a saved result file and persist the metrics."""
        data = load_json(result_path)
        predictions = data.get("predictions") or []
        samples = data.get("evaluation_samples") or []

        if not samples:
            samples = task.load_dataset()
            if len(samples) < len(predictions):
                raise ValueError(
                    f"Result file has {len(predictions)} predictions but only {len(samples)} samples were loaded."
                )
            samples = samples[: len(predictions)]

        metrics = task.evaluate(samples, predictions)
        evaluations = data.get("evaluations") or {}
        evaluations[evaluation_name] = {
            "metrics": metrics,
            "grader": grader_info or {},
        }
        data["metrics"] = metrics
        data["evaluations"] = evaluations

        destination = output_path or result_path
        save_json(data, destination)
        return data

    def summary(self, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Return a summary dict of all task metrics."""
        return {
            task_name: {
                "metrics": result.metrics,
                "num_samples": result.num_samples,
            }
            for task_name, result in results.items()
        }
