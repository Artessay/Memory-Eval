"""Main evaluator orchestrating model evaluation across tasks."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from memory_eval.models.base import BaseModel
from memory_eval.tasks.base import BaseTask, TaskResult
from memory_eval.utils.io import append_jsonl, ensure_dir, load_json, load_jsonl, save_json, save_jsonl


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


def build_records_path(result_path: str) -> str:
    return str(Path(result_path).with_suffix(".jsonl"))


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


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

        task_config = dict(task.config)
        generation_config = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "limit": limit,
        }
        model_info = {
            "backend": self.model_backend,
            "model_name": self.model.model_name,
        }
        result_path = build_result_path(
            output_dir=self.output_dir,
            task_name=task.task_name,
            model_backend=model_info.get("backend"),
            model_name=model_info.get("model_name"),
            task_config=task_config,
            generation_config=generation_config,
        )
        records_path = build_records_path(result_path)
        existing_records = self._load_sample_records_from_paths(
            result_path=result_path,
            records_path=records_path,
        )

        result = task.run(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            limit=limit,
            compute_metrics=False,
            existing_records=existing_records,
            record_callback=lambda record: append_jsonl(record, records_path),
        )
        result.metadata.update(
            {
                "task_config": task_config,
                "generation_config": generation_config,
                "model": model_info,
                "sample_results_path": records_path,
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

    def _legacy_records_from_summary(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        predictions = data.get("predictions") or []
        references = data.get("references") or []
        samples = data.get("evaluation_samples") or []
        if not predictions:
            return []

        records = []
        for index, prediction in enumerate(predictions):
            record = {
                "prediction": prediction,
                "reference": references[index] if index < len(references) else "",
                "evaluation_sample": samples[index] if index < len(samples) else {},
            }
            records.append(record)
        return records

    def _load_sample_records_from_paths(self, result_path: str, records_path: Optional[str] = None) -> List[Dict[str, Any]]:
        records_path = records_path or build_records_path(result_path)
        if os.path.exists(records_path):
            return load_jsonl(records_path)

        if not os.path.exists(result_path):
            return []

        data = load_json(result_path)
        records = self._legacy_records_from_summary(data)
        if records:
            save_jsonl(records, records_path)
        return records

    def _build_summary_payload(
        self,
        *,
        task_name: str,
        task_config: Dict[str, Any],
        model: Dict[str, Any],
        generation_config: Dict[str, Any],
        metrics: Dict[str, Any],
        num_samples: int,
        evaluations: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "task_name": task_name,
            "task_config": task_config,
            "model": model,
            "generation_config": generation_config,
            "metrics": metrics,
            "num_samples": num_samples,
            "evaluations": evaluations or {},
            "metadata": metadata or {},
        }

    def _merge_evaluated_records(
        self,
        source_records: List[Dict[str, Any]],
        persisted_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged_records: List[Dict[str, Any]] = []
        for index, source_record in enumerate(source_records[: len(persisted_records)]):
            persisted_record = persisted_records[index] or {}
            evaluation = persisted_record.get("evaluation")
            if evaluation is None:
                break

            merged_record = dict(source_record)
            merged_record["evaluation"] = evaluation
            for key, value in persisted_record.items():
                if key not in {"prediction", "reference", "evaluation_sample", "evaluation"}:
                    merged_record[key] = value
            merged_records.append(merged_record)
        return merged_records

    def _save_evaluated_records(
        self,
        *,
        destination_records_path: str,
        source_records: List[Dict[str, Any]],
        evaluated_records: List[Dict[str, Any]],
        in_place: bool,
    ) -> None:
        if in_place:
            persisted_records: List[Dict[str, Any]] = []
            completed = len(evaluated_records)
            for index, source_record in enumerate(source_records):
                persisted_record = dict(source_record)
                if index < completed:
                    persisted_record["evaluation"] = evaluated_records[index].get("evaluation")
                persisted_records.append(persisted_record)
            save_jsonl(persisted_records, destination_records_path)
            return

        compact_records = [
            {"evaluation": record.get("evaluation")}
            for record in evaluated_records
        ]
        save_jsonl(compact_records, destination_records_path)

    def _save_result(self, result: TaskResult) -> str:
        """Save a TaskResult to JSON and return the output path."""
        path = self._build_result_path(result)
        ensure_dir(os.path.dirname(path))
        extra_metadata = {
            key: value
            for key, value in result.metadata.items()
            if key not in {"task_config", "generation_config", "model", "result_path"}
        }
        extra_metadata.setdefault("sample_results_path", build_records_path(path))
        save_json(
            self._build_summary_payload(
                task_name=result.task_name,
                task_config=result.metadata.get("task_config", {}),
                model=result.metadata.get("model", {}),
                generation_config=result.metadata.get("generation_config", {}),
                metrics=result.metrics,
                num_samples=result.num_samples,
                evaluations={},
                metadata=extra_metadata,
            ),
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
        source_records = self._load_sample_records_from_paths(result_path)
        if not source_records:
            raise ValueError(f"No per-sample records found for result file: {result_path}")

        destination = output_path or result_path
        destination_records_path = build_records_path(destination)
        in_place = _normalize_path(destination) == _normalize_path(result_path)

        if os.path.exists(destination_records_path):
            persisted_records = load_jsonl(destination_records_path)
        elif not in_place and os.path.exists(destination):
            persisted_records = self._load_sample_records_from_paths(
                result_path=destination,
                records_path=destination_records_path,
            )
        else:
            persisted_records = []

        evaluated_records = self._merge_evaluated_records(source_records, persisted_records)
        resumed_evaluations = len(evaluated_records)

        for index in tqdm(
            range(len(evaluated_records), len(source_records)),
            initial=len(evaluated_records),
            total=len(source_records),
            desc=f"Evaluating {task.task_name}",
        ):
            source_record = source_records[index]
            evaluated_record = dict(source_record)
            evaluated_record["evaluation"] = task.evaluate_record(source_record)
            evaluated_records.append(evaluated_record)
            if in_place:
                self._save_evaluated_records(
                    destination_records_path=destination_records_path,
                    source_records=source_records,
                    evaluated_records=evaluated_records,
                    in_place=True,
                )
            else:
                append_jsonl({"evaluation": evaluated_record["evaluation"]}, destination_records_path)

        if in_place or persisted_records:
            self._save_evaluated_records(
                destination_records_path=destination_records_path,
                source_records=source_records,
                evaluated_records=evaluated_records,
                in_place=in_place,
            )

        metrics = task.aggregate_metrics_from_records(evaluated_records)
        evaluations = data.get("evaluations") or {}
        evaluations[evaluation_name] = {
            "metrics": metrics,
            "grader": grader_info or {},
        }

        metadata = dict(data.get("metadata") or {})
        metadata["source_result_path"] = result_path
        metadata["source_sample_results_path"] = build_records_path(result_path)
        metadata["sample_results_path"] = destination_records_path
        # metadata["resumed_evaluations"] = resumed_evaluations

        updated = self._build_summary_payload(
            task_name=data.get("task_name") or task.task_name,
            task_config=data.get("task_config") or {},
            model=data.get("model") or {},
            generation_config=data.get("generation_config") or {},
            metrics=metrics,
            num_samples=len(source_records),
            evaluations=evaluations,
            metadata=metadata,
        )
        save_json(updated, destination)
        return updated

    def summary(self, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Return a summary dict of all task metrics."""
        return {
            task_name: {
                "metrics": result.metrics,
                "num_samples": result.num_samples,
            }
            for task_name, result in results.items()
        }
