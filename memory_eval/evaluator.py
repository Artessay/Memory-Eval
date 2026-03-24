"""Main evaluator orchestrating model evaluation across tasks."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from memory_eval.models.base import BaseModel
from memory_eval.tasks.base import BaseTask, TaskResult
from memory_eval.utils.io import save_json, ensure_dir


class Evaluator:
    """Orchestrates evaluation of one or more tasks with a given model."""

    def __init__(
        self,
        model: BaseModel,
        output_dir: str = "results",
    ):
        self.model = model
        self.output_dir = output_dir

    def run_task(
        self,
        task: BaseTask,
        max_tokens: int = 512,
        temperature: float = 0.0,
        limit: Optional[int] = None,
    ) -> TaskResult:
        """Run a single task and save results."""
        result = task.run(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            limit=limit,
        )
        self._save_result(result)
        return result

    def run_tasks(
        self,
        tasks: List[BaseTask],
        max_tokens: int = 512,
        temperature: float = 0.0,
        limit: Optional[int] = None,
    ) -> Dict[str, TaskResult]:
        """Run multiple tasks and return a dict of results keyed by task name."""
        results = {}
        for task in tasks:
            result = self.run_task(task, max_tokens=max_tokens, temperature=temperature, limit=limit)
            results[task.task_name] = result
        return results

    def _save_result(self, result: TaskResult) -> None:
        """Save a TaskResult to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.task_name}_{timestamp}.json"
        path = os.path.join(self.output_dir, filename)
        ensure_dir(self.output_dir)
        save_json(
            {
                "task_name": result.task_name,
                "metrics": result.metrics,
                "num_samples": result.num_samples,
                "predictions": result.predictions,
                "references": result.references,
                "metadata": result.metadata,
            },
            path,
        )

    def summary(self, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Return a summary dict of all task metrics."""
        return {
            task_name: {
                "metrics": result.metrics,
                "num_samples": result.num_samples,
            }
            for task_name, result in results.items()
        }
