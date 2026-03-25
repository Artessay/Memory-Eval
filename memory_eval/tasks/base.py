"""Abstract base class for all evaluation tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from memory_eval.models.base import BaseModel


@dataclass
class TaskResult:
    """Holds the result of running a task."""
    task_name: str
    metrics: Dict[str, float]
    num_samples: int
    predictions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    evaluation_samples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    """Base class for all evaluation tasks."""

    task_name: str = "base"
    description: str = ""
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: str = "test"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and return the dataset samples as a list of dicts."""

    @abstractmethod
    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build the model input messages for a single sample.

        Returns:
            A list of message dicts with 'role' and 'content' keys.
        """

    @abstractmethod
    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            samples: The original dataset samples.
            predictions: The model predictions.

        Returns:
            A dict mapping metric names to values.
        """

    def serialize_sample_for_evaluation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-serializable sample payload needed for deferred evaluation."""
        return dict(sample)

    def build_result_record(self, sample: Dict[str, Any], prediction: str) -> Dict[str, Any]:
        """Build a JSON-serializable per-sample record for incremental persistence."""
        return {
            "prediction": prediction,
            "reference": self.get_reference(sample),
            "evaluation_sample": self.serialize_sample_for_evaluation(sample),
        }

    def evaluate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Compute per-sample evaluation details for a persisted record."""
        return {}

    def aggregate_metrics_from_records(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics from persisted per-sample records."""
        samples = [record.get("evaluation_sample", {}) for record in records]
        predictions = [str(record.get("prediction", "")) for record in records]
        return self.evaluate(samples, predictions)

    def run(
        self,
        model: BaseModel,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        limit: Optional[int] = None,
        compute_metrics: bool = True,
        existing_records: Optional[List[Dict[str, Any]]] = None,
        record_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TaskResult:
        """
        Run the full evaluation pipeline on the given model.

        Args:
            model: A BaseModel instance to evaluate.
            max_tokens: Max tokens to generate per sample.
            temperature: Sampling temperature.
            limit: If set, only evaluate on this many samples.
            compute_metrics: If True, compute task metrics immediately.
            existing_records: Existing persisted sample records to resume from.
            record_callback: Called after each newly generated sample record.

        Returns:
            A TaskResult with metrics and predictions.
        """
        from tqdm import tqdm

        samples = self.load_dataset()
        if limit is not None:
            samples = samples[:limit]

        existing_records = list(existing_records or [])
        completed = min(len(existing_records), len(samples))
        if len(existing_records) > len(samples):
            existing_records = existing_records[: len(samples)]

        predictions = [str(record.get("prediction", "")) for record in existing_records]
        references = [str(record.get("reference", "")) for record in existing_records]
        evaluation_samples = [
            record.get("evaluation_sample", {}) for record in existing_records
        ]
        new_records: List[Dict[str, Any]] = []

        for sample in tqdm(samples[completed:], initial=completed, total=len(samples), desc=f"Running {self.task_name}"):
            messages = self.build_messages(sample)
            pred = model.generate(messages, max_tokens=max_tokens, temperature=temperature)
            record = self.build_result_record(sample, pred)
            new_records.append(record)
            predictions.append(record["prediction"])
            references.append(str(record.get("reference", "")))
            evaluation_samples.append(record.get("evaluation_sample", {}))
            if record_callback is not None:
                record_callback(record)

        metrics = self.aggregate_metrics_from_records(existing_records + new_records) if compute_metrics else {}

        return TaskResult(
            task_name=self.task_name,
            metrics=metrics,
            num_samples=len(samples),
            predictions=predictions,
            references=references,
            evaluation_samples=evaluation_samples,
            metadata={"task_config": dict(self.config)},
        )

    def get_reference(self, sample: Dict[str, Any]) -> str:
        """Return the ground-truth reference for a sample (override if needed)."""
        return str(sample.get("answer", sample.get("label", "")))
