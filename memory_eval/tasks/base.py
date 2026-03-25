"""Abstract base class for all evaluation tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    def run(
        self,
        model: BaseModel,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        limit: Optional[int] = None,
        compute_metrics: bool = True,
    ) -> TaskResult:
        """
        Run the full evaluation pipeline on the given model.

        Args:
            model: A BaseModel instance to evaluate.
            max_tokens: Max tokens to generate per sample.
            temperature: Sampling temperature.
            limit: If set, only evaluate on this many samples.
            compute_metrics: If True, compute task metrics immediately.

        Returns:
            A TaskResult with metrics and predictions.
        """
        from tqdm import tqdm

        samples = self.load_dataset()
        if limit is not None:
            samples = samples[:limit]

        predictions = []
        for sample in tqdm(samples, desc=f"Running {self.task_name}"):
            messages = self.build_messages(sample)
            pred = model.generate(messages, max_tokens=max_tokens, temperature=temperature)
            predictions.append(pred)

        metrics = self.evaluate(samples, predictions) if compute_metrics else {}
        references = [self.get_reference(s) for s in samples]
        evaluation_samples = [self.serialize_sample_for_evaluation(sample) for sample in samples]

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
