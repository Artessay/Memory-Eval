"""HealthBench task: Medical conversation rubric-based evaluation."""

from typing import Any, Dict, List, Optional

from memory_eval.tasks.base import BaseTask
from memory_eval.tasks.registry import TaskRegistry
from memory_eval.metrics.rubric import (
    build_rubric_grader_prompt,
    compute_rubric_score,
    parse_rubric_judgment,
)


SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful medical AI assistant. "
    "Provide clear, accurate, and compassionate responses to health-related questions. "
    "Always recommend consulting a healthcare professional for personalized medical advice."
)


@TaskRegistry.register("healthbench")
class HealthBenchTask(BaseTask):
    """
    HealthBench: Medical conversation benchmark with rubric-based evaluation.

    The dataset contains realistic health conversations evaluated against
    physician-created rubrics.

    Dataset: openai/healthbench on HuggingFace
    Subsets: 'hard' (1000 examples) or 'consensus' (3671 examples)
    """

    task_name = "healthbench"
    description = (
        "Medical conversation benchmark evaluating AI responses against "
        "physician-created rubrics for accuracy, completeness, and safety."
    )
    dataset_name = "openai/healthbench"
    dataset_split = "test"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._grader_model = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load HealthBench dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets package required: pip install datasets") from exc

        subset = self.config.get("subset", "hard")
        ds = load_dataset(
            self.dataset_name,
            subset,
            split=self.config.get("split", self.dataset_split),
            trust_remote_code=True,
        )
        return [dict(item) for item in ds]

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build messages for the HealthBench task."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # The conversation history (all turns except the last assistant turn)
        conversation = sample.get("conversation") or []
        if isinstance(conversation, list):
            for turn in conversation:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant", "system"):
                    messages.append({"role": role, "content": content})
        elif isinstance(conversation, str):
            messages.append({"role": "user", "content": conversation})

        # If no conversation found, use prompt field
        if len(messages) == 1:
            prompt = sample.get("prompt") or sample.get("question", "")
            messages.append({"role": "user", "content": prompt})

        return messages

    def set_grader_model(self, model) -> None:
        """Set the LLM grader to use for rubric evaluation."""
        self._grader_model = model

    def _grade_single(
        self,
        sample: Dict[str, Any],
        prediction: str,
    ) -> float:
        """Grade a single prediction against the sample's rubric."""
        rubrics = sample.get("rubrics") or sample.get("criteria") or []
        if not rubrics:
            return 0.0

        judgments = []
        weights = []
        for rubric in rubrics:
            if isinstance(rubric, dict):
                criterion = rubric.get("criterion") or rubric.get("text") or str(rubric)
                weight = float(rubric.get("weight", 1.0))
            else:
                criterion = str(rubric)
                weight = 1.0

            if self._grader_model is not None:
                conversation = sample.get("conversation") or []
                prompt = build_rubric_grader_prompt(conversation, prediction, criterion)
                grader_messages = [{"role": "user", "content": prompt}]
                response = self._grader_model.generate(grader_messages, max_tokens=128, temperature=0.0)
                judgment = parse_rubric_judgment(response)
            else:
                judgment = None

            judgments.append(judgment)
            weights.append(weight)

        return compute_rubric_score(judgments, weights)

    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate HealthBench predictions.

        If a grader model has been set, uses rubric-based scoring.
        Otherwise returns placeholder metrics.
        """
        if self._grader_model is not None:
            scores = [
                self._grade_single(sample, pred)
                for sample, pred in zip(samples, predictions)
            ]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            return {
                "rubric_score": avg_score,
                "num_graded": len([s for s in scores if s is not None]),
            }

        # Without grader model: return length statistics as proxy
        avg_length = sum(len(p.split()) for p in predictions) / len(predictions) if predictions else 0.0
        return {
            "avg_response_length": avg_length,
            "num_samples": float(len(predictions)),
        }

    def get_reference(self, sample: Dict[str, Any]) -> str:
        """Return rubric count as a summary reference."""
        rubrics = sample.get("rubrics") or sample.get("criteria") or []
        return f"{len(rubrics)} rubrics"
