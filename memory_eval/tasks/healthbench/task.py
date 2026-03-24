"""HealthBench task: Medical conversation rubric-based evaluation."""

from pathlib import Path
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


SUBSET_FILE_PATTERNS = {
    "hard": "hard_*.jsonl",
    "consensus": "consensus_*.jsonl",
    "standard": "*_oss_eval.jsonl",
}


@TaskRegistry.register("healthbench")
class HealthBenchTask(BaseTask):
    """
    HealthBench: Medical conversation benchmark with rubric-based evaluation.

    The dataset contains realistic health conversations evaluated against
    physician-created rubrics.

    Dataset: openai/healthbench on HuggingFace
    Subsets: 'hard' (1000 examples), 'consensus' (3671 examples),
    or 'standard' (OSS eval split)
    """

    task_name = "healthbench"
    description = (
        "Medical conversation benchmark evaluating AI responses against "
        "physician-created rubrics for accuracy, completeness, and safety."
    )
    dataset_name = "openai/healthbench"
    dataset_split = "train"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._grader_model = None

    def _get_subset(self) -> str:
        subset = self.config.get("subset", "standard")
        if subset not in SUBSET_FILE_PATTERNS:
            valid_subsets = ", ".join(sorted(SUBSET_FILE_PATTERNS))
            raise ValueError(
                f"Invalid HealthBench subset: {subset}. "
                f"Expected one of: {valid_subsets}"
            )
        return subset

    def _get_data_files(self, subset: str) -> str:
        return SUBSET_FILE_PATTERNS[subset]

    def _load_from_local_dir(self, data_dir: Path, subset: str) -> List[Dict[str, Any]]:
        file_pattern = self._get_data_files(subset)
        if subset == "standard":
            files = [path for path in data_dir.glob(file_pattern) if "meta" not in path.name]
        else:
            files = list(data_dir.glob(file_pattern))

        if not files:
            raise FileNotFoundError(
                f"No HealthBench files matching '{file_pattern}' found in {data_dir}"
            )

        records = []
        for path in sorted(files):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        import json

                        records.append(json.loads(line))
        return records

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load HealthBench dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets package required: pip install datasets") from exc

        subset = self._get_subset()
        data_dir = self.config.get("data_dir")
        if data_dir:
            return self._load_from_local_dir(Path(data_dir), subset)

        ds = load_dataset(
            self.dataset_name,
            data_files=self._get_data_files(subset),
            split=self.config.get("split", "train"),
        )
        return [dict(item) for item in ds]

    def _append_chat_messages(
        self,
        messages: List[Dict[str, Any]],
        payload: Any,
        default_role: str = "user",
    ) -> int:
        """Append normalized chat messages from dataset payloads."""
        appended = 0

        if isinstance(payload, str):
            if payload:
                messages.append({"role": default_role, "content": payload})
                return 1
            return 0

        if isinstance(payload, dict):
            role = payload.get("role", default_role)
            content = payload.get("content", "")
            if role in ("user", "assistant", "system") and content not in (None, ""):
                messages.append({"role": role, "content": content})
                return 1
            return 0

        if isinstance(payload, list):
            for item in payload:
                appended += self._append_chat_messages(
                    messages,
                    item,
                    default_role=default_role,
                )

        return appended

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build messages for the HealthBench task."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # The conversation history (all turns except the last assistant turn)
        conversation = sample.get("conversation") or []
        appended = self._append_chat_messages(messages, conversation)

        # If no conversation found, use prompt field
        if appended == 0:
            prompt = sample.get("prompt") or sample.get("question", "")
            appended = self._append_chat_messages(messages, prompt)

        if appended == 0:
            messages.append({"role": "user", "content": ""})

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
            num_graded = sum(
                1 for sample in samples
                if sample.get("rubrics") or sample.get("criteria")
            )
            return {
                "rubric_score": avg_score,
                "num_graded": num_graded,
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
