"""HealthBench task: Medical conversation rubric-based evaluation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory_eval.tasks.base import BaseTask
from memory_eval.tasks.registry import TaskRegistry
from memory_eval.metrics.rubric import (
    aggregate_tag_scores,
    build_rubric_grader_prompt,
    compute_bootstrap_stats,
    compute_points_score,
    parse_json_judgment,
    parse_rubric_judgment,
)

logger = logging.getLogger(__name__)


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

# Maximum retries when the grader returns unparseable output
_GRADER_MAX_RETRIES = 3


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

    # ------------------------------------------------------------------
    # Grading helpers
    # ------------------------------------------------------------------

    def _grade_rubric_item(
        self,
        conversation: List[Dict[str, str]],
        prediction: str,
        criterion: str,
    ) -> Dict[str, Any]:
        """Call the grader model to evaluate one criterion.

        Returns a dict with ``criteria_met`` (bool) and ``explanation`` (str).
        Retries up to ``_GRADER_MAX_RETRIES`` times on unparseable output.
        """
        prompt = build_rubric_grader_prompt(conversation, prediction, criterion)
        grader_messages = [{"role": "user", "content": prompt}]

        for attempt in range(_GRADER_MAX_RETRIES):
            response = self._grader_model.generate(
                grader_messages, max_tokens=256, temperature=0.0,
            )
            met, explanation = parse_json_judgment(response)
            if met is not None:
                return {"criteria_met": met, "explanation": explanation}
            logger.debug(
                "Grader returned unparseable output (attempt %d/%d): %s",
                attempt + 1,
                _GRADER_MAX_RETRIES,
                response[:200],
            )

        # Exhausted retries – fall back to Yes/No keyword matching
        met = parse_rubric_judgment(response)
        return {
            "criteria_met": met if met is not None else False,
            "explanation": f"Fallback parse from: {response[:200]}",
        }

    def _grade_single(
        self,
        sample: Dict[str, Any],
        prediction: str,
    ) -> Dict[str, Any]:
        """Grade a single prediction against the sample's rubric.

        Returns a dict with ``score``, ``example_tags``, and ``rubric_results``
        suitable for downstream aggregation.
        """
        rubrics = sample.get("rubrics") or sample.get("criteria") or []
        if not rubrics:
            return {"score": None, "example_tags": [], "rubric_results": []}

        # Build the conversation used for the grader prompt
        conversation = sample.get("conversation") or []
        if not conversation:
            # Reconstruct from prompt + prompt_response if available
            prompt_msgs = sample.get("prompt") or []
            resp_msgs = sample.get("prompt_response") or []
            if isinstance(prompt_msgs, list):
                conversation = prompt_msgs + resp_msgs
            elif isinstance(prompt_msgs, str) and prompt_msgs:
                conversation = [{"role": "user", "content": prompt_msgs}]

        rubric_results: List[Dict[str, Any]] = []
        for rubric in rubrics:
            if isinstance(rubric, dict):
                criterion = rubric.get("criterion") or rubric.get("text") or str(rubric)
                points = rubric.get("points")
                tags = rubric.get("tags", [])
            else:
                criterion = str(rubric)
                points = None
                tags = []

            if self._grader_model is not None:
                judgment = self._grade_rubric_item(conversation, prediction, criterion)
            else:
                judgment = {"criteria_met": False, "explanation": "No grader model"}

            rubric_results.append({
                "criterion": criterion,
                "points": float(points) if points is not None else 1.0,
                "tags": tags,
                "criteria_met": judgment["criteria_met"],
                "explanation": judgment["explanation"],
            })

        score = compute_points_score(rubric_results)
        return {
            "score": score if score is not None else 0.0,
            "example_tags": sample.get("example_tags", []),
            "rubric_results": rubric_results,
        }

    # ------------------------------------------------------------------
    # Main evaluation entry-point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        predictions: List[str],
    ) -> Dict[str, float]:
        """Evaluate HealthBench predictions.

        When a grader model has been set via :meth:`set_grader_model`, each
        prediction is graded against physician-created rubrics.  The returned
        metrics include:

        * ``overall_score`` – mean rubric score across all samples (0–1)
        * ``overall_score:bootstrap_std`` – bootstrap standard error
        * ``overall_score:n_samples`` – number of graded samples
        * Per-tag mean scores (example-level and rubric-level tags)

        Without a grader model, only proxy metrics (average response length
        and sample count) are returned.
        """
        if self._grader_model is not None:
            sample_results: List[Dict[str, Any]] = []
            scores: List[float] = []

            for sample, pred in zip(samples, predictions):
                result = self._grade_single(sample, pred)
                sample_results.append(result)
                if result["score"] is not None:
                    scores.append(result["score"])

            # Overall score with bootstrap statistics
            stats = compute_bootstrap_stats(scores)
            metrics: Dict[str, float] = {
                "overall_score": stats["mean"],
                "overall_score:bootstrap_std": stats["bootstrap_std"],
                "overall_score:n_samples": stats["n_samples"],
            }

            # Per-tag scores
            tag_scores = aggregate_tag_scores(sample_results)
            for tag, tag_score in sorted(tag_scores.items()):
                metrics[f"tag:{tag}"] = tag_score

            return metrics

        # Without grader model: return length statistics as proxy
        avg_length = (
            sum(len(p.split()) for p in predictions) / len(predictions)
            if predictions
            else 0.0
        )
        return {
            "avg_response_length": avg_length,
            "num_samples": float(len(predictions)),
        }

    def get_reference(self, sample: Dict[str, Any]) -> str:
        """Return rubric count as a summary reference."""
        rubrics = sample.get("rubrics") or sample.get("criteria") or []
        return f"{len(rubrics)} rubrics"
