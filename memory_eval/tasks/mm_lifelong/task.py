"""MM-Lifelong task: Multimodal Lifelong Understanding Evaluation."""

import base64
import io
import json
from typing import Any, Dict, List, Optional

from memory_eval.tasks.base import BaseTask
from memory_eval.tasks.registry import TaskRegistry
from memory_eval.metrics.accuracy import compute_exact_match, compute_multiple_choice_accuracy


SYSTEM_PROMPT = (
    "You are a multimodal AI assistant with strong visual understanding capabilities. "
    "You will be presented with an image and a question. "
    "Choose the single best answer from the provided options."
)

QUESTION_TEMPLATE = """{question}

Options:
{options}

Answer with the letter of the correct option only (e.g., A, B, C, or D)."""

OPEN_ENDED_QUESTION_TEMPLATE = """{question}

Answer as concisely as possible using only the final answer."""


@TaskRegistry.register("mm_lifelong")
class MMLifelongTask(BaseTask):
    """
    MM-Lifelong: A multimodal benchmark for lifelong visual understanding.

    The dataset contains image-question pairs testing temporal reasoning,
    episodic memory, and visual knowledge that spans long time horizons.

    Dataset: CG-Bench/MM-Lifelong on HuggingFace
    """

    task_name = "mm_lifelong"
    description = (
        "Multimodal lifelong understanding benchmark testing temporal reasoning "
        "and visual knowledge retention over long time horizons."
    )
    dataset_name = "CG-Bench/MM-Lifelong"
    dataset_split = "test"
    split_files = {
        "train": [("month/train.json", "train")],
        "val": [("month/val.json", "val")],
        "test": [("day/test.json", "test_day"), ("week/test.json", "test_week")],
        "test_day": [("day/test.json", "test_day")],
        "test_week": [("week/test.json", "test_week")],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._dataset = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load MM-Lifelong dataset from raw HuggingFace JSON files."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub package required via datasets dependency") from exc

        split_name = self.config.get("split", self.dataset_split)
        repo_files = self.split_files.get(split_name)
        if repo_files is None:
            valid_splits = ", ".join(sorted(self.split_files))
            raise ValueError(f"Invalid MM-Lifelong split '{split_name}'. Expected one of: {valid_splits}")

        samples = []
        for repo_file, source_split in repo_files:
            local_path = hf_hub_download(self.dataset_name, repo_file, repo_type="dataset")
            with open(local_path, "r", encoding="utf-8") as file_obj:
                items = json.load(file_obj)
            for item in items:
                sample = dict(item)
                if "clue_interval" not in sample and "clue_intervals" in sample:
                    sample["clue_interval"] = sample["clue_intervals"]
                sample["source_split"] = source_split
                samples.append(sample)
        return samples

    def _encode_image(self, image) -> str:
        """Encode a PIL image or bytes to base64 string."""
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        # PIL Image
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _format_options(self, sample: Dict[str, Any]) -> str:
        """Format answer options as a lettered list."""
        option_keys = ["A", "B", "C", "D", "E"]
        options_text = []
        for key in option_keys:
            val = sample.get(f"option_{key}") or sample.get(key) or sample.get(f"choice_{key}")
            if val is not None:
                options_text.append(f"{key}. {val}")
            else:
                # Try generic options list
                options_list = sample.get("options") or sample.get("choices")
                if options_list and isinstance(options_list, list):
                    for i, opt in enumerate(options_list):
                        if i < len(option_keys):
                            options_text.append(f"{option_keys[i]}. {opt}")
                    break
        return "\n".join(options_text) if options_text else str(sample.get("options", ""))

    def _has_options(self, sample: Dict[str, Any]) -> bool:
        option_keys = ["option_A", "option_B", "option_C", "option_D", "option_E", "A", "B", "C", "D", "E"]
        return bool(
            sample.get("options")
            or sample.get("choices")
            or any(sample.get(key) is not None for key in option_keys)
        )

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build multimodal messages for the MM-Lifelong task."""
        question = sample.get("question", sample.get("Question", ""))
        if self._has_options(sample):
            options_text = self._format_options(sample)
            text_content = QUESTION_TEMPLATE.format(
                question=question,
                options=options_text,
            )
        else:
            text_content = OPEN_ENDED_QUESTION_TEMPLATE.format(question=question)

        # Build content parts (text + optional image)
        content = []
        image = sample.get("image") or sample.get("Image")
        if image is not None:
            try:
                encoded = self._encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                })
            except Exception:
                pass  # Skip image if encoding fails

        content.append({"type": "text", "text": text_content})

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def get_reference(self, sample: Dict[str, Any]) -> str:
        """Return the ground-truth answer letter."""
        answer = sample.get("answer") or sample.get("Answer") or sample.get("gt_answer", "")
        return str(answer).strip().upper()

    def serialize_sample_for_evaluation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Persist only the fields required to score deferred predictions."""
        serialized = {
            "answer": sample.get("answer") or sample.get("Answer") or sample.get("gt_answer", ""),
            "category": sample.get("category"),
            "type": sample.get("type"),
            "task_type": sample.get("task_type"),
        }
        if self._has_options(sample):
            serialized["options"] = sample.get("options") or sample.get("choices") or []
        return serialized

    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        predictions: List[str],
    ) -> Dict[str, float]:
        """Compute accuracy for MM-Lifelong."""
        references = [self.get_reference(s) for s in samples]
        if all(self._has_options(sample) for sample in samples):
            accuracy = compute_multiple_choice_accuracy(predictions, references)
        else:
            accuracy = sum(compute_exact_match(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions) if predictions else 0.0

        # Compute per-category accuracy if category info available
        metrics: Dict[str, float] = {"accuracy": accuracy}
        categories: Dict[str, List] = {}
        for sample, pred, ref in zip(samples, predictions, references):
            cat = sample.get("category") or sample.get("type") or sample.get("task_type")
            if cat:
                if cat not in categories:
                    categories[cat] = {"preds": [], "refs": []}
                categories[cat]["preds"].append(pred)
                categories[cat]["refs"].append(ref)

        for cat, data in categories.items():
            cat_acc = compute_multiple_choice_accuracy(data["preds"], data["refs"])
            metrics[f"accuracy_{cat}"] = cat_acc

        return metrics
