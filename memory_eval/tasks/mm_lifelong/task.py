"""MM-Lifelong task: Multimodal Lifelong Understanding Evaluation."""

import base64
import io
from typing import Any, Dict, List, Optional

from memory_eval.tasks.base import BaseTask
from memory_eval.tasks.registry import TaskRegistry
from memory_eval.metrics.accuracy import compute_multiple_choice_accuracy, extract_choice


SYSTEM_PROMPT = (
    "You are a multimodal AI assistant with strong visual understanding capabilities. "
    "You will be presented with an image and a question. "
    "Choose the single best answer from the provided options."
)

QUESTION_TEMPLATE = """{question}

Options:
{options}

Answer with the letter of the correct option only (e.g., A, B, C, or D)."""


@TaskRegistry.register("mm_lifelong")
class MMLonglivedTask(BaseTask):
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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._dataset = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load MM-Lifelong dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets package required: pip install datasets") from exc

        ds = load_dataset(
            self.dataset_name,
            split=self.config.get("split", self.dataset_split),
            trust_remote_code=True,
        )
        samples = []
        for item in ds:
            samples.append(dict(item))
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

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build multimodal messages for the MM-Lifelong task."""
        question = sample.get("question", sample.get("Question", ""))
        options_text = self._format_options(sample)
        text_content = QUESTION_TEMPLATE.format(
            question=question,
            options=options_text,
        )

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

    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        predictions: List[str],
    ) -> Dict[str, float]:
        """Compute accuracy for MM-Lifelong."""
        references = [self.get_reference(s) for s in samples]
        accuracy = compute_multiple_choice_accuracy(predictions, references)

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
