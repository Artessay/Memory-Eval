"""Accuracy-based metrics for evaluation."""

import re
import string
from typing import List, Optional


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, articles and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def compute_exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 if prediction exactly matches reference (after normalization)."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def extract_choice(text: str) -> Optional[str]:
    """Extract a letter choice (A/B/C/D/E) from model output."""
    text = text.strip()
    # Direct single-letter answer
    match = re.match(r"^([A-Ea-e])[.\):\s]", text)
    if match:
        return match.group(1).upper()
    # "The answer is X" style
    match = re.search(r"(?:answer is|answer:)\s*([A-Ea-e])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Standalone letter at start
    match = re.match(r"^([A-Ea-e])\s*$", text.strip())
    if match:
        return match.group(1).upper()
    return None


def compute_multiple_choice_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute accuracy for multiple-choice questions."""
    if not predictions:
        return 0.0
    correct = 0
    for pred, ref in zip(predictions, references):
        extracted = extract_choice(pred)
        if extracted is None:
            # Fallback: check if reference letter appears as a standalone token
            ref_letter = extract_choice(ref)
            if ref_letter and re.search(
                rf"(?<![A-Za-z]){re.escape(ref_letter)}(?![A-Za-z])", pred, re.IGNORECASE
            ):
                correct += 1
        elif extracted.upper() == ref.strip().upper():
            correct += 1
    return correct / len(predictions)
