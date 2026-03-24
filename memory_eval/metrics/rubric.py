"""Rubric-based scoring metrics (used for HealthBench-style evaluation)."""

import re
from typing import Dict, List, Optional


def parse_rubric_judgment(text: str) -> Optional[bool]:
    """
    Parse a rubric criterion judgment from a grader's response.
    Expected formats: 'Yes', 'No', or reasoning followed by a verdict.
    Returns True if criterion is met, False if not, None if unparseable.
    """
    text = text.strip().lower()
    # Check for explicit yes/no at end or start
    if re.search(r"\byes\b", text):
        return True
    if re.search(r"\bno\b", text):
        return False
    return None


def compute_rubric_score(
    judgments: List[Optional[bool]],
    weights: Optional[List[float]] = None,
) -> float:
    """
    Compute a rubric score from a list of criterion judgments.

    Args:
        judgments: List of boolean judgments (True=met, False=not met, None=skip).
        weights: Optional per-criterion weights. Defaults to uniform.

    Returns:
        Weighted fraction of criteria met (0.0 to 1.0).
    """
    valid = [(j, w) for j, w in zip(judgments, weights or [1.0] * len(judgments)) if j is not None]
    if not valid:
        return 0.0
    total_weight = sum(w for _, w in valid)
    if total_weight == 0:
        return 0.0
    met_weight = sum(w for j, w in valid if j)
    return met_weight / total_weight


def build_rubric_grader_prompt(
    conversation: List[Dict[str, str]],
    model_response: str,
    criterion: str,
) -> str:
    """
    Build a prompt for an LLM grader to evaluate a single rubric criterion.

    Args:
        conversation: The original conversation (list of {"role": ..., "content": ...}).
        model_response: The model's response to evaluate.
        criterion: The rubric criterion to check.

    Returns:
        A prompt string for the grader.
    """
    history = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation
    )
    return (
        "You are a medical expert evaluating an AI assistant's response.\n\n"
        f"Conversation:\n{history}\n\n"
        f"AI Response:\n{model_response}\n\n"
        f"Criterion: {criterion}\n\n"
        "Does the AI response satisfy the criterion above? "
        "Respond with 'Yes' or 'No', followed by a brief explanation."
    )
