"""Rubric-based scoring metrics (used for HealthBench-style evaluation)."""

import json
import re
from typing import Dict, List, Optional


def parse_rubric_judgment(text: str) -> Optional[bool]:
    """
    Parse a rubric criterion judgment from a grader's response.
    Expected formats: a JSON object with a boolean `criteria_met` field,
    or a textual Yes/No verdict.
    Returns True if criterion is met, False if not, None if unparseable.
    """
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if code_block_match:
        candidates.append(code_block_match.group(1))

    json_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
    if json_match:
        candidates.append(json_match.group(1))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("criteria_met"), bool):
            return data["criteria_met"]

    text = stripped.lower()
    if re.search(r'"criteria_met"\s*:\s*true', text):
        return True
    if re.search(r'"criteria_met"\s*:\s*false', text):
        return False
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
    history = "\n\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation
    )
    return (
        "You are evaluating an AI assistant's response to a health-related question.\n\n"
        "## Original Conversation\n"
        f"{history or 'User: '}\n\n"
        "## AI Response to Evaluate\n"
        f"{model_response}\n\n"
        "## Evaluation Criterion\n"
        f"{criterion}\n\n"
        "## Instructions\n"
        "Determine whether the AI response clearly satisfies the criterion above. "
        "Be precise and conservative.\n"
        "Respond with a JSON object using this schema: "
        '{"explanation": "Brief explanation of your reasoning", "criteria_met": true/false}'
    )
