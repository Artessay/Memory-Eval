"""Rubric-based scoring metrics (used for HealthBench-style evaluation)."""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Grader prompt template
# ---------------------------------------------------------------------------

GRADER_TEMPLATE = """\
You are evaluating an AI assistant's response to a health-related question.

## Conversation
{conversation}

## AI Response to Evaluate
{response}

## Evaluation Criterion
{criterion}

## Instructions
Determine whether the AI response meets the evaluation criterion above.
Be precise – the criterion must be clearly met for a positive judgment.

Respond **only** with a JSON object in the following format (no other text):
{{"criteria_met": true, "explanation": "Brief explanation of your reasoning"}}
or
{{"criteria_met": false, "explanation": "Brief explanation of your reasoning"}}
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_json_judgment(text: str) -> Tuple[Optional[bool], str]:
    """Parse a JSON grading response with ``criteria_met`` and ``explanation``.

    Returns:
        A tuple of (criteria_met, explanation).  *criteria_met* is ``None``
        when the response cannot be parsed.
    """
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip())

    # 1. Try full JSON parse
    try:
        data = json.loads(cleaned)
        met = data.get("criteria_met")
        if isinstance(met, bool):
            return met, data.get("explanation", "")
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Try to find an embedded JSON object
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            met = data.get("criteria_met")
            if isinstance(met, bool):
                return met, data.get("explanation", "")
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Keyword fallback
    lower = cleaned.lower()
    if '"criteria_met": true' in lower or '"criteria_met":true' in lower:
        return True, cleaned
    if '"criteria_met": false' in lower or '"criteria_met":false' in lower:
        return False, cleaned

    return None, cleaned


def parse_rubric_judgment(text: str) -> Optional[bool]:
    """Parse a rubric criterion judgment from a grader's response.

    Supports both JSON responses (``{"criteria_met": true/false, ...}``) and
    simple Yes/No text responses for backward compatibility.

    Returns ``True`` if criterion is met, ``False`` if not, ``None`` if
    unparseable.
    """
    # Try JSON parsing first
    met, _ = parse_json_judgment(text)
    if met is not None:
        return met

    # Fall back to Yes/No keyword matching
    lower = text.strip().lower()
    if re.search(r"\byes\b", lower):
        return True
    if re.search(r"\bno\b", lower):
        return False
    return None


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_rubric_score(
    judgments: List[Optional[bool]],
    weights: Optional[List[float]] = None,
) -> float:
    """Compute a rubric score from a list of criterion judgments.

    Args:
        judgments: List of boolean judgments (True=met, False=not met, None=skip).
        weights: Optional per-criterion weights.  Defaults to uniform.

    Returns:
        Weighted fraction of criteria met (0.0 to 1.0).
    """
    valid = [
        (j, w)
        for j, w in zip(judgments, weights or [1.0] * len(judgments))
        if j is not None
    ]
    if not valid:
        return 0.0
    total_weight = sum(w for _, w in valid)
    if total_weight == 0:
        return 0.0
    met_weight = sum(w for j, w in valid if j)
    return met_weight / total_weight


def compute_points_score(
    rubric_results: List[Dict[str, Any]],
) -> Optional[float]:
    """Compute a HealthBench-style points-based score.

    Each rubric result is a dict with at least ``"points"`` (int) and
    ``"criteria_met"`` (bool).  The score is the sum of points for met
    criteria divided by the sum of *positive* points.

    Returns:
        Score clipped to [0, 1], or ``None`` when there are no positive
        criteria.
    """
    total_possible = sum(
        r["points"] for r in rubric_results if r["points"] > 0
    )
    if total_possible == 0:
        return None
    achieved = sum(
        r["points"] for r in rubric_results if r["criteria_met"]
    )
    return float(np.clip(achieved / total_possible, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Bootstrap statistics
# ---------------------------------------------------------------------------


def compute_bootstrap_stats(
    values: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute mean, bootstrap standard-error, and sample count.

    Args:
        values: List of per-sample scores.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys ``mean``, ``bootstrap_std``, and ``n_samples``.
    """
    if not values:
        return {"mean": 0.0, "bootstrap_std": 0.0, "n_samples": 0}

    rng = np.random.RandomState(seed)
    arr = np.array(values)
    mean = float(np.clip(np.mean(arr), 0.0, 1.0))
    bootstrap_indices = rng.choice(len(arr), size=(n_bootstrap, len(arr)), replace=True)
    bootstrap_means = np.clip(np.mean(arr[bootstrap_indices], axis=1), 0.0, 1.0)
    return {
        "mean": mean,
        "bootstrap_std": float(np.std(bootstrap_means)),
        "n_samples": len(values),
    }


# ---------------------------------------------------------------------------
# Per-tag aggregation
# ---------------------------------------------------------------------------


def aggregate_tag_scores(
    sample_results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Aggregate per-tag scores across graded samples.

    Each element in *sample_results* should contain:

    * ``"score"`` – the overall sample score (float)
    * ``"example_tags"`` – list of example-level tag names
    * ``"rubric_results"`` – list of per-rubric dicts, each with ``"points"``,
      ``"criteria_met"``, and ``"tags"``

    Returns:
        Dict mapping tag names to their aggregated mean scores.
    """
    tag_values: Dict[str, List[float]] = defaultdict(list)

    for result in sample_results:
        score = result.get("score")
        if score is None:
            continue

        # Example-level tags share the overall score
        for tag in result.get("example_tags", []):
            tag_values[tag].append(score)

        # Rubric-level tags get their own sub-score
        rubric_tag_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rr in result.get("rubric_results", []):
            for tag in rr.get("tags", []):
                rubric_tag_items[tag].append(rr)

        for tag, items in rubric_tag_items.items():
            tag_score = compute_points_score(items)
            if tag_score is not None:
                tag_values[tag].append(tag_score)

    return {tag: float(np.clip(np.mean(vals), 0.0, 1.0)) for tag, vals in tag_values.items()}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_rubric_grader_prompt(
    conversation: List[Dict[str, str]],
    model_response: str,
    criterion: str,
) -> str:
    """Build a prompt for an LLM grader to evaluate a single rubric criterion.

    The prompt asks the grader to reply with a JSON object containing
    ``criteria_met`` (bool) and ``explanation`` (str).

    Args:
        conversation: The original conversation messages.
        model_response: The model's response to evaluate.
        criterion: The rubric criterion to check.

    Returns:
        A prompt string for the grader.
    """
    history = "\n\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation
    )
    return GRADER_TEMPLATE.format(
        conversation=history,
        response=model_response,
        criterion=criterion,
    )
