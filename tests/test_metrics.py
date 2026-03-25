"""Tests for evaluation metrics."""

import pytest
from memory_eval.metrics.accuracy import (
    normalize_answer,
    compute_exact_match,
    extract_choice,
    compute_multiple_choice_accuracy,
)
from memory_eval.metrics.rubric import (
    parse_rubric_judgment,
    parse_json_judgment,
    compute_rubric_score,
    compute_points_score,
    compute_bootstrap_stats,
    aggregate_tag_scores,
    build_rubric_grader_prompt,
)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_removes_articles(self):
        assert normalize_answer("The cat sat") == "cat sat"

    def test_removes_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_extra_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"


class TestComputeExactMatch:
    def test_exact_match(self):
        assert compute_exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert compute_exact_match("paris", "Paris") == 1.0

    def test_no_match(self):
        assert compute_exact_match("London", "Paris") == 0.0

    def test_with_articles(self):
        assert compute_exact_match("the cat", "cat") == 1.0


class TestExtractChoice:
    def test_direct_letter(self):
        assert extract_choice("A. Paris") == "A"

    def test_answer_is_pattern(self):
        assert extract_choice("The answer is B") == "B"

    def test_standalone_letter(self):
        assert extract_choice("C") == "C"

    def test_lowercase(self):
        assert extract_choice("d") == "D"

    def test_no_choice(self):
        assert extract_choice("Paris is the capital of France") is None


class TestComputeMultipleChoiceAccuracy:
    def test_all_correct(self):
        preds = ["A", "B", "C", "D"]
        refs = ["A", "B", "C", "D"]
        assert compute_multiple_choice_accuracy(preds, refs) == 1.0

    def test_all_wrong(self):
        preds = ["B", "C", "D", "A"]
        refs = ["A", "B", "C", "D"]
        assert compute_multiple_choice_accuracy(preds, refs) == 0.0

    def test_half_correct(self):
        preds = ["A", "C", "C", "D"]
        refs = ["A", "B", "C", "D"]
        assert compute_multiple_choice_accuracy(preds, refs) == 0.75

    def test_empty(self):
        assert compute_multiple_choice_accuracy([], []) == 0.0

    def test_with_sentence_preds(self):
        preds = ["The answer is A because...", "B. That is correct"]
        refs = ["A", "B"]
        assert compute_multiple_choice_accuracy(preds, refs) == 1.0


class TestParseRubricJudgment:
    def test_yes(self):
        assert parse_rubric_judgment("Yes, the response covers this.") is True

    def test_no(self):
        assert parse_rubric_judgment("No, it does not mention this.") is False

    def test_yes_alone(self):
        assert parse_rubric_judgment("yes") is True

    def test_no_alone(self):
        assert parse_rubric_judgment("no") is False

    def test_unparseable(self):
        assert parse_rubric_judgment("I cannot determine this.") is None

    def test_json_true(self):
        assert parse_rubric_judgment('{"criteria_met": true, "explanation": "ok"}') is True

    def test_json_false(self):
        assert parse_rubric_judgment('{"criteria_met": false, "explanation": "no"}') is False

    def test_json_in_markdown(self):
        text = '```json\n{"criteria_met": true, "explanation": "good"}\n```'
        assert parse_rubric_judgment(text) is True


class TestParseJsonJudgment:
    def test_valid_true(self):
        met, explanation = parse_json_judgment('{"criteria_met": true, "explanation": "Good."}')
        assert met is True
        assert explanation == "Good."

    def test_valid_false(self):
        met, explanation = parse_json_judgment('{"criteria_met": false, "explanation": "Bad."}')
        assert met is False
        assert explanation == "Bad."

    def test_markdown_wrapped(self):
        text = '```json\n{"criteria_met": true, "explanation": "ok"}\n```'
        met, _ = parse_json_judgment(text)
        assert met is True

    def test_embedded_json(self):
        text = 'Here is my judgment: {"criteria_met": false, "explanation": "nope"}'
        met, _ = parse_json_judgment(text)
        assert met is False

    def test_keyword_fallback_true(self):
        met, _ = parse_json_judgment('The "criteria_met": true in the response')
        assert met is True

    def test_keyword_fallback_false(self):
        met, _ = parse_json_judgment('"criteria_met": false')
        assert met is False

    def test_unparseable(self):
        met, _ = parse_json_judgment("I have no idea")
        assert met is None


class TestComputeRubricScore:
    def test_all_met(self):
        assert compute_rubric_score([True, True, True]) == 1.0

    def test_none_met(self):
        assert compute_rubric_score([False, False, False]) == 0.0

    def test_half_met(self):
        score = compute_rubric_score([True, False, True, False])
        assert score == 0.5

    def test_with_none(self):
        score = compute_rubric_score([True, None, False])
        assert score == 0.5  # Only True and False count

    def test_all_none(self):
        assert compute_rubric_score([None, None]) == 0.0

    def test_weighted(self):
        score = compute_rubric_score([True, False], weights=[2.0, 1.0])
        assert abs(score - 2 / 3) < 1e-9

    def test_empty(self):
        assert compute_rubric_score([]) == 0.0


class TestBuildRubricGraderPrompt:
    def test_basic(self):
        conversation = [{"role": "user", "content": "What should I do for a headache?"}]
        prompt = build_rubric_grader_prompt(conversation, "Take ibuprofen.", "Recommends OTC medication")
        assert "headache" in prompt
        assert "ibuprofen" in prompt
        assert "Recommends OTC medication" in prompt
        assert "criteria_met" in prompt

    def test_requests_json(self):
        conversation = [{"role": "user", "content": "test"}]
        prompt = build_rubric_grader_prompt(conversation, "response", "criterion")
        assert "JSON" in prompt or "json" in prompt


class TestComputePointsScore:
    def test_all_met(self):
        results = [
            {"points": 2, "criteria_met": True},
            {"points": 3, "criteria_met": True},
        ]
        assert compute_points_score(results) == 1.0

    def test_none_met(self):
        results = [
            {"points": 2, "criteria_met": False},
            {"points": 3, "criteria_met": False},
        ]
        assert compute_points_score(results) == 0.0

    def test_partial(self):
        results = [
            {"points": 2, "criteria_met": True},
            {"points": 3, "criteria_met": False},
        ]
        score = compute_points_score(results)
        assert abs(score - 2 / 5) < 1e-9

    def test_negative_points(self):
        # Negative rubrics (harmful) – met negative criteria reduce score
        results = [
            {"points": 3, "criteria_met": True},
            {"points": -1, "criteria_met": True},  # harmful criterion met
        ]
        # achieved = 3 + (-1) = 2, total_possible = 3
        score = compute_points_score(results)
        assert abs(score - 2 / 3) < 1e-9

    def test_no_positive_points(self):
        results = [
            {"points": -1, "criteria_met": True},
        ]
        assert compute_points_score(results) is None

    def test_empty(self):
        assert compute_points_score([]) is None

    def test_clipped_to_zero(self):
        results = [
            {"points": 1, "criteria_met": False},
            {"points": -3, "criteria_met": True},
        ]
        # achieved = -3, total_possible = 1 → raw = -3, clipped to 0
        score = compute_points_score(results)
        assert score == 0.0


class TestComputeBootstrapStats:
    def test_basic(self):
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        stats = compute_bootstrap_stats(values)
        assert "mean" in stats
        assert "bootstrap_std" in stats
        assert "n_samples" in stats
        assert stats["n_samples"] == 5
        assert abs(stats["mean"] - 0.7) < 1e-9

    def test_empty(self):
        stats = compute_bootstrap_stats([])
        assert stats["mean"] == 0.0
        assert stats["n_samples"] == 0

    def test_single_value(self):
        stats = compute_bootstrap_stats([0.42])
        assert abs(stats["mean"] - 0.42) < 1e-9
        assert stats["bootstrap_std"] == 0.0


class TestAggregateTagScores:
    def test_example_tags(self):
        results = [
            {"score": 0.8, "example_tags": ["safety"], "rubric_results": []},
            {"score": 0.6, "example_tags": ["safety"], "rubric_results": []},
        ]
        tags = aggregate_tag_scores(results)
        assert "safety" in tags
        assert abs(tags["safety"] - 0.7) < 1e-9

    def test_rubric_tags(self):
        results = [
            {
                "score": 1.0,
                "example_tags": [],
                "rubric_results": [
                    {"points": 2, "criteria_met": True, "tags": ["accuracy"]},
                    {"points": 1, "criteria_met": False, "tags": ["completeness"]},
                ],
            }
        ]
        tags = aggregate_tag_scores(results)
        assert "accuracy" in tags
        assert tags["accuracy"] == 1.0
        assert "completeness" in tags
        assert tags["completeness"] == 0.0

    def test_skips_none_scores(self):
        results = [
            {"score": None, "example_tags": ["x"], "rubric_results": []},
        ]
        tags = aggregate_tag_scores(results)
        assert len(tags) == 0
