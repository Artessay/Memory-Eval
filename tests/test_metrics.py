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
    compute_rubric_score,
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
    def test_json_true(self):
        assert parse_rubric_judgment('{"explanation": "ok", "criteria_met": true}') is True

    def test_json_false_in_code_block(self):
        response = '```json\n{"explanation": "missing", "criteria_met": false}\n```'
        assert parse_rubric_judgment(response) is False

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
        assert '"criteria_met"' in prompt
