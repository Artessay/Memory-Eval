"""Tests for task implementations."""

import pytest
from unittest.mock import MagicMock, patch
from memory_eval.tasks.base import BaseTask, TaskResult
from memory_eval.tasks.registry import TaskRegistry
import memory_eval.tasks.mm_lifelong  # noqa: F401 - register tasks
import memory_eval.tasks.healthbench  # noqa: F401 - register tasks


class TestTaskRegistry:
    def test_tasks_registered(self):
        tasks = TaskRegistry.list_tasks()
        assert "mm_lifelong" in tasks
        assert "healthbench" in tasks

    def test_get_task(self):
        task_cls = TaskRegistry.get("mm_lifelong")
        assert issubclass(task_cls, BaseTask)

    def test_get_unknown_task(self):
        with pytest.raises(ValueError, match="not found"):
            TaskRegistry.get("nonexistent_task")


class TestMMLifelongTask:
    def setup_method(self):
        from memory_eval.tasks.mm_lifelong.task import MMLifelongTask
        self.task = MMLifelongTask()

    def test_task_name(self):
        assert self.task.task_name == "mm_lifelong"

    def test_build_messages_text_only(self):
        sample = {
            "question": "What is shown in the image?",
            "options": ["A cat", "A dog", "A bird", "A fish"],
            "answer": "A",
        }
        messages = self.task.build_messages(sample)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_format_options_list(self):
        sample = {
            "question": "Which is correct?",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "answer": "B",
        }
        messages = self.task.build_messages(sample)
        user_content = messages[1]["content"]
        # content is a list of parts
        text_part = next(p for p in user_content if p["type"] == "text")
        assert "A." in text_part["text"]
        assert "B." in text_part["text"]

    def test_get_reference(self):
        sample = {"answer": "b"}
        assert self.task.get_reference(sample) == "B"

    def test_evaluate(self):
        samples = [
            {"answer": "A"},
            {"answer": "B"},
            {"answer": "C"},
        ]
        predictions = ["A", "B", "D"]
        metrics = self.task.evaluate(samples, predictions)
        assert "accuracy" in metrics
        assert abs(metrics["accuracy"] - 2/3) < 1e-9

    def test_evaluate_with_categories(self):
        samples = [
            {"answer": "A", "category": "temporal"},
            {"answer": "B", "category": "temporal"},
            {"answer": "C", "category": "spatial"},
        ]
        predictions = ["A", "B", "D"]
        metrics = self.task.evaluate(samples, predictions)
        assert "accuracy" in metrics
        assert "accuracy_temporal" in metrics
        assert "accuracy_spatial" in metrics
        assert metrics["accuracy_temporal"] == 1.0
        assert metrics["accuracy_spatial"] == 0.0


class TestHealthBenchTask:
    def setup_method(self):
        from memory_eval.tasks.healthbench.task import HealthBenchTask
        self.task = HealthBenchTask()

    def test_task_name(self):
        assert self.task.task_name == "healthbench"

    def test_build_messages_with_conversation(self):
        sample = {
            "conversation": [
                {"role": "user", "content": "I have a headache, what should I do?"}
            ],
            "rubrics": [{"criterion": "Suggests OTC pain relief", "weight": 1.0}],
        }
        messages = self.task.build_messages(sample)
        assert messages[0]["role"] == "system"
        assert any(m["role"] == "user" for m in messages)

    def test_build_messages_with_prompt(self):
        sample = {
            "prompt": "What are the symptoms of diabetes?",
            "rubrics": [],
        }
        messages = self.task.build_messages(sample)
        assert len(messages) == 2
        assert messages[1]["content"] == "What are the symptoms of diabetes?"

    def test_evaluate_without_grader(self):
        samples = [{"rubrics": [{"criterion": "test", "weight": 1.0}]}]
        predictions = ["This is a response with several words."]
        metrics = self.task.evaluate(samples, predictions)
        assert "avg_response_length" in metrics

    def test_evaluate_with_grader(self):
        mock_grader = MagicMock()
        mock_grader.generate.return_value = "Yes, this criterion is met."
        self.task.set_grader_model(mock_grader)

        samples = [
            {
                "conversation": [{"role": "user", "content": "headache?"}],
                "rubrics": [
                    {"criterion": "Recommends rest", "weight": 1.0},
                    {"criterion": "Suggests hydration", "weight": 1.0},
                ],
            }
        ]
        predictions = ["Rest and drink water."]
        metrics = self.task.evaluate(samples, predictions)
        assert "rubric_score" in metrics
        assert metrics["rubric_score"] == 1.0

    def test_subset_config(self):
        from memory_eval.tasks.healthbench.task import HealthBenchTask
        task = HealthBenchTask(config={"subset": "consensus"})
        assert task.config["subset"] == "consensus"


class TestBaseTaskResult:
    def test_task_result_creation(self):
        result = TaskResult(
            task_name="test",
            metrics={"accuracy": 0.8},
            num_samples=100,
            predictions=["A", "B"],
            references=["A", "C"],
        )
        assert result.task_name == "test"
        assert result.metrics["accuracy"] == 0.8
        assert result.num_samples == 100
