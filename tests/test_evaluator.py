from pathlib import Path
from unittest.mock import MagicMock

import memory_eval.evaluator as evaluator_module
from memory_eval.evaluator import Evaluator, build_records_path
from memory_eval.tasks.base import BaseTask
from memory_eval.tasks.healthbench.task import HealthBenchTask
from memory_eval.utils.io import load_json, load_jsonl, save_json, save_jsonl


class DummyTask(BaseTask):
    task_name = "dummy"

    def load_dataset(self):
        return [{"question": "What is 1+1?", "answer": "2"}]

    def build_messages(self, sample):
        return [{"role": "user", "content": sample["question"]}]

    def evaluate(self, samples, predictions):
        return {"accuracy": float(predictions == ["2"])}


class DummyModel:
    model_name = "dummy-model"

    def generate(self, messages, max_tokens=4096, temperature=0.0):
        return "2"


class TestEvaluatorResultStorage:
    def test_run_task_writes_sidecar_jsonl_and_lightweight_summary(self, tmp_path: Path):
        evaluator = Evaluator(
            model=DummyModel(),
            output_dir=str(tmp_path / "results"),
            model_backend="openai",
        )

        result = evaluator.run_task(DummyTask())

        result_path = Path(result.metadata["result_path"])
        summary = load_json(str(result_path))
        records = load_jsonl(build_records_path(str(result_path)))

        assert "predictions" not in summary
        assert "references" not in summary
        assert "evaluation_samples" not in summary
        assert summary["metadata"]["sample_results_path"].endswith(".jsonl")

        assert records == [
            {
                "prediction": "2",
                "reference": "2",
                "evaluation_sample": {"question": "What is 1+1?", "answer": "2"},
            }
        ]

    def test_evaluate_result_reads_sidecar_records_and_writes_evaluated_sidecar(self, tmp_path: Path):
        result_path = (
            tmp_path
            / "results"
            / "healthbench"
            / "openai"
            / "gpt-4o"
            / "subset-standard__max_tokens-128__temperature-0.0__limit-1.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "schema_version": 2,
                "task_name": "healthbench",
                "task_config": {"subset": "standard"},
                "model": {"backend": "openai", "model_name": "gpt-4o"},
                "generation_config": {"max_tokens": 128, "temperature": 0.0, "limit": 1},
                "metrics": {},
                "num_samples": 1,
                "evaluations": {},
                "metadata": {},
            },
            str(result_path),
        )
        save_jsonl(
            [
                {
                    "prediction": "Rest and hydrate.",
                    "reference": "1 rubrics",
                    "evaluation_sample": {
                        "conversation": [{"role": "user", "content": "headache?"}],
                        "prompt": None,
                        "question": None,
                        "rubrics": [{"criterion": "Recommends rest", "points": 5}],
                        "criteria": [],
                    },
                }
            ],
            build_records_path(str(result_path)),
        )

        task = HealthBenchTask(config={"subset": "standard"})
        mock_grader = MagicMock()
        mock_grader.generate.return_value = '{"explanation": "criterion satisfied", "criteria_met": true}'
        task.set_grader_model(mock_grader)

        evaluated_path = (
            tmp_path
            / "results"
            / "evaluated"
            / "healthbench"
            / "openai"
            / "gpt-4o"
            / "subset-standard__max_tokens-128__temperature-0.0__limit-1__graded-by-openai-gpt-4o.json"
        )
        evaluator = Evaluator()

        updated = evaluator.evaluate_result(
            result_path=str(result_path),
            task=task,
            evaluation_name="grader__healthbench__openai__gpt-4o",
            grader_info={"backend": "openai", "model_name": "gpt-4o"},
            output_path=str(evaluated_path),
        )

        assert updated["metrics"]["rubric_score"] == 1.0
        assert "predictions" not in updated
        assert evaluated_path.exists()

        evaluated_records = load_jsonl(build_records_path(str(evaluated_path)))
        assert evaluated_records == [{"evaluation": {"rubric_score": 1.0, "criterion_results": [{"criterion": "Recommends rest", "weight": 1.0, "criteria_met": True}]}}]
        assert evaluated_records[0]["evaluation"]["rubric_score"] == 1.0

    def test_evaluate_result_resumes_from_compact_sidecar_records(self, tmp_path: Path):
        result_path = tmp_path / "results" / "dummy" / "openai" / "dummy-model" / "max_tokens-128__temperature-0.0__limit-2.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "schema_version": 2,
                "task_name": "dummy",
                "task_config": {},
                "model": {"backend": "openai", "model_name": "dummy-model"},
                "generation_config": {"max_tokens": 128, "temperature": 0.0, "limit": 2},
                "metrics": {},
                "num_samples": 2,
                "evaluations": {},
                "metadata": {},
            },
            str(result_path),
        )
        save_jsonl(
            [
                {
                    "prediction": "2",
                    "reference": "2",
                    "evaluation_sample": {"question": "What is 1+1?", "answer": "2"},
                },
                {
                    "prediction": "3",
                    "reference": "4",
                    "evaluation_sample": {"question": "What is 2+2?", "answer": "4"},
                },
            ],
            build_records_path(str(result_path)),
        )

        evaluated_path = tmp_path / "results" / "evaluated" / "dummy" / "openai" / "dummy-model" / "max_tokens-128__temperature-0.0__limit-2__graded.json"
        save_jsonl(
            [{"evaluation": {"correct": True}}],
            build_records_path(str(evaluated_path)),
        )

        class DummyEvaluationTask(DummyTask):
            def evaluate_record(self, record):
                return {"correct": record.get("prediction") == record.get("reference")}

            def aggregate_metrics_from_records(self, records):
                return {
                    "accuracy": sum(1 for record in records if record["evaluation"]["correct"]) / len(records)
                }

        evaluator = Evaluator()
        updated = evaluator.evaluate_result(
            result_path=str(result_path),
            task=DummyEvaluationTask(),
            evaluation_name="default",
            output_path=str(evaluated_path),
        )

        assert updated["metadata"]["resumed_evaluations"] == 1
        assert updated["metrics"]["accuracy"] == 0.5
        assert load_jsonl(build_records_path(str(evaluated_path))) == [
            {"evaluation": {"correct": True}},
            {"evaluation": {"correct": False}},
        ]

    def test_evaluate_result_shows_progress_bar(self, tmp_path: Path, monkeypatch):
        result_path = tmp_path / "results" / "dummy" / "openai" / "dummy-model" / "max_tokens-128__temperature-0.0__limit-1.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "schema_version": 2,
                "task_name": "dummy",
                "task_config": {},
                "model": {"backend": "openai", "model_name": "dummy-model"},
                "generation_config": {"max_tokens": 128, "temperature": 0.0, "limit": 1},
                "metrics": {},
                "num_samples": 1,
                "evaluations": {},
                "metadata": {},
            },
            str(result_path),
        )
        save_jsonl(
            [
                {
                    "prediction": "2",
                    "reference": "2",
                    "evaluation_sample": {"question": "What is 1+1?", "answer": "2"},
                }
            ],
            build_records_path(str(result_path)),
        )

        calls = []

        def fake_tqdm(iterable, **kwargs):
            calls.append(kwargs)
            return iterable

        class DummyEvaluationTask(DummyTask):
            def evaluate_record(self, record):
                return {"correct": True}

            def aggregate_metrics_from_records(self, records):
                return {"accuracy": 1.0}

        monkeypatch.setattr(evaluator_module, "tqdm", fake_tqdm)

        Evaluator().evaluate_result(
            result_path=str(result_path),
            task=DummyEvaluationTask(),
            evaluation_name="default",
            output_path=str(tmp_path / "results" / "evaluated" / "dummy.json"),
        )

        assert calls
        assert calls[0]["desc"] == "Evaluating dummy"
        assert calls[0]["total"] == 1