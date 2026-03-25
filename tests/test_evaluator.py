from pathlib import Path
from unittest.mock import MagicMock

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
                        "rubrics": [{"criterion": "Recommends rest", "weight": 1.0}],
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
        assert evaluated_records[0]["prediction"] == "Rest and hydrate."
        assert evaluated_records[0]["evaluation"]["rubric_score"] == 1.0