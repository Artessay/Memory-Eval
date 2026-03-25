"""Tests for the Memory-Eval CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memory_eval.cli import (
    DEFAULT_MODEL_CONFIG_PATH,
    _resolve_evaluate_output_file,
    _resolve_model_spec,
    _resolve_result_file,
    main,
)


class TestModelConfigResolution:
    def test_resolve_model_spec_from_default_config(self):
        backend, model_name, model_kwargs = _resolve_model_spec(
            model_config="azure-gpt-4o",
            model_backend=None,
            model_name=None,
            base_url=None,
            api_key_env=None,
            config_path=DEFAULT_MODEL_CONFIG_PATH,
        )

        assert backend == "azure"
        assert model_name == "gpt-4o"
        assert model_kwargs["endpoint_env"] == "AZURE_OPENAI_ENDPOINT"
        assert model_kwargs["api_key_env"] == "AZURE_OPENAI_API_KEY"
        assert model_kwargs["api_version"] == "2024-08-01-preview"

    def test_resolve_model_spec_allows_cli_override(self):
        backend, model_name, model_kwargs = _resolve_model_spec(
            model_config="gemini-2.0-flash",
            model_backend=None,
            model_name=None,
            base_url="https://override.example/v1",
            api_key_env="CUSTOM_API_KEY",
            config_path=DEFAULT_MODEL_CONFIG_PATH,
        )

        assert backend == "openai"
        assert model_name == "gemini-2.0-flash"
        assert model_kwargs["base_url"] == "https://override.example/v1"
        assert model_kwargs["api_key_env"] == "CUSTOM_API_KEY"

    def test_resolve_model_spec_requires_complete_input(self):
        with patch("memory_eval.cli._load_model_configs", return_value={}):
            try:
                _resolve_model_spec(
                    model_config=None,
                    model_backend="openai",
                    model_name=None,
                    base_url=None,
                    api_key_env=None,
                    config_path=DEFAULT_MODEL_CONFIG_PATH,
                )
            except Exception as exc:
                assert "--model-config or both --model-backend and --model-name" in str(exc)
            else:
                raise AssertionError("Expected incomplete model specification to fail")


class TestResultPathResolution:
    def test_resolve_result_file_uses_healthbench_default_subset(self, tmp_path: Path):
        result_path = tmp_path / "results" / "healthbench" / "azure" / "gpt-4o" / "subset-standard__max_tokens-4096__temperature-0.0__limit-10.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}", encoding="utf-8")

        resolved_path, backend, model_name = _resolve_result_file(
            result_file=None,
            task_name="healthbench",
            subset=None,
            model_config=None,
            model_backend="azure",
            model_name="gpt-4o",
            base_url=None,
            api_key_env=None,
            output_dir=str(tmp_path / "results"),
            max_tokens=4096,
            temperature=0.0,
            limit=10,
        )

        assert resolved_path == result_path
        assert backend == "azure"
        assert model_name == "gpt-4o"

    def test_resolve_result_file_from_run_parameters(self, tmp_path: Path):
        result_path = tmp_path / "results" / "healthbench" / "azure" / "gpt-4o" / "subset-standard__max_tokens-4096__temperature-0.0__limit-10.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}", encoding="utf-8")

        resolved_path, backend, model_name = _resolve_result_file(
            result_file=None,
            task_name="healthbench",
            subset="standard",
            model_config=None,
            model_backend="azure",
            model_name="gpt-4o",
            base_url=None,
            api_key_env=None,
            output_dir=str(tmp_path / "results"),
            max_tokens=4096,
            temperature=0.0,
            limit=10,
        )

        assert resolved_path == result_path
        assert backend == "azure"
        assert model_name == "gpt-4o"

    def test_resolve_evaluate_output_file_defaults_to_evaluated_tree(self):
        output_path = _resolve_evaluate_output_file(
            output_file=None,
            result_file=Path("results/healthbench/azure/gpt-4o/subset-standard__max_tokens-4096__temperature-0.0__limit-10.json"),
            grader_backend="azure",
            grader_model="gpt-4o",
            evaluated_output_dir="results/evaluated",
        )

        assert output_path == Path(
            "results/evaluated/healthbench/azure/gpt-4o/subset-standard__max_tokens-4096__temperature-0.0__limit-10__graded-by-azure-gpt-4o.json"
        )


class TestCliEvaluate:
    def test_evaluate_saved_healthbench_result_with_grader(self, tmp_path: Path):
        result_file = tmp_path / "healthbench.json"
        result_file.write_text(
            json.dumps(
                {
                    "task_name": "healthbench",
                    "task_config": {"subset": "hard"},
                    "predictions": ["Rest and stay hydrated."],
                    "num_samples": 1,
                    "evaluation_samples": [
                        {
                            "conversation": [{"role": "user", "content": "headache?"}],
                            "rubrics": [{"criterion": "Recommends rest", "weight": 1.0}],
                            "criteria": [],
                        }
                    ],
                    "metrics": {},
                    "evaluations": {},
                    "references": ["1 rubrics"],
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        mock_grader = MagicMock()
        mock_grader.generate.return_value = '{"explanation": "criterion satisfied", "criteria_met": true}'

        with patch("memory_eval.cli.register_builtin_models"), patch("memory_eval.cli.register_builtin_tasks"), patch(
            "memory_eval.models.registry.ModelRegistry.get"
        ) as mock_get_model, patch("memory_eval.tasks.registry.TaskRegistry.get") as mock_get_task:
            from memory_eval.tasks.healthbench.task import HealthBenchTask

            mock_get_model.return_value = MagicMock(return_value=mock_grader)
            mock_get_task.return_value = HealthBenchTask

            result = runner.invoke(
                main,
                [
                    "evaluate",
                    "--result-file",
                    str(result_file),
                    "--grader-backend",
                    "openai",
                    "--grader-model",
                    "gpt-4o",
                ],
            )

        assert result.exit_code == 0, result.output
        evaluated_path = _resolve_evaluate_output_file(
            output_file=None,
            result_file=result_file,
            grader_backend="openai",
            grader_model="gpt-4o",
            evaluated_output_dir="results/evaluated",
        )
        updated = json.loads(evaluated_path.read_text(encoding="utf-8"))
        assert updated["metrics"]["rubric_score"] == 1.0
        assert "grader__healthbench__openai__gpt-4o" in updated["evaluations"]

    def test_evaluate_can_locate_result_file_from_parameters(self, tmp_path: Path):
        result_path = tmp_path / "results" / "healthbench" / "azure" / "gpt-4o" / "subset-standard__max_tokens-4096__temperature-0.0__limit-10.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "task_name": "healthbench",
                    "task_config": {"subset": "standard"},
                    "predictions": ["Rest and stay hydrated."],
                    "num_samples": 1,
                    "evaluation_samples": [
                        {
                            "conversation": [{"role": "user", "content": "headache?"}],
                            "rubrics": [{"criterion": "Recommends rest", "weight": 1.0}],
                            "criteria": [],
                        }
                    ],
                    "metrics": {},
                    "evaluations": {},
                    "references": ["1 rubrics"],
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        mock_grader = MagicMock()
        mock_grader.generate.return_value = '{"explanation": "criterion satisfied", "criteria_met": true}'

        with patch("memory_eval.cli.register_builtin_models"), patch("memory_eval.cli.register_builtin_tasks"), patch(
            "memory_eval.models.registry.ModelRegistry.get"
        ) as mock_get_model, patch("memory_eval.tasks.registry.TaskRegistry.get") as mock_get_task:
            from memory_eval.tasks.healthbench.task import HealthBenchTask

            mock_get_model.return_value = MagicMock(return_value=mock_grader)
            mock_get_task.return_value = HealthBenchTask

            result = runner.invoke(
                main,
                [
                    "evaluate",
                    "--task",
                    "healthbench",
                    "--subset",
                    "standard",
                    "--model-backend",
                    "azure",
                    "--model-name",
                    "gpt-4o",
                    "--output-dir",
                    str(tmp_path / "results"),
                    "--evaluated-output-dir",
                    str(tmp_path / "results" / "evaluated"),
                    "--max-tokens",
                    "4096",
                    "--limit",
                    "10",
                    "--grader-backend",
                    "azure",
                    "--grader-model",
                    "gpt-4o",
                ],
            )

        assert result.exit_code == 0, result.output
        evaluated_path = tmp_path / "results" / "evaluated" / "healthbench" / "azure" / "gpt-4o" / "subset-standard__max_tokens-4096__temperature-0.0__limit-10__graded-by-azure-gpt-4o.json"
        assert evaluated_path.exists()
        updated = json.loads(evaluated_path.read_text(encoding="utf-8"))
        assert updated["metrics"]["rubric_score"] == 1.0

    def test_evaluate_saved_healthbench_result_with_grader_config(self, tmp_path: Path):
        result_file = tmp_path / "healthbench.json"
        result_file.write_text(
            json.dumps(
                {
                    "task_name": "healthbench",
                    "task_config": {"subset": "hard"},
                    "predictions": ["Rest and stay hydrated."],
                    "num_samples": 1,
                    "evaluation_samples": [
                        {
                            "conversation": [{"role": "user", "content": "headache?"}],
                            "rubrics": [{"criterion": "Recommends rest", "weight": 1.0}],
                            "criteria": [],
                        }
                    ],
                    "metrics": {},
                    "evaluations": {},
                    "references": ["1 rubrics"],
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        mock_grader = MagicMock()
        mock_grader.generate.return_value = '{"explanation": "criterion satisfied", "criteria_met": true}'

        with patch("memory_eval.cli.register_builtin_models"), patch("memory_eval.cli.register_builtin_tasks"), patch(
            "memory_eval.tasks.registry.TaskRegistry.get"
        ) as mock_get_task, patch("memory_eval.cli._build_model", return_value=mock_grader) as mock_build_model:
            from memory_eval.tasks.healthbench.task import HealthBenchTask

            mock_get_task.return_value = HealthBenchTask

            result = runner.invoke(
                main,
                [
                    "evaluate",
                    "--result-file",
                    str(result_file),
                    "--grader-config",
                    "azure-gpt-4o",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_build_model.assert_called_once()
        assert mock_build_model.call_args.kwargs["model_backend"] == "azure"
        assert mock_build_model.call_args.kwargs["model_name"] == "gpt-4o"

    def test_evaluate_requires_complete_grader_config(self, tmp_path: Path):
        result_file = tmp_path / "result.json"
        result_file.write_text(
            json.dumps(
                {
                    "task_name": "mm_lifelong",
                    "predictions": ["A"],
                    "num_samples": 1,
                    "evaluation_samples": [{"answer": "A"}],
                    "metrics": {},
                    "evaluations": {},
                    "references": ["A"],
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--result-file", str(result_file), "--grader-backend", "openai"])
        assert result.exit_code != 0
        assert "--model-config or both --model-backend and --model-name" in result.output