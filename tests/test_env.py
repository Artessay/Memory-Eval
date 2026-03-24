"""Tests for environment loading and secret precedence."""

import os
from unittest.mock import patch


class TestProjectEnv:
    def test_load_project_env_reads_dotenv_without_overriding_existing_env(self, tmp_path, monkeypatch):
        from memory_eval.utils.env import load_project_env

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")

        load_project_env(force=True)

        assert os.environ["OPENAI_API_KEY"] == "from-env"

    def test_load_project_env_sets_missing_env_from_dotenv(self, tmp_path, monkeypatch):
        from memory_eval.utils.env import load_project_env

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        load_project_env(force=True)

        assert os.environ["OPENAI_API_KEY"] == "from-dotenv"

    def test_openai_model_prefers_injected_env_over_dotenv(self, tmp_path, monkeypatch):
        from memory_eval.models.openai_model import OpenAIModel

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")

        with patch("openai.OpenAI") as mock_openai:
            OpenAIModel(model_name="gpt-4o")

        mock_openai.assert_called_once_with(api_key="from-env", base_url=None)