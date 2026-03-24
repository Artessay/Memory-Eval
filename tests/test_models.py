"""Tests for model backends and registry."""

import pytest
from unittest.mock import MagicMock, patch
from memory_eval.models import register_builtin_models
from memory_eval.models.base import BaseModel
from memory_eval.models.registry import ModelRegistry


register_builtin_models()


class TestModelRegistry:
    def test_backends_registered(self):
        backends = ModelRegistry.list_backends()
        assert "openai" in backends
        assert "hf" in backends

    def test_get_backend(self):
        backend_cls = ModelRegistry.get("openai")
        assert issubclass(backend_cls, BaseModel)

    def test_get_unknown_backend(self):
        with pytest.raises(ValueError, match="not found"):
            ModelRegistry.get("nonexistent_backend")


class TestOpenAIModel:
    def test_generate(self):
        with patch("memory_eval.models.openai_model.OpenAIModel.__init__", return_value=None):
            from memory_eval.models.openai_model import OpenAIModel
            model = OpenAIModel.__new__(OpenAIModel)
            model.model_name = "gpt-4o"

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "B"
            mock_client.chat.completions.create.return_value = mock_response
            model._client = mock_client

            messages = [{"role": "user", "content": "What is 2+2?"}]
            result = model.generate(messages)
            assert result == "B"

    def test_batch_generate(self):
        with patch("memory_eval.models.openai_model.OpenAIModel.__init__", return_value=None):
            from memory_eval.models.openai_model import OpenAIModel
            model = OpenAIModel.__new__(OpenAIModel)
            model.model_name = "gpt-4o"

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Answer"
            mock_client.chat.completions.create.return_value = mock_response
            model._client = mock_client

            all_messages = [
                [{"role": "user", "content": "Q1"}],
                [{"role": "user", "content": "Q2"}],
            ]
            results = model.batch_generate(all_messages)
            assert len(results) == 2
            assert all(r == "Answer" for r in results)
