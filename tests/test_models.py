"""Tests for model backends and registry."""

import sys
import types

import pytest
from unittest.mock import MagicMock, patch
from memory_eval.models import register_builtin_models
from memory_eval.models.base import BaseModel
from memory_eval.models.registry import ModelRegistry
from memory_eval.utils.retry import RetryHandler


register_builtin_models()


class TestModelRegistry:
    def test_backends_registered(self):
        backends = ModelRegistry.list_backends()
        assert "azure" in backends
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
            model.retry_handler = RetryHandler(max_retries=2, base_delay=0.0, max_delay=0.0, jitter=0.0)

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
            model.retry_handler = RetryHandler(max_retries=2, base_delay=0.0, max_delay=0.0, jitter=0.0)

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

class TestAzureOpenAIModel:
    def test_generate(self):
        with patch("memory_eval.models.azure_model.AzureOpenAIModel.__init__", return_value=None):
            from memory_eval.models.azure_model import AzureOpenAIModel

            model = AzureOpenAIModel.__new__(AzureOpenAIModel)
            model.model_name = "gpt-4o"
            model.retry_handler = RetryHandler(max_retries=2, base_delay=0.0, max_delay=0.0, jitter=0.0)

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Azure answer"
            mock_client.chat.completions.create.return_value = mock_response
            model._client = mock_client

            messages = [{"role": "user", "content": "Ping"}]
            result = model.generate(messages)
            assert result == "Azure answer"

    def test_init_prefers_api_key_when_available(self):
        from memory_eval.models.azure_model import AzureOpenAIModel

        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "secret",
                "AZURE_OPENAI_API_VERSION": "2024-10-21",
            },
            clear=True,
        ):
            with patch("openai.AzureOpenAI") as mock_client:
                AzureOpenAIModel(model_name="gpt-4o")

        mock_client.assert_called_once_with(
            azure_endpoint="https://example.openai.azure.com/",
            api_key="secret",
            api_version="2024-10-21",
        )

    def test_init_uses_aad_when_api_key_missing(self):
        from memory_eval.models.azure_model import AzureOpenAIModel

        token_provider = object()
        mock_credential = MagicMock()
        azure_identity = types.ModuleType("azure.identity")
        azure_identity.DefaultAzureCredential = MagicMock(return_value=mock_credential)
        azure_identity.get_bearer_token_provider = MagicMock(return_value=token_provider)
        azure_package = types.ModuleType("azure")
        azure_package.identity = azure_identity

        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
                "OPENAI_API_VERSION": "2024-10-21",
            },
            clear=True,
        ):
            with patch.dict(
                sys.modules,
                {
                    "azure": azure_package,
                    "azure.identity": azure_identity,
                },
            ):
                with patch("openai.AzureOpenAI") as mock_client:
                    AzureOpenAIModel(model_name="gpt-4o")

        azure_identity.DefaultAzureCredential.assert_called_once_with()
        azure_identity.get_bearer_token_provider.assert_called_once_with(
            mock_credential,
            "https://cognitiveservices.azure.com/.default",
        )
        mock_client.assert_called_once_with(
            azure_endpoint="https://example.openai.azure.com/",
            azure_ad_token_provider=token_provider,
            api_version="2024-10-21",
        )
