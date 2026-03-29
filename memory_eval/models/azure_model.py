"""Azure OpenAI model backend."""

import os
from typing import Any, Dict, List, Optional

from memory_eval.models.base import BaseModel
from memory_eval.models.registry import ModelRegistry
from memory_eval.utils.env import load_project_env


@ModelRegistry.register("azure")
class AzureOpenAIModel(BaseModel):
    """Model backend for Azure OpenAI."""

    def __init__(
        self,
        model_name: str,
        azure_endpoint: Optional[str] = None,
        endpoint_env: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        api_version: Optional[str] = None,
        credential_scope: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        load_project_env()
        self._default_generate_kwargs = dict(kwargs)

        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc

        resolved_endpoint = (
            azure_endpoint
            or (os.environ.get(endpoint_env) if endpoint_env else None)
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not resolved_endpoint:
            raise ValueError(
                "Azure endpoint not configured. Set AZURE_OPENAI_ENDPOINT, "
                "or pass azure_endpoint."
            )

        resolved_key = api_key or (
            os.environ.get(api_key_env) if api_key_env else None
        ) or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_api_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or os.environ.get("OPENAI_API_VERSION")
        )
        if not resolved_api_version:
            # raise warning for backward compatibility, since API version is required for Azure but not for OpenAI
            print(
                "Azure API version not configured. Set AZURE_OPENAI_API_VERSION "
                "with default value `2024-08-01-preview`."
            )
            resolved_api_version = "2024-08-01-preview"

        client_kwargs = {
            "azure_endpoint": resolved_endpoint,
            "api_version": resolved_api_version,
        }

        if resolved_key:
            client_kwargs["api_key"] = resolved_key
        else:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except ImportError as exc:
                raise ImportError(
                    "azure-identity package required for Azure AD auth: pip install azure-identity"
                ) from exc

            token_scope = credential_scope or os.environ.get("AZURE_OPENAI_SCOPE") or "https://cognitiveservices.azure.com/.default"
            client_kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(),
                token_scope,
            )

        self._client = AzureOpenAI(**client_kwargs)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        merged = {**self._default_generate_kwargs, **kwargs}
        response = self.retry_handler.run(
            lambda: self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **merged,
            )
        )
        return response.choices[0].message.content or ""

if __name__ == "__main__":
    model = AzureOpenAIModel(model_name="gpt-4o")
    messages = [{'role': 'user', 'content': 'What are the differences between Azure Machine Learning and Azure AI services?'}]
    response = model.generate(messages)
    print(response)