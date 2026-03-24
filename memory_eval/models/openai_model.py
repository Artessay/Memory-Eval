"""OpenAI-compatible API model backend."""

import base64
import os
from typing import Any, Dict, List, Optional

from memory_eval.models.base import BaseModel
from memory_eval.models.registry import ModelRegistry


@ModelRegistry.register("openai")
class OpenAIModel(BaseModel):
    """Model backend for OpenAI and OpenAI-compatible APIs."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc

        resolved_key = api_key or (
            os.environ.get(api_key_env) if api_key_env else None
        ) or os.environ.get("OPENAI_API_KEY")

        self._client = OpenAI(api_key=resolved_key, base_url=base_url)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""
