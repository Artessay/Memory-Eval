"""Abstract base class for all model backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModel(ABC):
    """Base class for all model backends."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Generate a response given a list of messages.

        Args:
            messages: A list of message dicts with 'role' and 'content'.
                      Content can be a string or a list of content parts
                      (for multimodal inputs).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated text response.
        """

    def batch_generate(
        self,
        all_messages: List[List[Dict[str, Any]]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[str]:
        """Generate responses for a batch of message lists."""
        return [
            self.generate(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
            for messages in all_messages
        ]
