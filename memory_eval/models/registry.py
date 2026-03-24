"""Model registry for managing available model backends."""

from typing import Dict, Type

from memory_eval.models.base import BaseModel


class ModelRegistry:
    """Registry for model backend classes."""

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model backend class."""
        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        if name not in cls._registry:
            raise ValueError(
                f"Model backend '{name}' not found. "
                f"Available backends: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_backends(cls) -> list:
        return list(cls._registry.keys())
