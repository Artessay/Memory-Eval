"""Task registry for managing available evaluation tasks."""

from typing import Dict, Type

from memory_eval.tasks.base import BaseTask


class TaskRegistry:
    """Registry for task classes."""

    _registry: Dict[str, Type[BaseTask]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a task class."""
        def decorator(task_cls: Type[BaseTask]) -> Type[BaseTask]:
            task_cls.task_name = name
            cls._registry[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseTask]:
        if name not in cls._registry:
            raise ValueError(
                f"Task '{name}' not found. "
                f"Available tasks: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_tasks(cls) -> list:
        return list(cls._registry.keys())
