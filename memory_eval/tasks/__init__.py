from importlib import import_module

from memory_eval.tasks.base import BaseTask, TaskResult
from memory_eval.tasks.registry import TaskRegistry


def register_builtin_tasks() -> None:
	"""Import bundled task modules so they register themselves."""
	import_module("memory_eval.tasks.mm_lifelong")
	import_module("memory_eval.tasks.healthbench")


__all__ = ["BaseTask", "TaskResult", "TaskRegistry", "register_builtin_tasks"]
