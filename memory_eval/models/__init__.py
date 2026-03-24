from importlib import import_module

from memory_eval.models.base import BaseModel
from memory_eval.models.registry import ModelRegistry


def register_builtin_models() -> None:
	"""Import bundled model modules so they register themselves."""
	import_module("memory_eval.models.openai_model")
	import_module("memory_eval.models.hf_model")


__all__ = ["BaseModel", "ModelRegistry", "register_builtin_models"]
