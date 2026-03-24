"""HuggingFace Transformers model backend."""

from typing import Any, Dict, List, Optional

from memory_eval.models.base import BaseModel
from memory_eval.models.registry import ModelRegistry


@ModelRegistry.register("hf")
class HuggingFaceModel(BaseModel):
    """Model backend using HuggingFace Transformers."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: str = "auto",
        **kwargs,
    ):
        super().__init__(model_name)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch required: pip install memory-eval[hf]"
            ) from exc

        import torch as _torch

        _device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        _dtype = getattr(_torch, torch_dtype) if torch_dtype != "auto" else "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            # trust_remote_code=True, --- IGNORE ---
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=_device,
            torch_dtype=_dtype,
            # trust_remote_code=True, --- IGNORE ---
        )
        self._model.eval()

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        import torch

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-7),
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
