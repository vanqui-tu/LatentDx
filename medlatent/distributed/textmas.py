"""TextMAS prompts, generation adapters, and answer parsing."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from ..prompts import TEXTMAS_SYSTEM_PROMPT, build_textmas_agent_prompt, build_textmas_host_prompt


@dataclass(frozen=True, slots=True)
class TextGeneration:
    text: str
    prompt_tokens: int
    completion_tokens: int


class TextGenerator(Protocol):
    def generate(self, system_prompt: str, user_prompt: str, *, max_new_tokens: int) -> TextGeneration:
        ...


class TransformersTextGenerator:
    """Greedy Transformers backend for reproducible TextMAS inference."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        dtype_name: str = "bfloat16",
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_name)
        if dtype is None:
            raise ValueError("dtype_name must be bfloat16, float16, or float32")
        resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(resolved_device)
        self.model.eval()
        self.device = resolved_device

    def generate(self, system_prompt: str, user_prompt: str, *, max_new_tokens: int) -> TextGeneration:
        import torch

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            rendered = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = output[0, encoded["input_ids"].shape[1] :]
        return TextGeneration(
            text=self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
            prompt_tokens=int(encoded["input_ids"].shape[1]),
            completion_tokens=int(generated_ids.shape[0]),
        )


def build_agent_prompt(*, hospital_id: int, case_disease: str, case_phenotype: str, test_phenotype: str) -> str:
    return build_textmas_agent_prompt(
        hospital_id=hospital_id,
        case_disease=case_disease,
        case_phenotype=case_phenotype,
        test_phenotype=test_phenotype,
    )


def build_host_prompt(*, context: str, test_phenotype: str) -> str:
    return build_textmas_host_prompt(context=context, test_phenotype=test_phenotype)


def parse_textmas_answer(text: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        answer = " ".join(match.group(1).split())
        return answer or None
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line or None
