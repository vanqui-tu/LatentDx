"""TextMAS prompts, generation adapters, and answer parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Protocol
from urllib import error as urlerror
from urllib import request as urlrequest

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


class VllmTextGenerator:
    """OpenAI-compatible vLLM chat-completions adapter.

    The server owns model loading and GPU placement. This adapter is only used
    when the runner is given ``--vllm_base_url``; Transformers remains the
    default local backend.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://127.0.0.1:8000",
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str, *, max_new_tokens: int) -> TextGeneration:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        request = urlrequest.Request(
            self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=300.0) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"vLLM request failed at {self._endpoint()}: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("vLLM returned invalid JSON") from exc
        if "error" in body:
            raise RuntimeError(f"vLLM error: {body['error']}")
        try:
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage") or {}
            return TextGeneration(
                text=str(text).strip(),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("vLLM response has no valid chat completion") from exc

    def _endpoint(self) -> str:
        suffix = "/v1/chat/completions"
        return self.base_url if self.base_url.endswith(suffix) else f"{self.base_url}{suffix}"


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
