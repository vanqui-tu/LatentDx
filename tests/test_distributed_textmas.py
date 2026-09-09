import json
from unittest.mock import patch

from medlatent.distributed import VllmTextGenerator, build_agent_prompt, build_host_prompt, parse_textmas_answer


def test_textmas_prompts_match_agent_and_host_contracts():
    agent_prompt = build_agent_prompt(
        hospital_id=2,
        case_disease="Disease A",
        case_phenotype="Ataxia",
        test_phenotype="Seizure",
    )
    host_prompt = build_host_prompt(context="Hospital 2: Disease A because ataxia.", test_phenotype="Seizure")

    assert "Hospital 2" in agent_prompt
    assert "one line" in agent_prompt
    assert "Hospital evidence and independent assessments" in host_prompt
    assert "<answer>" in host_prompt


def test_textmas_answer_parser_prefers_answer_tags_and_has_line_fallback():
    assert parse_textmas_answer("reason\n<answer>\nDisease A\n</answer>") == "Disease A"
    assert parse_textmas_answer("Disease B\nBecause...") == "Disease B"


def test_vllm_generator_uses_openai_compatible_chat_contract():
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "<answer>Disease A</answer>"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            }).encode()

    def request(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode())
        seen["timeout"] = timeout
        return Response()

    with patch("medlatent.distributed.textmas.urlrequest.urlopen", request):
        generator = VllmTextGenerator("Qwen/Qwen3-4B-Instruct-2507", base_url="http://localhost:8000")
        result = generator.generate("system", "user", max_new_tokens=16)

    assert seen["url"] == "http://localhost:8000/v1/chat/completions"
    assert seen["body"]["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert seen["body"]["temperature"] == 0.0
    assert seen["body"]["seed"] == 42
    assert seen["body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert result.text == "<answer>Disease A</answer>"
    assert (result.prompt_tokens, result.completion_tokens) == (11, 3)
