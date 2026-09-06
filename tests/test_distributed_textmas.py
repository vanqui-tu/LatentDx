from medlatent.distributed import build_agent_prompt, build_host_prompt, parse_textmas_answer


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
