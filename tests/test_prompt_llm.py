"""Разбор промта необязателен, поэтому тесты проверяют мягкую деградацию.

Они намеренно не требуют запущенной Ollama: набор должен проходить на машине,
где локальной модели нет вовсе.
"""
import pytest

import prompt_llm
from prompt_llm import PromptLLMError

DEAD = "http://127.0.0.1:9"  # заведомо закрытый порт


def test_empty_prompt_rejected():
    with pytest.raises(PromptLLMError, match="пустой"):
        prompt_llm.parse_prompt("   ")


def test_overlong_prompt_rejected():
    with pytest.raises(PromptLLMError, match="длиннее"):
        prompt_llm.parse_prompt("а" * (prompt_llm.MAX_PROMPT_CHARS + 1))


def test_unreachable_ollama_gives_readable_error():
    with pytest.raises(PromptLLMError, match="недоступна"):
        prompt_llm.parse_prompt("поверни направо", base_url=DEAD, timeout=2)


def test_probe_helpers_degrade_quietly():
    assert prompt_llm.available_models(base_url=DEAD, timeout=1) == []
    assert prompt_llm.is_available(base_url=DEAD) is False


def test_schema_covers_all_supported_axes():
    from motion import AXES
    allowed = prompt_llm.SCHEMA["properties"]["segments"]["items"]["properties"]["axis"]["enum"]
    assert set(allowed) == set(AXES)


def test_server_reports_llm_status_without_crashing(monkeypatch):
    from fastapi.testclient import TestClient
    import server

    monkeypatch.setattr(server.prompt_llm, "available_models", lambda *a, **k: [])
    response = TestClient(server.app).get("/api/llm")
    assert response.status_code == 200
    assert response.json()["available"] is False
