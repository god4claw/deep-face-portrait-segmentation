"""Опциональный разбор промта локальной моделью через Ollama.

Строго необязательный слой. Источник истины - таймлайн сегментов; эта модель
только заполняет его черновиком. Если Ollama не запущена или модель не стоит,
функция возвращает понятную ошибку и не роняет ни сервер, ни UI.

Почему не регулярки: предыдущая версия разбирала текст набором regex и молча
теряла команды ("поверни" ловилось как "вернуться"). Локальная модель со
структурированным выводом либо возвращает валидный JSON, либо честно падает.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests

OLLAMA_URL = "http://127.0.0.1:11434"
# По бенчмарку (tools/bench_llm.py): смысл 21/24 против 15/24 у qwen3.5:2b
# при сопоставимой задержке. Схему обе держат идеально, разница в понимании.
DEFAULT_MODEL = "gemma4:e2b"
TIMEOUT_SEC = 60
MAX_PROMPT_CHARS = 2000

# Ollama умеет structured output: модель обязана вернуть JSON этой формы.
SCHEMA = {
    "type": "object",
    "properties": {
        "duration": {"type": "number"},
        "blink_rate_per_min": {"type": "number"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "axis": {"type": "string", "enum": ["yaw", "pitch", "roll", "zoom"]},
                    "frm": {"type": "number"},
                    "to": {"type": "number"},
                    "start": {"type": "number"},
                    "duration": {"type": "number"},
                },
                "required": ["axis", "frm", "to", "start", "duration"],
            },
        },
    },
    "required": ["segments"],
}

SYSTEM = """Ты переводишь описание движения головы в JSON-таймлайн. Отвечай только JSON.

Оси и знаки:
  yaw   поворот влево-вправо. Положительное значение - ВПРАВО, отрицательное - ВЛЕВО.
  pitch наклон вверх-вниз. Положительное - ВВЕРХ, отрицательное - ВНИЗ.
  roll  наклон головы к плечу. Положительное - к правому плечу.
  zoom  приближение в ПРОЦЕНТАХ. Положительное - ближе к камере, отрицательное - дальше.

Поля сегмента:
  frm      угол в начале движения (обычно текущий, для первого движения 0)
  to       угол в конце движения
  start    секунда начала
  duration длительность движения в секундах

Правила:
  Комфортный поворот 20-40 градусов занимает 0.5-0.6 с, медленный до 1.0 с, резкий 0.3 с.
  Первое движение начинай не раньше 0.5 с, между движениями оставляй паузу около 0.4 с.
  Если размер не указан - бери 30 градусов для поворота и 10 процентов для zoom.
  Цепочку движений по одной оси связывай: frm следующего равен to предыдущего.
  Возврат в исходное - это сегмент с to равным 0.
  blink_rate_per_min: покой 17, разговор 23, чтение 6, волнение 32.
  duration - общая длина клипа в секундах, по умолчанию с запасом 0.7 с после конца.

Пример. Запрос: "поверни направо на 30, подержи, вернись, моргай как в разговоре"
{"duration":4.8,"blink_rate_per_min":23,"segments":[
{"axis":"yaw","frm":0,"to":30,"start":0.6,"duration":0.55},
{"axis":"yaw","frm":30,"to":0,"start":3.0,"duration":0.55}]}"""


class PromptLLMError(RuntimeError):
    """Понятная ошибка вместо падения: UI показывает её и продолжает работать."""


@dataclass
class LLMResult:
    segments: list
    duration: float | None
    blink_rate_per_min: float | None
    model: str
    elapsed_sec: float
    raw: str


def available_models(base_url: str = OLLAMA_URL, timeout: float = 3.0) -> list:
    """Список установленных моделей. Пустой список = Ollama недоступна."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        return [item["name"] for item in response.json().get("models", [])]
    except Exception:
        return []


def is_available(model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL) -> bool:
    names = available_models(base_url)
    return any(name == model or name.startswith(model.split(":")[0] + ":") for name in names)


def parse_prompt(text: str, *, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL,
                 timeout: int = TIMEOUT_SEC) -> LLMResult:
    """Промт -> черновик таймлайна. Бросает PromptLLMError с внятным текстом."""
    text = (text or "").strip()
    if not text:
        raise PromptLLMError("пустой запрос")
    if len(text) > MAX_PROMPT_CHARS:
        raise PromptLLMError(f"запрос длиннее {MAX_PROMPT_CHARS} символов")

    payload = {
        "model": model,
        "prompt": text,
        "system": SYSTEM,
        "format": SCHEMA,
        "stream": False,
        # У рассуждающих моделей (qwen3.x) при включённом think ответ уезжает
        # в поле thinking, а response приходит пустым. Нам рассуждение не нужно.
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 700},
    }

    started = time.perf_counter()
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise PromptLLMError(f"Ollama недоступна на {base_url} - запусти 'ollama serve'") from exc
    except requests.exceptions.Timeout as exc:
        raise PromptLLMError(f"модель {model} не ответила за {timeout} с") from exc
    except requests.exceptions.HTTPError as exc:
        detail = getattr(exc.response, "text", "")[:200]
        raise PromptLLMError(f"Ollama вернула ошибку: {detail}") from exc
    elapsed = time.perf_counter() - started

    body = response.json()
    raw = body.get("response") or ""
    if not raw.strip():
        # запасной путь: не все сборки уважают think=false, тогда JSON лежит в thinking
        raw = body.get("thinking") or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptLLMError(f"модель вернула не JSON: {raw[:200]}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise PromptLLMError("в ответе нет списка segments")

    segments = []
    for item in data["segments"][:64]:
        if not isinstance(item, dict):
            continue
        try:
            segments.append({
                "target": "face",
                "axis": str(item["axis"]),
                "frm": float(item["frm"]),
                "to": float(item["to"]),
                "start": max(0.0, float(item["start"])),
                "duration": max(0.05, float(item["duration"])),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if not segments:
        raise PromptLLMError("модель не вернула ни одного корректного сегмента")

    def number(key):
        value = data.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return LLMResult(segments=segments, duration=number("duration"),
                     blink_rate_per_min=number("blink_rate_per_min"),
                     model=model, elapsed_sec=elapsed, raw=raw)
