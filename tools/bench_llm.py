"""Сравнение локальных моделей на разборе промта.

Меряет не только скорость, но и смысл: правильный ли знак у оси, разложилась ли
цепочка на несколько сегментов, связаны ли углы между собой. Модель, которая
отвечает быстро и невпопад, здесь получает ноль.

Запуск:  python tools\\bench_llm.py
         python tools\\bench_llm.py --models qwen3.5:2b gemma4:e2b --repeat 3
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motion import Segment, segments_from_data  # noqa: E402
from prompt_llm import PromptLLMError, available_models, parse_prompt  # noqa: E402


def _axis(segments, axis):
    return [s for s in segments if s["axis"] == axis]


CASES = [
    ("поверни голову направо на 30 градусов",
     lambda s: len(_axis(s, "yaw")) >= 1 and _axis(s, "yaw")[0]["to"] > 15,
     "yaw положительный"),

    ("поверни налево на 20 градусов, потом направо на 40",
     lambda s: len(_axis(s, "yaw")) >= 2 and _axis(s, "yaw")[0]["to"] < 0 and _axis(s, "yaw")[1]["to"] > 0,
     "два сегмента, сначала минус потом плюс"),

    ("наклон 10 градусов вниз и поворот 40 градусов вправо",
     lambda s: bool(_axis(s, "pitch")) and _axis(s, "pitch")[0]["to"] < 0
               and bool(_axis(s, "yaw")) and _axis(s, "yaw")[0]["to"] > 25,
     "число привязано к своей оси"),

    ("приблизься к камере на 15 процентов",
     lambda s: bool(_axis(s, "zoom")) and _axis(s, "zoom")[0]["to"] > 5,
     "zoom положительный"),

    ("отдались от камеры подальше",
     lambda s: bool(_axis(s, "zoom")) and _axis(s, "zoom")[0]["to"] < 0,
     "zoom отрицательный"),

    ("поверни направо на 30, подержи две секунды, потом вернись обратно",
     lambda s: len(_axis(s, "yaw")) >= 2 and _axis(s, "yaw")[0]["to"] > 15
               and abs(_axis(s, "yaw")[-1]["to"]) < 5
               and _axis(s, "yaw")[-1]["start"] >= _axis(s, "yaw")[0]["start"] + 2.0,
     "возврат в ноль с паузой 2 с"),

    ("быстро направо на 45, потом медленно налево на 15",
     lambda s: len(_axis(s, "yaw")) >= 2
               and _axis(s, "yaw")[0]["duration"] < _axis(s, "yaw")[1]["duration"],
     "первый рывок короче второго"),

    ("подними голову вверх на 15 градусов",
     lambda s: bool(_axis(s, "pitch")) and _axis(s, "pitch")[0]["to"] > 5,
     "pitch положительный"),
]


def run_case(model: str, prompt: str, check, timeout: int):
    """Возвращает (ok_schema, ok_semantic, elapsed, note)."""
    try:
        result = parse_prompt(prompt, model=model, timeout=timeout)
    except PromptLLMError as exc:
        return False, False, None, str(exc)[:60]

    try:  # проходит ли валидацию боевого таймлайна
        segments_from_data(result.segments)
    except Exception as exc:
        return False, False, result.elapsed_sec, f"невалидный сегмент: {exc}"[:60]

    try:
        ok = bool(check(result.segments))
    except Exception:
        ok = False
    note = " ".join(f"{s['axis']}:{s['frm']:g}->{s['to']:g}@{s['start']:g}" for s in result.segments[:3])
    return True, ok, result.elapsed_sec, note[:70]


def bench(models: list, repeat: int, timeout: int) -> None:
    installed = available_models()
    if not installed:
        print("Ollama недоступна. Запусти 'ollama serve' и повтори.")
        return

    print(f"установлено моделей: {len(installed)}\n")
    summary = []

    for model in models:
        if not any(name == model for name in installed):
            print(f"!! {model} не установлена, пропускаю (ollama pull {model})\n")
            continue

        print("=" * 78)
        print(f"МОДЕЛЬ: {model}")
        print("=" * 78)
        times, schema_ok, semantic_ok = [], 0, 0
        total = 0

        for prompt, check, description in CASES:
            for attempt in range(repeat):
                ok_schema, ok_sem, elapsed, note = run_case(model, prompt, check, timeout)
                total += 1
                schema_ok += int(ok_schema)
                semantic_ok += int(ok_sem)
                if elapsed:
                    times.append(elapsed)
                if attempt == 0:
                    mark = "OK  " if ok_sem else ("СХЕМА" if ok_schema else "СБОЙ")
                    seconds = f"{elapsed:5.2f}s" if elapsed else "  -  "
                    print(f"  [{mark:5s}] {seconds}  {description}")
                    print(f"           {note}")

        median = statistics.median(times) if times else float("nan")
        print(f"\n  схема   : {schema_ok}/{total}")
        print(f"  смысл   : {semantic_ok}/{total}")
        print(f"  медиана : {median:.2f} с\n")
        summary.append((model, schema_ok, semantic_ok, total, median))

    if summary:
        print("=" * 78)
        print("ИТОГ")
        print("=" * 78)
        print(f"{'модель':<20}{'схема':>10}{'смысл':>10}{'медиана':>12}")
        for model, schema_ok, semantic_ok, total, median in summary:
            print(f"{model:<20}{schema_ok}/{total:<8}{semantic_ok}/{total:<8}{median:>10.2f} с")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen3.5:2b", "gemma4:e2b"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    bench(args.models, max(1, args.repeat), args.timeout)


if __name__ == "__main__":
    main()
