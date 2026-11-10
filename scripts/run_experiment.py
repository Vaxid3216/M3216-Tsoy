"""Эксперимент лабораторной 1: 4 температуры x 3 запуска и сравнение инструкций.

Запуск: python -m scripts.run_experiment
Настройки модели читаются из .env (ключ не печатается и не сохраняется).
Результат пишется в docs/experiment_results.json.
Каждый запрос идёт без истории, как после /reset.
"""

import asyncio
import json
import re
from datetime import date
from pathlib import Path

import aiohttp
from aiohttp_socks import ProxyConnector

from app.config import Settings
from app.prompts import JOKE_PROMPT, JOKE_PROMPT_BASIC

QUERY = "Что такое рекурсия?"
TEMPERATURES = (0.0, 0.3, 0.7, 1.0)
RUNS = 3
COMPARE_TEMPERATURE = 0.3
OUT = Path("docs/experiment_results.json")


def check_format(text: str) -> dict:
    """Заранее названные признаки хорошего ответа для режима /joke."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    words = len(text.split())
    has_joke_line = any(line.startswith("Шутка:") for line in lines)
    no_markdown = not re.search(r"[*#`_]{1,}", text)
    return {
        "words": words,
        "joke_line": has_joke_line,
        "max_60_words": words <= 60,
        "no_markdown": no_markdown,
        "format_ok": has_joke_line and words <= 60 and no_markdown,
    }


async def ask(session, settings, system: str, temperature: float) -> dict:
    body = {
        "model": settings.llm_model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": QUERY},
        ],
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    for attempt in range(4):
        try:
            async with session.post(
                f"{settings.llm_base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status == 200:
                    data = await response.json(content_type=None)
                    text = (data["choices"][0]["message"]["content"] or "").strip()
                    if text:
                        return {"text": text, "usage": data.get("usage")}
                print(f"  попытка {attempt + 1}: HTTP {response.status} или пустой ответ")
        except (TimeoutError, aiohttp.ClientError) as exc:
            print(f"  попытка {attempt + 1}: {type(exc).__name__}")
        await asyncio.sleep(5)
    return {"text": "", "usage": None}


async def main() -> None:
    settings = Settings.load()
    settings.require_llm()
    proxy = settings.llm_proxy_url
    connector = ProxyConnector.from_url(proxy) if proxy else None
    results = {
        "date": date.today().isoformat(),
        "model": settings.llm_model,
        "query": QUERY,
        "temperature_runs": [],
        "prompt_comparison": [],
    }
    async with aiohttp.ClientSession(connector=connector) as session:
        number = 0
        for temperature in TEMPERATURES:
            for _ in range(RUNS):
                number += 1
                answer = await ask(session, settings, JOKE_PROMPT, temperature)
                row = {"run": number, "temperature": temperature, **answer}
                row["format"] = check_format(answer["text"])
                results["temperature_runs"].append(row)
                print(f"запуск {number}: t={temperature}, слов={row['format']['words']}")
        for name, prompt in (("basic", JOKE_PROMPT_BASIC), ("improved", JOKE_PROMPT)):
            for run in range(1, RUNS + 1):
                answer = await ask(session, settings, prompt, COMPARE_TEMPERATURE)
                row = {"prompt": name, "run": run, **answer}
                row["format"] = check_format(answer["text"])
                results["prompt_comparison"].append(row)
                print(f"сравнение {name} {run}: слов={row['format']['words']}")
    await asyncio.to_thread(
        OUT.write_text, json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Готово: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
