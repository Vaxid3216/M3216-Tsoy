"""Разовая проверка LLM: системная инструкция и параметр temperature.

Запуск из корня проекта:
    .venv\\Scripts\\python.exe scripts\\check_llm.py

Скрипт читает LLM_BASE_URL, LLM_MODEL и LLM_API_KEY из файла .env.
Ключ нигде не печатается.
"""

import asyncio
import os
import sys

import aiohttp
from aiohttp_socks import ProxyConnector
from dotenv import load_dotenv

# Подгружаем переменные из .env в окружение процесса.
load_dotenv()

BASE_URL = os.environ["LLM_BASE_URL"].rstrip("/")
MODEL = os.environ["LLM_MODEL"]
API_KEY = os.environ["LLM_API_KEY"]
# Если сервис недоступен напрямую, пробуем тот же прокси, что и для Telegram.
PROXY_URL = os.environ.get("TELEGRAM_PROXY_URL")


async def ask(session, messages, temperature=None):
    """Отправляет один запрос модели и возвращает (HTTP-статус, тело ответа)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": 500}
    if temperature is not None:
        body["temperature"] = temperature
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{BASE_URL}/chat/completions"
    async with session.post(url, json=body, headers=headers) as resp:
        return resp.status, await resp.json()


def reply_text(data):
    """Достаёт текст ответа модели из JSON."""
    return data["choices"][0]["message"]["content"] or ""


async def run_checks(session):
    # Проверка 1. Системная инструкция.
    # Просим отвечать одним словом. Если модель следует системной
    # инструкции, ответ будет коротким и с этим словом.
    messages = [
        {"role": "system", "content": "Отвечай только одним словом: ПИНГ."},
        {"role": "user", "content": "Привет! Расскажи, как у тебя дела?"},
    ]
    status, data = await ask(session, messages)
    print(f"[1] системная инструкция: HTTP {status}")
    if status == 200:
        text = reply_text(data).strip()
        print(f"    ответ модели: {text[:80]!r}")
        print("    следует инструкции:", "да" if "ПИНГ" in text.upper() else "нет")
    else:
        print("    ошибка:", data)

    # Проверка 2. Параметр temperature.
    # Если параметр не принят, API обычно вернёт ошибку 400.
    for temp in (0.0, 1.0):
        messages = [{"role": "user", "content": "Назови любой цвет одним словом."}]
        status, data = await ask(session, messages, temperature=temp)
        print(f"[2] temperature={temp}: HTTP {status}")
        if status == 200:
            print(f"    ответ модели: {reply_text(data).strip()[:80]!r}")
        else:
            print("    ошибка:", data)


async def main():
    print(f"Модель: {MODEL}")
    print(f"Адрес:  {BASE_URL}")
    if "--proxy" in sys.argv and PROXY_URL:
        # Принудительно идём через прокси (флаг --proxy).
        connector = ProxyConnector.from_url(PROXY_URL)
        async with aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            await run_checks(session)
        return
    try:
        # Сначала пробуем напрямую.
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            await run_checks(session)
    except aiohttp.ClientConnectionError as exc:
        print(f"Прямое подключение не удалось: {exc}")
        if not PROXY_URL:
            return
        print("Пробую через прокси из TELEGRAM_PROXY_URL...")
        connector = ProxyConnector.from_url(PROXY_URL)
        async with aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=60)
        ) as session:
            await run_checks(session)


if __name__ == "__main__":
    asyncio.run(main())
