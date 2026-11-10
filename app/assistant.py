"""Логика ассистента без привязки к Telegram: режимы, настройки, история, вызов модели."""

import asyncpg

from app import storage
from app.config import Settings
from app.llm import LLMClient, LLMError
from app.prompts import MODE_DESCRIPTIONS, PROMPTS

TELEGRAM_LIMIT = 4096
TEMPERATURES = (0.0, 0.3, 0.7, 1.0)

GENERIC_ERROR = "Что-то пошло не так. Попробуйте ещё раз чуть позже."


class PgStorage:
    """Хранилище на PostgreSQL; в тестах его заменяет объект с теми же методами."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def get_user(self, user_id: int) -> dict:
        row = await storage.get_or_create_user(self._pool, user_id)
        return {"mode": row["mode"], "temperature": float(row["temperature"])}

    async def set_mode(self, user_id: int, mode: str) -> None:
        await storage.set_mode(self._pool, user_id, mode)

    async def set_temperature(self, user_id: int, temperature: float) -> None:
        await storage.set_temperature(self._pool, user_id, temperature)

    async def history(self, user_id: int, limit: int) -> list[dict]:
        rows = await storage.get_history(self._pool, user_id, limit)
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def add_exchange(self, user_id: int, question: str, answer: str) -> None:
        await storage.add_exchange(self._pool, user_id, question, answer)

    async def reset_history(self, user_id: int) -> None:
        await storage.reset_history(self._pool, user_id)


def trim_history(
    history: list[dict], current: str, max_messages: int, max_chars: int
) -> list[dict]:
    """Оставляет самые новые сообщения в пределах лимитов, сохраняя порядок.

    Объём оценивается числом символов. Текущий запрос в историю не входит, но
    его длина вычитается из бюджета, а история начинается с реплики пользователя.
    """
    budget = max_chars - len(current)
    kept: list[dict] = []
    total = 0
    for message in reversed(history[-max_messages:]):
        total += len(message["content"])
        if total > budget:
            break
        kept.append(message)
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Режет текст на части не длиннее limit без потери символов."""
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit) + 1 or text.rfind(" ", 0, limit) + 1 or limit
        parts.append(text[:cut])
        text = text[cut:]
    if text:
        parts.append(text)
    return parts


class Assistant:
    def __init__(self, store: PgStorage, llm: LLMClient, settings: Settings):
        self._store = store
        self._llm = llm
        self._settings = settings

    async def start_text(self, user_id: int) -> str:
        user = await self._store.get_user(user_id)
        modes = "\n".join(f"/{mode} — {text}" for mode, text in MODE_DESCRIPTIONS.items())
        return (
            "Привет! Я AI-ассистент студента. Напишите сообщение, и я отвечу "
            "в выбранном режиме.\n\n"
            f"Режимы:\n{modes}\n\n"
            "Другие команды:\n"
            "/settings — показать настройки или выбрать temperature\n"
            "/reset — очистить историю диалога\n\n"
            f"Сейчас включён режим: {user['mode']}."
        )

    async def switch_mode(self, user_id: int, mode: str) -> str:
        await self._store.get_user(user_id)
        await self._store.set_mode(user_id, mode)
        return (
            f"Режим переключён: /{mode} — {MODE_DESCRIPTIONS[mode]}. "
            "История предыдущего режима очищена."
        )

    async def show_settings(self, user_id: int) -> str:
        user = await self._store.get_user(user_id)
        return (
            f"Режим: {user['mode']}\n"
            f"Модель: {self._settings.llm_model}\n"
            f"Temperature: {user['temperature']:.1f}\n\n"
            "Чтобы изменить temperature, отправьте, например, /settings 0.7. "
            f"Допустимые значения: {', '.join(f'{t:.1f}' for t in TEMPERATURES)}."
        )

    async def set_temperature(self, user_id: int, raw: str) -> str:
        try:
            value = float(raw.strip().replace(",", "."))
        except ValueError:
            value = None
        if value not in TEMPERATURES:
            allowed = ", ".join(f"{t:.1f}" for t in TEMPERATURES)
            return f"Такое значение не подходит. Допустимые значения temperature: {allowed}."
        await self._store.get_user(user_id)
        await self._store.set_temperature(user_id, value)
        return f"Temperature: {value:.1f}. Новое значение применится со следующего запроса."

    async def reset(self, user_id: int) -> str:
        await self._store.reset_history(user_id)
        return "История диалога очищена. Режим и temperature сохранены."

    async def answer(self, user_id: int, text: str) -> list[str]:
        """Возвращает части ответа; при сбое модели одно безопасное сообщение."""
        settings = self._settings
        user = await self._store.get_user(user_id)
        history = trim_history(
            await self._store.history(user_id, settings.history_max_messages),
            text,
            settings.history_max_messages,
            settings.history_max_chars,
        )
        prompt = PROMPTS.get(user["mode"], PROMPTS["study"])
        messages = [
            {"role": "system", "content": prompt},
            *history,
            {"role": "user", "content": text},
        ]
        try:
            reply = (await self._llm.complete(messages, user["temperature"])).strip()
        except LLMError as exc:
            return [str(exc)]
        await self._store.add_exchange(user_id, text, reply)
        return split_message(reply)
