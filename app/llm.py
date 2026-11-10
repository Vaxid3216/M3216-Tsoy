import logging
import time
import uuid

import aiohttp
from aiohttp_socks import ProxyConnector

from app.config import Settings

logger = logging.getLogger("app.llm")


class LLMError(Exception):
    """Сбой вызова модели; текст безопасен для показа пользователю."""


class LLMClient:
    """Клиент OpenAI-совместимого chat/completions API."""

    def __init__(self, settings: Settings, session: aiohttp.ClientSession | None = None):
        self._settings = settings
        self._session = session
        self._owns_session = session is None

    async def close(self) -> None:
        if self._session and self._owns_session:
            await self._session.close()

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        """Отправляет диалог модели и возвращает непустой текст ответа."""
        settings = self._settings
        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        logger.info("LLM-запрос %s: модель %s, начало.", request_id, settings.llm_model)
        try:
            text = await self._request(messages, temperature)
        except LLMError as exc:
            logger.warning(
                "LLM-запрос %s: модель %s, ошибка за %.1f с: %s",
                request_id,
                settings.llm_model,
                time.monotonic() - started,
                exc,
            )
            raise
        logger.info(
            "LLM-запрос %s: модель %s, успех за %.1f с.",
            request_id,
            settings.llm_model,
            time.monotonic() - started,
        )
        return text

    async def _request(self, messages: list[dict[str, str]], temperature: float) -> str:
        settings = self._settings
        if self._session is None:
            proxy = settings.llm_proxy_url
            connector = ProxyConnector.from_url(proxy) if proxy else None
            self._session = aiohttp.ClientSession(connector=connector)
        body = {"model": settings.llm_model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        timeout = aiohttp.ClientTimeout(total=settings.llm_timeout_seconds)
        try:
            async with self._session.post(
                f"{settings.llm_base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    raise LLMError(f"Сервис модели вернул ошибку (HTTP {response.status}).")
                try:
                    data = await response.json(content_type=None)
                except ValueError:
                    raise LLMError("Модель вернула ответ в непонятном формате.") from None
        except TimeoutError:
            raise LLMError("Модель отвечает слишком долго. Попробуйте ещё раз.") from None
        except aiohttp.ClientError:
            raise LLMError("Сервис модели недоступен. Попробуйте позже.") from None
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("Модель вернула ответ в непонятном формате.") from None
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Модель вернула пустой ответ. Попробуйте переформулировать вопрос.")
        return text
