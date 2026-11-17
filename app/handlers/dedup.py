"""Защита от повторной доставки одного Telegram update."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Update

logger = logging.getLogger("app.handlers")


class UpdateDedupMiddleware(BaseMiddleware):
    """Пропускает update, который уже обработан (таблица `processed_updates`)."""

    def __init__(self, store: Any):
        self._store = store

    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        if not await self._store.mark_update_processed(event.update_id):
            logger.info("Повторная доставка update пропущена.")
            return None
        return await handler(event, data)
