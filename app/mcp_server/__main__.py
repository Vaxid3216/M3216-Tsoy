"""Запуск MCP-сервера на stdio: `python -m app.mcp_server`."""

import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
from mcp.server.stdio import stdio_server

from app import storage
from app.config import Settings
from app.db import create_pool
from app.mcp_server import weather
from app.mcp_server.server import Dependencies, build_server


async def main() -> None:
    # stdout занят протоколом MCP, журнал только в stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    settings = Settings.load()
    pool: asyncpg.Pool | None = None

    async def create_reminder(
        owner_id: int, text: str, remind_at: datetime, timezone: str, key: str
    ) -> tuple[int, bool]:
        nonlocal pool
        if pool is None:
            pool = await create_pool(settings)
        return await storage.create_reminder(pool, owner_id, text, remind_at, timezone, key)

    deps = Dependencies(
        schedule_file=Path(settings.schedule_file),
        schedule_timezone=settings.schedule_timezone,
        owner_secret=settings.mcp_owner_secret,
        fetch_json=weather.make_fetch_json(settings.weather_proxy_url),
        create_reminder=create_reminder,
        clock=lambda: datetime.now(UTC),
    )
    server = build_server(deps)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        if pool is not None:
            await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
