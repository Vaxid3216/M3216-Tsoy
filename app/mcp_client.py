"""MCP-клиент: запускает сервер, обнаруживает инструменты и вызывает их.

Бот является хостом, этот модуль клиентом. Схемы инструментов берутся из
обнаружения (`tools/list`) и нигде в боте не дублируются.
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.config import Settings
from app.mcp_server import owner

logger = logging.getLogger("app.mcp_client")

CALL_TIMEOUT_SECONDS = 40
START_TIMEOUT_SECONDS = 20
# Дочерний процесс получает только нужные ему переменные, без токена бота и ключа LLM.
SERVER_ENV_KEYS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "SCHEDULE_FILE",
    "SCHEDULE_TIMEZONE",
    "MCP_OWNER_SECRET",
    "WEATHER_PROXY_URL",
)


class McpUnavailable(Exception):
    """Сервер MCP не запущен или соединение оборвано."""


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool


@dataclass(frozen=True)
class ToolResult:
    is_error: bool
    structured: dict[str, Any] | None
    text: str


@dataclass
class McpClient:
    """Подключение к серверу по stdio; при обрыве следующий вызов пробует переподключиться."""

    settings: Settings
    tools: list[ToolInfo] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    _task: asyncio.Task | None = None
    _stop: asyncio.Event | None = None
    _session: ClientSession | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def available(self) -> bool:
        return self._session is not None

    def _server_params(self) -> StdioServerParameters:
        settings = self.settings
        values = {
            "POSTGRES_HOST": settings.postgres_host,
            "POSTGRES_PORT": str(settings.postgres_port),
            "POSTGRES_DB": settings.postgres_db,
            "POSTGRES_USER": settings.postgres_user,
            "POSTGRES_PASSWORD": settings.postgres_password,
            "SCHEDULE_FILE": settings.schedule_file,
            "SCHEDULE_TIMEZONE": settings.schedule_timezone,
            "MCP_OWNER_SECRET": settings.mcp_owner_secret,
            "WEATHER_PROXY_URL": settings.weather_proxy_url,
        }
        env = {key: value for key, value in values.items() if key in SERVER_ENV_KEYS and value}
        # PATH и системные переменные нужны интерпретатору (например, SYSTEMROOT в Windows).
        for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE"):
            if key in os.environ:
                env[key] = os.environ[key]
        return StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], env=env)

    async def start(self) -> bool:
        """Запускает сервер и обнаруживает инструменты; False при неудаче (без исключения)."""
        async with self._lock:
            return await self._connect()

    async def _connect(self) -> bool:
        if self._session is not None:
            return True
        ready: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()
        task = asyncio.create_task(self._run(ready, stop))
        try:
            async with asyncio.timeout(START_TIMEOUT_SECONDS):
                started = await ready
        except TimeoutError:
            started = False
            logger.warning("MCP-сервер не ответил за %d с.", START_TIMEOUT_SECONDS)
        if not started:
            stop.set()
            await asyncio.gather(task, return_exceptions=True)
            return False
        self._task, self._stop = task, stop
        return True

    async def _run(self, ready: "asyncio.Future[bool]", stop: asyncio.Event) -> None:
        """Одна задача владеет соединением от открытия до закрытия (требование anyio)."""
        try:
            async with stdio_client(self._server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self.tools = [
                        ToolInfo(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=tool.inputSchema,
                            read_only=bool(tool.annotations and tool.annotations.readOnlyHint),
                        )
                        for tool in listed.tools
                    ]
                    listed_resources = await session.list_resources()
                    self.resources = [str(r.uri) for r in listed_resources.resources]
                    self._session = session
                    logger.info("MCP-сервер запущен, инструментов: %d.", len(self.tools))
                    if not ready.done():
                        ready.set_result(True)
                    await stop.wait()
        except Exception as exc:
            logger.warning("MCP-соединение закрыто: %s.", type(exc).__name__)
        finally:
            self._session = None
            self.tools = []
            self.resources = []
            if not ready.done():
                ready.set_result(False)

    async def _drop(self) -> None:
        task, stop = self._task, self._stop
        self._task = self._stop = self._session = None
        self.tools = []
        self.resources = []
        if stop is not None:
            stop.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def ensure_started(self) -> bool:
        """Возвращает готовность; при отсутствии связи пробует подключиться заново."""
        if self._session is not None:
            return True
        return await self.start()

    async def call_tool(
        self, name: str, arguments: dict[str, Any], meta: dict[str, Any] | None = None
    ) -> ToolResult:
        if not await self.ensure_started():
            raise McpUnavailable("MCP-сервер недоступен.")
        session = self._session
        assert session is not None
        try:
            async with asyncio.timeout(CALL_TIMEOUT_SECONDS):
                result = await session.call_tool(name, arguments, meta=meta)
        except Exception as exc:
            logger.warning("Вызов MCP-инструмента %s оборвался: %s.", name, type(exc).__name__)
            async with self._lock:
                await self._drop()
            raise McpUnavailable("Связь с MCP-сервером потеряна.") from None
        text = "".join(block.text for block in result.content if getattr(block, "text", None))
        return ToolResult(
            is_error=bool(result.isError), structured=result.structuredContent, text=text
        )

    async def call_add_reminder(
        self, arguments: dict[str, Any], owner_id: int, key: str, timezone: str
    ) -> ToolResult:
        """Вызов `add_reminder` с подписанным контекстом владельца."""
        meta = owner.build_meta(self.settings.mcp_owner_secret, owner_id, key, timezone)
        return await self.call_tool("add_reminder", arguments, meta=meta)

    async def close(self) -> None:
        async with self._lock:
            await self._drop()
