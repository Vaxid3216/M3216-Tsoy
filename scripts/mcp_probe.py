"""Независимая проверка MCP-сервера клиентом SDK (без кода бота и агента).

Запуск: python -m scripts.mcp_probe
Сервер стартует дочерним процессом со своими настройками: БД не нужна (напоминания
не создаются, а проверяется отказ без подписи), погода не запрашивается.
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> int:
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE")
        if key in os.environ
    }
    env["MCP_OWNER_SECRET"] = "probe-secret-not-known-to-client"
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], env=env)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools
        for tool in tools:
            kind = "чтение" if tool.annotations and tool.annotations.readOnlyHint else "изменение"
            print(f"инструмент {tool.name} ({kind}): вход {sorted(tool.inputSchema['properties'])}")
        for resource in (await session.list_resources()).resources:
            print(f"ресурс {resource.uri}")
        schedule = await session.call_tool("get_schedule", {"date": "2026-10-13"})
        print("get_schedule:", schedule.structuredContent)
        bad = await session.call_tool("get_schedule", {"date": "13.10.2026"})
        print("неверная дата отклонена:", bad.isError)
        args = {"text": "проверка", "remind_at": "2036-01-01T10:00:00+03:00"}
        unsigned = await session.call_tool("add_reminder", args)
        print("add_reminder без подписи отклонён:", unsigned.isError)
        return 0 if bad.isError and unsigned.isError else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
