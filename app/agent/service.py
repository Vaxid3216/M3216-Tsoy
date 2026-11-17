"""Агентный цикл: решение модели, проверка, вызов MCP, подтверждение изменений."""

import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.agent.decision import (
    CLARIFY_SPEC,
    Confirmation,
    Decision,
    DecisionError,
    parse_decision,
    summarize_args,
    validate_arguments,
)
from app.agent.prompts import build_system_prompt
from app.assistant import GENERIC_ERROR, split_message, trim_history
from app.config import Settings
from app.llm import LLMError
from app.mcp_client import McpUnavailable, ToolInfo
from app.mcp_server.reminders import ReminderError, resolve_remind_at
from app.prompts import PROMPTS

logger = logging.getLogger("app.agent")

MAX_MCP_CALLS = 2
MAX_STEPS = MAX_MCP_CALLS + 1  # после последнего вызова нужен шаг с итоговым ответом
CONFIRM_TTL_SECONDS = 300
MAX_RESULT_CHARS = 4000

UNAVAILABLE = "Эта функция сейчас недоступна. Попробуйте позже."
LIMIT_REACHED = (
    "Для этого запроса понадобилось слишком много обращений к инструментам, поэтому я "
    "остановился. Попробуйте сформулировать запрос короче."
)
STALE_ACTION = "Это действие уже недействительно."


@dataclass(frozen=True)
class AgentReply:
    parts: list[str]
    confirmation: Confirmation | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def to_llm_tools(tools: list[ToolInfo]) -> list[dict[str, Any]]:
    """Схемы обнаруженных MCP-инструментов в формате `tools` LLM API, плюс уточнение."""
    specs = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]
    return [*specs, CLARIFY_SPEC]


def wrap_tool_result(name: str, result: Any) -> str:
    """Результат инструмента передаётся как недоверенные данные с указанием источника."""
    body = json.dumps(
        {"source": f"инструмент {name}", "untrusted_data": result}, ensure_ascii=False
    )
    return body[:MAX_RESULT_CHARS]


class Agent:
    def __init__(
        self,
        store: Any,
        llm: Any,
        mcp: Any,
        settings: Settings,
        clock: Callable[[], datetime] = _utc_now,
    ):
        self._store = store
        self._llm = llm
        self._mcp = mcp
        self._settings = settings
        self._clock = clock

    # ---- построение запроса ----

    def build_messages(
        self,
        mode: str,
        text: str,
        history: list[dict],
        now: datetime,
        timezone: str | None,
        tools_available: bool,
    ) -> list[dict[str, Any]]:
        zone = timezone or self._settings.schedule_timezone
        system = build_system_prompt(
            PROMPTS.get(mode, PROMPTS["study"]), now, zone, timezone is not None, tools_available
        )
        return [{"role": "system", "content": system}, *history, {"role": "user", "content": text}]

    async def plan(
        self,
        text: str,
        *,
        mode: str = "study",
        timezone: str | None = None,
        now: datetime | None = None,
        temperature: float = 0.3,
    ) -> Decision:
        """Первое решение модели без выполнения: для эксперимента по маршрутизации."""
        available = await self._mcp.ensure_started()
        tools = self._mcp.tools if available else []
        messages = self.build_messages(mode, text, [], now or self._clock(), timezone, bool(tools))
        message = await self._llm.chat(messages, temperature, to_llm_tools(tools))
        return parse_decision(message)

    # ---- обработка сообщения ----

    async def handle(self, user_id: int, update_id: int, text: str) -> AgentReply:
        request_id = uuid.uuid4().hex[:8]
        settings = self._settings
        user = await self._store.get_user(user_id)
        available = await self._mcp.ensure_started()
        tools: list[ToolInfo] = list(self._mcp.tools) if available else []
        history = trim_history(
            await self._store.history(user_id, settings.history_max_messages),
            text,
            settings.history_max_messages,
            settings.history_max_chars,
        )
        timezone = user.get("timezone")
        messages = self.build_messages(
            user["mode"], text, history, self._clock(), timezone, bool(tools)
        )
        await self._store.add_message(user_id, "user", text)
        reply = await self._loop(request_id, user_id, update_id, user, messages, tools)
        if reply is None:
            return AgentReply([GENERIC_ERROR])
        await self._store.add_message(user_id, "assistant", "\n".join(reply.parts))
        return reply

    async def _loop(
        self,
        request_id: str,
        user_id: int,
        update_id: int,
        user: dict,
        messages: list[dict[str, Any]],
        tools: list[ToolInfo],
    ) -> AgentReply | None:
        by_name = {tool.name: tool for tool in tools}
        specs = to_llm_tools(tools)
        calls = 0
        for _ in range(MAX_STEPS):
            started = time.monotonic()
            try:
                message = await self._llm.chat(messages, user["temperature"], specs)
            except LLMError as exc:
                return AgentReply([str(exc)])
            try:
                decision = parse_decision(message)
            except DecisionError as exc:
                logger.warning("Агент %s: решение отклонено: %s", request_id, exc)
                await self._audit(user_id, "call_tool", "rejected", "not_executed", reason=str(exc))
                return AgentReply(["Не удалось понять решение модели. Переформулируйте запрос."])
            logger.info(
                "Агент %s: решение %s, инструмент %s.",
                request_id,
                decision.kind,
                decision.tool or "-",
            )
            if decision.kind == "respond":
                await self._audit(user_id, "respond", "not_applicable", "not_executed")
                return AgentReply(split_message(decision.text))
            if decision.kind == "clarify":
                await self._audit(user_id, "clarify", "not_applicable", "not_executed")
                return AgentReply(split_message(decision.text))

            summary = summarize_args(decision.tool, decision.arguments)
            tool = by_name.get(decision.tool)
            if tool is None:
                await self._audit(
                    user_id,
                    "call_tool",
                    "rejected",
                    "not_executed",
                    tool=decision.tool[:60],
                    reason="инструмент не обнаружен через MCP",
                )
                return AgentReply(["Такого инструмента у меня нет, поэтому я его не вызывал."])
            problem = validate_arguments(tool.input_schema, decision.arguments)
            if problem:
                await self._audit(
                    user_id,
                    "call_tool",
                    "rejected",
                    "not_executed",
                    tool=tool.name,
                    args_summary=summary,
                    reason=f"аргументы не прошли схему ({problem})",
                )
                return AgentReply(["Не получилось корректно подготовить запрос. Уточните детали."])
            if not tool.read_only:
                return await self._prepare(user_id, update_id, user, tool, decision, summary)
            if calls >= MAX_MCP_CALLS:
                await self._audit(
                    user_id,
                    "call_tool",
                    "rejected",
                    "not_executed",
                    tool=tool.name,
                    args_summary=summary,
                    reason="превышен лимит вызовов инструментов",
                )
                return AgentReply([LIMIT_REACHED])
            calls += 1
            try:
                result = await self._mcp.call_tool(tool.name, decision.arguments)
            except McpUnavailable:
                await self._audit(
                    user_id,
                    "call_tool",
                    "passed",
                    "error",
                    tool=tool.name,
                    args_summary=summary,
                    reason="MCP-сервер недоступен",
                )
                return AgentReply([UNAVAILABLE])
            await self._audit(
                user_id,
                "call_tool",
                "passed",
                "error" if result.is_error else "success",
                tool=tool.name,
                args_summary=summary,
                reason="ошибка инструмента" if result.is_error else "",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            payload = {"error": result.text[:300]} if result.is_error else result.structured
            call = message["tool_calls"][0]
            call_id = str(call.get("id") or "call_0")
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": [{**call, "id": call_id}],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": wrap_tool_result(tool.name, payload),
                }
            )
        await self._audit(
            user_id,
            "call_tool",
            "rejected",
            "not_executed",
            reason="превышен лимит шагов агента",
        )
        return AgentReply([LIMIT_REACHED])

    # ---- изменение данных: подготовка и подтверждение ----

    async def _prepare(
        self,
        user_id: int,
        update_id: int,
        user: dict,
        tool: ToolInfo,
        decision: Decision,
        summary: str,
    ) -> AgentReply:
        async def reject(message: str, reason: str) -> AgentReply:
            await self._audit(
                user_id,
                "prepare",
                "rejected",
                "not_executed",
                tool=tool.name,
                args_summary=summary,
                reason=reason,
            )
            return AgentReply([message])

        timezone = user.get("timezone")
        if not timezone:
            return await reject(
                "Сначала задайте часовой пояс, например: /timezone Europe/Moscow. "
                "Без него я не могу точно поставить напоминание.",
                "часовой пояс не задан",
            )
        args = decision.arguments
        reminder_text = str(args.get("text", "")).strip()
        if not reminder_text:
            return await reject("Текст напоминания пустой. О чём напомнить?", "пустой текст")
        try:
            moment = datetime.fromisoformat(str(args["remind_at"]))
            when = resolve_remind_at(moment, timezone, self._clock())
        except (ValueError, KeyError) as exc:
            reason = str(exc) if isinstance(exc, ReminderError) else "время не распознано"
            return await reject(f"{reason} Укажите дату и время точнее.", reason)
        normalized = {"text": reminder_text, "remind_at": when.isoformat()}
        action_id, _ = await self._store.prepare_action(
            user_id, update_id, tool.name, normalized, CONFIRM_TTL_SECONDS
        )
        await self._audit(
            user_id,
            "prepare",
            "passed",
            "not_executed",
            tool=tool.name,
            args_summary=summary,
            reason="ожидает подтверждения",
        )
        prompt = (
            "Создать напоминание?\n"
            f"Текст: {reminder_text}\n"
            f"Когда: {when:%d.%m.%Y %H:%M} ({timezone})\n"
            "Подтвердите в течение 5 минут."
        )
        return AgentReply([prompt], Confirmation(str(action_id)))

    async def confirm(self, user_id: int, action_id: str) -> str:
        try:
            parsed = uuid.UUID(action_id)
        except ValueError:
            return STALE_ACTION
        action = await self._store.begin_execution(user_id, parsed)
        if action is None:
            known = await self._store.get_action(user_id, parsed)
            if known and known["status"] == "pending":
                await self._store.expire_action(user_id, parsed)
                return "Время подтверждения истекло (5 минут). Повторите запрос."
            return STALE_ACTION
        started = time.monotonic()
        user = await self._store.get_user(user_id)
        timezone = user.get("timezone") or self._settings.schedule_timezone
        summary = summarize_args(action["tool"], action["args"])

        async def fail(message: str, reason: str) -> str:
            await self._store.finish_action(user_id, parsed, False)
            await self._audit(
                user_id,
                "call_tool",
                "passed",
                "error",
                tool=action["tool"],
                args_summary=summary,
                reason=reason,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return message

        try:
            result = await self._mcp.call_add_reminder(
                action["args"], user_id, str(parsed), timezone
            )
        except McpUnavailable:
            return await fail("Не удалось создать напоминание: сервис сейчас недоступен.", "MCP")
        data = result.structured or {}
        if result.is_error or "reminder_id" not in data:
            return await fail(
                "Не удалось создать напоминание. Проверьте время и попробуйте снова.",
                "ошибка инструмента",
            )
        await self._store.finish_action(user_id, parsed, True)
        await self._audit(
            user_id,
            "call_tool",
            "passed",
            "success",
            tool=action["tool"],
            args_summary=summary,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        when = datetime.fromisoformat(data["remind_at"])
        return (
            f"Напоминание №{data['reminder_id']} создано: {action['args']['text']}\n"
            f"Когда: {when:%d.%m.%Y %H:%M} ({data.get('timezone', timezone)})."
        )

    async def cancel(self, user_id: int, action_id: str) -> str:
        try:
            parsed = uuid.UUID(action_id)
        except ValueError:
            return STALE_ACTION
        if await self._store.cancel_action(user_id, parsed):
            return "Хорошо, напоминание не создано."
        return STALE_ACTION

    # ---- команды ----

    async def tools_text(self) -> str:
        if not await self._mcp.ensure_started() or not self._mcp.tools:
            return "Инструменты сейчас недоступны."
        lines = [
            f"• {tool.name} — {tool.description} ({'чтение' if tool.read_only else 'изменение'})"
            for tool in self._mcp.tools
        ]
        resources = getattr(self._mcp, "resources", [])
        if resources:
            lines.append("Ресурсы: " + ", ".join(resources))
        return "Доступные инструменты:\n" + "\n".join(lines)

    async def why_text(self, user_id: int) -> str:
        row = await self._store.last_audit(user_id)
        if row is None:
            return "Запросов к агенту пока не было."
        labels = {
            "respond": "ответ без инструмента",
            "clarify": "уточняющий вопрос",
            "call_tool": f"вызов инструмента {row['tool'] or '-'}",
            "prepare": f"подготовка действия {row['tool'] or '-'}",
        }
        checks = {"passed": "пройдена", "rejected": "отклонено", "not_applicable": "не требуется"}
        execs = {"success": "успешно", "error": "ошибка", "not_executed": "не выполнялось"}
        lines = [
            f"Время: {row['created_at']:%d.%m.%Y %H:%M:%S} UTC",
            f"Действие: {labels[row['action']]}",
        ]
        if row["args_summary"]:
            lines.append(f"Аргументы: {row['args_summary']}")
        lines.append(f"Проверка: {checks[row['check_status']]}")
        lines.append(f"Выполнение: {execs[row['exec_status']]}")
        if row["reason"]:
            lines.append(f"Причина: {row['reason']}")
        return "\n".join(lines)

    async def timezone_text(self, user_id: int, raw: str | None) -> str:
        if not raw:
            user = await self._store.get_user(user_id)
            zone = user.get("timezone")
            return f"Ваш часовой пояс: {zone}." if zone else "Часовой пояс не задан."
        name = raw.strip()
        try:
            zone = ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            return "Неизвестная зона. Нужен формат IANA, например /timezone Europe/Moscow."
        await self._store.set_timezone(user_id, name)
        local = self._clock().astimezone(zone)
        return f"Часовой пояс сохранён: {name}. Сейчас там {local:%d.%m.%Y %H:%M}."

    async def _audit(self, user_id: int, action: str, check: str, execution: str, **kw: Any):
        try:
            await self._store.add_audit(user_id, action, check, execution, **kw)
        except Exception:
            logger.exception("Не удалось записать аудит.")
