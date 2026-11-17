"""Сборка MCP-сервера: схемы, обработчики и проверка аргументов."""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.mcp_server import owner, reminders, schedule, weather

logger = logging.getLogger("app.mcp_server")

WEEK_URI = "schedule://current-week"
MAX_TEXT = 500

CreateReminder = Callable[[int, str, datetime, str, str], Awaitable[tuple[int, bool]]]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class WeatherIn(_Strict):
    city: str = Field(min_length=1, max_length=100, description="Город, можно «Город, Страна».")


class Candidate(_Strict):
    name: str
    country: str
    admin1: str


class WeatherOut(_Strict):
    status: Literal["ok", "ambiguous", "not_found", "error"]
    place: str | None = None
    country: str | None = None
    temperature_c: float | None = None
    conditions: str | None = None
    wind_speed: float | None = None
    wind_unit: str | None = None
    observed_at: str | None = None
    timezone: str | None = None
    candidates: list[Candidate] | None = None
    reason: Literal["timeout", "unavailable", "bad_response"] | None = None


class ScheduleIn(_Strict):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Дата YYYY-MM-DD.")

    @field_validator("date")
    @classmethod
    def _real_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


class LessonOut(_Strict):
    start: str
    end: str
    title: str
    location: str


class ScheduleOut(_Strict):
    date: str
    lessons: list[LessonOut]


class ReminderIn(_Strict):
    text: str = Field(min_length=1, max_length=MAX_TEXT, description="Текст напоминания.")
    remind_at: datetime = Field(description="Время ISO 8601, только будущее.")


class ReminderOut(_Strict):
    reminder_id: int
    remind_at: str
    timezone: str
    created: bool


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Dependencies:
    schedule_file: Path
    schedule_timezone: str
    owner_secret: str
    fetch_json: weather.FetchJson
    create_reminder: CreateReminder
    clock: Callable[[], datetime] = field(default=_utc_now)


TOOLS = [
    types.Tool(
        name="get_weather",
        description="Текущая погода в городе (температура, ветер, условия). Только чтение.",
        inputSchema=WeatherIn.model_json_schema(),
        outputSchema=WeatherOut.model_json_schema(),
        annotations=types.ToolAnnotations(readOnlyHint=True, openWorldHint=True),
    ),
    types.Tool(
        name="get_schedule",
        description="Занятия на выбранную дату (YYYY-MM-DD). Только чтение.",
        inputSchema=ScheduleIn.model_json_schema(),
        outputSchema=ScheduleOut.model_json_schema(),
        annotations=types.ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    ),
    types.Tool(
        name="add_reminder",
        description=(
            "Создать напоминание на будущее время. Меняет данные: вызывается "
            "только после подтверждения пользователя."
        ),
        inputSchema=ReminderIn.model_json_schema(),
        outputSchema=ReminderOut.model_json_schema(),
        annotations=types.ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False
        ),
    ),
]


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def build_server(deps: Dependencies) -> Server:
    server: Server = Server("lab2-assistant-tools")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "get_weather":
                args = WeatherIn.model_validate(arguments)
                result = await weather.get_weather(args.city, deps.fetch_json)
                return _clean(WeatherOut.model_validate(result).model_dump())
            if name == "get_schedule":
                day = ScheduleIn.model_validate(arguments)
                lessons = schedule.lessons_on(deps.schedule_file, date.fromisoformat(day.date))
                return {"date": day.date, "lessons": lessons}
            if name == "add_reminder":
                return await _add_reminder(deps, server, arguments)
        except ValidationError as exc:
            raise ValueError(f"Неверные аргументы: {exc.error_count()} ошибок проверки.") from None
        except schedule.ScheduleError as exc:
            raise ValueError(str(exc)) from None
        raise ValueError(f"Неизвестный инструмент: {name}")

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=AnyUrl(WEEK_URI),
                name="Расписание на текущую неделю",
                description="Занятия с понедельника по воскресенье, сгруппированы по датам.",
                mimeType="application/json",
            )
        ]

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        if str(uri) != WEEK_URI:
            raise ValueError("Неизвестный ресурс.")
        try:
            week = schedule.current_week(deps.schedule_file, deps.clock, deps.schedule_timezone)
        except schedule.ScheduleError as exc:
            raise ValueError(str(exc)) from None
        body = json.dumps({"days": week}, ensure_ascii=False)
        return [ReadResourceContents(content=body, mime_type="application/json")]

    return server


async def _add_reminder(
    deps: Dependencies, server: Server, arguments: dict[str, Any]
) -> dict[str, Any]:
    meta = server.request_context.meta
    extra = (meta.model_extra or {}) if meta is not None else {}
    try:
        context = owner.verify(deps.owner_secret, extra)
    except owner.OwnerError as exc:
        logger.warning("add_reminder отклонён: %s", exc)
        raise ValueError(str(exc)) from None
    args = ReminderIn.model_validate(arguments)
    try:
        when = reminders.resolve_remind_at(args.remind_at, context.timezone, deps.clock())
    except reminders.ReminderError as exc:
        raise ValueError(str(exc)) from None
    reminder_id, created = await deps.create_reminder(
        context.owner_id, args.text, when, context.timezone, context.idempotency_key
    )
    logger.info("add_reminder: напоминание %s, новое=%s.", reminder_id, created)
    return ReminderOut(
        reminder_id=reminder_id,
        remind_at=when.isoformat(),
        timezone=context.timezone,
        created=created,
    ).model_dump()
