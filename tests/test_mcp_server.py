"""Тесты MCP-сервера через настоящий протокол в памяти; внешние границы подменены."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from app.mcp_server import owner, weather
from app.mcp_server.server import Dependencies, build_server

SECRET = "test-secret-0123456789"
NOW = datetime(2026, 10, 12, 6, 0, tzinfo=UTC)  # 09:00 в Москве
SCHEDULE = Path("data/schedule.json")


def place(name: str, country: str, admin1: str, population: int | None = None) -> dict:
    return {
        "name": name,
        "country": country,
        "country_code": country[:2].upper(),
        "admin1": admin1,
        "latitude": 55.0,
        "longitude": 49.0,
        "population": population,
    }


FORECAST = {
    "utc_offset_seconds": 10800,
    "timezone": "Europe/Moscow",
    "current": {
        "time": "2026-10-12T09:00",
        "temperature_2m": 7.4,
        "weather_code": 61,
        "wind_speed_10m": 3.2,
    },
}


def fake_fetch(results: list[dict] | None = None, error: Exception | None = None):
    calls: list[str] = []

    async def fetch(url: str, params: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if error:
            raise error
        if url == weather.GEOCODING_URL:
            return {"results": results} if results is not None else {}
        return FORECAST

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


class FakeReminders:
    def __init__(self):
        self.rows: dict[tuple[int, str], int] = {}
        self.created: list[tuple] = []

    async def __call__(self, owner_id, text, remind_at, timezone, key):
        if (owner_id, key) in self.rows:
            return self.rows[(owner_id, key)], False
        self.rows[(owner_id, key)] = len(self.rows) + 1
        self.created.append((owner_id, text, remind_at, timezone))
        return self.rows[(owner_id, key)], True


def make_deps(fetch=None, reminders=None, schedule_file=SCHEDULE, now=NOW) -> Dependencies:
    return Dependencies(
        schedule_file=schedule_file,
        schedule_timezone="Europe/Moscow",
        owner_secret=SECRET,
        fetch_json=fetch or fake_fetch([]),
        create_reminder=reminders or FakeReminders(),
        clock=lambda: now,
    )


async def call(deps: Dependencies, name: str, arguments: dict, meta: dict | None = None):
    server = build_server(deps)
    async with create_connected_server_and_client_session(server) as client:
        return await client.call_tool(name, arguments, meta=meta)


def meta(key: str = "k1", owner_id: int = 1, zone: str = "Europe/Moscow") -> dict:
    return owner.build_meta(SECRET, owner_id, key, zone)


# ---- обнаружение ----


async def test_tools_and_resource_are_discoverable_with_schemas():
    server = build_server(make_deps())
    async with create_connected_server_and_client_session(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        resources = (await client.list_resources()).resources
    assert set(tools) == {"get_weather", "get_schedule", "add_reminder"}
    for tool in tools.values():
        assert tool.inputSchema["additionalProperties"] is False
        assert tool.outputSchema is not None
    assert tools["get_weather"].annotations.readOnlyHint is True
    assert tools["get_schedule"].annotations.readOnlyHint is True
    assert tools["add_reminder"].annotations.readOnlyHint is False
    assert [str(r.uri) for r in resources] == ["schedule://current-week"]


# ---- T1-T3 погода ----


async def test_weather_ok_returns_structured_result():
    fetch = fake_fetch([place("Казань", "Россия", "Татарстан", 1_200_000)])
    result = await call(make_deps(fetch), "get_weather", {"city": "Казань"})
    assert not result.isError
    data = result.structuredContent
    assert data["status"] == "ok"
    assert data["place"] == "Казань"
    assert data["temperature_c"] == 7.4
    assert data["conditions"] == "слабый дождь"
    assert data["observed_at"] == "2026-10-12T09:00:00+03:00"
    assert json.loads(result.content[0].text) == data


async def test_weather_ambiguous_lists_candidates_without_choosing():
    fetch = fake_fetch(
        [place("Springfield", "США", "Иллинойс", 1000), place("Springfield", "США", "Миссури", 900)]
    )
    data = (await call(make_deps(fetch), "get_weather", {"city": "Springfield"})).structuredContent
    assert data["status"] == "ambiguous"
    assert len(data["candidates"]) == 2
    assert fetch.calls == [weather.GEOCODING_URL]  # прогноз не запрашивался


async def test_weather_country_hint_filters_candidates():
    fetch = fake_fetch(
        [place("Тбилиси", "Грузия", "Тбилиси", 1_100_000), place("Тбилиси", "Россия", "Крым", 10)]
    )
    data = (
        await call(make_deps(fetch), "get_weather", {"city": "Тбилиси, Грузия"})
    ).structuredContent
    assert data["status"] == "ok"
    assert data["country"] == "Грузия"


async def test_weather_not_found():
    data = (
        await call(make_deps(fake_fetch(None)), "get_weather", {"city": "Нигде-на-Дону"})
    ).structuredContent
    assert data == {"status": "not_found"}


@pytest.mark.parametrize("arguments", [{}, {"city": ""}, {"city": "   "}, {"city": "x" * 101}])
async def test_weather_rejects_bad_city(arguments):
    assert (await call(make_deps(), "get_weather", arguments)).isError


async def test_weather_rejects_extra_arguments():
    result = await call(make_deps(), "get_weather", {"city": "Казань", "units": "x"})
    assert result.isError


# ---- N4 недоступность погодного API ----


@pytest.mark.parametrize("reason", ["timeout", "unavailable", "bad_response"])
async def test_weather_service_failure_is_structured_error(reason):
    fetch = fake_fetch(error=weather.WeatherUnavailable(reason))
    result = await call(make_deps(fetch), "get_weather", {"city": "Казань"})
    assert not result.isError
    assert result.structuredContent == {"status": "error", "reason": reason}


async def test_weather_malformed_forecast_is_bad_response():
    async def fetch(url, params):
        if url == weather.GEOCODING_URL:
            return {"results": [place("Казань", "Россия", "Татарстан", 1_200_000)]}
        return {"current": {}}

    result = await call(make_deps(fetch), "get_weather", {"city": "Казань"})
    assert result.structuredContent == {"status": "error", "reason": "bad_response"}


# ---- T4-T6 расписание ----


async def test_schedule_returns_lessons_for_date():
    data = (await call(make_deps(), "get_schedule", {"date": "2026-10-13"})).structuredContent
    assert data["date"] == "2026-10-13"
    assert [lesson["title"] for lesson in data["lessons"]] == ["Алгоритмы и структуры данных"]
    assert set(data["lessons"][0]) == {"start", "end", "title", "location"}


async def test_schedule_empty_day_is_valid_result():
    result = await call(make_deps(), "get_schedule", {"date": "2026-10-18"})
    assert not result.isError
    assert result.structuredContent == {"date": "2026-10-18", "lessons": []}


@pytest.mark.parametrize("value", ["12.10.2026", "2026-13-01", "2026-02-30", "2026-1-5", ""])
async def test_schedule_rejects_invalid_date(value):
    assert (await call(make_deps(), "get_schedule", {"date": value})).isError


async def test_schedule_rejects_path_argument():
    result = await call(make_deps(), "get_schedule", {"date": "2026-10-12", "path": "/etc/passwd"})
    assert result.isError


async def test_schedule_broken_file_is_error_without_details(tmp_path):
    broken = tmp_path / "schedule.json"
    broken.write_text("{not json", encoding="utf-8")
    result = await call(make_deps(schedule_file=broken), "get_schedule", {"date": "2026-10-12"})
    assert result.isError
    assert str(tmp_path) not in result.content[0].text


# ---- T7-T9 напоминание ----

REMINDER = {"text": "отправить отчёт", "remind_at": "2026-10-13T18:30:00+03:00"}


async def test_add_reminder_creates_record_for_signed_owner():
    store = FakeReminders()
    result = await call(make_deps(reminders=store), "add_reminder", REMINDER, meta("k1", 42))
    assert not result.isError
    data = result.structuredContent
    assert data["created"] is True
    assert data["remind_at"] == "2026-10-13T18:30:00+03:00"
    assert data["timezone"] == "Europe/Moscow"
    owner_id, text, when, zone = store.created[0]
    assert (owner_id, text, zone) == (42, "отправить отчёт", "Europe/Moscow")
    assert when.isoformat() == "2026-10-13T18:30:00+03:00"


async def test_add_reminder_naive_time_is_local_in_owner_zone():
    arguments = {"text": "x", "remind_at": "2026-10-13T18:30:00"}
    result = await call(make_deps(), "add_reminder", arguments, meta())
    assert result.structuredContent["remind_at"] == "2026-10-13T18:30:00+03:00"


async def test_add_reminder_without_meta_is_rejected():
    store = FakeReminders()
    assert (await call(make_deps(reminders=store), "add_reminder", REMINDER)).isError
    assert not store.created


async def test_add_reminder_with_wrong_signature_is_rejected():
    store = FakeReminders()
    forged = {**meta(), "owner_id": 999}  # подпись выдана на другого владельца
    result = await call(make_deps(reminders=store), "add_reminder", REMINDER, forged)
    assert result.isError
    assert not store.created


async def test_add_reminder_rejects_owner_in_arguments():
    store = FakeReminders()
    arguments = {**REMINDER, "owner_id": 7}
    assert (await call(make_deps(reminders=store), "add_reminder", arguments, meta())).isError
    assert not store.created


async def test_add_reminder_is_idempotent_for_same_key():
    store = FakeReminders()
    deps = make_deps(reminders=store)
    first = await call(deps, "add_reminder", REMINDER, meta("same"))
    second = await call(deps, "add_reminder", REMINDER, meta("same"))
    assert first.structuredContent["created"] is True
    assert second.structuredContent["created"] is False
    assert first.structuredContent["reminder_id"] == second.structuredContent["reminder_id"]
    assert len(store.created) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"text": "", "remind_at": "2026-10-13T18:30:00+03:00"},
        {"text": "   ", "remind_at": "2026-10-13T18:30:00+03:00"},
        {"text": "x" * 501, "remind_at": "2026-10-13T18:30:00+03:00"},
        {"text": "x", "remind_at": "вчера"},
        {"text": "x", "remind_at": "2026-10-12T08:00:00+03:00"},  # в прошлом
        {"text": "x", "remind_at": "2026-10-13T18:30:00+05:00"},  # чужое смещение
        {"text": "x"},
    ],
)
async def test_add_reminder_rejects_invalid_input(arguments):
    store = FakeReminders()
    assert (await call(make_deps(reminders=store), "add_reminder", arguments, meta())).isError
    assert not store.created


async def test_add_reminder_unknown_timezone_in_meta_is_rejected():
    result = await call(make_deps(), "add_reminder", REMINDER, meta(zone="Mars/Base"))
    assert result.isError


async def test_unknown_tool_is_rejected():
    assert (await call(make_deps(), "delete_all", {})).isError


# ---- T12 ресурс ----


async def read_week(deps: Dependencies) -> dict:
    server = build_server(deps)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.read_resource("schedule://current-week")
    return json.loads(result.contents[0].text)


async def test_resource_returns_monday_to_sunday_with_all_dates():
    week = (await read_week(make_deps()))["days"]
    assert list(week) == [f"2026-10-{day}" for day in range(12, 19)]
    assert len(week["2026-10-12"]) == 2
    assert week["2026-10-18"] == []


async def test_resource_week_crosses_month_and_year(tmp_path):
    source = tmp_path / "s.json"
    source.write_text(
        json.dumps(
            {
                "lessons": [
                    {
                        "date": "2027-01-01",
                        "start": "10:00",
                        "end": "11:30",
                        "title": "Праздничный семинар",
                        "location": "ауд. 1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    now = datetime(2026, 12, 31, 12, 0, tzinfo=UTC)  # четверг
    week = (await read_week(make_deps(schedule_file=source, now=now)))["days"]
    assert list(week) == [
        "2026-12-28",
        "2026-12-29",
        "2026-12-30",
        "2026-12-31",
        "2027-01-01",
        "2027-01-02",
        "2027-01-03",
    ]
    assert week["2027-01-01"][0]["title"] == "Праздничный семинар"


async def test_resource_uses_configured_timezone_for_today(tmp_path):
    # 23:30 воскресенья по UTC уже понедельник следующей недели в Москве.
    now = datetime(2026, 10, 18, 23, 30, tzinfo=UTC)
    week = (await read_week(make_deps(now=now)))["days"]
    assert next(iter(week)) == "2026-10-19"


# ---- проверка времени с переходом на зимнее/летнее ----


async def test_add_reminder_rejects_nonexistent_and_ambiguous_local_time():
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    deps = make_deps(now=now)
    gap = {"text": "x", "remind_at": "2026-03-08T02:30:00"}  # Нью-Йорк: пропущенный час
    fold = {"text": "x", "remind_at": "2026-11-01T01:30:00"}  # повторяющийся час
    zone = "America/New_York"
    assert (await call(deps, "add_reminder", gap, meta(zone=zone))).isError
    assert (await call(deps, "add_reminder", fold, meta("k2", zone=zone))).isError
