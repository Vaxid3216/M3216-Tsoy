"""Расписание из JSON-файла: занятия на дату и на текущую неделю."""

import json
from collections import defaultdict
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

Lesson = dict[str, str]
Clock = Callable[[], datetime]


class ScheduleError(Exception):
    """Файл расписания недоступен или повреждён."""


def load_lessons(path: Path) -> dict[str, list[Lesson]]:
    """Читает файл и группирует занятия по датам, внутри дня по времени начала."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw["lessons"]
        by_date: dict[str, list[Lesson]] = defaultdict(list)
        for item in items:
            lesson = {key: str(item[key]) for key in ("start", "end", "title", "location")}
            by_date[str(item["date"])].append(lesson)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ScheduleError("Расписание сейчас недоступно.") from exc
    for lessons in by_date.values():
        lessons.sort(key=lambda lesson: lesson["start"])
    return dict(by_date)


def lessons_on(path: Path, day: date) -> list[Lesson]:
    return load_lessons(path).get(day.isoformat(), [])


def current_week(path: Path, clock: Clock, timezone: str) -> dict[str, list[Lesson]]:
    """Семь дат с понедельника по воскресенье той недели, где находится «сегодня»."""
    today = clock().astimezone(ZoneInfo(timezone)).date()
    monday = today - timedelta(days=today.weekday())
    lessons = load_lessons(path)
    days = [monday + timedelta(days=offset) for offset in range(7)]
    return {day.isoformat(): lessons.get(day.isoformat(), []) for day in days}
