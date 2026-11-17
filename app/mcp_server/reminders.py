"""Проверка времени напоминания относительно зоны пользователя."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ReminderError(ValueError):
    """Время напоминания не прошло проверку; текст безопасен для пользователя."""


def load_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise ReminderError("Неизвестный часовой пояс.") from None


def resolve_remind_at(value: datetime, zone_name: str, now: datetime) -> datetime:
    """Возвращает момент напоминания с зоной пользователя.

    Время без смещения считается локальным; неоднозначное и несуществующее местное
    время отклоняется. Смещение в значении должно совпадать со смещением зоны.
    """
    zone = load_zone(zone_name)
    if value.tzinfo is None:
        first = value.replace(tzinfo=zone, fold=0)
        second = value.replace(tzinfo=zone, fold=1)
        if first.utcoffset() != second.utcoffset():
            raise ReminderError("Это местное время неоднозначно из-за перехода времени.")
        roundtrip = first.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
        if roundtrip != value:
            raise ReminderError("Такого местного времени нет из-за перехода времени.")
        local = first
    else:
        local = value.astimezone(zone)
        if local.utcoffset() != value.utcoffset():
            raise ReminderError("Смещение времени не совпадает с часовым поясом пользователя.")
    if local <= now:
        raise ReminderError("Время напоминания должно быть в будущем.")
    return local
