"""Погода через Open-Meteo: геокодирование и текущие условия."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
from aiohttp_socks import ProxyConnector

logger = logging.getLogger("app.mcp_server.weather")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 8
MAX_RETRIES = 2
DOMINANCE_RATIO = 10
MIN_DOMINANT_POPULATION = 100_000

# Коды погоды WMO (Open-Meteo) в текст.
WMO = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозевый туман",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "слабая ледяная морось",
    57: "ледяная морось",
    61: "слабый дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "слабый ледяной дождь",
    67: "ледяной дождь",
    71: "слабый снег",
    73: "снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "слабый ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "слабый снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}

FetchJson = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class WeatherUnavailable(Exception):
    """Погодный сервис недоступен; `reason`: timeout, unavailable или bad_response."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def make_fetch_json(proxy_url: str = "") -> FetchJson:
    """HTTP-клиент с конечным тайм-аутом; повторы только для чтения (GET)."""

    async def fetch(url: str, params: dict[str, Any]) -> dict[str, Any]:
        last = "unavailable"
        for attempt in range(MAX_RETRIES + 1):
            connector = ProxyConnector.from_url(proxy_url) if proxy_url else None
            timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
            try:
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    async with session.get(url, params=params) as response:
                        if response.status >= 500:
                            last = "unavailable"
                            continue
                        if response.status != 200:
                            raise WeatherUnavailable("bad_response")
                        data = await response.json(content_type=None)
                        if not isinstance(data, dict):
                            raise WeatherUnavailable("bad_response")
                        return data
            except TimeoutError:
                last = "timeout"
            except (aiohttp.ContentTypeError, ValueError):
                raise WeatherUnavailable("bad_response") from None
            except aiohttp.ClientError:
                last = "unavailable"
            logger.warning("Погодный сервис: попытка %d неудачна (%s).", attempt + 1, last)
        raise WeatherUnavailable(last)

    return fetch


def _split_city(city: str) -> tuple[str, str]:
    name, _, country = city.partition(",")
    return name.strip(), country.strip()


def _matches_country(place: dict[str, Any], hint: str) -> bool:
    hint = hint.casefold()
    return hint in {
        str(place.get("country", "")).casefold(),
        str(place.get("country_code", "")).casefold(),
    }


def _population(place: dict[str, Any]) -> int:
    value = place.get("population")
    return value if isinstance(value, int) else 0


def _distinct(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for place in places:
        key = (str(place.get("country", "")), str(place.get("admin1", "")))
        if key not in seen:
            seen.add(key)
            result.append(place)
    return result


async def get_weather(city: str, fetch: FetchJson) -> dict[str, Any]:
    """Возвращает результат со `status`: ok, ambiguous, not_found или error."""
    name, country_hint = _split_city(city)
    if not name:
        return {"status": "not_found"}
    try:
        found = await fetch(
            GEOCODING_URL, {"name": name, "count": 10, "language": "ru", "format": "json"}
        )
        places = [p for p in found.get("results") or [] if isinstance(p, dict)]
        if country_hint:
            places = [p for p in places if _matches_country(p, country_hint)]
        places = _distinct(places)
        if not places:
            return {"status": "not_found"}
        if len(places) > 1:
            first, second = places[0], places[1]
            dominant = _population(first) >= MIN_DOMINANT_POPULATION and _population(
                first
            ) >= DOMINANCE_RATIO * _population(second)
            if not dominant:
                return {
                    "status": "ambiguous",
                    "candidates": [
                        {
                            "name": str(p.get("name", "")),
                            "country": str(p.get("country", "")),
                            "admin1": str(p.get("admin1", "")),
                        }
                        for p in places[:5]
                    ],
                }
        place = places[0]
        forecast = await fetch(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "wind_speed_unit": "ms",
                "timezone": "auto",
            },
        )
        return _ok(place, forecast)
    except WeatherUnavailable as exc:
        return {"status": "error", "reason": exc.reason}
    except (KeyError, TypeError, ValueError):
        return {"status": "error", "reason": "bad_response"}


def _format_offset(seconds: int) -> str:
    sign = "+" if seconds >= 0 else "-"
    hours, rest = divmod(abs(seconds), 3600)
    return f"{sign}{hours:02d}:{rest // 60:02d}"


def _ok(place: dict[str, Any], forecast: dict[str, Any]) -> dict[str, Any]:
    current = forecast["current"]
    offset = _format_offset(int(forecast.get("utc_offset_seconds", 0)))
    observed = str(current["time"])
    if len(observed) == 16:  # YYYY-MM-DDTHH:MM
        observed += ":00"
    return {
        "status": "ok",
        "place": str(place["name"]),
        "country": str(place.get("country", "")),
        "temperature_c": float(current["temperature_2m"]),
        "conditions": WMO.get(int(current["weather_code"]), "условия не определены"),
        "wind_speed": float(current["wind_speed_10m"]),
        "wind_unit": "м/с",
        "observed_at": observed + offset,
        "timezone": str(forecast.get("timezone", "")),
    }
