"""Единственное место чтения и проверки настроек приложения."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_proxy_url: str = field(default="", repr=False)
    llm_timeout_seconds: int = 30
    history_max_messages: int = 20
    history_max_chars: int = 6000

    @classmethod
    def load(
        cls, env_file: Path | str = ".env", *, environ: Mapping[str, str] | None = None
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        def positive_int(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if result < 1:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно целое число больше нуля.") from None

        def proxy_url(key: str) -> str:
            result = value(key)
            if result:
                try:
                    parsed = urlsplit(result)
                    if (
                        parsed.scheme not in {"http", "socks5"}
                        or not parsed.hostname
                        or not parsed.port
                        or parsed.path not in {"", "/"}
                        or parsed.query
                        or parsed.fragment
                    ):
                        raise ValueError
                except ValueError:
                    raise ConfigError(
                        f"{key}: нужен http://host:port или socks5://host:port; "
                        "при необходимости добавьте user:password@."
                    ) from None
            return result

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        proxy = proxy_url("TELEGRAM_PROXY_URL")
        llm_proxy = proxy_url("LLM_PROXY_URL")
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        return cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
            llm_base_url=value("LLM_BASE_URL").rstrip("/"),
            llm_model=value("LLM_MODEL"),
            llm_api_key=value("LLM_API_KEY"),
            llm_proxy_url=llm_proxy,
            llm_timeout_seconds=positive_int("LLM_TIMEOUT_SECONDS", "30"),
            history_max_messages=positive_int("HISTORY_MAX_MESSAGES", "20"),
            history_max_chars=positive_int("HISTORY_MAX_CHARS", "6000"),
        )

    def require_llm(self) -> None:
        """Проверяет настройки языковой модели; нужна только самому боту."""
        parsed = urlsplit(self.llm_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigError("LLM_BASE_URL: нужен адрес API вида https://host/v1.")
        if not self.llm_model:
            raise ConfigError("LLM_MODEL: имя модели не задано.")
        if not self.llm_api_key:
            raise ConfigError("LLM_API_KEY: ключ доступа к API не задан.")
