"""Подпись доверенного контекста владельца напоминания.

Бот кладёт владельца в служебное поле `_meta` MCP-вызова и подписывает его общим
секретом. Модель владельца не задаёт, а клиент без секрета (например, Inspector)
не может подделать подпись.
"""

import hmac
from dataclasses import dataclass
from hashlib import sha256


class OwnerError(ValueError):
    """Контекст владельца отсутствует или подпись неверна."""


@dataclass(frozen=True)
class OwnerContext:
    owner_id: int
    idempotency_key: str
    timezone: str


def _payload(owner_id: int, idempotency_key: str, timezone: str) -> bytes:
    return f"{owner_id}\n{idempotency_key}\n{timezone}".encode()


def sign(secret: str, owner_id: int, idempotency_key: str, timezone: str) -> str:
    return hmac.new(
        secret.encode(), _payload(owner_id, idempotency_key, timezone), sha256
    ).hexdigest()


def build_meta(
    secret: str, owner_id: int, idempotency_key: str, timezone: str
) -> dict[str, object]:
    """Поле `_meta` для вызова `add_reminder` (используется MCP-клиентом бота)."""
    return {
        "owner_id": owner_id,
        "idempotency_key": idempotency_key,
        "timezone": timezone,
        "signature": sign(secret, owner_id, idempotency_key, timezone),
    }


def verify(secret: str, meta: dict[str, object] | None) -> OwnerContext:
    if not secret:
        raise OwnerError("Подпись владельца не настроена на сервере.")
    if not meta:
        raise OwnerError("Нет подписанного контекста владельца.")
    owner_id = meta.get("owner_id")
    key = meta.get("idempotency_key")
    zone = meta.get("timezone")
    signature = meta.get("signature")
    if (
        not isinstance(owner_id, int)
        or isinstance(owner_id, bool)
        or not isinstance(key, str)
        or not 1 <= len(key) <= 128
        or not isinstance(zone, str)
        or not isinstance(signature, str)
    ):
        raise OwnerError("Контекст владельца неполный или неверного формата.")
    expected = sign(secret, owner_id, key, zone)
    if not hmac.compare_digest(expected, signature):
        raise OwnerError("Подпись владельца неверна.")
    return OwnerContext(owner_id=owner_id, idempotency_key=key, timezone=zone)
