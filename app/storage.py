from datetime import datetime

import asyncpg

DEFAULT_MODE = "study"
DEFAULT_TEMPERATURE = "0.3"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users (
    user_id     BIGINT PRIMARY KEY,
    mode        TEXT NOT NULL DEFAULT '{DEFAULT_MODE}',
    temperature NUMERIC(2, 1) NOT NULL DEFAULT {DEFAULT_TEMPERATURE}
        CHECK (temperature IN (0.0, 0.3, 0.7, 1.0))
);

CREATE TABLE IF NOT EXISTS messages (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_user_id_id_idx ON messages (user_id, id);

ALTER TABLE users ADD COLUMN IF NOT EXISTS timezone TEXT;

CREATE TABLE IF NOT EXISTS reminders (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id        BIGINT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    text            TEXT NOT NULL CHECK (char_length(text) BETWEEN 1 AND 500),
    remind_at       TIMESTAMPTZ NOT NULL,
    timezone        TEXT NOT NULL,
    idempotency_key TEXT NOT NULL CHECK (char_length(idempotency_key) BETWEEN 1 AND 128),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (owner_id, idempotency_key)
);
"""


async def ensure_schema(db: asyncpg.Pool) -> None:
    await db.execute(SCHEMA)


async def get_or_create_user(db: asyncpg.Pool, user_id: int) -> asyncpg.Record:
    await db.execute(
        "INSERT INTO users (user_id) VALUES ($1) ON CONFLICT DO NOTHING",
        user_id,
    )
    return await db.fetchrow("SELECT mode, temperature FROM users WHERE user_id = $1", user_id)


async def set_temperature(db: asyncpg.Pool, user_id: int, temperature: float) -> None:
    await db.execute(
        "UPDATE users SET temperature = $2 WHERE user_id = $1",
        user_id,
        temperature,
    )


async def set_mode(db: asyncpg.Pool, user_id: int, mode: str) -> None:
    await db.execute(
        "UPDATE users SET mode = $2 WHERE user_id = $1",
        user_id,
        mode,
    )
    await db.execute("DELETE FROM messages WHERE user_id = $1", user_id)


async def add_message(db: asyncpg.Pool, user_id: int, role: str, content: str) -> None:
    await db.execute(
        "INSERT INTO messages (user_id, role, content) VALUES ($1, $2, $3)",
        user_id,
        role,
        content,
    )


async def get_history(db: asyncpg.Pool, user_id: int, limit: int) -> list[asyncpg.Record]:
    rows = await db.fetch(
        "SELECT role, content, created_at FROM messages "
        "WHERE user_id = $1 ORDER BY id DESC LIMIT $2",
        user_id,
        limit,
    )
    return list(reversed(rows))


async def reset_history(db: asyncpg.Pool, user_id: int) -> None:
    await db.execute(
        "DELETE FROM messages WHERE user_id = $1",
        user_id,
    )


async def create_reminder(
    db: asyncpg.Pool,
    owner_id: int,
    text: str,
    remind_at: datetime,
    timezone: str,
    idempotency_key: str,
) -> tuple[int, bool]:
    """Создаёт напоминание; повтор с тем же ключом возвращает прежнее (id, False)."""
    row = await db.fetchrow(
        "INSERT INTO reminders (owner_id, text, remind_at, timezone, idempotency_key) "
        "VALUES ($1, $2, $3, $4, $5) "
        "ON CONFLICT (owner_id, idempotency_key) DO NOTHING RETURNING id",
        owner_id,
        text,
        remind_at,
        timezone,
        idempotency_key,
    )
    if row is not None:
        return row["id"], True
    existing = await db.fetchval(
        "SELECT id FROM reminders WHERE owner_id = $1 AND idempotency_key = $2",
        owner_id,
        idempotency_key,
    )
    return existing, False
