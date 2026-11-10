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
