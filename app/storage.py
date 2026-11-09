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
