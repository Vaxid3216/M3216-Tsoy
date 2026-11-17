import json
from datetime import datetime
from uuid import UUID, uuid4

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

CREATE TABLE IF NOT EXISTS pending_actions (
    id         UUID PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    update_id  BIGINT NOT NULL,
    tool       TEXT NOT NULL,
    args       JSONB NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'executing', 'executed', 'failed', 'cancelled', 'expired')),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, update_id)
);

CREATE TABLE IF NOT EXISTS processed_updates (
    update_id   BIGINT PRIMARY KEY,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_audit (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id      BIGINT NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    action       TEXT NOT NULL CHECK (action IN ('respond', 'clarify', 'call_tool', 'prepare')),
    tool         TEXT,
    args_summary TEXT NOT NULL DEFAULT '',
    check_status TEXT NOT NULL CHECK (check_status IN ('passed', 'rejected', 'not_applicable')),
    exec_status  TEXT NOT NULL CHECK (exec_status IN ('success', 'error', 'not_executed')),
    reason       TEXT NOT NULL DEFAULT '',
    duration_ms  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS agent_audit_user_id_id_idx ON agent_audit (user_id, id);
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


async def set_timezone(db: asyncpg.Pool, user_id: int, timezone: str) -> None:
    await db.execute(
        "INSERT INTO users (user_id, timezone) VALUES ($1, $2) "
        "ON CONFLICT (user_id) DO UPDATE SET timezone = EXCLUDED.timezone",
        user_id,
        timezone,
    )


async def get_timezone(db: asyncpg.Pool, user_id: int) -> str | None:
    return await db.fetchval("SELECT timezone FROM users WHERE user_id = $1", user_id)


async def mark_update_processed(db: asyncpg.Pool, update_id: int) -> bool:
    """True, если update встречен впервые; повторная доставка даёт False."""
    row = await db.fetchrow(
        "INSERT INTO processed_updates (update_id) VALUES ($1) "
        "ON CONFLICT DO NOTHING RETURNING update_id",
        update_id,
    )
    return row is not None


async def prepare_action(
    db: asyncpg.Pool,
    user_id: int,
    update_id: int,
    tool: str,
    args: dict,
    ttl_seconds: int = 300,
) -> tuple[UUID, bool]:
    """Готовит действие к подтверждению; тот же update даёт прежнее действие (id, False)."""
    action_id = uuid4()
    row = await db.fetchrow(
        "INSERT INTO pending_actions (id, user_id, update_id, tool, args, expires_at) "
        "VALUES ($1, $2, $3, $4, $5::jsonb, now() + make_interval(secs => $6)) "
        "ON CONFLICT (user_id, update_id) DO NOTHING RETURNING id",
        action_id,
        user_id,
        update_id,
        tool,
        json.dumps(args, ensure_ascii=False),
        float(ttl_seconds),
    )
    if row is not None:
        return action_id, True
    existing = await db.fetchval(
        "SELECT id FROM pending_actions WHERE user_id = $1 AND update_id = $2",
        user_id,
        update_id,
    )
    return existing, False


async def get_action(db: asyncpg.Pool, user_id: int, action_id: UUID) -> dict | None:
    row = await db.fetchrow(
        "SELECT id, tool, args, status, expires_at FROM pending_actions "
        "WHERE id = $1 AND user_id = $2",
        action_id,
        user_id,
    )
    return None if row is None else {**row, "args": json.loads(row["args"])}


async def begin_execution(db: asyncpg.Pool, user_id: int, action_id: UUID) -> dict | None:
    """Атомарно pending -> executing; None, если действие чужое, не в pending или просрочено."""
    row = await db.fetchrow(
        "UPDATE pending_actions SET status = 'executing' "
        "WHERE id = $1 AND user_id = $2 AND status = 'pending' AND expires_at > now() "
        "RETURNING tool, args",
        action_id,
        user_id,
    )
    return None if row is None else {"tool": row["tool"], "args": json.loads(row["args"])}


async def finish_action(db: asyncpg.Pool, user_id: int, action_id: UUID, success: bool) -> bool:
    status = "executed" if success else "failed"
    result = await db.execute(
        "UPDATE pending_actions SET status = $3 "
        "WHERE id = $1 AND user_id = $2 AND status = 'executing'",
        action_id,
        user_id,
        status,
    )
    return result.endswith(" 1")


async def cancel_action(db: asyncpg.Pool, user_id: int, action_id: UUID) -> bool:
    result = await db.execute(
        "UPDATE pending_actions SET status = 'cancelled' "
        "WHERE id = $1 AND user_id = $2 AND status = 'pending'",
        action_id,
        user_id,
    )
    return result.endswith(" 1")


async def expire_action(db: asyncpg.Pool, user_id: int, action_id: UUID) -> bool:
    result = await db.execute(
        "UPDATE pending_actions SET status = 'expired' "
        "WHERE id = $1 AND user_id = $2 AND status = 'pending' AND expires_at <= now()",
        action_id,
        user_id,
    )
    return result.endswith(" 1")


async def add_audit(
    db: asyncpg.Pool,
    user_id: int,
    action: str,
    check_status: str,
    exec_status: str,
    *,
    tool: str | None = None,
    args_summary: str = "",
    reason: str = "",
    duration_ms: int = 0,
) -> None:
    await db.execute(
        "INSERT INTO agent_audit "
        "(user_id, action, tool, args_summary, check_status, exec_status, reason, duration_ms) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        user_id,
        action,
        tool,
        args_summary,
        check_status,
        exec_status,
        reason,
        duration_ms,
    )


async def last_audit(db: asyncpg.Pool, user_id: int) -> asyncpg.Record | None:
    return await db.fetchrow(
        "SELECT created_at, action, tool, args_summary, check_status, exec_status, reason "
        "FROM agent_audit WHERE user_id = $1 ORDER BY id DESC LIMIT 1",
        user_id,
    )


async def list_reminders(db: asyncpg.Pool, owner_id: int) -> list[asyncpg.Record]:
    return await db.fetch(
        "SELECT id, text, remind_at, timezone FROM reminders WHERE owner_id = $1 ORDER BY id",
        owner_id,
    )
