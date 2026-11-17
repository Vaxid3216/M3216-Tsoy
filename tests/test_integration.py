"""RUN_INTEGRATION=1 python -m pytest -m integration -v.

Тест создаёт собственные контейнер и volume; пользовательскую БД не использует.
"""

import asyncio
import json
import os
import secrets
import socket
import time
from dataclasses import replace
from datetime import UTC, datetime

import asyncpg
import pytest

from app.config import Settings
from app.db import create_pool
from app.health import HealthState, health_result
from scripts.common import CommandError, run_command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]


@pytest.fixture
def database():
    name = "itmo-test-" + secrets.token_hex(6)
    volume = name + "-data"
    password = secrets.token_urlsafe(24)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def docker(*args):
        return run_command(["docker", *args], label="Интеграционный Docker-тест", timeout=180)

    def wait_until_ready():
        deadline = time.monotonic() + 60
        while True:
            try:
                docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "bot", "-d", "bot")
                return
            except CommandError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    docker("info")
    docker("volume", "create", volume)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:5432",
            "-e",
            "POSTGRES_DB=bot",
            "-e",
            "POSTGRES_USER=bot",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/postgresql/data",
            "postgres:16-bookworm",
        )
        info = json.loads(docker("inspect", name))[0]
        assert int(info["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]) == port
        wait_until_ready()
        yield (
            Settings(
                bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                postgres_password=password,
                postgres_port=port,
            ),
            docker,
            wait_until_ready,
            name,
        )
    finally:
        try:
            docker("rm", "-f", name)
        finally:
            docker("volume", "rm", volume)


async def test_real_db_password_restart_health_and_persistence(database):
    # Arrange
    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    task = asyncio.create_task(asyncio.Event().wait())
    state = HealthState(pool=pool, initialized=True, polling_task=task)
    try:
        await pool.execute("CREATE TABLE integration_probe (value text NOT NULL)")
        await pool.execute("INSERT INTO integration_probe VALUES ($1)", "Сохранилось 👋")
        # Act / Assert: реальное подключение и ошибка пароля.
        assert (await health_result(state))[0] == 200
        with pytest.raises(asyncpg.InvalidPasswordError):
            await create_pool(replace(settings, postgres_password="wrong-password"))
        # Act / Assert: потеря соединения и сохранность данных после запуска.
        await asyncio.to_thread(docker, "stop", name)
        assert (await health_result(state))[0] == 503
        await asyncio.to_thread(docker, "start", name)
        await asyncio.to_thread(wait_until_ready)
        for _ in range(60):
            if (await health_result(state))[0] == 200:
                break
            await asyncio.sleep(1)
        assert (await health_result(state))[0] == 200
        assert await pool.fetchval("SELECT value FROM integration_probe") == "Сохранилось 👋"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()


async def test_pending_actions_confirmation_is_atomic_isolated_and_idempotent(database):
    # Arrange
    from app import storage

    settings, *_ = database
    pool = await create_pool(settings)
    try:
        await storage.ensure_schema(pool)
        await storage.ensure_schema(pool)  # повторное применение схемы безопасно
        for user_id in (1, 2):
            await storage.get_or_create_user(pool, user_id)
        args = {"text": "отправить отчёт", "remind_at": "2030-01-01T18:30:00+03:00"}

        # Act / Assert: один и тот же update даёт одно действие.
        action_id, created = await storage.prepare_action(pool, 1, 100, "add_reminder", args)
        again, created_again = await storage.prepare_action(pool, 1, 100, "add_reminder", args)
        assert (created, created_again, again) == (True, False, action_id)

        # Чужой пользователь не видит и не подтверждает действие (T11).
        assert await storage.get_action(pool, 2, action_id) is None
        assert await storage.begin_execution(pool, 2, action_id) is None
        assert await storage.cancel_action(pool, 2, action_id) is False

        # Конкурентное двойное подтверждение: выигрывает ровно одно (T10).
        results = await asyncio.gather(
            *(storage.begin_execution(pool, 1, action_id) for _ in range(5))
        )
        winners = [r for r in results if r is not None]
        assert len(winners) == 1
        assert winners[0]["args"] == args

        # Два вызова add_reminder с ключом действия дают одно напоминание.
        when = datetime(2030, 1, 1, 15, 30, tzinfo=UTC)
        key = str(action_id)
        first = await storage.create_reminder(pool, 1, "x", when, "Europe/Moscow", key)
        second = await storage.create_reminder(pool, 1, "x", when, "Europe/Moscow", key)
        assert first[1] is True and second == (first[0], False)
        assert len(await storage.list_reminders(pool, 1)) == 1
        assert await storage.list_reminders(pool, 2) == []

        # Завершение и повторные переходы.
        assert await storage.finish_action(pool, 1, action_id, success=True) is True
        assert await storage.finish_action(pool, 1, action_id, success=False) is False
        assert (await storage.get_action(pool, 1, action_id))["status"] == "executed"
        assert await storage.cancel_action(pool, 1, action_id) is False

        # Отмена блокирует подтверждение; просроченное действие не выполняется.
        cancelled, _ = await storage.prepare_action(pool, 1, 101, "add_reminder", args)
        assert await storage.cancel_action(pool, 1, cancelled) is True
        assert await storage.begin_execution(pool, 1, cancelled) is None
        expired, _ = await storage.prepare_action(pool, 1, 102, "add_reminder", args, ttl_seconds=0)
        assert await storage.begin_execution(pool, 1, expired) is None
        assert await storage.expire_action(pool, 1, expired) is True
        assert (await storage.get_action(pool, 1, expired))["status"] == "expired"

        # Дедупликация update и аудит.
        assert await storage.mark_update_processed(pool, 555) is True
        assert await storage.mark_update_processed(pool, 555) is False
        await storage.add_audit(
            pool, 1, "call_tool", "passed", "success", tool="get_weather", args_summary="Казань"
        )
        last = await storage.last_audit(pool, 1)
        assert (last["tool"], last["args_summary"]) == ("get_weather", "Казань")
        assert await storage.last_audit(pool, 2) is None

        # Часовой пояс сохраняется, ограничения БД не пускают неверный статус.
        await storage.set_timezone(pool, 1, "Europe/Moscow")
        assert await storage.get_timezone(pool, 1) == "Europe/Moscow"
        assert await storage.get_timezone(pool, 2) is None
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute("UPDATE pending_actions SET status = 'weird'")
    finally:
        await pool.close()
