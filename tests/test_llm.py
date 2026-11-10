import asyncio
import logging

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from app.config import Settings
from app.llm import LLMClient, LLMError

API_KEY = "sk-very-secret"
MESSAGES = [
    {"role": "system", "content": "Ты помощник."},
    {"role": "user", "content": "Привет"},
]


def make_settings(base_url, timeout=5):
    return Settings(
        bot_token="unused",
        postgres_password="unused",
        llm_base_url=base_url,
        llm_model="test-model",
        llm_api_key=API_KEY,
        llm_timeout_seconds=timeout,
    )


@pytest.fixture
async def serve():
    servers = []

    async def start(handler):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        return make_settings(f"http://{server.host}:{server.port}/v1")

    yield start
    for server in servers:
        await server.close()


def reply(content):
    return web.json_response({"choices": [{"message": {"role": "assistant", "content": content}}]})


async def test_complete_sends_roles_order_and_parameters(serve):
    # Arrange
    seen = {}

    async def handler(request):
        seen["body"] = await request.json()
        seen["auth"] = request.headers["Authorization"]
        return reply("Привет, студент!")

    client = LLMClient(await serve(handler))
    # Act
    text = await client.complete(MESSAGES, 0.7)
    await client.close()
    # Assert
    assert text == "Привет, студент!"
    assert seen["body"] == {"model": "test-model", "messages": MESSAGES, "temperature": 0.7}
    assert seen["auth"] == f"Bearer {API_KEY}"


async def test_service_error_becomes_safe_message(serve):
    # Arrange
    async def handler(request):
        return web.json_response({"error": f"leak {API_KEY}"}, status=500)

    client = LLMClient(await serve(handler))
    # Act / Assert
    with pytest.raises(LLMError) as exc:
        await client.complete(MESSAGES, 0.3)
    await client.close()
    assert str(exc.value) == "Сервис модели вернул ошибку (HTTP 500)."
    assert API_KEY not in str(exc.value)


@pytest.mark.parametrize(
    "content",
    [None, "", "   \n"],
)
async def test_empty_answer_is_rejected(serve, content):
    # Arrange
    async def handler(request):
        return reply(content)

    client = LLMClient(await serve(handler))
    # Act / Assert
    with pytest.raises(LLMError, match="пустой ответ"):
        await client.complete(MESSAGES, 0.3)
    await client.close()


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{}]}, [1, 2]])
async def test_unexpected_json_shape_is_rejected(serve, payload):
    # Arrange
    async def handler(request):
        return web.json_response(payload)

    client = LLMClient(await serve(handler))
    # Act / Assert
    with pytest.raises(LLMError, match="непонятном формате"):
        await client.complete(MESSAGES, 0.3)
    await client.close()


async def test_non_json_body_is_rejected(serve):
    # Arrange
    async def handler(request):
        return web.Response(text="<html>oops</html>", content_type="text/html")

    client = LLMClient(await serve(handler))
    # Act / Assert
    with pytest.raises(LLMError, match="непонятном формате"):
        await client.complete(MESSAGES, 0.3)
    await client.close()


async def test_timeout_becomes_safe_message(serve):
    # Arrange
    async def handler(request):
        await asyncio.sleep(5)
        return reply("поздно")

    settings = await serve(handler)
    client = LLMClient(make_settings(settings.llm_base_url, timeout=1))
    # Act / Assert
    with pytest.raises(LLMError, match="слишком долго"):
        await client.complete(MESSAGES, 0.3)
    await client.close()


async def test_unreachable_service_becomes_safe_message():
    # Arrange: порт 1 никто не слушает.
    client = LLMClient(make_settings("http://127.0.0.1:1/v1"))
    # Act / Assert
    with pytest.raises(LLMError, match="недоступен"):
        await client.complete(MESSAGES, 0.3)
    await client.close()


async def test_logs_have_no_secret_or_dialog_text(serve, caplog):
    # Arrange
    async def handler(request):
        return reply("секретный ответ модели")

    client = LLMClient(await serve(handler))
    caplog.set_level(logging.INFO, logger="app.llm")
    # Act
    await client.complete(MESSAGES, 0.3)
    await client.close()
    # Assert
    logged = caplog.text
    assert "test-model" in logged
    for forbidden in (API_KEY, "Привет", "Ты помощник", "секретный ответ модели"):
        assert forbidden not in logged
