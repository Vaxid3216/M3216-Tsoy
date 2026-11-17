"""Обработчики лабы 2: кнопки подтверждения, команды агента и защита от повторов."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from aiogram import Bot, Dispatcher
from aiogram.methods import AnswerCallbackQuery, EditMessageReplyMarkup, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from app.agent.service import Agent
from app.config import Settings
from app.handlers.assistant import build_router
from app.handlers.dedup import UpdateDedupMiddleware
from tests.test_agent import NOW, REMINDER, FakeMcp, FakeStore, ScriptedLLM, tool_call

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


def make_agent(*script):
    store = FakeStore()
    settings = Settings(bot_token="x", postgres_password="x", llm_model="m")
    mcp = FakeMcp()
    return Agent(store, ScriptedLLM(*script), mcp, settings, clock=lambda: NOW), store, mcp


def user(user_id=42):
    return User(id=user_id, is_bot=False, first_name="Студент")


def message_update(text, update_id=1, user_id=42):
    message = Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=user_id, type="private"),
        from_user=user(user_id),
        text=text,
    )
    return Update(update_id=update_id, message=message)


def callback_update(data, update_id=2, user_id=42):
    message = Message(
        message_id=5, date=datetime.now(UTC), chat=Chat(id=user_id, type="private"), text="x"
    )
    callback = CallbackQuery(
        id="cb", from_user=user(user_id), chat_instance="ci", message=message, data=data
    )
    return Update(update_id=update_id, callback_query=callback)


async def feed(agent, *updates, middleware=None):
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    if middleware:
        dispatcher.update.outer_middleware(middleware)
    dispatcher.include_router(build_router())
    for update in updates:
        await dispatcher.feed_update(bot, update, agent=agent, assistant=None)
    methods = [call.args[1] for call in bot.session.call_args_list]
    return methods


def sent_texts(methods):
    return [m.text for m in methods if isinstance(m, SendMessage)]


async def test_reminder_request_shows_confirm_and_cancel_buttons():
    agent, store, mcp = make_agent(tool_call("add_reminder", REMINDER))
    methods = await feed(agent, message_update("Напомни завтра в 18:30 отправить отчёт"))
    (send,) = [m for m in methods if isinstance(m, SendMessage)]
    buttons = [b for row in send.reply_markup.inline_keyboard for b in row]
    assert [b.text for b in buttons] == ["Подтвердить", "Отменить"]
    action_id = str(next(iter(store.actions)))
    assert [b.callback_data for b in buttons] == [f"ok:{action_id}", f"no:{action_id}"]
    assert mcp.reminder_calls == []


async def test_confirm_button_creates_reminder_and_removes_keyboard():
    agent, store, mcp = make_agent(tool_call("add_reminder", REMINDER))
    await feed(agent, message_update("Напомни"))
    action_id = str(next(iter(store.actions)))
    methods = await feed(agent, callback_update(f"ok:{action_id}"))
    assert any("Напоминание №7 создано" in t for t in sent_texts(methods))
    assert any(isinstance(m, EditMessageReplyMarkup) for m in methods)
    assert any(isinstance(m, AnswerCallbackQuery) for m in methods)
    assert len(mcp.reminder_calls) == 1


async def test_cancel_button_and_foreign_button_press():
    agent, store, mcp = make_agent(tool_call("add_reminder", REMINDER))
    await feed(agent, message_update("Напомни"))
    action_id = str(next(iter(store.actions)))
    foreign = await feed(agent, callback_update(f"ok:{action_id}", user_id=99))
    assert sent_texts(foreign) == ["Это действие уже недействительно."]
    cancelled = await feed(agent, callback_update(f"no:{action_id}"))
    assert "не создано" in sent_texts(cancelled)[0]
    assert mcp.reminder_calls == []


async def test_commands_tools_why_timezone():
    agent, store, mcp = make_agent()
    methods = await feed(
        agent,
        message_update("/tools"),
        message_update("/timezone Europe/Moscow", update_id=2),
        message_update("/why", update_id=3),
    )
    tools, zone, why = sent_texts(methods)
    assert "get_weather" in tools and "сохранён" in zone and "не было" in why


async def test_repeated_update_is_processed_once():
    agent, store, mcp = make_agent(tool_call("add_reminder", REMINDER))

    class Store:
        seen: set[int] = set()

        async def mark_update_processed(self, update_id):
            new = update_id not in self.seen
            self.seen.add(update_id)
            return new

    update = message_update("Напомни", update_id=77)
    methods = await feed(agent, update, update, middleware=UpdateDedupMiddleware(Store()))
    assert len(sent_texts(methods)) == 1
    assert len(store.actions) == 1
