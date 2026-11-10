from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.assistant import (
    GENERIC_ERROR,
    TELEGRAM_LIMIT,
    Assistant,
    split_message,
    trim_history,
)
from app.config import Settings
from app.handlers.assistant import build_router
from app.llm import LLMError
from app.prompts import PROMPTS

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


class FakeStorage:
    """Хранилище в памяти с тем же интерфейсом, что и PgStorage."""

    def __init__(self):
        self.users = {}
        self.messages = {}

    async def get_user(self, user_id):
        return dict(self.users.setdefault(user_id, {"mode": "study", "temperature": 0.3}))

    async def set_mode(self, user_id, mode):
        self.users[user_id]["mode"] = mode
        self.messages[user_id] = []

    async def set_temperature(self, user_id, temperature):
        self.users[user_id]["temperature"] = temperature

    async def history(self, user_id, limit):
        return list(self.messages.get(user_id, []))[-limit:]

    async def add_exchange(self, user_id, question, answer):
        self.messages.setdefault(user_id, []).extend(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        )

    async def reset_history(self, user_id):
        self.messages[user_id] = []


class FakeLLM:
    def __init__(self, reply="Ответ модели", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    async def complete(self, messages, temperature):
        self.calls.append({"messages": messages, "temperature": temperature})
        if self.error:
            raise self.error
        return self.reply


def make_settings(**overrides):
    return Settings(
        bot_token="unused",
        postgres_password="unused",
        llm_model="test-model",
        **overrides,
    )


def make_assistant(llm=None, store=None, **overrides):
    store = store or FakeStorage()
    llm = llm or FakeLLM()
    return Assistant(store, llm, make_settings(**overrides)), store, llm


# --- Сценарий 1: сообщение передаётся модели с инструкцией активного режима ---


async def test_message_goes_to_model_with_active_mode_prompt():
    # Arrange
    assistant, store, llm = make_assistant()
    await assistant.switch_mode(1, "translate")
    # Act
    parts = await assistant.answer(1, "на английский: привет")
    # Assert
    assert parts == ["Ответ модели"]
    assert llm.calls[0]["messages"] == [
        {"role": "system", "content": PROMPTS["translate"]},
        {"role": "user", "content": "на английский: привет"},
    ]
    assert llm.calls[0]["temperature"] == 0.3


async def test_history_is_sent_in_chronological_order_and_saved():
    # Arrange
    assistant, store, llm = make_assistant()
    # Act
    await assistant.answer(1, "первый")
    await assistant.answer(1, "второй")
    # Assert
    roles_and_texts = [(m["role"], m["content"]) for m in llm.calls[1]["messages"][1:]]
    assert roles_and_texts == [
        ("user", "первый"),
        ("assistant", "Ответ модели"),
        ("user", "второй"),
    ]


# --- Сценарий 2: истории двух пользователей не смешиваются ---


async def test_histories_of_two_users_are_separate():
    # Arrange
    assistant, store, llm = make_assistant()
    # Act
    await assistant.answer(1, "вопрос Ани")
    await assistant.answer(2, "вопрос Бори")
    # Assert
    second_call = llm.calls[1]["messages"]
    assert [m["content"] for m in second_call[1:]] == ["вопрос Бори"]
    assert [m["content"] for m in store.messages[1]] == ["вопрос Ани", "Ответ модели"]


# --- Сценарий 3: сокращение истории ---


def msg(role, text):
    return {"role": role, "content": text}


def test_trim_by_message_count_keeps_newest_in_order():
    # Arrange
    history = [msg("user", "1"), msg("assistant", "2"), msg("user", "3"), msg("assistant", "4")]
    # Act
    kept = trim_history(history, "новый", max_messages=3, max_chars=1000)
    # Assert: берутся 3 последних, ведущая реплика ассистента отбрасывается
    assert kept == [msg("user", "3"), msg("assistant", "4")]


def test_trim_by_size_drops_oldest_and_reserves_room_for_current_request():
    # Arrange
    history = [msg("user", "a" * 40), msg("assistant", "b" * 40), msg("user", "c" * 40)]
    # Act: бюджет 100 - 30 (текущий запрос) = 70 символов на историю
    kept = trim_history(history, "x" * 30, max_messages=10, max_chars=100)
    # Assert
    assert kept == [msg("user", "c" * 40)]


def test_trim_current_request_is_never_lost_even_if_huge():
    # Arrange
    history = [msg("user", "старое")]
    # Act
    kept = trim_history(history, "x" * 500, max_messages=10, max_chars=100)
    # Assert
    assert kept == []


async def test_model_request_respects_configured_limits_and_keeps_current_text():
    # Arrange
    assistant, store, llm = make_assistant(history_max_messages=2, history_max_chars=1000)
    for text in ("один", "два", "три"):
        await assistant.answer(1, text)
    # Act
    await assistant.answer(1, "четыре")
    # Assert: система + (не больше 2 сообщений истории) + текущий запрос
    sent = llm.calls[-1]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[-1] == {"role": "user", "content": "четыре"}
    assert len(sent) - 2 <= 2


# --- Сценарий 4: смена режима очищает историю и сохраняет настройку ---


async def test_mode_switch_clears_history_and_saves_mode():
    # Arrange
    assistant, store, llm = make_assistant()
    await assistant.answer(1, "вопрос")
    await assistant.set_temperature(1, "0.7")
    # Act
    reply = await assistant.switch_mode(1, "joke")
    # Assert
    assert reply.startswith("Режим переключён: /joke")
    assert store.users[1]["mode"] == "joke"
    assert store.messages[1] == []
    assert store.users[1]["temperature"] == 0.7
    assert len(llm.calls) == 1  # сама команда в модель не отправлялась


# --- Сценарий 5: /reset очищает только историю вызвавшего ---


async def test_reset_clears_only_callers_history_and_keeps_settings():
    # Arrange
    assistant, store, llm = make_assistant()
    await assistant.switch_mode(1, "translate")
    await assistant.set_temperature(1, "1.0")
    await assistant.answer(1, "вопрос 1")
    await assistant.answer(2, "вопрос 2")
    # Act
    reply = await assistant.reset(1)
    # Assert
    assert reply == "История диалога очищена. Режим и temperature сохранены."
    assert store.messages[1] == []
    assert [m["content"] for m in store.messages[2]] == ["вопрос 2", "Ответ модели"]
    assert store.users[1] == {"mode": "translate", "temperature": 1.0}


# --- Сценарий 6: /settings принимает разрешённое значение и отклоняет неверное ---


@pytest.mark.parametrize("raw", ["0.0", "0.3", "0.7", "1.0", "0,7", " 1 "])
async def test_settings_accepts_allowed_values(raw):
    # Arrange
    assistant, store, llm = make_assistant()
    # Act
    reply = await assistant.set_temperature(1, raw)
    # Assert
    assert reply.startswith("Temperature: ")
    assert store.users[1]["temperature"] == float(raw.strip().replace(",", "."))


@pytest.mark.parametrize("raw", ["0.5", "2", "abc", "-0.3", "nan"])
async def test_settings_rejects_invalid_values_before_any_api_call(raw):
    # Arrange
    assistant, store, llm = make_assistant()
    # Act
    reply = await assistant.set_temperature(1, raw)
    # Assert
    assert (
        reply == "Такое значение не подходит. Допустимые значения temperature: 0.0, 0.3, 0.7, 1.0."
    )
    assert 1 not in store.users
    assert llm.calls == []


async def test_new_temperature_is_applied_to_next_request_only_for_that_user():
    # Arrange
    assistant, store, llm = make_assistant()
    await assistant.set_temperature(1, "0.7")
    # Act
    await assistant.answer(1, "вопрос")
    await assistant.answer(2, "вопрос")
    # Assert
    assert [call["temperature"] for call in llm.calls] == [0.7, 0.3]


async def test_settings_shows_mode_model_and_temperature():
    # Arrange
    assistant, store, llm = make_assistant()
    await assistant.switch_mode(1, "joke")
    # Act
    text = await assistant.show_settings(1)
    # Assert
    assert "Режим: joke" in text
    assert "Модель: test-model" in text
    assert "Temperature: 0.3" in text


# --- Сценарий 7: ошибка модели превращается в безопасное сообщение ---


async def test_llm_error_becomes_safe_message_and_is_not_saved():
    # Arrange
    llm = FakeLLM(error=LLMError("Сервис модели недоступен. Попробуйте позже."))
    assistant, store, _ = make_assistant(llm=llm)
    # Act
    parts = await assistant.answer(1, "вопрос")
    # Assert
    assert parts == ["Сервис модели недоступен. Попробуйте позже."]
    assert store.messages.get(1, []) == []


# --- Сценарий 8: длинный ответ делится без потери текста ---


def test_split_message_keeps_every_character_and_order():
    # Arrange
    text = ("слово " * 20 + "\n") * 60
    # Act
    parts = split_message(text)
    # Assert
    assert len(parts) > 1
    assert all(len(part) <= TELEGRAM_LIMIT for part in parts)
    assert "".join(parts) == text


def test_split_message_handles_text_without_spaces():
    # Arrange
    text = "я" * (TELEGRAM_LIMIT * 2 + 10)
    # Act
    parts = split_message(text)
    # Assert
    assert [len(part) for part in parts] == [TELEGRAM_LIMIT, TELEGRAM_LIMIT, 10]
    assert "".join(parts) == text


async def test_long_answer_is_returned_in_parts_not_longer_than_limit():
    # Arrange
    long_reply = ("абзац " * 100 + "\n") * 20
    assistant, store, llm = make_assistant(llm=FakeLLM(reply=long_reply))
    # Act
    parts = await assistant.answer(1, "вопрос")
    # Assert
    assert len(parts) > 1
    assert all(len(part) <= TELEGRAM_LIMIT for part in parts)
    assert "".join(parts) == long_reply.strip()


# --- Обработчики Telegram ---


def make_update(text, chat_type="private", user_id=42):
    message = Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=user_id, type=chat_type),
        from_user=User(id=user_id, is_bot=False, first_name="Студент"),
        text=text,
    )
    return Update(update_id=1, message=message)


async def feed(update, assistant):
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router())
    await dispatcher.feed_update(bot, update, assistant=assistant)
    calls = [call.args[1] for call in bot.session.call_args_list]
    return [method.text for method in calls if isinstance(method, SendMessage)]


async def test_handler_sends_model_reply_for_plain_text():
    # Arrange
    assistant, store, llm = make_assistant()
    # Act
    sent = await feed(make_update("что такое стек?"), assistant)
    # Assert
    assert sent == ["Ответ модели"]
    assert llm.calls[0]["messages"][-1] == {"role": "user", "content": "что такое стек?"}


async def test_handler_splits_long_answer_into_several_messages():
    # Arrange
    long_reply = "я" * (TELEGRAM_LIMIT + 5)
    assistant, store, llm = make_assistant(llm=FakeLLM(reply=long_reply))
    # Act
    sent = await feed(make_update("вопрос"), assistant)
    # Assert
    assert [len(part) for part in sent] == [TELEGRAM_LIMIT, 5]


async def test_handler_command_is_not_sent_to_model():
    # Arrange
    assistant, store, llm = make_assistant()
    # Act
    sent = await feed(make_update("/translate"), assistant)
    # Assert
    assert sent[0].startswith("Режим переключён: /translate")
    assert llm.calls == []


async def test_handler_ignores_group_chats():
    # Arrange
    assistant, store, llm = make_assistant()
    # Act
    sent = await feed(make_update("привет", chat_type="group"), assistant)
    # Assert
    assert sent == []
    assert llm.calls == []


async def test_handler_hides_unexpected_errors():
    # Arrange
    assistant, store, llm = make_assistant()
    assistant._store.get_user = AsyncMock(side_effect=RuntimeError("password=secret"))
    # Act
    sent = await feed(make_update("вопрос"), assistant)
    # Assert
    assert sent == [GENERIC_ERROR]
