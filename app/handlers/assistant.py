import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.assistant import GENERIC_ERROR, Assistant
from app.prompts import PROMPTS

logger = logging.getLogger("app.handlers")


async def reply_text(message: Message, text: str) -> None:
    await message.answer(text, parse_mode=None)


async def start(message: Message, assistant: Assistant) -> None:
    await reply_text(message, await assistant.start_text(message.from_user.id))


async def switch_mode(message: Message, command: CommandObject, assistant: Assistant) -> None:
    await reply_text(message, await assistant.switch_mode(message.from_user.id, command.command))


async def settings(message: Message, command: CommandObject, assistant: Assistant) -> None:
    user_id = message.from_user.id
    if command.args:
        text = await assistant.set_temperature(user_id, command.args)
    else:
        text = await assistant.show_settings(user_id)
    await reply_text(message, text)


async def reset(message: Message, assistant: Assistant) -> None:
    await reply_text(message, await assistant.reset(message.from_user.id))


async def unknown_command(message: Message) -> None:
    await reply_text(message, "Неизвестная команда. Список команд: /start")


async def chat(message: Message, assistant: Assistant) -> None:
    try:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            parts = await assistant.answer(message.from_user.id, message.text)
    except Exception:
        logger.exception("Не удалось обработать сообщение.")
        parts = [GENERIC_ERROR]
    for part in parts:
        await reply_text(message, part)


def build_router() -> Router:
    """Новый роутер на каждый вызов: aiogram не позволяет подключить один дважды."""
    router = Router(name="assistant")
    # Только текстовые сообщения из личных чатов.
    router.message.filter(F.chat.type == "private", F.from_user, F.text)
    # Порядок важен: команды раньше общего обработчика текста.
    router.message.register(start, CommandStart())
    router.message.register(switch_mode, Command(*PROMPTS))
    router.message.register(settings, Command("settings"))
    router.message.register(reset, Command("reset"))
    router.message.register(unknown_command, F.text.startswith("/"))
    router.message.register(chat)
    return router
