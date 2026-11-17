import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from aiogram.utils.chat_action import ChatActionSender

from app.agent.service import Agent
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


async def tools(message: Message, agent: Agent) -> None:
    await reply_text(message, await agent.tools_text())


async def why(message: Message, agent: Agent) -> None:
    await reply_text(message, await agent.why_text(message.from_user.id))


async def timezone(message: Message, command: CommandObject, agent: Agent) -> None:
    await reply_text(message, await agent.timezone_text(message.from_user.id, command.args))


def confirmation_keyboard(action_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Подтвердить", callback_data=f"ok:{action_id}"),
                InlineKeyboardButton(text="Отменить", callback_data=f"no:{action_id}"),
            ]
        ]
    )


async def chat(message: Message, agent: Agent, event_update: Update) -> None:
    try:
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            reply = await agent.handle(message.from_user.id, event_update.update_id, message.text)
    except Exception:
        logger.exception("Не удалось обработать сообщение.")
        await reply_text(message, GENERIC_ERROR)
        return
    for index, part in enumerate(reply.parts):
        is_last = index == len(reply.parts) - 1
        markup = (
            confirmation_keyboard(reply.confirmation.action_id)
            if (is_last and reply.confirmation)
            else None
        )
        await message.answer(part, parse_mode=None, reply_markup=markup)


async def _finish_callback(callback: CallbackQuery, text: str) -> None:
    await callback.answer()
    if callback.message is not None and hasattr(callback.message, "edit_reply_markup"):
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            logger.debug("Не удалось убрать кнопки.")
        await callback.message.answer(text, parse_mode=None)


async def confirm_callback(callback: CallbackQuery, agent: Agent) -> None:
    try:
        text = await agent.confirm(callback.from_user.id, callback.data.removeprefix("ok:"))
    except Exception:
        logger.exception("Не удалось подтвердить действие.")
        text = GENERIC_ERROR
    await _finish_callback(callback, text)


async def cancel_callback(callback: CallbackQuery, agent: Agent) -> None:
    try:
        text = await agent.cancel(callback.from_user.id, callback.data.removeprefix("no:"))
    except Exception:
        logger.exception("Не удалось отменить действие.")
        text = GENERIC_ERROR
    await _finish_callback(callback, text)


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
    router.message.register(tools, Command("tools"))
    router.message.register(why, Command("why"))
    router.message.register(timezone, Command("timezone"))
    router.message.register(unknown_command, F.text.startswith("/"))
    router.message.register(chat)
    router.callback_query.register(confirm_callback, F.data.startswith("ok:"))
    router.callback_query.register(cancel_callback, F.data.startswith("no:"))
    return router
