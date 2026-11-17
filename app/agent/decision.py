"""Решение модели: respond, clarify или call_tool, и проверка его перед выполнением."""

import json
from dataclasses import dataclass
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CLARIFY_TOOL = "clarify"

CLARIFY_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": CLARIFY_TOOL,
        "description": (
            "Задать пользователю один уточняющий вопрос, когда не хватает данных для "
            "инструмента (например, время напоминания не названо точно)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "minLength": 1, "maxLength": 300}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


class Decision(BaseModel):
    """Структурированное решение одного шага агента."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["respond", "clarify", "call_tool"]
    text: str = ""
    tool: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""


class DecisionError(ValueError):
    """Решение модели не соответствует схеме; текст безопасен для журнала."""


def parse_decision(message: dict[str, Any]) -> Decision:
    """Переводит сообщение модели (нативный tool calling) в `Decision`."""
    calls = message.get("tool_calls") or []
    if calls:
        call = calls[0]  # параллельные вызовы не поддерживаются: берётся первый
        try:
            function = call["function"]
            name = function["name"]
            raw = function.get("arguments") or "{}"
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            call_id = str(call.get("id") or "")
        except (KeyError, TypeError, ValueError):
            raise DecisionError("Не удалось разобрать вызов инструмента.") from None
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise DecisionError("Вызов инструмента неверной структуры.")
        if name == CLARIFY_TOOL:
            question = arguments.get("question")
            if (
                not isinstance(question, str)
                or not question.strip()
                or set(arguments) != {"question"}
            ):
                raise DecisionError("Неверные аргументы уточнения.")
            return Decision(kind="clarify", text=question.strip(), call_id=call_id)
        return Decision(kind="call_tool", tool=name, arguments=arguments, call_id=call_id)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise DecisionError("Модель вернула пустое решение.")
    try:
        return Decision(kind="respond", text=content.strip())
    except ValidationError:
        raise DecisionError("Ответ модели не прошёл проверку.") from None


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """None, если аргументы подходят под схему обнаруженного инструмента; иначе причина."""
    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "аргументы"
        return f"{path}: {exc.validator}"
    return None


@dataclass(frozen=True)
class Confirmation:
    """Подготовленное действие, ожидающее нажатия кнопки."""

    action_id: str


def summarize_args(tool: str, arguments: dict[str, Any]) -> str:
    """Безопасное резюме для аудита: без полных сообщений пользователя."""
    if tool == "get_weather":
        return f"город: {str(arguments.get('city', ''))[:60]}"
    if tool == "get_schedule":
        return f"дата: {str(arguments.get('date', ''))[:10]}"
    if tool == "add_reminder":
        text = str(arguments.get("text", ""))
        return f"время: {str(arguments.get('remind_at', ''))[:30]}, текст: {len(text)} симв."
    return f"аргументов: {len(arguments)}"
