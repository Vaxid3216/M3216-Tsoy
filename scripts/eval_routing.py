"""Эксперимент лабораторной 2: маршрутизация 12 запросов (respond / clarify / инструмент).

Запуск: python -m scripts.eval_routing --version 1 --note "исходная инструкция"
Настройки модели читаются из .env (ключ не печатается и не сохраняется). Время и зона
берутся из docs/evaluation/lab2-routing.json, инструменты обнаруживаются у настоящего
MCP-сервера, но не вызываются: проверяется первое решение модели.
Результат дописывается в docs/lab2-routing-results.json (версии решения).
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.agent.decision import Decision, DecisionError
from app.agent.service import Agent
from app.config import Settings
from app.llm import LLMClient, LLMError
from app.mcp_client import McpClient

CASES = Path("docs/evaluation/lab2-routing.json")
OUT = Path("docs/lab2-routing-results.json")
TEMPERATURE = 0.0
ATTEMPTS = 3
ROUTING_THRESHOLD = 10
ARGUMENT_THRESHOLD = 8


def actual_action(decision: Decision) -> str:
    return decision.tool if decision.kind == "call_tool" else decision.kind


def _norm_text(value: Any) -> str:
    return str(value).strip().strip(".!").casefold()


def _moment(value: Any, zone: ZoneInfo) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed


def arguments_match(expected: dict[str, Any], actual: dict[str, Any], zone: ZoneInfo) -> bool:
    if set(actual) != set(expected):
        return False
    for key, wanted in expected.items():
        got = actual[key]
        if key == "remind_at":
            if _moment(got, zone) != _moment(wanted, zone) or _moment(got, zone) is None:
                return False
        elif _norm_text(got) != _norm_text(wanted):
            return False
    return True


async def run_case(agent: Agent, case: dict, context: dict) -> dict:
    now = datetime.fromisoformat(context["now"])
    zone = ZoneInfo(context["timezone"])
    decision = None
    error = ""
    for _ in range(ATTEMPTS):
        try:
            decision = await agent.plan(
                case["query"],
                timezone=context["timezone"],
                now=now,
                temperature=TEMPERATURE,
            )
            break
        except (LLMError, DecisionError) as exc:
            error = str(exc)
            await asyncio.sleep(2)
    expected = case["expected_action"]
    result: dict[str, Any] = {
        "id": case["id"],
        "query": case["query"],
        "expected_action": expected,
        "expected_arguments": case["expected_arguments"],
    }
    if decision is None:
        return {
            **result,
            "actual_action": "error",
            "actual_arguments": {},
            "error": error,
            "routing_ok": False,
            "arguments_ok": None,
        }
    actual = actual_action(decision)
    routing_ok = actual == expected
    arguments_ok = None
    if expected not in ("respond", "clarify"):
        arguments_ok = routing_ok and arguments_match(
            case["expected_arguments"], decision.arguments, zone
        )
    return {
        **result,
        "actual_action": actual,
        "actual_arguments": decision.arguments,
        "routing_ok": routing_ok,
        "arguments_ok": arguments_ok,
    }


async def main(version: int, note: str) -> None:
    settings = Settings.load()
    settings.require_llm()
    spec = json.loads(await asyncio.to_thread(CASES.read_text, "utf-8"))
    context = spec["evaluation_context"]
    llm = LLMClient(settings)
    mcp = McpClient(settings)
    try:
        if not await mcp.start():
            raise SystemExit("MCP-сервер не запустился: эксперимент невозможен.")
        agent = Agent(PlannerStore(), llm, mcp, settings)
        rows = []
        for case in spec["cases"]:
            row = await run_case(agent, case, context)
            rows.append(row)
            mark = "OK " if row["routing_ok"] and row["arguments_ok"] is not False else "ERR"
            print(
                f"{mark} {row['id']}: ожидалось {row['expected_action']}, "
                f"получено {row['actual_action']} {row['actual_arguments']}"
            )
    finally:
        await mcp.close()
        await llm.close()
    tool_rows = [r for r in rows if r["arguments_ok"] is not None]
    summary = {
        "routing_accuracy": f"{sum(r['routing_ok'] for r in rows)}/{len(rows)}",
        "argument_accuracy": f"{sum(bool(r['arguments_ok']) for r in tool_rows)}/{len(tool_rows)}",
        "routing_ok": sum(r["routing_ok"] for r in rows) >= ROUTING_THRESHOLD,
        "argument_ok": sum(bool(r["arguments_ok"]) for r in tool_rows) >= ARGUMENT_THRESHOLD,
    }
    print(summary)
    existing = await asyncio.to_thread(_read_existing)
    data = existing or {"versions": []}
    data["versions"] = [v for v in data["versions"] if v["version"] != version]
    data["versions"].append(
        {
            "version": version,
            "note": note,
            "model": settings.llm_model,
            "temperature": TEMPERATURE,
            "context": context,
            "summary": summary,
            "cases": rows,
        }
    )
    data["versions"].sort(key=lambda v: v["version"])
    await asyncio.to_thread(
        OUT.write_text, json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )


def _read_existing() -> dict | None:
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else None


class PlannerStore:
    """Заглушка хранилища: `plan` не обращается к БД, но Agent требует объект."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    asyncio.run(main(args.version, args.note))
