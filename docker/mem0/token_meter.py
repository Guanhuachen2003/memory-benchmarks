"""Request-local token accounting for Mem0's model calls."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any


_current_usage: ContextVar[dict[str, dict[str, int]] | None] = ContextVar(
    "mem0_token_usage",
    default=None,
)


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
    }


def start_meter() -> Token:
    return _current_usage.set({})


def record_usage(category: str, usage: Any) -> None:
    meter = _current_usage.get()
    if meter is None or usage is None:
        return

    def get_value(name: str) -> int:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return int(value or 0)

    prompt_tokens = get_value("prompt_tokens") or get_value("input_tokens")
    completion_tokens = get_value("completion_tokens") or get_value("output_tokens")
    total_tokens = get_value("total_tokens") or prompt_tokens + completion_tokens
    if total_tokens <= 0:
        return

    target = meter.setdefault(category, _empty_usage())
    target["prompt_tokens"] += prompt_tokens
    target["completion_tokens"] += completion_tokens
    target["total_tokens"] += total_tokens
    target["requests"] += 1


def finish_meter(token: Token) -> tuple[dict[str, int] | None, dict[str, dict[str, int]]]:
    breakdown = _current_usage.get() or {}
    _current_usage.reset(token)
    if not breakdown:
        return None, {}

    total = _empty_usage()
    for usage in breakdown.values():
        for key in total:
            total[key] += usage.get(key, 0)
    return total, breakdown
