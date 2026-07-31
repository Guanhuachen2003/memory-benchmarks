"""Configurable OpenAI-compatible gate that decides whether to call Mem0 add."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a memory-write gate.
Decide whether the conversation contains durable information worth storing in
long-term memory.

Store specific user facts, preferences, relationships, goals, commitments,
plans, important events, experiences, and meaningful state changes.
Skip greetings, acknowledgements, generic questions, transient requests,
repeated information, and assistant-only content that reveals nothing durable
about the user.

Return only JSON:
{"should_add": true or false}"""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class MemoryWriteGate:
    """Memory gate supporting hosted APIs and local OpenAI-compatible vLLM."""

    def __init__(self) -> None:
        self.enabled = _env_bool("MEMORY_GATE_ENABLED", True)
        self.fail_open = _env_bool("MEMORY_GATE_FAIL_OPEN", True)
        self.model = os.getenv("MEMORY_GATE_MODEL", "deepseek-chat")
        self.base_url = (
            os.getenv("MEMORY_GATE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.deepseek.com"
        )
        self.api_key = (
            os.getenv("MEMORY_GATE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "local-vllm"
        )
        self.system_prompt = os.getenv("MEMORY_GATE_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30.0)
            if self.enabled
            else None
        )
        logger.info(
            "Memory gate: enabled=%s, model=%s, base_url=%s, fail_open=%s",
            self.enabled,
            self.model,
            self.base_url,
            self.fail_open,
        )

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines = []
        for message in messages:
            role = str(message.get("role", "unknown")).upper()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_decision(content: str) -> bool:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("gate response did not contain a JSON object")
        payload = json.loads(match.group(0))
        should_add = payload.get("should_add")
        if not isinstance(should_add, bool):
            raise ValueError("gate response should_add must be boolean")
        return should_add

    def evaluate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.enabled:
            return {"should_add": True}

        conversation = self._format_messages(messages)
        if not conversation:
            return {"should_add": False}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Evaluate this conversation:\n\n{conversation}",
                    },
                ],
                temperature=0,
                max_tokens=120,
            )
            content = response.choices[0].message.content or ""
            return {"should_add": self._parse_decision(content)}
        except Exception:
            logger.exception("Memory gate decision failed")
            return {"should_add": self.fail_open}
