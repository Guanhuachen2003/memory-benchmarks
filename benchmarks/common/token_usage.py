"""Helpers for recording and aggregating provider token usage."""

from __future__ import annotations

from typing import Any


TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def empty_usage() -> dict[str, int]:
    return {key: 0 for key in TOKEN_KEYS}


def normalize_usage(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None

    prompt_tokens = int(
        usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    )
    completion_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or 0
    )
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)

    if total_tokens <= 0:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def add_usage(target: dict[str, int], usage: Any) -> None:
    normalized = normalize_usage(usage)
    if not normalized:
        return
    for key in TOKEN_KEYS:
        target[key] = target.get(key, 0) + normalized.get(key, 0)


def extract_usage(payload: Any) -> dict[str, int] | None:
    """Find token usage in common OpenAI/Mem0 response shapes."""
    if not isinstance(payload, dict):
        return None

    direct = normalize_usage(payload.get("usage") or payload.get("token_usage"))
    if direct:
        return direct

    aggregate = empty_usage()
    found = False
    for value in payload.values():
        if isinstance(value, dict):
            nested = extract_usage(value)
            if nested:
                add_usage(aggregate, nested)
                found = True
        elif isinstance(value, list):
            for item in value:
                nested = extract_usage(item)
                if nested:
                    add_usage(aggregate, nested)
                    found = True

    return aggregate if found else None


def summarize_token_usage(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize memory construction per question and answer generation per call."""
    construction_total = empty_usage()
    answer_total = empty_usage()
    construction_samples = 0
    answer_samples = 0

    for evaluation in evaluations:
        ingestion = evaluation.get("ingestion")
        if isinstance(ingestion, dict):
            usage = normalize_usage(ingestion.get("token_usage"))
            if usage:
                add_usage(construction_total, usage)
                construction_samples += 1

        cutoff_results = evaluation.get("cutoff_results")
        if isinstance(cutoff_results, dict):
            for cutoff_result in cutoff_results.values():
                if not isinstance(cutoff_result, dict):
                    continue
                usage = normalize_usage(cutoff_result.get("answer_token_usage"))
                if usage:
                    add_usage(answer_total, usage)
                    answer_samples += 1

    def build_summary(total: dict[str, int], samples: int, unit: str) -> dict[str, Any]:
        average = {
            key: round(total[key] / samples, 2) if samples else 0.0
            for key in TOKEN_KEYS
        }
        return {
            "unit": unit,
            "samples": samples,
            "total": total,
            "average": average,
        }

    return {
        "memory_construction": build_summary(
            construction_total,
            construction_samples,
            "benchmark_question",
        ),
        "answer_generation": build_summary(
            answer_total,
            answer_samples,
            "answer_api_call",
        ),
    }


def display_token_usage_summary(summary: dict[str, Any]) -> None:
    """Print compact average token usage for the completed benchmark."""
    print("\n=== Average Token Usage ===")
    for key, label in (
        ("memory_construction", "Memory construction / question"),
        ("answer_generation", "Answer generation / API call"),
    ):
        section = summary[key]
        average = section["average"]
        print(
            f"{label}: prompt={average['prompt_tokens']:.2f}, "
            f"completion={average['completion_tokens']:.2f}, "
            f"total={average['total_tokens']:.2f} "
            f"(n={section['samples']})"
        )
