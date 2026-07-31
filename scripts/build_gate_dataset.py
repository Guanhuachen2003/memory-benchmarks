#!/usr/bin/env python3
"""Build a small memory-gate dataset with DeepSeek teacher labels.

The script samples real LongMemEval turns, labels them in batches through an
OpenAI-compatible DeepSeek endpoint, and writes conversation-grouped
train/validation/test JSONL files. Intermediate labels are append-only so an
interrupted run can resume without paying for completed examples again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


SYSTEM_PROMPT = """You label training examples for a cross-conversation memory gate.

Return should_add=true only when the CURRENT TURN contains information worth
retaining for future conversations, such as a durable user fact, preference,
relationship, recurring constraint, long-term plan, meaningful experience,
state change, correction, or an explicit future-facing instruction.

Return should_add=false for greetings, thanks, generic knowledge requests,
one-off task instructions, hypothetical/role-play facts, assistant guesses,
facts about unrelated third parties, transient momentary details, or content
that does not reveal reusable user information.

Important boundaries:
- "I like X" is usually true; "tell me about X" is false.
- "I am moving next month" can be true; "imagine I moved" is false.
- A durable correction or changed preference is true.
- Do not infer a user fact merely from a question topic.
- Judge the current turn, using recent context only to resolve references.

Output valid JSON only:
{"labels":[{"id":"...","should_add":true,"confidence":0.0,
"information_type":"preference|profile|relationship|constraint|plan|experience|state_change|instruction|none",
"operation_type":"new|update|correction|repeat|none"}]}

Do not include explanations or additional keys."""


LIKELY_MEMORY_RE = re.compile(
    r"\b(i am|i'm|i was|i have|i've|i like|i love|i hate|i prefer|"
    r"my |we are|we're|i plan|i will|i usually|i always|i never|"
    r"allergic|diagnosed|moved|moving|work as|studied|graduated|"
    r"remember that|from now on|in the future)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default="datasets/longmemeval/longmemeval_s_cleaned.json",
    )
    parser.add_argument("--output-dir", default="datasets/gate_deepseek_small")
    parser.add_argument("--target-size", type=int, default=300)
    parser.add_argument("--candidate-multiplier", type=float, default=1.8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-context-messages", type=int, default=4)
    parser.add_argument("--max-message-chars", type=int, default=1800)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=int, default=90)
    return parser.parse_args()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pair_messages(session: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, str]]]]:
    pairs: list[tuple[int, list[dict[str, str]]]] = []
    current: list[dict[str, str]] = []
    start_index = 0
    for index, message in enumerate(session):
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if not content or role not in {"user", "assistant"}:
            continue
        if role == "user" and current:
            pairs.append((start_index, current))
            current = []
        if not current:
            start_index = index
        current.append({"role": role, "content": content})
        if role == "assistant":
            pairs.append((start_index, current))
            current = []
    if current:
        pairs.append((start_index, current))
    return pairs


def truncate_message(message: dict[str, str], limit: int) -> dict[str, str]:
    content = message["content"]
    if len(content) > limit:
        content = content[:limit] + "…"
    return {"role": message["role"], "content": content}


def collect_candidates(
    dataset: list[dict[str, Any]],
    target: int,
    seed: int,
    context_messages: int,
    message_chars: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    likely: list[dict[str, Any]] = []
    unlikely: list[dict[str, Any]] = []
    seen: set[str] = set()

    for record in dataset:
        question_id = str(record["question_id"])
        session_ids = record.get("haystack_session_ids", [])
        dates = record.get("haystack_dates", [])
        for session_index, session in enumerate(record.get("haystack_sessions", [])):
            history: list[dict[str, str]] = []
            for pair_index, (_, current) in enumerate(pair_messages(session)):
                current_text = "\n".join(m["content"] for m in current)
                fingerprint = stable_hash(current_text)
                if fingerprint in seen:
                    history.extend(current)
                    continue
                seen.add(fingerprint)
                example_id = f"{question_id}:s{session_index}:p{pair_index}"
                candidate = {
                    "id": example_id,
                    "source": "longmemeval",
                    "conversation_id": question_id,
                    "question_type": record.get("question_type"),
                    "session_id": (
                        session_ids[session_index]
                        if session_index < len(session_ids)
                        else f"session_{session_index}"
                    ),
                    "session_index": session_index,
                    "pair_index": pair_index,
                    "session_date": dates[session_index] if session_index < len(dates) else None,
                    "recent_context": [
                        truncate_message(m, message_chars)
                        for m in history[-context_messages:]
                    ],
                    "current_messages": [
                        truncate_message(m, message_chars) for m in current
                    ],
                    "sampling_bucket": (
                        "likely_memory"
                        if LIKELY_MEMORY_RE.search(
                            "\n".join(
                                m["content"] for m in current if m["role"] == "user"
                            )
                        )
                        else "likely_non_memory"
                    ),
                }
                (likely if candidate["sampling_bucket"] == "likely_memory" else unlikely).append(
                    candidate
                )
                history.extend(current)

    rng.shuffle(likely)
    rng.shuffle(unlikely)
    positive_quota = target // 2
    selected = likely[:positive_quota] + unlikely[: target - positive_quota]
    if len(selected) < target:
        used = {item["id"] for item in selected}
        remainder = [item for item in likely + unlikely if item["id"] not in used]
        rng.shuffle(remainder)
        selected.extend(remainder[: target - len(selected)])
    rng.shuffle(selected)
    return selected


def render_batch(batch: list[dict[str, Any]]) -> str:
    payload = []
    for item_index, item in enumerate(batch):
        payload.append(
            {
                # Short batch-local IDs prevent teacher models from subtly
                # corrupting long source IDs while copying them to the output.
                "id": str(item_index),
                "recent_context": item["recent_context"],
                "current_turn": item["current_messages"],
            }
        )
    return "Label every example in this JSON array:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def call_teacher(
    *,
    base_url: str,
    api_key: str,
    model: str,
    batch: list[dict[str, Any]],
    timeout: int,
    max_retries: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_batch(batch)},
        ],
        "temperature": 0,
        "max_tokens": max(512, len(batch) * 120),
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read())
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed_labels = parsed if isinstance(parsed, list) else parsed["labels"]
            labels = {
                str(label["id"]): label
                for label in parsed_labels
                if isinstance(label, dict) and "id" in label
            }
            expected = {str(index) for index in range(len(batch))}
            if set(labels) != expected:
                missing = sorted(expected - set(labels))
                extra = sorted(set(labels) - expected)
                raise ValueError(f"Teacher ID mismatch: missing={missing}, extra={extra}")
            labels = {
                item["id"]: labels[str(index)] for index, item in enumerate(batch)
            }
            usage = result.get("usage") or {}
            return labels, {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "requests": 1,
            }
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Teacher request failed after {max_retries} attempts: {last_error}")


def load_existing(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    rows: dict[str, dict[str, Any]] = {}
    usage: Counter[str] = Counter()
    if not path.exists():
        return rows, usage
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["id"]] = row
            usage.update(row.get("teacher_usage", {}))
    return rows, usage


def split_for(conversation_id: str, seed: int) -> str:
    value = int(stable_hash(f"{seed}:{conversation_id}")[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "validation"
    return "test"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    if "deepseek.com" not in base_url:
        raise SystemExit(
            f"Refusing non-DeepSeek teacher endpoint: {base_url}. "
            "Use the official https://api.deepseek.com endpoint."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "teacher_labels.jsonl"
    dataset = json.loads(Path(args.dataset_path).read_text(encoding="utf-8"))
    candidate_target = max(
        args.target_size, round(args.target_size * args.candidate_multiplier)
    )
    candidates = collect_candidates(
        dataset,
        candidate_target,
        args.seed,
        args.max_context_messages,
        args.max_message_chars,
    )
    existing, usage = load_existing(raw_path)
    pending = [item for item in candidates if item["id"] not in existing]

    print(
        f"Candidates={len(candidates)} existing={len(existing)} pending={len(pending)} "
        f"teacher={args.model} endpoint={base_url}"
    )
    with raw_path.open("a", encoding="utf-8") as raw:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            labels, batch_usage = call_teacher(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                batch=batch,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            usage.update(batch_usage)
            for item_index, item in enumerate(batch):
                label = labels[item["id"]]
                row = {
                    **item,
                    "label": bool(label["should_add"]),
                    "confidence": max(0.0, min(1.0, float(label.get("confidence", 0.5)))),
                    "information_type": str(label.get("information_type", "none")),
                    "operation_type": str(label.get("operation_type", "none")),
                    "teacher_model": args.model,
                    # Store batch usage once so resumed runs can reconstruct the
                    # exact cumulative API usage without double-counting it.
                    "teacher_usage": batch_usage if item_index == 0 else {},
                }
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing[row["id"]] = row
            raw.flush()
            done = min(offset + len(batch), len(pending))
            print(
                f"Labeled {done}/{len(pending)} | "
                f"tokens={usage['total_tokens']}"
            )

    candidate_ids = {item["id"] for item in candidates}
    labeled = [row for key, row in existing.items() if key in candidate_ids]
    labeled.sort(key=lambda row: stable_hash(f"{args.seed}:{row['id']}"))

    positives = [row for row in labeled if row["label"]]
    negatives = [row for row in labeled if not row["label"]]
    per_class = min(len(positives), len(negatives), args.target_size // 2)
    if per_class * 2 < args.target_size:
        print(
            f"Warning: balanced target reduced from {args.target_size} "
            f"to {per_class * 2} due to teacher label distribution"
        )
    final_rows = positives[:per_class] + negatives[:per_class]
    final_rows.sort(key=lambda row: stable_hash(f"final:{args.seed}:{row['id']}"))
    for row in final_rows:
        row["information_type"] = {
            "habit": "preference",
            "correction": "state_change",
        }.get(row["information_type"], row["information_type"])
        # Teacher models do not interpret an unconstrained "confidence" field
        # consistently (some emit class confidence, others positive-class
        # probability). Preserve it for audit only and never use it as a
        # training weight without a separate calibration pass.
        row["teacher_confidence_raw"] = row.pop("confidence", None)
        row["split"] = split_for(row["conversation_id"], args.seed)

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        by_split[row["split"]].append(row)
    for split in ("train", "validation", "test"):
        write_jsonl(output_dir / f"{split}.jsonl", by_split[split])
    write_jsonl(output_dir / "all.jsonl", final_rows)

    conversation_splits: dict[str, set[str]] = defaultdict(set)
    for row in final_rows:
        conversation_splits[row["conversation_id"]].add(row["split"])
    leaked = [key for key, splits in conversation_splits.items() if len(splits) > 1]
    summary = {
        "dataset_path": args.dataset_path,
        "teacher_model": args.model,
        "teacher_base_url": base_url,
        "seed": args.seed,
        "requested_size": args.target_size,
        "candidate_count": len(candidates),
        "final_count": len(final_rows),
        "label_distribution": dict(Counter(str(row["label"]).lower() for row in final_rows)),
        "sampling_bucket_distribution": dict(
            Counter(row["sampling_bucket"] for row in final_rows)
        ),
        "information_type_distribution": dict(
            Counter(row["information_type"] for row in final_rows)
        ),
        "operation_type_distribution": dict(
            Counter(row["operation_type"] for row in final_rows)
        ),
        "split_counts": {split: len(rows) for split, rows in by_split.items()},
        "split_conversations": {
            split: len({row["conversation_id"] for row in rows})
            for split, rows in by_split.items()
        },
        "conversation_leakage_count": len(leaked),
        "teacher_usage": dict(usage),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
