#!/usr/bin/env python3
"""Export base and fine-tuned gate decisions for a fixed LongMemEval sample."""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are a high-recall memory-write gate. Output true when the current "
    "turn contains durable or potentially useful personal information for "
    "future conversations. Missing an important memory is more costly than "
    "performing an unnecessary memory write, so when genuinely uncertain, "
    "prefer true. Output exactly true or false. Do not explain."
)

QUESTION_TYPES = {
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--dataset-path",
        default="datasets/longmemeval/longmemeval_s_cleaned.json",
    )
    parser.add_argument("--output-dir", default="results/gate_offline_18")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--per-type", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def render_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{message.get('role', 'unknown')}: {message.get('content', '').strip()}"
        for message in messages
        if message.get("content", "").strip()
    )


def load_dataset(path: str) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sample_questions_stratified(
    questions: list[dict[str, Any]], per_type: int, seed: int
) -> list[dict[str, Any]]:
    """Mirror benchmarks.longmemeval.run.sample_questions_stratified exactly."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        if question["question_type"] in QUESTION_TYPES:
            groups[question["question_type"]].append(question)
    for group in groups.values():
        group.sort(key=lambda question: question["question_id"])
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for question_type in sorted(groups):
        group = groups[question_type]
        sampled.extend(rng.sample(group, min(per_type, len(group))))
    sampled.sort(key=lambda question: question["question_id"])
    return sampled


def parse_date(date: str) -> int | None:
    try:
        cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date).strip()
        parsed = datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(
            tzinfo=timezone.utc
        )
        return int(parsed.timestamp())
    except (TypeError, ValueError):
        return None


def sort_sessions_chronologically(
    question: dict[str, Any],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    sessions = list(
        zip(
            question["haystack_session_ids"],
            question["haystack_dates"],
            question["haystack_sessions"],
        )
    )

    def sort_key(item: tuple[str, str, list[dict[str, Any]]]) -> tuple[Any, ...]:
        parsed = parse_date(item[1])
        return (0, parsed, item[1]) if parsed is not None else (1, 0, item[1])

    sessions.sort(key=sort_key)
    return sessions


def pair_turns(session: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
    cleaned = [
        {"role": turn["role"], "content": turn["content"]} for turn in session
    ]
    return [cleaned[index : index + 2] for index in range(0, len(cleaned), 2)]


def make_chat(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    current = render_messages(messages)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Decide whether this turn should be written to long-term memory.\n\n"
                "<recent_context>\n(empty)\n</recent_context>\n\n"
                f"<current_turn>\n{current}\n</current_turn>"
            ),
        },
    ]


def build_rows(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        for session_index, (session_id, date, session) in enumerate(
            sort_sessions_chronologically(question)
        ):
            for pair_index, messages in enumerate(pair_turns(session)):
                if any(not message.get("content", "").strip() for message in messages):
                    continue
                rows.append(
                    {
                        "id": f"{question['question_id']}:s{session_index}:p{pair_index}",
                        "question_id": question["question_id"],
                        "question_type": question["question_type"],
                        "session_id": session_id,
                        "session_date": date,
                        "session_index": session_index,
                        "pair_index": pair_index,
                        "messages": messages,
                    }
                )
    return rows


def parse_boolean(text: str) -> bool | None:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    matches = re.findall(r"\b(true|false)\b", cleaned.lower())
    if not matches:
        return None
    return matches[-1] == "true"


def load_model(args: argparse.Namespace, with_adapter: bool):
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    if with_adapter:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model


def load_prediction_cache(
    path: Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cached = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cached) > len(rows):
        raise ValueError(f"Cache has more rows than expected: {path}")
    for index, item in enumerate(cached):
        if item.get("id") != rows[index]["id"]:
            raise ValueError(f"Cache does not match selected inputs at row {index}: {path}")
    return [item["prediction"] for item in cached]


@torch.inference_mode()
def predict(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    label: str,
    cache_path: Path,
) -> list[dict[str, Any]]:
    predictions = load_prediction_cache(cache_path, rows)
    if predictions:
        print(f"[{label}] resuming from {len(predictions)}/{len(rows)}", flush=True)
    started = time.time()
    for offset in range(len(predictions), len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                make_chat(row["messages"]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        prompt_counts = encoded["attention_mask"].sum(dim=1).tolist()
        generated_counts = (new_tokens != tokenizer.pad_token_id).sum(dim=1).tolist()
        batch_predictions = []
        for text, prompt_tokens, completion_tokens in zip(
            texts, prompt_counts, generated_counts
        ):
            decision = parse_boolean(text)
            batch_predictions.append(
                {
                    "decision": decision,
                    "valid": decision is not None,
                    "raw_output": text.strip(),
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                }
            )
        with cache_path.open("a", encoding="utf-8") as handle:
            for row, prediction in zip(batch, batch_predictions):
                handle.write(
                    json.dumps(
                        {"id": row["id"], "prediction": prediction},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
        predictions.extend(batch_predictions)
        done = min(offset + batch_size, len(rows))
        print(f"[{label}] {done}/{len(rows)}", flush=True)
    print(f"[{label}] completed in {time.time() - started:.1f}s", flush=True)
    return predictions


def decision_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        "invalid" if item["decision"] is None else str(item["decision"]).lower()
        for item in predictions
    )
    return {key: counts.get(key, 0) for key in ("true", "false", "invalid")}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dataset = load_dataset(args.dataset_path)
    questions = sample_questions_stratified(
        dataset,
        per_type=args.per_type,
        seed=args.seed,
    )
    rows = build_rows(questions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Selected {len(questions)} questions and {len(rows)} memory-write pairs")
    print("Question IDs:")
    for question in questions:
        print(f"  {question['question_type']}: {question['question_id']}")

    print("Loading base model...", flush=True)
    base_cache = output_dir / "base_predictions.jsonl"
    base_predictions = load_prediction_cache(base_cache, rows)
    if len(base_predictions) < len(rows):
        model = load_model(args, with_adapter=False)
        base_predictions = predict(
            model,
            tokenizer,
            rows,
            args.batch_size,
            args.max_input_tokens,
            args.max_new_tokens,
            "base",
            base_cache,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("[base] cache complete; skipping model load", flush=True)

    print("Loading base model with LoRA adapter...", flush=True)
    trained_cache = output_dir / "trained_predictions.jsonl"
    trained_predictions = load_prediction_cache(trained_cache, rows)
    if len(trained_predictions) < len(rows):
        model = load_model(args, with_adapter=True)
        trained_predictions = predict(
            model,
            tokenizer,
            rows,
            args.batch_size,
            args.max_input_tokens,
            args.max_new_tokens,
            "trained",
            trained_cache,
        )
    else:
        print("[trained] cache complete; skipping model load", flush=True)

    disagreements = 0
    output_path = output_dir / "decisions.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row, base, trained in zip(rows, base_predictions, trained_predictions):
            item = {**row, "base": base, "trained": trained}
            item["disagreement"] = base["decision"] != trained["decision"]
            disagreements += int(item["disagreement"])
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = [
        {
            "question_id": question["question_id"],
            "question_type": question["question_type"],
            "question": question.get("question"),
        }
        for question in questions
    ]
    (output_dir / "selected_questions.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "seed": args.seed,
        "per_type": args.per_type,
        "question_count": len(questions),
        "memory_write_pair_count": len(rows),
        "base_counts": decision_counts(base_predictions),
        "trained_counts": decision_counts(trained_predictions),
        "disagreement_count": disagreements,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "load_in_4bit": args.load_in_4bit,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
