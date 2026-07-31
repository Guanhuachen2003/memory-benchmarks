#!/usr/bin/env python3
"""Build a recall-oriented gate dataset from existing DeepSeek labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


INFO_TYPE_MAP = {
    "habit": "preference",
    "correction": "state_change",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="datasets/gate_deepseek_small",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/gate_deepseek_recall_v2",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_for(conversation_id: str, seed: int) -> str:
    value = int(stable_hash(f"{seed}:{conversation_id}")[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "validation"
    return "test"


def normalize_raw(row: dict[str, Any], split: str) -> dict[str, Any]:
    normalized = dict(row)
    normalized["information_type"] = INFO_TYPE_MAP.get(
        normalized.get("information_type"),
        normalized.get("information_type", "none"),
    )
    if "teacher_confidence_raw" not in normalized:
        normalized["teacher_confidence_raw"] = normalized.pop("confidence", None)
    normalized["split"] = split
    return normalized


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_jsonl(source_dir / "teacher_labels.jsonl")
    old_train = read_jsonl(source_dir / "train.jsonl")
    validation = read_jsonl(source_dir / "validation.jsonl")
    test = read_jsonl(source_dir / "test.jsonl")

    # Keep every unique, teacher-labeled positive example assigned to the
    # training conversation split. This adds coverage without duplicating data.
    train_positives = {
        row["id"]: normalize_raw(row, "train")
        for row in raw
        if row["label"]
        and split_for(row["conversation_id"], args.seed) == "train"
    }

    # Preserve the original negative training examples. Their coverage was
    # already selected to balance easy and harder non-memory turns.
    train_negatives = {
        row["id"]: row for row in old_train if not row["label"]
    }

    train = list(train_positives.values()) + list(train_negatives.values())
    train.sort(key=lambda row: stable_hash(f"{args.seed}:v2:{row['id']}"))

    validation = sorted(
        validation,
        key=lambda row: stable_hash(f"{args.seed}:validation:{row['id']}"),
    )
    test = sorted(
        test,
        key=lambda row: stable_hash(f"{args.seed}:test:{row['id']}"),
    )

    eval_conversations = {
        row["conversation_id"] for row in validation + test
    }
    train_conversations = {row["conversation_id"] for row in train}
    leakage = sorted(train_conversations & eval_conversations)
    if leakage:
        raise RuntimeError(f"Conversation leakage detected: {leakage[:10]}")

    all_rows = train + validation + test
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("Duplicate example IDs detected across splits")

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation)
    write_jsonl(output_dir / "test.jsonl", test)
    write_jsonl(output_dir / "all.jsonl", all_rows)

    summary = {
        "source_dir": str(source_dir),
        "seed": args.seed,
        "strategy": (
            "all unique train-split positives from 2000 teacher labels; "
            "original train negatives; unchanged validation and test"
        ),
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "label_distribution": {
            split: dict(Counter(str(row["label"]).lower() for row in rows))
            for split, rows in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "train_positive_negative_ratio": round(
            len(train_positives) / max(1, len(train_negatives)),
            4,
        ),
        "train_information_types": dict(
            Counter(row["information_type"] for row in train if row["label"])
        ),
        "train_operation_types": dict(
            Counter(row["operation_type"] for row in train if row["label"])
        ),
        "conversation_counts": {
            "train": len(train_conversations),
            "validation": len({row["conversation_id"] for row in validation}),
            "test": len({row["conversation_id"] for row in test}),
        },
        "conversation_leakage_count": len(leakage),
        "unique_id_count": len({row["id"] for row in all_rows}),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
