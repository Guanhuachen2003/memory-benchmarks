#!/usr/bin/env python3
"""Convert gate JSONL splits to LLaMA-Factory Alpaca JSON and register them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a high-recall memory-write gate. Output true when the current "
    "turn contains durable or potentially useful personal information for "
    "future conversations. Missing an important memory is more costly than "
    "performing an unnecessary memory write, so when genuinely uncertain, "
    "prefer true. Output exactly true or false. Do not explain."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--llamafactory-dir", required=True)
    parser.add_argument("--dataset-name", default="mem0_gate_v2")
    return parser.parse_args()


def render(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    data_dir = Path(args.llamafactory_dir) / "data"
    target_dir = data_dir / args.dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (source_dir / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        converted = []
        for row in rows:
            context = render(row.get("recent_context", [])) or "(empty)"
            current = render(row["current_messages"])
            converted.append(
                {
                    "system": SYSTEM_PROMPT,
                    "instruction": (
                        "Decide whether this turn should be written to "
                        "long-term memory."
                    ),
                    "input": (
                        f"<recent_context>\n{context}\n</recent_context>\n\n"
                        f"<current_turn>\n{current}\n</current_turn>"
                    ),
                    "output": "true" if row["label"] else "false",
                }
            )
        (target_dir / f"{split}.json").write_text(
            json.dumps(converted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"{split}: {len(converted)} -> {target_dir / f'{split}.json'}")

    info_path = data_dir / "dataset_info.json"
    dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
    columns = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
    }
    for split in ("train", "validation", "test"):
        dataset_info[f"{args.dataset_name}_{split}"] = {
            "file_name": f"{args.dataset_name}/{split}.json",
            "columns": columns,
        }
    info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Registered {args.dataset_name}_* in {info_path}")


if __name__ == "__main__":
    main()
