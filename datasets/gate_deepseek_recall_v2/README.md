# Memory Gate Dataset — Recall V2

This variant addresses the low positive recall observed with the original
balanced training split.

- Training: 495 unique examples (303 add, 192 skip)
- Validation: unchanged 46 examples
- Test: unchanged 64 examples
- Positive-to-negative training ratio: 1.5781
- Conversation leakage: 0

The additional positive training examples come from the existing 2,000
DeepSeek teacher labels and belong only to the deterministic training
conversation split. No validation/test examples were moved, duplicated, or
re-labeled.

Use `scripts/build_recall_gate_dataset.py` to reproduce this dataset.
