# DeepSeek Memory Gate Dataset (Small)

This dataset contains real LongMemEval conversation turns labeled by the
official DeepSeek API (`deepseek-chat`) for binary memory-gate training.

## Files

- `train.jsonl`: training split
- `validation.jsonl`: threshold selection and early stopping
- `test.jsonl`: held-out evaluation
- `all.jsonl`: union of the three final splits
- `teacher_labels.jsonl`: all teacher-labeled candidates, including examples
  not selected into the balanced final dataset
- `summary.json`: counts, distributions, split validation, and API token usage

All examples sharing a `conversation_id` are assigned to the same split.

## Training fields

- `recent_context`: up to four preceding messages
- `current_messages`: the user/assistant messages evaluated by the gate
- `label`: `true` means call Mem0 add; `false` means skip
- `information_type`: coarse memory category
- `operation_type`: new/update/correction/repeat/none

`teacher_confidence_raw` is retained only for auditing. DeepSeek did not use
that field consistently, so it should not be used as a loss weight.

The model input should be constructed from `recent_context` and
`current_messages`. The target should be exactly `true` or `false`.
