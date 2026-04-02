# Domain Dataset Placeholder

Provide your domain dataset here in JSONL format.

Each record should contain:

```json
{
  "prompt": "Question or instruction",
  "answer": "Reference answer",
  "text": "Question: ...\nAnswer: ..."
}
```

Recommended split files:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`

For reproducibility, document:

1. Data source
2. Annotation rules
3. Split policy
4. Answer normalization rules
