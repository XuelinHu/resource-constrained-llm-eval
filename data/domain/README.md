# Railway Domain QA Dataset

This dataset is generated from local railway bilingual corpus documents under `data/corpus/railway`.

## Splits

- train: 22164
- valid: 2770
- test: 2770

## Categories

- en_to_zh_translation: 1161
- terminology_en_to_zh: 12691
- terminology_zh_to_en: 12691
- zh_to_en_translation: 1161

## Schema

Each JSONL record contains `prompt`, `answer`, `text`, `category`, and `source`.
`text` is formatted as `Question: ...\nAnswer: ...` for QLoRA training.
