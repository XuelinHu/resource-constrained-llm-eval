# Railway Domain QA Dataset

This dataset is generated from local railway bilingual corpus documents under `data/corpus/railway`.
Terminology samples are extracted from the classified vocabulary Word document and keep the original railway subdomain labels.

## Splits

- train: 20832
- valid: 2602
- test: 2602
- unique classified term pairs: 11857

## Categories

- en_to_zh_translation: 1161
- terminology_en_to_zh: 11857
- terminology_zh_to_en: 11857
- zh_to_en_translation: 1161

## Terminology Subdomains

- 1. 通用词汇 / General vocabulary: 550
- 2. 通信 / Communication: 499
- 3. 信号 / Signaling: 875
- 4. 机车 / Locomotive: 791
- 5. 车辆 / Rolling stock: 1133
- 6. 牵引供电 / Traction power supply: 1771
- 7. 工务工程 / Permanent way: 4363
- 8. 运输与经济 / Transportation and economy: 550
- 9. 相关科学 / Related sciences: 908
- 10. 常用缩写 / Abbreviation: 312
- 11. 相关国际组织名称 / Railways and International Organizations: 105

## Schema

Each JSONL record contains `prompt`, `answer`, `text`, `category`, and `source`.
Terminology records also contain `domain_category_id`, `domain_category_key`,
`domain_category`, `domain_category_en`, `term_zh`, and `term_en`.
`text` is formatted as `Question: ...\nAnswer: ...` for QLoRA training.
The unique classified term inventory is written to `terminology_inventory.jsonl`.
