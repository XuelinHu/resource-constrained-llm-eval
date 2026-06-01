"""Build extractive railway regulation QA data without LLM generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "corpus" / "railway"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "domain_regqa"

SPACE_RE = re.compile(r"\s+")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT_RE = re.compile(r"[。！？；：]")
RULE_KEYWORD_RE = re.compile(
    r"(应|须|必须|不得|严禁|禁止|负责|职责|原则|方针|标准|要求|范围|包括|分为|"
    r"检查|检测|检修|维修|维护|试验|管理|安全|可靠|定期|周期|组织|执行|制定|落实|确保|适用于)"
)
COMPLETE_ENDINGS = ("。", "！", "？", "；", "）", ")")
TOPIC_STOPWORDS = {"该", "其", "由", "对", "对于", "并", "且", "或", "及", "和", "不", "均", "凡"}
LIST_ITEM_MARK_RE = re.compile(
    r"^\s*(?:"
    r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇①②③④⑤⑥⑦⑧⑨⑩]"
    r"|[（(]\s*[0-9一二三四五六七八九十]{1,3}\s*[）)]"
    r"|[0-9一二三四五六七八九十]{1,3}[、.．]"
    r")\s*"
)
LEADING_MARK_RE = re.compile(
    r"^\s*(第[一二三四五六七八九十百千万0-9]+条|[一二三四五六七八九十0-9]+[、.．]|"
    r"[（(][一二三四五六七八九十0-9]+[）)]|[-—]+)\s*"
)
SENTENCE_RE = re.compile(r"[^。！？]+[。！？]?")


@dataclass(frozen=True)
class RegulationSample:
    prompt: str
    answer: str
    category: str
    source: str
    paragraph_id: str
    evidence: str
    answer_start: int

    def to_record(self) -> dict[str, str | int]:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "text": f"Question: {self.prompt}\nAnswer: {self.answer}",
            "category": self.category,
            "source": self.source,
            "paragraph_id": self.paragraph_id,
            "evidence": self.evidence,
            "answer_start": self.answer_start,
            "generation_method": "rule_based_extractive",
        }


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    return SPACE_RE.sub(" ", text).strip()


def read_docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        text = normalize_text("".join((node.text or "") for node in para.findall(".//w:t", ns)))
        if text:
            paragraphs.append(text)
    return paragraphs


def is_heading_or_toc(text: str) -> bool:
    if PUNCT_RE.search(text):
        return False
    if re.search(r"(目录|附件\s*\d+|第[一二三四五六七八九十0-9]+[章节部分])", text):
        return True
    if len(text) <= 35 and re.search(r"(管理|标准|要求|记录|办法)\d*$", text):
        return True
    return False


def is_rule_paragraph(text: str) -> bool:
    if not (16 <= len(text) <= 650):
        return False
    if not HAN_RE.search(text):
        return False
    if is_heading_or_toc(text):
        return False
    if not RULE_KEYWORD_RE.search(text):
        return False
    alpha_or_han = sum(ch.isalpha() or "\u4e00" <= ch <= "\u9fff" for ch in text)
    return alpha_or_han / max(1, len(text)) >= 0.35


def answer_only_prompt(question: str) -> str:
    return (
        "根据铁路规章回答问题。\n"
        "只返回最终答案，不要解释，不要添加前缀、后缀、引号或标签。\n"
        f"{question}"
    )


def strip_list_item_marker(text: str) -> str:
    return LIST_ITEM_MARK_RE.sub("", text).strip()


def clean_topic(text: str, marker: str | None = None) -> str:
    topic = text
    if marker and marker in topic:
        if marker == "负责":
            topic = topic.rsplit(marker, 1)[0]
        else:
            topic = topic.split(marker, 1)[0]
    topic = strip_list_item_marker(LEADING_MARK_RE.sub("", topic))
    topic = re.split(r"[。！？；，,：:]", topic)[-1]
    topic = topic.strip(" ，,；;：:。！？（）()“”\"'《》")
    topic = re.sub(r"^(其中|同时|并|且|或|以及|对|对于|凡|各|有关|由)", "", topic).strip()
    topic = re.sub(r"(的|和|与|及|或|并)+$", "", topic).strip()
    if (
        len(topic) < 2
        or topic in TOPIC_STOPWORDS
        or not HAN_RE.search(topic)
        or topic.count("（") != topic.count("）")
        or topic.count("(") != topic.count(")")
    ):
        return "该条款"
    if len(topic) > 32:
        topic = topic[-32:]
    return topic


def split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    for sentence in SENTENCE_RE.findall(paragraph):
        sentence = normalize_text(sentence)
        if 12 <= len(sentence) <= 420 and sentence.endswith(COMPLETE_ENDINGS) and RULE_KEYWORD_RE.search(sentence):
            sentences.append(sentence)
    if not sentences and 16 <= len(paragraph) <= 420 and paragraph.endswith(COMPLETE_ENDINGS):
        sentences.append(paragraph)
    return sentences


def make_sample(
    *,
    question: str,
    answer: str,
    category: str,
    source: str,
    paragraph_id: str,
    evidence: str,
) -> RegulationSample | None:
    answer = strip_list_item_marker(normalize_text(answer))
    evidence = normalize_text(evidence)
    answer_start = evidence.find(answer)
    if answer_start < 0:
        return None
    if not (8 <= len(answer) <= 650):
        return None
    if not answer.endswith(COMPLETE_ENDINGS):
        return None
    if not (6 <= len(question) <= 180):
        return None
    return RegulationSample(
        prompt=answer_only_prompt(question),
        answer=answer,
        category=category,
        source=source,
        paragraph_id=paragraph_id,
        evidence=evidence,
        answer_start=answer_start,
    )


def sentence_questions(sentence: str) -> list[tuple[str, str]]:
    questions: list[tuple[str, str]] = []

    if "适用于" in sentence:
        topic = clean_topic(sentence, "适用于")
        questions.append(("regulation_scope_qa", f"本规章中，{topic}适用于什么范围？"))

    if "坚持" in sentence and "方针" in sentence:
        topic = clean_topic(sentence, "应") if "应" in sentence else clean_topic(sentence, "坚持")
        questions.append(("regulation_principle_qa", f"规章对{topic}提出了什么方针？"))

    if "原则" in sentence and ("按照" in sentence or "遵循" in sentence):
        topic = clean_topic(sentence, "应") if "应" in sentence else clean_topic(sentence, "按照")
        questions.append(("regulation_principle_qa", f"规章要求{topic}遵循什么原则？"))

    for marker in ("不得", "严禁", "禁止", "不应"):
        if marker in sentence:
            topic = clean_topic(sentence, marker)
            questions.append(("regulation_prohibition_qa", f"根据规章，{topic}有哪些禁止性要求？"))
            break

    if "负责" in sentence or "职责" in sentence:
        marker = "负责" if "负责" in sentence else "职责"
        topic = clean_topic(sentence, marker)
        if topic == "该条款":
            questions.append(("regulation_responsibility_qa", "根据规章，相关责任或分工是什么？"))
        else:
            questions.append(("regulation_responsibility_qa", f"根据规章，{topic}负责哪些工作或职责？"))

    if "包括" in sentence and "不包括" not in sentence:
        topic = clean_topic(sentence, "包括")
        if topic == "该条款":
            questions.append(("regulation_definition_qa", "规章中，该条款说明了哪些内容？"))
        else:
            questions.append(("regulation_definition_qa", f"规章中，{topic}包括哪些内容？"))

    if "分为" in sentence:
        topic = clean_topic(sentence, "分为")
        questions.append(("regulation_definition_qa", f"规章中，{topic}分为哪些类型？"))

    if "标准" in sentence or "要求" in sentence:
        topic = clean_topic(sentence, "应") if "应" in sentence else clean_topic(sentence, "要求")
        questions.append(("regulation_standard_qa", f"根据规章，{topic}的标准或要求是什么？"))

    if re.search(r"(检查|检测|检修|维修|维护|试验)", sentence):
        topic = clean_topic(sentence, "应") if "应" in sentence else clean_topic(sentence)
        questions.append(("regulation_inspection_qa", f"根据规章，{topic}的检查、检测或维修要求是什么？"))

    requirement_match = re.search(r"(?<!不)(必须|须|应)|确保|制定|执行|落实", sentence)
    if requirement_match:
        marker = requirement_match.group(0)
        topic = clean_topic(sentence, marker)
        questions.append(("regulation_requirement_qa", f"根据规章，{topic}应满足什么要求？"))

    return questions


def generic_questions(paragraph: str) -> list[str]:
    topic = paragraph
    if "：" in paragraph:
        prefix = paragraph.split("：", 1)[0]
        if 2 <= len(prefix) <= 36 and HAN_RE.search(prefix):
            topic = prefix
    topic = clean_topic(topic)
    if topic == "该条款":
        return [
            "根据规章，该条款的原文规定是什么？",
            "请给出该规章条款的原文要求。",
        ]
    return [
        f"根据规章，关于{topic}的规定是什么？",
        f"请给出规章中关于{topic}的原文要求。",
    ]


def build_candidates_for_paragraph(
    *,
    paragraph: str,
    source: str,
    paragraph_id: str,
    max_per_paragraph: int,
) -> list[RegulationSample]:
    output: list[RegulationSample] = []
    seen: set[tuple[str, str]] = set()

    for sentence in split_sentences(paragraph):
        for category, question in sentence_questions(sentence):
            sample = make_sample(
                question=question,
                answer=sentence,
                category=category,
                source=source,
                paragraph_id=paragraph_id,
                evidence=paragraph,
            )
            if sample is None:
                continue
            key = (sample.prompt, sample.answer)
            if key not in seen:
                seen.add(key)
                output.append(sample)
            if len(output) >= max_per_paragraph:
                return output

    for question in generic_questions(paragraph):
        sample = make_sample(
            question=question,
            answer=paragraph,
            category="regulation_clause_qa",
            source=source,
            paragraph_id=paragraph_id,
            evidence=paragraph,
        )
        if sample is not None and (sample.prompt, sample.answer) not in seen:
            output.append(sample)
        if len(output) >= max_per_paragraph:
            break

    return output[:max_per_paragraph]


def source_prefix(path: Path) -> str:
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:8]
    return f"{path.stem[:12]}_{digest}"


def build_candidates(corpus_dir: Path, source_glob: str, max_per_paragraph: int) -> list[RegulationSample]:
    samples: list[RegulationSample] = []
    for docx_file in sorted(corpus_dir.glob(source_glob)):
        if "词汇" in docx_file.name:
            continue
        prefix = source_prefix(docx_file)
        paragraph_index = 0
        for raw_paragraph in read_docx_paragraphs(docx_file):
            paragraph = normalize_text(raw_paragraph)
            if not is_rule_paragraph(paragraph):
                continue
            paragraph_index += 1
            paragraph_id = f"{prefix}_p{paragraph_index:04d}"
            samples.extend(
                build_candidates_for_paragraph(
                    paragraph=paragraph,
                    source=docx_file.name,
                    paragraph_id=paragraph_id,
                    max_per_paragraph=max_per_paragraph,
                )
            )

    deduped: list[RegulationSample] = []
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        key = (sample.prompt, sample.answer)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def select_balanced(samples: list[RegulationSample], target: int, rng: random.Random) -> list[RegulationSample]:
    by_category: dict[str, list[RegulationSample]] = defaultdict(list)
    for sample in samples:
        by_category[sample.category].append(sample)
    for category_samples in by_category.values():
        rng.shuffle(category_samples)

    selected: list[RegulationSample] = []
    while len(selected) < target and by_category:
        progressed = False
        for category in sorted(list(by_category)):
            if not by_category[category]:
                del by_category[category]
                continue
            selected.append(by_category[category].pop())
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def split_by_paragraph(
    samples: list[RegulationSample],
    limit: int,
    seed: int,
) -> tuple[list[RegulationSample], list[RegulationSample], list[RegulationSample]]:
    rng = random.Random(seed)
    by_paragraph: dict[str, list[RegulationSample]] = defaultdict(list)
    for sample in samples:
        by_paragraph[sample.paragraph_id].append(sample)

    paragraph_ids = list(by_paragraph)
    rng.shuffle(paragraph_ids)
    train_end = round(len(paragraph_ids) * 0.8)
    valid_end = train_end + round(len(paragraph_ids) * 0.1)
    split_ids = {
        "train": set(paragraph_ids[:train_end]),
        "valid": set(paragraph_ids[train_end:valid_end]),
        "test": set(paragraph_ids[valid_end:]),
    }

    targets = {
        "train": round(limit * 0.8),
        "valid": round(limit * 0.1),
        "test": limit - round(limit * 0.8) - round(limit * 0.1),
    }
    split_samples: dict[str, list[RegulationSample]] = {}
    for split, ids in split_ids.items():
        pool = [sample for pid in ids for sample in by_paragraph[pid]]
        if len(pool) < targets[split]:
            raise ValueError(f"Not enough {split} samples: need {targets[split]}, got {len(pool)}")
        split_samples[split] = select_balanced(pool, targets[split], rng)

    return split_samples["train"], split_samples["valid"], split_samples["test"]


def write_jsonl(path: Path, samples: list[RegulationSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")


def write_review_csv(path: Path, rows: Iterable[tuple[str, RegulationSample]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "category", "paragraph_id", "source", "prompt", "answer", "evidence"])
        for split, sample in rows:
            writer.writerow(
                [
                    split,
                    sample.category,
                    sample.paragraph_id,
                    sample.source,
                    sample.prompt,
                    sample.answer,
                    sample.evidence,
                ]
            )


def validate_split_disjoint(train: list[RegulationSample], valid: list[RegulationSample], test: list[RegulationSample]) -> None:
    train_ids = {sample.paragraph_id for sample in train}
    valid_ids = {sample.paragraph_id for sample in valid}
    test_ids = {sample.paragraph_id for sample in test}
    if train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids:
        raise ValueError("Paragraph leakage detected across train/valid/test")
    for sample in train + valid + test:
        if sample.answer not in sample.evidence:
            raise ValueError(f"Non-extractive answer detected: {sample.paragraph_id}")


def write_readme(path: Path, train: list[RegulationSample], valid: list[RegulationSample], test: list[RegulationSample]) -> None:
    all_samples = train + valid + test
    counts = Counter(sample.category for sample in all_samples)
    sources = Counter(sample.source for sample in all_samples)
    paragraph_count = len({sample.paragraph_id for sample in all_samples})
    lines = [
        "# Railway Regulation QA Dataset",
        "",
        "This dataset is generated by deterministic rule-based extraction from local railway regulation documents.",
        "No LLM is used during generation. Every answer is an exact substring of the stored `evidence` field.",
        "",
        "## Splits",
        "",
        f"- train: {len(train)}",
        f"- valid: {len(valid)}",
        f"- test: {len(test)}",
        f"- total: {len(all_samples)}",
        f"- unique evidence paragraphs: {paragraph_count}",
        "",
        "Paragraph IDs are split disjointly across train, valid, and test to reduce same-paragraph leakage.",
        "",
        "## Categories",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Sources", ""])
    for source, count in sorted(sources.items()):
        lines.append(f"- {source}: {count}")
    lines.extend(
        [
            "",
            "## Schema",
            "",
            "Each JSONL record contains `prompt`, `answer`, `text`, `category`, `source`,",
            "`paragraph_id`, `evidence`, `answer_start`, and `generation_method`.",
            "`text` is formatted as `Question: ...\\nAnswer: ...` for QLoRA training.",
            "",
            "A full `human_review.csv` file is also written for manual verification.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build extractive railway regulation QA JSONL data.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--source-glob", default="*规章*.docx")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-paragraph", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidates = build_candidates(args.corpus_dir, args.source_glob, args.max_per_paragraph)
    if len(candidates) < args.limit:
        raise ValueError(f"Need {args.limit} samples, but only generated {len(candidates)} candidates")

    train, valid, test = split_by_paragraph(candidates, args.limit, args.seed)
    validate_split_disjoint(train, valid, test)

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    write_jsonl(args.output_dir / "test.jsonl", test)
    write_review_csv(
        args.output_dir / "human_review.csv",
        [("train", sample) for sample in train]
        + [("valid", sample) for sample in valid]
        + [("test", sample) for sample in test],
    )
    write_readme(args.output_dir / "README.md", train, valid, test)

    all_samples = train + valid + test
    print(f"Generated candidates: {len(candidates)}")
    print(f"Selected total: {len(all_samples)}")
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")
    print("categories:")
    for category, count in sorted(Counter(sample.category for sample in all_samples).items()):
        print(f"  {category}: {count}")
    print(f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
