"""Build the local railway domain QA dataset from copied corpus documents."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "corpus" / "railway"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "domain"

HAN_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")
DOT_LEADER_RE = re.compile(r"\.{4,}")
TERM_PAIR_RE = re.compile(
    r"([\u4e00-\u9fff][\u4e00-\u9fff0-9（）()、/·\-\s]{1,28}?)\s+"
    r"([A-Za-z][A-Za-z0-9,;:/().&'\-\s]{2,90})(?=\s{2,}[\u4e00-\u9fff]|$)"
)
EN_PUNCT_RE = re.compile(r"^[A-Za-z0-9,;:/().&'\-\s]+$")
EN_LEADING_FRAGMENT_RE = re.compile(
    r"^(and|or|the|of|to|for|with|without|after|before|when|while|if|in|on|at|from|by)\s+",
    re.IGNORECASE,
)
INCOMPLETE_EN_TERM_SUFFIXES = {
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
    "working",
}
INCOMPLETE_EN_TERM_ENDINGS = {
    "broken",
}


@dataclass(frozen=True)
class Sample:
    prompt: str
    answer: str
    category: str
    source: str

    def to_record(self) -> dict[str, str]:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "text": f"Question: {self.prompt}\nAnswer: {self.answer}",
            "category": self.category,
            "source": self.source,
        }


def answer_only_prompt(instruction: str, question: str) -> str:
    """Build a prompt that asks for the answer only, reducing verbose generations."""
    return (
        f"{instruction}\n"
        "Return only the final answer. Do not explain. Do not add prefixes, suffixes, quotes, or labels.\n"
        f"{question}"
    )


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def normalize_chinese_term(text: str) -> str:
    text = normalize_text(text)
    # OCR often inserts spaces between Chinese characters. Remove only those spaces.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text


def is_chinese(text: str) -> bool:
    return bool(HAN_RE.search(text))


def is_english(text: str) -> bool:
    return bool(LATIN_RE.search(text)) and not is_chinese(text)


def latin_ratio(text: str) -> float:
    return sum(ch.isascii() and ch.isalpha() for ch in text) / max(1, len(text))


def han_ratio(text: str) -> float:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in text) / max(1, len(text))


def has_sentence_boundary(text: str, language: str) -> bool:
    if language == "zh":
        return text.endswith(("。", "；", "：", "）", ")", "”", "」", "】"))
    return text.endswith((".", ";", ":", ")", '"', "'"))


def has_balanced_brackets(text: str) -> bool:
    pairs = [("(", ")"), ("（", "）"), ("[", "]"), ("【", "】")]
    return all(text.count(left) == text.count(right) for left, right in pairs)


def is_complete_translation_pair(chinese: str, english: str) -> bool:
    if not has_balanced_brackets(chinese) or not has_balanced_brackets(english):
        return False
    if EN_LEADING_FRAGMENT_RE.match(english) and len(english) < 80:
        return False
    if len(chinese) > 24 and not has_sentence_boundary(chinese, "zh"):
        return False
    if len(english) > 40 and not has_sentence_boundary(english, "en"):
        return False
    return True


def is_useful_text(text: str) -> bool:
    if len(text) < 4 or len(text) > 700:
        return False
    if DOT_LEADER_RE.search(text):
        return False
    if text.isdigit():
        return False
    alpha_or_han = sum(ch.isalpha() or "\u4e00" <= ch <= "\u9fff" for ch in text)
    return alpha_or_han / max(1, len(text)) >= 0.35


def read_docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        text = normalize_text("".join((node.text or "") for node in para.findall(".//w:t", ns)))
        if is_useful_text(text):
            paragraphs.append(text)
    return paragraphs


def build_translation_samples(paragraphs: list[str], source: str) -> Iterable[Sample]:
    if "词汇" in source:
        return
    for first, second in zip(paragraphs, paragraphs[1:]):
        if not (12 <= len(first) <= 450 and 12 <= len(second) <= 700):
            continue
        if han_ratio(first) < 0.25 or latin_ratio(second) < 0.35:
            continue
        if not is_complete_translation_pair(first, second):
            continue
        if is_chinese(first) and is_english(second):
            yield Sample(
                prompt=answer_only_prompt(
                    "Translate the following railway-domain Chinese text into English.",
                    first,
                ),
                answer=second,
                category="zh_to_en_translation",
                source=source,
            )
            yield Sample(
                prompt=answer_only_prompt(
                    "Translate the following railway-domain English text into Chinese.",
                    second,
                ),
                answer=first,
                category="en_to_zh_translation",
                source=source,
            )


def clean_term(value: str) -> str:
    value = normalize_text(value)
    value = value.strip(" ,;:/.-")
    return value


def is_clean_english_term(value: str) -> bool:
    if not (3 <= len(value) <= 80):
        return False
    if not has_balanced_brackets(value):
        return False
    if not EN_PUNCT_RE.fullmatch(value):
        return False
    if value.count("(") != value.count(")"):
        return False
    words = [part for part in re.split(r"[\s/\-]+", value) if part]
    if not 1 <= len(words) <= 8:
        return False
    if any(word.isdigit() for word in words):
        return False
    if words[-1].lower().strip(".,;:()'\"") in INCOMPLETE_EN_TERM_SUFFIXES:
        return False
    if len(words) <= 3 and words[-1].lower().strip(".,;:()'\"") in INCOMPLETE_EN_TERM_ENDINGS:
        return False
    joined = "".join(words)
    # Very long unbroken lowercase strings are usually OCR/page-layout artifacts.
    if re.search(r"[a-z]{18,}", joined):
        return False
    return latin_ratio(value) >= 0.45


def is_clean_chinese_term(value: str) -> bool:
    if not (2 <= len(value) <= 24):
        return False
    if not has_balanced_brackets(value):
        return False
    if LATIN_RE.search(value):
        return False
    if han_ratio(value) < 0.45:
        return False
    return not any(token in value for token in ("目录", "附录", "页", "章"))


def build_term_samples(paragraphs: list[str], source: str) -> Iterable[Sample]:
    for paragraph in paragraphs:
        if len(paragraph) > 260 or DOT_LEADER_RE.search(paragraph):
            continue
        for zh, en in TERM_PAIR_RE.findall(paragraph):
            zh = normalize_chinese_term(clean_term(zh))
            en = clean_term(en)
            if not is_clean_chinese_term(zh) or not is_clean_english_term(en):
                continue
            en_words = [part for part in re.split(r"[\s/\-]+", en) if part]
            if len(zh) >= 8 and len(en_words) <= 1:
                continue
            if len(zh) >= 12 and len(en_words) <= 2:
                continue
            yield Sample(
                prompt=answer_only_prompt(
                    "Provide the English railway technical term.",
                    zh,
                ),
                answer=en,
                category="terminology_zh_to_en",
                source=source,
            )
            yield Sample(
                prompt=answer_only_prompt(
                    "Provide the Chinese railway technical term.",
                    en,
                ),
                answer=zh,
                category="terminology_en_to_zh",
                source=source,
            )


def deduplicate(samples: Iterable[Sample]) -> list[Sample]:
    seen: set[tuple[str, str]] = set()
    output: list[Sample] = []
    for sample in samples:
        key = (sample.prompt, sample.answer)
        if key in seen:
            continue
        seen.add(key)
        output.append(sample)
    return output


def split_samples(samples: list[Sample], seed: int) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)
    by_category: dict[str, list[Sample]] = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)

    train: list[Sample] = []
    valid: list[Sample] = []
    test: list[Sample] = []
    for category_samples in by_category.values():
        rng.shuffle(category_samples)
        total = len(category_samples)
        valid_count = max(1, round(total * 0.1)) if total >= 10 else max(0, total // 10)
        test_count = max(1, round(total * 0.1)) if total >= 10 else max(0, total // 10)
        valid.extend(category_samples[:valid_count])
        test.extend(category_samples[valid_count : valid_count + test_count])
        train.extend(category_samples[valid_count + test_count :])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def write_jsonl(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")


def write_readme(path: Path, train: list[Sample], valid: list[Sample], test: list[Sample]) -> None:
    counts: dict[str, int] = {}
    for sample in train + valid + test:
        counts[sample.category] = counts.get(sample.category, 0) + 1
    lines = [
        "# Railway Domain QA Dataset",
        "",
        "This dataset is generated from local railway bilingual corpus documents under `data/corpus/railway`.",
        "",
        "## Splits",
        "",
        f"- train: {len(train)}",
        f"- valid: {len(valid)}",
        f"- test: {len(test)}",
        "",
        "## Categories",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(
        [
            "",
            "## Schema",
            "",
            "Each JSONL record contains `prompt`, `answer`, `text`, `category`, and `source`.",
            "`text` is formatted as `Question: ...\\nAnswer: ...` for QLoRA training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build railway domain JSONL data from local corpus documents.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    docx_files = sorted(args.corpus_dir.glob("*.docx"))
    if not docx_files:
        raise FileNotFoundError(f"No .docx files found under {args.corpus_dir}")

    samples: list[Sample] = []
    for docx_file in docx_files:
        paragraphs = read_docx_paragraphs(docx_file)
        source = docx_file.name
        samples.extend(build_translation_samples(paragraphs, source))
        samples.extend(build_term_samples(paragraphs, source))

    samples = deduplicate(samples)
    train, valid, test = split_samples(samples, args.seed)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    write_jsonl(args.output_dir / "test.jsonl", test)
    write_readme(args.output_dir / "README.md", train, valid, test)

    print(f"Generated {len(samples)} samples")
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
