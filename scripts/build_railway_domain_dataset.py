"""Build the local railway domain QA dataset from copied corpus documents."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
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
    r"([A-Za-z0-9（）()、/·\-\s]*?[\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9（）()、/·\-\s]{0,40}?)\s+"
    r"([A-Za-z0-9(][^\u4e00-\u9fff]{1,120}?)(?=\s+(?:[A-Za-z0-9（）()、/·\-]*[\u4e00-\u9fff])|$)"
)
EN_PUNCT_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ0-9,;:/().&'\-\[\]\s]+$")
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
    "under",
}
INCOMPLETE_EN_TERM_ENDINGS = {
    "broken",
}
VOCAB_CATEGORIES = [
    {
        "id": 1,
        "key": "general_vocabulary",
        "zh": "通用词汇",
        "en": "General vocabulary",
        "heading": "通用词汇 General vocabulary",
    },
    {"id": 2, "key": "communication", "zh": "通信", "en": "Communication", "heading": "通 信 Communication"},
    {"id": 3, "key": "signaling", "zh": "信号", "en": "Signaling", "heading": "信 号 Signaling"},
    {"id": 4, "key": "locomotive", "zh": "机车", "en": "Locomotive", "heading": "机 车 Locomotive"},
    {"id": 5, "key": "rolling_stock", "zh": "车辆", "en": "Rolling stock", "heading": "车 辆 Rolling stock"},
    {
        "id": 6,
        "key": "traction_power_supply",
        "zh": "牵引供电",
        "en": "Traction power supply",
        "heading": "牵引供电 Traction power supply",
    },
    {"id": 7, "key": "permanent_way", "zh": "工务工程", "en": "Permanent way", "heading": "工务工程 Permanent way"},
    {
        "id": 8,
        "key": "transportation_economy",
        "zh": "运输与经济",
        "en": "Transportation and economy",
        "heading": "运输与经济 Transportation and economy",
    },
    {"id": 9, "key": "related_sciences", "zh": "相关科学", "en": "Related sciences", "heading": "相关科学 Related sciences"},
    {"id": 10, "key": "abbreviation", "zh": "常用缩写", "en": "Abbreviation", "heading": "附件1常用缩写 Abbreviation"},
    {
        "id": 11,
        "key": "international_organizations",
        "zh": "相关国际组织名称",
        "en": "Railways and International Organizations",
        "heading": "相关国际组织名称 Railways and International Organizations",
    },
]


@dataclass(frozen=True)
class Sample:
    prompt: str
    answer: str
    category: str
    source: str
    domain_category_id: int | None = None
    domain_category_key: str | None = None
    domain_category: str | None = None
    domain_category_en: str | None = None
    term_zh: str | None = None
    term_en: str | None = None

    def to_record(self) -> dict[str, str | int]:
        record: dict[str, str | int] = {
            "prompt": self.prompt,
            "answer": self.answer,
            "text": f"Question: {self.prompt}\nAnswer: {self.answer}",
            "category": self.category,
            "source": self.source,
        }
        for key in (
            "domain_category_id",
            "domain_category_key",
            "domain_category",
            "domain_category_en",
            "term_zh",
            "term_en",
        ):
            value = getattr(self, key)
            if value is not None:
                record[key] = value
        return record


@dataclass(frozen=True)
class TermPair:
    zh: str
    en: str
    source: str
    domain_category_id: int
    domain_category_key: str
    domain_category: str
    domain_category_en: str
    source_block: int
    abbreviation: str | None = None
    full_name_en: str | None = None

    def to_record(self) -> dict[str, str | int]:
        record: dict[str, str | int] = {
            "term_zh": self.zh,
            "term_en": self.en,
            "source": self.source,
            "domain_category_id": self.domain_category_id,
            "domain_category_key": self.domain_category_key,
            "domain_category": self.domain_category,
            "domain_category_en": self.domain_category_en,
            "source_block": self.source_block,
        }
        if self.abbreviation:
            record["abbreviation"] = self.abbreviation
        if self.full_name_en:
            record["full_name_en"] = self.full_name_en
        return record


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


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def prepare_term_line(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"(?<=[\u4e00-\u9fff）)])(?=[A-Za-z0-9]+(?:\s*[-/]\s*|\s+[A-Za-z]))", " ", text)
    text = re.sub(r"(?<=[a-z)])(?=[\u4e00-\u9fff])", "  ", text)
    return text.strip()


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


def raw_text_from_element(element: ET.Element) -> str:
    return "".join((node.text or "") for node in element.findall(".//w:t", {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"})).strip()


def table_cells(row: ET.Element) -> list[str]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    cells: list[str] = []
    for cell in row.findall("./w:tc", ns):
        paragraphs: list[str] = []
        for para in cell.findall("./w:p", ns):
            text = normalize_text("".join((node.text or "") for node in para.findall(".//w:t", ns)))
            if text:
                paragraphs.append(text)
        cells.append(normalize_text(" ".join(paragraphs)))
    return cells


def is_section_heading(text: str, category: dict[str, str | int]) -> bool:
    compact = compact_text(text)
    heading = compact_text(str(category["heading"]))
    return len(text) <= 120 and (compact == heading or compact.startswith(heading + heading))


def find_vocab_category_ranges(body_children: list[ET.Element]) -> list[tuple[dict[str, str | int], int, int]]:
    starts: list[tuple[dict[str, str | int], int]] = []
    cursor = 0
    for category in VOCAB_CATEGORIES:
        for index in range(cursor, len(body_children)):
            text = normalize_text(raw_text_from_element(body_children[index]))
            if is_section_heading(text, category):
                starts.append((category, index))
                cursor = index + 1
                break
        else:
            raise ValueError(f"Cannot locate vocabulary section heading: {category['heading']}")

    pinyin_start = len(body_children)
    for index in range(starts[-1][1] + 1, len(body_children)):
        text = normalize_text(raw_text_from_element(body_children[index]))
        if text == "A":
            pinyin_start = index
            break

    ranges: list[tuple[dict[str, str | int], int, int]] = []
    for current, next_item in zip(starts, starts[1:] + [(None, pinyin_start)]):  # type: ignore[list-item]
        category, start = current
        _, end = next_item
        ranges.append((category, start, end))
    return ranges


def is_repeated_header_or_footer(text: str) -> bool:
    compact = compact_text(text)
    if not text:
        return True
    if "chinese-englishdictionaryforrailwaytechnicalstandards" in compact:
        return True
    if "续上表" in compact:
        return True
    return any(is_section_heading(text, category) for category in VOCAB_CATEGORIES)


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
    if not (2 <= len(value) <= 140):
        return False
    if not has_balanced_brackets(value):
        return False
    if not EN_PUNCT_RE.fullmatch(value):
        return False
    if value.count("(") != value.count(")"):
        return False
    words = [part for part in re.split(r"[\s/\-]+", value) if part]
    if not 1 <= len(words) <= 16:
        return False
    if words[-1].lower().strip(".,;:()'\"") in INCOMPLETE_EN_TERM_SUFFIXES:
        return False
    if len(words) <= 3 and words[-1].lower().strip(".,;:()'\"") in INCOMPLETE_EN_TERM_ENDINGS:
        return False
    joined = "".join(words)
    # Very long unbroken lowercase strings are usually OCR/page-layout artifacts.
    if re.search(r"[a-z]{18,}", joined):
        return False
    return latin_ratio(value) >= 0.25


def is_clean_chinese_term(value: str) -> bool:
    if not (2 <= len(value) <= 48):
        return False
    if not has_balanced_brackets(value):
        return False
    if re.search(r"^[a-z]", value):
        return False
    if re.search(r"[a-z]{2,}\s+[a-z]{2,}", value.lower()):
        return False
    if han_ratio(value) < 0.35:
        return False
    return not any(token in value for token in ("目录", "附录", "页", "章", "续上表", "Dictionary"))


def english_label(abbreviation: str, full_name: str) -> str:
    abbreviation = normalize_text(abbreviation)
    full_name = normalize_text(full_name)
    if not abbreviation:
        return full_name
    if not full_name or abbreviation.lower() == full_name.lower():
        return abbreviation
    return f"{abbreviation} ({full_name})"


def term_pair_from_values(
    *,
    zh: str,
    en: str,
    source: str,
    category: dict[str, str | int],
    source_block: int,
    abbreviation: str | None = None,
    full_name_en: str | None = None,
) -> TermPair | None:
    zh = normalize_chinese_term(clean_term(zh))
    en = clean_term(en)
    if not is_clean_chinese_term(zh) or not is_clean_english_term(en):
        return None
    return TermPair(
        zh=zh,
        en=en,
        source=source,
        domain_category_id=int(category["id"]),
        domain_category_key=str(category["key"]),
        domain_category=str(category["zh"]),
        domain_category_en=str(category["en"]),
        source_block=source_block,
        abbreviation=normalize_text(abbreviation) if abbreviation else None,
        full_name_en=normalize_text(full_name_en) if full_name_en else None,
    )


def term_pairs_from_paragraph(text: str, source: str, category: dict[str, str | int], source_block: int) -> Iterable[TermPair]:
    if len(text) > 500 or DOT_LEADER_RE.search(text) or is_repeated_header_or_footer(text):
        return
    for zh, en in TERM_PAIR_RE.findall(prepare_term_line(text)):
        pair = term_pair_from_values(
            zh=zh,
            en=en,
            source=source,
            category=category,
            source_block=source_block,
        )
        if pair is not None:
            yield pair


def term_pairs_from_table(table: ET.Element, source: str, category: dict[str, str | int], source_block: int) -> Iterable[TermPair]:
    rows = table.findall(".//w:tr", {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"})
    for row in rows[1:]:
        cells = table_cells(row)
        if len(cells) < 3:
            continue
        if category["key"] == "abbreviation":
            abbreviation, full_name, zh = cells[:3]
            en = english_label(abbreviation, full_name)
            pair = term_pair_from_values(
                zh=zh,
                en=en,
                source=source,
                category=category,
                source_block=source_block,
                abbreviation=abbreviation,
                full_name_en=full_name,
            )
        else:
            zh, abbreviation, full_name = cells[:3]
            en = english_label(abbreviation, full_name)
            pair = term_pair_from_values(
                zh=zh,
                en=en,
                source=source,
                category=category,
                source_block=source_block,
                abbreviation=abbreviation,
                full_name_en=full_name,
            )
        if pair is not None:
            yield pair


def build_term_pairs_from_docx(path: Path) -> list[TermPair]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find("w:body", ns)
    if body is None:
        raise ValueError(f"Missing Word body: {path}")
    body_children = list(body)
    output: list[TermPair] = []
    seen: set[tuple[str, str, str]] = set()

    for category, start, end in find_vocab_category_ranges(body_children):
        for index in range(start + 1, end):
            child = body_children[index]
            tag = child.tag.rsplit("}", 1)[-1]
            pairs: Iterable[TermPair]
            if category["key"] in {"abbreviation", "international_organizations"} and tag == "tbl":
                pairs = term_pairs_from_table(child, path.name, category, index)
            elif category["key"] not in {"abbreviation", "international_organizations"} and tag == "p":
                pairs = term_pairs_from_paragraph(raw_text_from_element(child), path.name, category, index)
            else:
                continue
            for pair in pairs:
                key = (pair.domain_category_key, pair.zh, pair.en)
                if key in seen:
                    continue
                seen.add(key)
                output.append(pair)
    return output


def build_term_samples(term_pairs: Iterable[TermPair]) -> Iterable[Sample]:
    for pair in term_pairs:
        yield Sample(
            prompt=answer_only_prompt(
                f"Provide the English railway technical term for the {pair.domain_category_en} subdomain.",
                pair.zh,
            ),
            answer=pair.en,
            category="terminology_zh_to_en",
            source=pair.source,
            domain_category_id=pair.domain_category_id,
            domain_category_key=pair.domain_category_key,
            domain_category=pair.domain_category,
            domain_category_en=pair.domain_category_en,
            term_zh=pair.zh,
            term_en=pair.en,
        )
        yield Sample(
            prompt=answer_only_prompt(
                f"Provide the Chinese railway technical term for the {pair.domain_category_en} subdomain.",
                pair.en,
            ),
            answer=pair.zh,
            category="terminology_en_to_zh",
            source=pair.source,
            domain_category_id=pair.domain_category_id,
            domain_category_key=pair.domain_category_key,
            domain_category=pair.domain_category,
            domain_category_en=pair.domain_category_en,
            term_zh=pair.zh,
            term_en=pair.en,
        )


def deduplicate(samples: Iterable[Sample]) -> list[Sample]:
    seen: set[tuple[str, str, str, str | None]] = set()
    output: list[Sample] = []
    for sample in samples:
        key = (sample.prompt, sample.answer, sample.category, sample.domain_category_key)
        if key in seen:
            continue
        seen.add(key)
        output.append(sample)
    return output


def split_samples(samples: list[Sample], seed: int) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)
    by_category: dict[tuple[str, str], list[Sample]] = {}
    for sample in samples:
        by_category.setdefault((sample.category, sample.domain_category_key or ""), []).append(sample)

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


def write_term_inventory(path: Path, term_pairs: list[TermPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in term_pairs:
            handle.write(json.dumps(pair.to_record(), ensure_ascii=False) + "\n")


def write_readme(path: Path, train: list[Sample], valid: list[Sample], test: list[Sample]) -> None:
    all_samples = train + valid + test
    counts = Counter(sample.category for sample in all_samples)
    unique_terms = {
        (
            sample.domain_category_id,
            sample.domain_category,
            sample.domain_category_en,
            sample.domain_category_key,
            sample.term_zh,
            sample.term_en,
        )
        for sample in all_samples
        if sample.term_zh and sample.term_en
    }
    domain_counts = Counter((term[0], term[1], term[2]) for term in unique_terms)
    lines = [
        "# Railway Domain QA Dataset",
        "",
        "This dataset is generated from local railway bilingual corpus documents under `data/corpus/railway`.",
        "Terminology samples are extracted from the classified vocabulary Word document and keep the original railway subdomain labels.",
        "",
        "## Splits",
        "",
        f"- train: {len(train)}",
        f"- valid: {len(valid)}",
        f"- test: {len(test)}",
        f"- unique classified term pairs: {len(unique_terms)}",
        "",
        "## Categories",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Terminology Subdomains", ""])
    for category_id, category, category_en in sorted(domain_counts):
        lines.append(f"- {category_id}. {category} / {category_en}: {domain_counts[(category_id, category, category_en)]}")
    lines.extend(
        [
            "",
            "## Schema",
            "",
            "Each JSONL record contains `prompt`, `answer`, `text`, `category`, and `source`.",
            "Terminology records also contain `domain_category_id`, `domain_category_key`,",
            "`domain_category`, `domain_category_en`, `term_zh`, and `term_en`.",
            "`text` is formatted as `Question: ...\\nAnswer: ...` for QLoRA training.",
            "The unique classified term inventory is written to `terminology_inventory.jsonl`.",
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
    all_term_pairs: list[TermPair] = []
    for docx_file in docx_files:
        source = docx_file.name
        if "词汇" in source:
            term_pairs = build_term_pairs_from_docx(docx_file)
            all_term_pairs.extend(term_pairs)
            samples.extend(build_term_samples(term_pairs))
            continue
        paragraphs = read_docx_paragraphs(docx_file)
        samples.extend(build_translation_samples(paragraphs, source))

    samples = deduplicate(samples)
    train, valid, test = split_samples(samples, args.seed)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    write_jsonl(args.output_dir / "test.jsonl", test)
    write_term_inventory(args.output_dir / "terminology_inventory.jsonl", all_term_pairs)
    write_readme(args.output_dir / "README.md", train, valid, test)

    print(f"Generated {len(samples)} samples")
    print(f"Classified term pairs: {len(all_term_pairs)}")
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")
    print("term subdomains:")
    for (category_id, category, category_en), count in sorted(
        Counter((pair.domain_category_id, pair.domain_category, pair.domain_category_en) for pair in all_term_pairs).items()
    ):
        print(f"  {category_id}. {category} / {category_en}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
