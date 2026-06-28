from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pandas as pd


HEADER_HINTS = ("tosp", "description", "lower", "upper", "ward type", "drg", "ccs", "icd", "diagnosis")

QUERY_EXPANSIONS = {
    "cost": ("fee", "fees", "bill", "benchmark", "lower", "upper"),
    "price": ("fee", "bill", "benchmark"),
    "bill": ("fee", "cost", "benchmark"),
    "surgery": ("surgical", "procedure", "operation", "tosp"),
    "procedure": ("surgery", "operation", "tosp"),
    "doctor": ("surgeon", "anaesthetist", "attendance"),
    "anesthesia": ("anaesthesia", "anaesthetist"),
    "anaesthetic": ("anaesthesia", "anaesthetist"),
    "ward": ("inpatient", "attendance", "hospital"),
    "icu": ("intensive", "care", "unit"),
    "hdu": ("high", "dependency", "unit"),
    "diagnosis": ("drg", "ccs", "icd", "condition"),
    "condition": ("diagnosis", "drg", "ccs", "medical"),
}

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "for",
    "from",
    "how",
    "into",
    "much",
    "need",
    "the",
    "this",
    "what",
    "when",
    "with",
    "would",
}

GENERIC_QUERY_TERMS = {
    "benchmark",
    "bill",
    "cost",
    "estimate",
    "fee",
    "fees",
    "hospital",
    "medical",
    "price",
    "procedure",
    "surgery",
    "surgical",
}


@dataclass
class BenchmarkRecord:
    sheet: str
    row_number: int
    fields: dict[str, str]
    searchable_text: str
    tokens: tuple[str, ...]

    def as_context(self) -> dict[str, Any]:
        return {
            "sheet": self.sheet,
            "row_number": self.row_number,
            "fields": self.fields,
        }


@dataclass
class BenchmarkIndex:
    records: list[BenchmarkRecord]
    document_frequency: dict[str, int]
    average_length: float


def load_benchmark_records(path: str) -> list[BenchmarkRecord]:
    workbook = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    records: list[BenchmarkRecord] = []
    for sheet_name, frame in workbook.items():
        frame = frame.fillna("")
        header_index = find_header_row(frame)
        if header_index is None:
            records.extend(load_note_records(sheet_name, frame))
        else:
            records.extend(load_tabular_records(sheet_name, frame, header_index))
    return records


def build_benchmark_index(records: list[BenchmarkRecord]) -> BenchmarkIndex:
    document_frequency: Counter[str] = Counter()
    total_length = 0
    for record in records:
        document_frequency.update(set(record.tokens))
        total_length += len(record.tokens)
    average_length = total_length / max(len(records), 1)
    return BenchmarkIndex(records=records, document_frequency=dict(document_frequency), average_length=average_length)


def search_benchmark_records(index: BenchmarkIndex, query: str, mode: str, limit: int = 10) -> list[tuple[BenchmarkRecord, float]]:
    raw_terms = tokenize(query)
    query_terms = expand_query_terms(raw_terms)
    specific_terms = expand_query_terms([term for term in raw_terms if term not in GENERIC_QUERY_TERMS])
    if not query_terms:
        return []

    intents = infer_query_intents(query, mode)
    query_phrases = extract_query_phrases(query)
    query_codes = extract_codes(query)

    candidates: list[tuple[BenchmarkRecord, float]] = []
    for record in index.records:
        score = hybrid_score(record, index, query_terms, specific_terms, query_phrases, query_codes, intents)
        if score > 0:
            candidates.append((record, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return diversify_results(candidates[: max(limit * 5, limit)], limit)


def hybrid_score(
    record: BenchmarkRecord,
    index: BenchmarkIndex,
    query_terms: list[str],
    specific_terms: list[str],
    query_phrases: list[str],
    query_codes: list[str],
    intents: set[str],
) -> float:
    if specific_terms and not has_specific_match(record.searchable_text, specific_terms, query_codes):
        return 0.0

    bm25 = bm25_score(record, index, query_terms)
    if bm25 == 0:
        fuzzy = fuzzy_score(record.searchable_text, query_terms)
        if fuzzy < 0.55:
            return 0
    else:
        fuzzy = fuzzy_score(record.searchable_text, query_terms)

    phrase = phrase_score(record.searchable_text, query_phrases)
    code = code_score(record.searchable_text, query_codes)
    sheet = sheet_intent_boost(record.sheet, intents)
    field = field_boost(record, query_terms)
    specificity = specific_term_score(record.searchable_text, specific_terms)
    return (bm25 * 1.0) + (phrase * 2.5) + (code * 8.0) + (fuzzy * 1.4) + sheet + field + (specificity * 3.0)


def has_specific_match(text: str, specific_terms: list[str], query_codes: list[str]) -> bool:
    if any(code in text.upper() for code in query_codes):
        return True
    return any(term in text for term in specific_terms)


def specific_term_score(text: str, specific_terms: list[str]) -> float:
    if not specific_terms:
        return 0.0
    matched = sum(1 for term in specific_terms if term in text)
    return matched / len(specific_terms)


def bm25_score(record: BenchmarkRecord, index: BenchmarkIndex, query_terms: list[str]) -> float:
    k1 = 1.5
    b = 0.75
    counts = Counter(record.tokens)
    score = 0.0
    doc_len = max(len(record.tokens), 1)
    for term in query_terms:
        tf = counts.get(term, 0)
        if tf == 0:
            continue
        df = index.document_frequency.get(term, 0)
        idf = math.log(1 + ((len(index.records) - df + 0.5) / (df + 0.5)))
        denominator = tf + k1 * (1 - b + b * (doc_len / max(index.average_length, 1)))
        score += idf * ((tf * (k1 + 1)) / denominator)
    return score


def phrase_score(text: str, phrases: list[str]) -> float:
    if not phrases:
        return 0.0
    return sum(1.0 for phrase in phrases if phrase and phrase in text)


def code_score(text: str, codes: list[str]) -> float:
    if not codes:
        return 0.0
    upper_text = text.upper()
    return sum(1.0 for code in codes if code in upper_text)


def fuzzy_score(text: str, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    best_scores = []
    for term in query_terms[:12]:
        if term in text_tokens:
            best_scores.append(1.0)
            continue
        best_scores.append(max(SequenceMatcher(None, term, token).ratio() for token in text_tokens))
    return sum(best_scores) / len(best_scores)


def sheet_intent_boost(sheet_name: str, intents: set[str]) -> float:
    sheet = sheet_name.lower()
    boost = 0.0
    if "doctor_fee" in intents and any(term in sheet for term in ("surg", "ana", "inpatient")):
        boost += 1.8
    if "hospital_fee" in intents and "hosp" in sheet:
        boost += 1.8
    if "medical_condition" in intents and any(term in sheet for term in ("medical", "ccs", "icd")):
        boost += 1.8
    if "surgical" in intents and any(term in sheet for term in ("surg", "tosp")):
        boost += 1.2
    if "inpatient" in intents and "inpatient" in sheet:
        boost += 1.2
    return boost


def field_boost(record: BenchmarkRecord, query_terms: list[str]) -> float:
    boost = 0.0
    important_fields = ("description", "tosp", "drg", "ccs", "diagnosis", "ward_type", "anatomical")
    for key, value in record.fields.items():
        if not any(field in key for field in important_fields):
            continue
        field_text = value.lower()
        boost += sum(0.35 for term in query_terms if term in field_text)
    return min(boost, 3.0)


def diversify_results(candidates: list[tuple[BenchmarkRecord, float]], limit: int) -> list[tuple[BenchmarkRecord, float]]:
    selected: list[tuple[BenchmarkRecord, float]] = []
    seen_keys: set[tuple[str, str]] = set()
    for record, score in candidates:
        signature = (
            record.sheet,
            first_matching_field(record, ("tosp", "drg", "ccs", "description", "note"))[:120].lower(),
        )
        if signature in seen_keys and len(selected) >= max(3, limit // 2):
            continue
        seen_keys.add(signature)
        selected.append((record, round(score, 3)))
        if len(selected) >= limit:
            break
    return selected


def build_context(matches: list[tuple[BenchmarkRecord, float]]) -> str:
    rows = [record.as_context() | {"retrieval_score": score} for record, score in matches]
    return json.dumps(rows, indent=2, ensure_ascii=False)


def estimate_amounts(record: BenchmarkRecord) -> tuple[int | None, int | None]:
    numbers: list[int] = []
    for key, value in record.fields.items():
        if any(token in key for token in ("lower", "upper", "fee", "bound", "cost")):
            numbers.extend(int(num.replace(",", "")) for num in re.findall(r"\d[\d,]*", value))
    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def first_matching_field(record: BenchmarkRecord, names: tuple[str, ...]) -> str:
    for name in names:
        for key, value in record.fields.items():
            if name in key and value:
                return value
    return next(iter(record.fields.values()), "")


def find_header_row(frame: pd.DataFrame) -> int | None:
    best_index: int | None = None
    best_score = 0
    for idx, row in frame.iterrows():
        text_cells = [str(cell).strip().lower() for cell in row.tolist() if str(cell).strip()]
        score = sum(any(hint in cell for cell in text_cells) for hint in HEADER_HINTS)
        if score > best_score:
            best_index = int(idx)
            best_score = score
    return best_index if best_score >= 2 else None


def load_note_records(sheet_name: str, frame: pd.DataFrame) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for idx, row in frame.iterrows():
        text = " ".join(str(cell).strip() for cell in row.tolist() if str(cell).strip())
        if len(text) < 20:
            continue
        records.append(make_record(sheet_name, int(idx) + 1, {"note": clean_text(text)}))
    return records


def load_tabular_records(sheet_name: str, frame: pd.DataFrame, header_index: int) -> list[BenchmarkRecord]:
    headers = make_headers(frame.iloc[header_index].tolist())
    records: list[BenchmarkRecord] = []
    for idx, row in frame.iloc[header_index + 1 :].iterrows():
        values = [clean_text(str(cell)) for cell in row.tolist()]
        fields = {
            headers[col_index]: value
            for col_index, value in enumerate(values)
            if col_index < len(headers) and value
        }
        if len(fields) < 2:
            continue
        records.append(make_record(sheet_name, int(idx) + 1, fields))
    return records


def make_record(sheet_name: str, row_number: int, fields: dict[str, str]) -> BenchmarkRecord:
    row_text = " ".join(fields.values())
    searchable_text = clean_text(f"{sheet_name} {row_text}").lower()
    return BenchmarkRecord(
        sheet=sheet_name,
        row_number=row_number,
        fields=fields,
        searchable_text=searchable_text,
        tokens=tuple(tokenize(searchable_text)),
    )


def make_headers(values: list[Any]) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for idx, value in enumerate(values):
        header = clean_text(str(value)).lower()
        header = re.sub(r"[^a-z0-9]+", "_", header).strip("_")
        if not header:
            header = f"column_{idx + 1}"
        seen[header] = seen.get(header, 0) + 1
        if seen[header] > 1:
            header = f"{header}_{seen[header]}"
        headers.append(header)
    return headers


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def tokenize(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(term) > 1 and term not in STOPWORDS]


def expand_query_terms(terms: list[str]) -> list[str]:
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        expanded.extend(QUERY_EXPANSIONS.get(term, ()))
    return list(dict.fromkeys(expanded))


def extract_query_phrases(query: str) -> list[str]:
    cleaned = clean_text(query).lower()
    phrases = re.findall(r'"([^"]+)"', cleaned)
    terms = tokenize(cleaned)
    phrases.extend(" ".join(terms[idx : idx + 2]) for idx in range(len(terms) - 1))
    phrases.extend(" ".join(terms[idx : idx + 3]) for idx in range(len(terms) - 2))
    return [phrase for phrase in phrases if len(phrase) > 4]


def extract_codes(query: str) -> list[str]:
    return [code.upper() for code in re.findall(r"\b[A-Z]{1,3}\d{2,4}[A-Z]?\b", query.upper())]


def infer_query_intents(query: str, mode: str) -> set[str]:
    text = f"{mode} {query}".lower()
    intents: set[str] = set()
    if any(term in text for term in ("hospital", "room", "facility", "ward", "length of stay", "stay")):
        intents.add("hospital_fee")
    if any(term in text for term in ("surgeon", "anaesthetist", "doctor", "attendance", "consult")):
        intents.add("doctor_fee")
    if any(term in text for term in ("diagnosis", "condition", "medical", "asthma", "bronchitis", "tonsillitis")):
        intents.add("medical_condition")
    if any(term in text for term in ("surgery", "procedure", "operation", "tosp", "colonoscopy", "endoscopy")):
        intents.add("surgical")
    if any(term in text for term in ("inpatient", "icu", "hdu", "general ward")):
        intents.add("inpatient")
    if "procedure cost estimate" in text:
        intents.update({"hospital_fee", "doctor_fee", "surgical"})
    if not intents:
        intents.update({"hospital_fee", "doctor_fee", "medical_condition", "surgical"})
    return intents
