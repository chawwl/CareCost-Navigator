from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


HEADER_HINTS = ("tosp", "description", "lower", "upper", "ward type", "drg", "ccs", "icd", "diagnosis")
FETCH_K_FACTOR = 5
MIN_RETRIEVAL_SCORE = 0.15
MMR_LAMBDA = 0.72
RRF_K = 60
ROW_CHUNK_SIZE = 900
ROW_CHUNK_OVERLAP = 90
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

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

MEDICAL_ALIASES = {
    "appendicitis": ("appendicectomy", "appendectomy"),
    "appendicectomy": ("appendicitis", "appendectomy"),
    "appendectomy": ("appendicitis", "appendicectomy"),
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
    "benchmarks",
    "bill",
    "bills",
    "charge",
    "charges",
    "cost",
    "costs",
    "estimate",
    "estimates",
    "fee",
    "fees",
    "hospital",
    "lower",
    "medical",
    "price",
    "prices",
    "procedure",
    "stay",
    "surgery",
    "surgical",
    "upper",
}

ANCHOR_CONTEXT_TERMS = GENERIC_QUERY_TERMS | {
    "additional",
    "both",
    "care",
    "component",
    "condition",
    "context",
    "day",
    "diagnosis",
    "doctor",
    "does",
    "exact",
    "in",
    "inpatient",
    "is",
    "looking",
    "me",
    "my",
    "none",
    "not",
    "of",
    "outpatient",
    "please",
    "professional",
    "range",
    "setting",
    "should",
    "sure",
    "to",
    "tosp",
    "which",
}

GUIDANCE_SHEET_PREFIX = "general principles"
MAPPING_SHEETS = {"ccs-icd mapping"}


@dataclass
class BenchmarkRecord:
    sheet: str
    row_number: int
    fields: dict[str, str]
    searchable_text: str
    tokens: tuple[str, ...]

    @property
    def record_id(self) -> str:
        return make_record_id(self.sheet, self.row_number)

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
    documents: list[Document]
    bm25_retriever: BM25Retriever
    vector_store: InMemoryVectorStore | None = None
    retrieval_backend: str = "LangChain BM25"
    retrieval_note: str = ""


@dataclass
class RetrievalCandidate:
    record: BenchmarkRecord
    score: float = 0.0
    best_chunk: str = ""
    ranks: list[int] | None = None

    def add(self, score: float, chunk: str, rank: int) -> None:
        self.score += score
        if not self.best_chunk or len(chunk) > len(self.best_chunk):
            self.best_chunk = chunk
        if self.ranks is None:
            self.ranks = []
        self.ranks.append(rank)


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


def build_benchmark_index(
    records: list[BenchmarkRecord],
    *,
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> BenchmarkIndex:
    document_frequency: Counter[str] = Counter()
    total_length = 0
    for record in records:
        document_frequency.update(set(record.tokens))
        total_length += len(record.tokens)

    documents = split_benchmark_documents(records)
    bm25_retriever = BM25Retriever.from_documents(
        documents,
        k=max(20, min(len(documents), 50)),
        preprocess_func=tokenize,
    )

    vector_store: InMemoryVectorStore | None = None
    retrieval_backend = "LangChain BM25"
    retrieval_note = ""
    embeddings = make_embedding_model(provider, api_key, base_url, embedding_model)
    if embeddings is not None:
        try:
            vector_store = InMemoryVectorStore(embeddings)
            vector_store.add_documents(documents, ids=[doc.metadata["chunk_id"] for doc in documents])
            retrieval_backend = f"LangChain BM25 + InMemoryVectorStore ({embedding_model})"
        except Exception as exc:
            retrieval_note = f"Semantic vector index was skipped: {exc}"

    return BenchmarkIndex(
        records=records,
        document_frequency=dict(document_frequency),
        average_length=total_length / max(len(records), 1),
        documents=documents,
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        retrieval_backend=retrieval_backend,
        retrieval_note=retrieval_note,
    )


def make_embedding_model(provider: str, api_key: str, base_url: str, embedding_model: str) -> Embeddings | None:
    if not api_key or provider not in {"OpenAI", "OpenAI-compatible"}:
        return None

    kwargs: dict[str, str] = {"model": embedding_model or DEFAULT_EMBEDDING_MODEL, "api_key": api_key}
    if provider == "OpenAI-compatible" and base_url:
        kwargs["base_url"] = normalize_openai_compatible_base_url(base_url)
    return OpenAIEmbeddings(**kwargs)


def split_benchmark_documents(records: list[BenchmarkRecord]) -> list[Document]:
    row_documents = [record_to_document(record) for record in records]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=ROW_CHUNK_SIZE,
        chunk_overlap=ROW_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "; ", " | ", " ", ""],
    )
    chunks = splitter.split_documents(row_documents)
    for chunk_number, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['record_id']}::chunk-{chunk_number}"
    return chunks


def record_to_document(record: BenchmarkRecord) -> Document:
    page_content = serialize_record(record)
    return Document(
        page_content=page_content,
        metadata={
            "record_id": record.record_id,
            "sheet": record.sheet,
            "row_number": record.row_number,
            "fields_json": json.dumps(record.fields, ensure_ascii=False),
        },
    )


def serialize_record(record: BenchmarkRecord) -> str:
    field_text = "\n".join(f"{key}: {value}" for key, value in record.fields.items() if value)
    return f"Sheet: {record.sheet}\nRow: {record.row_number}\n{field_text}"


def search_benchmark_records(
    index: BenchmarkIndex,
    query: str,
    mode: str,
    limit: int = 10,
    *,
    anchor: str | None = None,
) -> list[tuple[BenchmarkRecord, float]]:
    search_anchor = clean_text(anchor) if anchor is not None else extract_search_anchor(query)
    raw_anchor_terms = [term for term in tokenize(search_anchor) if term not in ANCHOR_CONTEXT_TERMS]
    anchor_terms = expand_query_terms(raw_anchor_terms)
    anchor_codes = extract_codes(search_anchor)
    anchor_phrases = extract_query_phrases(search_anchor)
    focused_anchor = search_anchor if anchor_terms or anchor_codes else None
    query_variants = dedupe_queries(build_multi_query_variants(query, mode, focused_anchor))
    record_lookup = {record.record_id: record for record in index.records}
    candidates: dict[str, RetrievalCandidate] = {}
    fetch_k = max(limit * FETCH_K_FACTOR, limit)
    intents = infer_query_intents(query, mode)
    restrict_to_benchmarks = bool(anchor_terms or anchor_codes) and mode in {"Procedure cost estimate", "Both"}

    for variant in query_variants:
        if not tokenize(variant):
            continue

        bm25_docs = index.bm25_retriever.invoke(variant)[:fetch_k]
        merge_ranked_documents(
            candidates,
            record_lookup,
            bm25_docs,
            anchor_terms,
            anchor_codes,
            restrict_to_benchmarks,
            source_weight=0.9,
        )

        if index.vector_store is None:
            continue

        vector_docs_with_scores = index.vector_store.similarity_search_with_score(variant, k=fetch_k)
        merge_scored_documents(
            candidates,
            record_lookup,
            vector_docs_with_scores,
            anchor_terms,
            anchor_codes,
            restrict_to_benchmarks,
            source_weight=1.25,
        )

    for candidate in candidates.values():
        candidate.score = retrieval_score(
            candidate.record,
            "",
            anchor_terms,
            anchor_terms,
            anchor_phrases,
            anchor_codes,
            intents,
            candidate.score,
        )

    ranked = sorted(candidates.values(), key=lambda candidate: candidate.score, reverse=True)
    filtered = [(candidate.record, candidate.score) for candidate in ranked if candidate.score >= MIN_RETRIEVAL_SCORE]
    if not filtered and ranked:
        filtered = [(candidate.record, candidate.score) for candidate in ranked[:limit]]
    return mmr_select(filtered[: max(limit * FETCH_K_FACTOR, limit)], limit)


def merge_ranked_documents(
    candidates: dict[str, RetrievalCandidate],
    record_lookup: dict[str, BenchmarkRecord],
    documents: list[Document],
    anchor_terms: list[str],
    anchor_codes: list[str],
    restrict_to_benchmarks: bool,
    *,
    source_weight: float,
) -> None:
    seen_records: set[str] = set()
    for rank, document in enumerate(documents):
        record = record_lookup.get(str(document.metadata.get("record_id", "")))
        if record is None or record.record_id in seen_records:
            continue
        seen_records.add(record.record_id)
        if restrict_to_benchmarks and not is_benchmark_evidence_record(record):
            continue
        if (anchor_terms or anchor_codes) and not has_specific_match(
            record.searchable_text, anchor_terms, anchor_codes
        ):
            continue
        get_candidate(candidates, record).add(
            source_weight * reciprocal_rank(rank), document.page_content, rank
        )


def merge_scored_documents(
    candidates: dict[str, RetrievalCandidate],
    record_lookup: dict[str, BenchmarkRecord],
    documents_with_scores: list[tuple[Document, float]],
    anchor_terms: list[str],
    anchor_codes: list[str],
    restrict_to_benchmarks: bool,
    *,
    source_weight: float,
) -> None:
    seen_records: set[str] = set()
    for rank, (document, raw_score) in enumerate(documents_with_scores):
        record = record_lookup.get(str(document.metadata.get("record_id", "")))
        if record is None or record.record_id in seen_records:
            continue
        seen_records.add(record.record_id)
        if restrict_to_benchmarks and not is_benchmark_evidence_record(record):
            continue
        if (anchor_terms or anchor_codes) and not has_specific_match(
            record.searchable_text, anchor_terms, anchor_codes
        ):
            continue
        vector_score = normalize_vector_score(raw_score)
        score = source_weight * vector_score + (0.35 * reciprocal_rank(rank))
        if score <= 0:
            continue
        get_candidate(candidates, record).add(score, document.page_content, rank)


def retrieval_score(
    record: BenchmarkRecord,
    chunk_text: str,
    query_terms: list[str],
    specific_terms: list[str],
    query_phrases: list[str],
    query_codes: list[str],
    intents: set[str],
    base_score: float,
) -> float:
    searchable_text = f"{record.searchable_text} {chunk_text.lower()}"
    if specific_terms and not has_specific_match(searchable_text, specific_terms, query_codes):
        return 0.0

    phrase = phrase_score(searchable_text, query_phrases)
    code = code_score(searchable_text, query_codes)
    fuzzy = fuzzy_score(searchable_text, query_terms)
    sheet = sheet_intent_boost(record.sheet, intents)
    field = field_boost(record, query_terms)
    amount = amount_boost(record, intents)
    specificity = specific_term_score(searchable_text, specific_terms)
    return base_score + (phrase * 0.8) + (code * 4.0) + (fuzzy * 0.5) + sheet + field + amount + specificity


def get_candidate(candidates: dict[str, RetrievalCandidate], record: BenchmarkRecord) -> RetrievalCandidate:
    if record.record_id not in candidates:
        candidates[record.record_id] = RetrievalCandidate(record=record)
    return candidates[record.record_id]


def normalize_vector_score(score: float) -> float:
    if math.isnan(score):
        return 0.0
    if 0 <= score <= 1:
        return score
    if score < 0:
        return 0.0
    return 1 / (1 + score)


def reciprocal_rank(rank: int) -> float:
    return 1 / (RRF_K + rank + 1)


def has_specific_match(text: str, specific_terms: list[str], query_codes: list[str]) -> bool:
    text_tokens = set(tokenize(text))
    if query_codes:
        return any(code.lower() in text_tokens for code in query_codes)
    return any(term in text_tokens for term in specific_terms)


def specific_term_score(text: str, specific_terms: list[str]) -> float:
    if not specific_terms:
        return 0.0
    text_tokens = set(tokenize(text))
    matched = sum(1 for term in specific_terms if term in text_tokens)
    return matched / len(specific_terms)


def phrase_score(text: str, phrases: list[str]) -> float:
    if not phrases:
        return 0.0
    normalized_text = f" {' '.join(tokenize(text))} "
    return sum(
        1.0
        for phrase in phrases
        if phrase and f" {' '.join(tokenize(phrase))} " in normalized_text
    )


def code_score(text: str, codes: list[str]) -> float:
    if not codes:
        return 0.0
    text_tokens = set(tokenize(text))
    return sum(1.0 for code in codes if code.lower() in text_tokens)


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
        field_tokens = set(tokenize(value))
        boost += sum(0.35 for term in query_terms if term in field_tokens)
    return min(boost, 3.0)


def amount_boost(record: BenchmarkRecord, intents: set[str]) -> float:
    if not ({"hospital_fee", "doctor_fee", "inpatient"} & intents):
        return 0.0
    lower, upper = estimate_amounts(record)
    return 2.0 if lower is not None and upper is not None else 0.0


def mmr_select(candidates: list[tuple[BenchmarkRecord, float]], limit: int) -> list[tuple[BenchmarkRecord, float]]:
    selected: list[tuple[BenchmarkRecord, float]] = []
    remaining = sorted(candidates, key=lambda item: item[1], reverse=True)
    while remaining and len(selected) < limit:
        if not selected:
            record, relevance = remaining.pop(0)
            selected.append((record, round(relevance, 3)))
            continue
        best_index = 0
        best_score = float("-inf")
        for idx, (record, relevance) in enumerate(remaining):
            diversity_penalty = max(token_jaccard(record, chosen) for chosen, _ in selected)
            mmr_score = (MMR_LAMBDA * relevance) - ((1 - MMR_LAMBDA) * diversity_penalty)
            if mmr_score > best_score:
                best_index = idx
                best_score = mmr_score
        record, relevance = remaining.pop(best_index)
        selected.append((record, round(relevance, 3)))
    return selected


def token_jaccard(left: BenchmarkRecord, right: BenchmarkRecord) -> float:
    left_tokens = set(left.tokens)
    right_tokens = set(right.tokens)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def build_multi_query_variants(query: str, mode: str, anchor: str | None = None) -> list[str]:
    if anchor:
        anchor_terms = [term for term in tokenize(anchor) if term not in ANCHOR_CONTEXT_TERMS]
        expanded_anchor = " ".join(expand_query_terms(anchor_terms))
        variants = [anchor, expanded_anchor, *extract_codes(anchor)]
        return dedupe_queries(variants)

    variants = [
        query,
        f"{query} {mode}",
    ]
    codes = extract_codes(query)
    if codes:
        variants.extend(codes)

    intent = infer_query_intents(query, mode)
    if "hospital_fee" in intent:
        variants.append(f"{query} hospital fee lower upper average length of stay")
    if "doctor_fee" in intent:
        variants.append(f"{query} surgeon anaesthetist doctor fee lower upper")
    if "medical_condition" in intent:
        variants.append(f"{query} DRG CCS ICD diagnosis medical condition")
    if "surgical" in intent:
        variants.append(f"{query} TOSP procedure description surgical")

    return dedupe_queries(variants)


def dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for variant in queries:
        normalized = clean_text(variant).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(variant)
    return deduped


def extract_search_anchor(query: str) -> str:
    cleaned = clean_text(query)
    form_match = re.search(
        r"\bestimate\s+the\s+fee\s+benchmark\s+for\s+(.+?)"
        r"(?:\.\s*(?:care\s+setting|fee\s+component|additional\s+context)\s*:|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if form_match:
        anchor_parts = [clean_text(form_match.group(1))]
        for label in ("Hospital", "Ward type"):
            match = re.search(
                rf"\b{re.escape(label)}\s*:\s*(.+?)(?:\.\s*(?:Ward type|Additional context)\s*:|$)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if match:
                value = clean_text(match.group(1))
                if value and not value.lower().startswith("any "):
                    anchor_parts.append(value)
        return " ".join(anchor_parts)
    return cleaned


def is_guidance_record(record: BenchmarkRecord) -> bool:
    return record.sheet.strip().lower().startswith(GUIDANCE_SHEET_PREFIX)


def is_mapping_record(record: BenchmarkRecord) -> bool:
    return record.sheet.strip().lower() in MAPPING_SHEETS


def is_benchmark_evidence_record(record: BenchmarkRecord) -> bool:
    return not is_guidance_record(record) and not is_mapping_record(record)


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
        expanded.extend(MEDICAL_ALIASES.get(term, ()))
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


def make_record_id(sheet_name: str, row_number: int) -> str:
    return f"{sheet_name}::{row_number}"


def normalize_openai_compatible_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    return url
