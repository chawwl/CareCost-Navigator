from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import requests


# NLM's documented MeSH descriptor lookup endpoint.
MESH_LOOKUP_URL = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
MESH_SOURCE_NAME = "U.S. National Library of Medicine — Medical Subject Headings (MeSH)"
MESH_SOURCE_URL = "https://www.nlm.nih.gov/mesh/"
DEFAULT_LIMIT = 5
DEFAULT_MIN_SCORE = 0.45
REMOTE_LIMIT = 10
REQUEST_TIMEOUT = 5
MAX_CANDIDATES = 12
MAX_LOOKUP_REQUESTS = 3

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "had", "has", "have", "having", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with", "would", "can", "could", "do", "does", "did",
    "how", "much", "what", "where", "when", "why", "which", "will", "you",
    "your", "after", "before", "during", "into", "than", "then", "there",
}

# These are form and pricing labels used by the application, not clinical
# terminology. Removing them prevents a generic MeSH result such as "Fees"
# from outranking a symptom or anatomical term supplied by the user.
RETRIEVAL_NOISE_TERMS = {
    "additional", "benchmark", "bill", "both", "care", "component", "context", "cost", "day",
    "doctor", "estimate", "fee", "fees", "hospital", "inpatient", "none", "outpatient", "price",
    "procedure", "professional", "setting", "surgery",
}

MORPHOLOGY = {
    "abdomen": ("abdominal",),
    "abdominal": ("abdomen",),
    "stomach": ("gastric",),
    "gastric": ("stomach",),
    "oesophagus": ("esophagus",),
    "esophagus": ("oesophagus",),
    "tumour": ("tumor",),
    "tumor": ("tumour",),
    "haemorrhage": ("hemorrhage",),
    "hemorrhage": ("haemorrhage",),
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", _normalise(text)))


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _tokens(text) if token not in STOPWORDS)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = _normalise(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _extract_mesh_id(uri: str) -> str:
    match = re.search(r"/mesh/([^/?#]+)$", uri)
    return match.group(1) if match else uri


@dataclass(frozen=True)
class MeshRecord:
    ui: str
    preferred_term: str
    terms: tuple[str, ...]
    tree_numbers: tuple[str, ...] = ()
    scope_note: str = ""

    @property
    def all_terms(self) -> tuple[str, ...]:
        return _unique((self.preferred_term, *self.terms))

    def as_dict(self) -> dict[str, object]:
        return {
            "ui": self.ui,
            "preferred_term": self.preferred_term,
            "terms": list(self.terms),
            "tree_numbers": list(self.tree_numbers),
            "scope_note": self.scope_note,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "MeshRecord":
        return cls(
            ui=str(value.get("ui", "")),
            preferred_term=str(value.get("preferred_term", "")),
            terms=tuple(str(x) for x in value.get("terms", [])),
            tree_numbers=tuple(str(x) for x in value.get("tree_numbers", [])),
            scope_note=str(value.get("scope_note", "")),
        )


@dataclass(frozen=True)
class MeshMatch:
    record: MeshRecord
    matched_term: str
    score: float

    def as_context(self) -> dict[str, object]:
        return {
            "mesh_id": self.record.ui,
            "preferred_term": self.record.preferred_term,
            "matched_term": self.matched_term,
            "score": round(self.score, 4),
            "tree_numbers": list(self.record.tree_numbers),
            "scope_note": self.record.scope_note[:500],
            "source": MESH_SOURCE_NAME,
            "source_url": MESH_SOURCE_URL,
        }


class MeshIndex:
    """MeSH terminology retrieval using NLM's official RDF/SPARQL service.

    The local XML corpus is no longer parsed at startup. User text is first
    normalised into a small set of candidate phrases, those candidates are
    retrieved in a single batched SPARQL request, and ranking is performed
    locally. Only the small ranked result is passed to the agent workflow.
    """

    def __init__(self, records: Iterable[MeshRecord] = (), source_files: Iterable[str] = ()) -> None:
        self.records = tuple(records)
        self.source_files = tuple(source_files)

    @classmethod
    def from_data_folder(
        cls,
        data_folder: Path,
        cache_path: Path | None = None,
        rebuild: bool = False,
    ) -> "MeshIndex":
        # Kept for compatibility with the existing application API. NLM is now
        # the source of truth, so the downloaded XML files are not parsed.
        return cls()

    def search(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[MeshMatch]:
        candidates = generate_query_candidates(query)
        if not candidates:
            return []

        records = _fetch_mesh_candidates_batch(candidates)
        # The original sentence can contain fee-form boilerplate and multiple
        # ideas. Rank each compact clinical candidate as well, so an exact
        # match such as "stomach" is not discarded merely because it appears
        # in a long request about costs and care settings.
        return _rank_records_for_candidates(candidates, records, limit=limit, min_score=min_score)

    def build_query_context(self, query: str, limit: int = DEFAULT_LIMIT) -> str:
        matches = self.search(query, limit=limit)
        if not matches:
            return (
                f"No sufficiently relevant MeSH terminology matches were found "
                f"(threshold: {DEFAULT_MIN_SCORE:.2f}).\n"
                f"Source: {MESH_SOURCE_NAME}\nSource URL: {MESH_SOURCE_URL}"
            )

        lines = [
            "MeSH terminology matches (official NLM MeSH RDF/SPARQL):",
            f"Source: {MESH_SOURCE_NAME}",
            f"Source URL: {MESH_SOURCE_URL}",
        ]
        for match in matches:
            line = (
                f"- {match.record.preferred_term} | matched: {match.matched_term} "
                f"| score: {match.score:.2f}"
            )
            if match.record.tree_numbers:
                line += f" | tree: {', '.join(match.record.tree_numbers[:3])}"
            lines.append(line)
        return "\n".join(lines)


def build_mesh_index(data_folder: Path, rebuild: bool = False) -> MeshIndex:
    return MeshIndex.from_data_folder(data_folder, rebuild=rebuild)


def generate_query_candidates(query: str, max_candidates: int = MAX_CANDIDATES) -> tuple[str, ...]:
    """Create a bounded set of useful phrase/token variants before the API call."""
    normalised = _normalise(query)
    tokens = [token for token in _content_tokens(normalised) if token not in RETRIEVAL_NOISE_TERMS]
    if not tokens:
        return ()

    candidates: list[str] = [normalised]

    # Prefer adjacent 3- and 2-token phrases. This preserves meaning better
    # than generating every permutation.
    for n in (3, 2):
        if len(tokens) >= n:
            candidates.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    candidates.extend(tokens)

    # Add a small set of deterministic morphology variants.
    variants: list[str] = []
    for token in tokens:
        variants.extend(MORPHOLOGY.get(token, ()))
    candidates.extend(variants)

    # For a two-word phrase, also test combinations with a morphology variant.
    for i, token in enumerate(tokens):
        for variant in MORPHOLOGY.get(token, ()):
            replaced = tokens.copy()
            replaced[i] = variant
            candidates.append(" ".join(replaced))

    return _unique(candidates)[:max_candidates]


@lru_cache(maxsize=256)
def _fetch_mesh_candidates_batch(candidates: tuple[str, ...]) -> tuple[MeshRecord, ...]:
    """Retrieve descriptor candidates from NLM's official MeSH Lookup API."""
    if not candidates:
        return ()

    records: dict[str, dict[str, object]] = {}
    # The endpoint is one-label-per-request. Start with the most meaningful
    # single-token candidates (e.g. "stomach", "endoscopy") and bound the
    # request count so an unavailable external service cannot stall a turn.
    lookup_candidates = sorted(
        (candidate for candidate in candidates if len(_content_tokens(candidate)) == 1),
        key=lambda candidate: (-len(candidate), candidate),
    )[:MAX_LOOKUP_REQUESTS]
    if not lookup_candidates:
        lookup_candidates = list(candidates[:MAX_LOOKUP_REQUESTS])

    for candidate in lookup_candidates:
        try:
            response = requests.get(
                MESH_LOOKUP_URL,
                params={"label": candidate, "match": "contains", "limit": REMOTE_LIMIT},
                headers={"Accept": "application/json", "User-Agent": "CareCost-Navigator/1.0"},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError, TypeError):
            continue

        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            record_uri = str(item.get("resource", "")).strip()
            preferred = str(item.get("label", "")).strip()
            if not record_uri or not preferred:
                continue
            records.setdefault(
                record_uri,
                {
                    "ui": _extract_mesh_id(record_uri),
                    "preferred_term": _normalise(preferred),
                    "terms": set(),
                    "tree_numbers": set(),
                },
            )

    return tuple(
        MeshRecord(
            ui=str(entry["ui"]),
            preferred_term=str(entry["preferred_term"]),
            terms=tuple(sorted(entry["terms"])),
            tree_numbers=tuple(sorted(entry["tree_numbers"])),
        )
        for entry in records.values()
    )


def _rank_records(
    query: str,
    records: Iterable[MeshRecord],
    *,
    limit: int,
    min_score: float,
) -> list[MeshMatch]:
    query_n = _normalise(query)
    query_tokens = set(_content_tokens(query_n))
    if not query_tokens:
        return []

    matches: list[MeshMatch] = []
    for record in records:
        best_score = 0.0
        best_term = record.preferred_term
        for term in record.all_terms:
            term_n = _normalise(term)
            term_tokens = set(_content_tokens(term_n))
            if not term_tokens:
                continue
            overlap = len(query_tokens & term_tokens) / max(1, len(query_tokens | term_tokens))
            sequence = SequenceMatcher(None, query_n, term_n).ratio()
            contains_bonus = 0.12 if term_n in query_n or query_n in term_n else 0.0
            exact_bonus = 0.35 if term_n == query_n else 0.0
            score = min(1.0, max(overlap, sequence * 0.85) + contains_bonus + exact_bonus)
            if len(query_tokens) >= 3 and len(query_tokens & term_tokens) < 2 and term_n != query_n:
                score *= 0.75
            if score > best_score:
                best_score = score
                best_term = term
        if best_score >= min_score:
            matches.append(MeshMatch(record, best_term, best_score))

    matches.sort(key=lambda item: (-item.score, item.record.preferred_term))
    return matches[:max(1, limit)]


def _rank_records_for_candidates(
    candidates: Iterable[str],
    records: Iterable[MeshRecord],
    *,
    limit: int,
    min_score: float,
) -> list[MeshMatch]:
    """Keep the strongest score for each record across bounded query candidates."""
    records = tuple(records)
    best_by_id: dict[str, MeshMatch] = {}
    for candidate in candidates:
        for match in _rank_records(candidate, records, limit=max(1, len(records)), min_score=min_score):
            current = best_by_id.get(match.record.ui)
            if current is None or match.score > current.score:
                best_by_id[match.record.ui] = match
    return sorted(best_by_id.values(), key=lambda item: (-item.score, item.record.preferred_term))[:max(1, limit)]


def clear_mesh_cache() -> None:
    """Clear the in-memory NLM candidate cache."""
    _fetch_mesh_candidates_batch.cache_clear()


def mesh_cache_info() -> dict[str, int]:
    info = _fetch_mesh_candidates_batch.cache_info()
    return {"hits": info.hits, "misses": info.misses, "maxsize": info.maxsize, "currsize": info.currsize}
