"""Compatibility helpers for medical terminology retrieval.

Hard-coded symptom/diagnosis/procedure mappings have been removed.
Use the local MeSH XML RAG as the terminology source.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from utils.mesh_rag import MeshMatch, build_mesh_index

DATA_FOLDER = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class QueryCandidate:
    stage: str
    term: str
    source: str
    confidence: float
    related_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryResolution:
    original_query: str
    retrieval_terms: tuple[str, ...]
    symptom_candidates: tuple[QueryCandidate, ...] = ()
    diagnosis_candidates: tuple[QueryCandidate, ...] = ()
    procedure_candidates: tuple[QueryCandidate, ...] = ()
    notes: tuple[str, ...] = ()


def resolve_medical_query(query: str, limit: int = 5, min_score: float = 0.45) -> QueryResolution:
    index = build_mesh_index(DATA_FOLDER)
    matches = index.search(query, limit=limit, min_score=min_score)
    candidates = tuple(
        QueryCandidate(
            stage="MeSH terminology",
            term=match.record.preferred_term,
            source="U.S. National Library of Medicine — Medical Subject Headings (MeSH)",
            confidence=match.score,
            related_terms=(match.matched_term,),
        )
        for match in matches
    )
    terms: list[str] = []
    for match in matches:
        terms.extend((match.record.preferred_term, match.matched_term))
    return QueryResolution(
        original_query=query,
        retrieval_terms=tuple(dict.fromkeys(term for term in terms if term)),
        diagnosis_candidates=candidates,
        notes=("Terminology expansion is sourced from the local MeSH XML RAG; no hard-coded medical mappings are used.",),
    )


def build_retrieval_expansion(query: str) -> str:
    resolution = resolve_medical_query(query)
    return " ".join((query, *resolution.retrieval_terms)).strip()


def infer_chat_title(query: str) -> str:
    resolution = resolve_medical_query(query, limit=1)
    if resolution.retrieval_terms:
        return resolution.retrieval_terms[0].title()
    cleaned = " ".join(query.lower().strip().split())
    return cleaned[:60].capitalize() if cleaned else "New chat"
