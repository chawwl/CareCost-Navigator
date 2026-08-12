from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from agentic_workflow import (
    AgentStep,
    MODEL_DEFAULTS,
    PROVIDER_ENV_KEYS,
    OfficialSource,
    resolve_api_key,
)
from utils.benchmark_rag import (
    BenchmarkIndex,
    BenchmarkRecord,
    DEFAULT_EMBEDDING_MODEL,
    build_benchmark_index,
    estimate_amounts,
    first_matching_field,
    load_benchmark_records,
)


APP_TITLE = "CareCost Navigator"
DATA_PATH = Path(__file__).parent / "data" / "feebenchmarks.xlsx"
MOH_SOURCE_URL = "https://www.moh.gov.sg/managing-expenses/bills-and-fee-benchmarks/hospital-bills-and-fee-benchmarks/"


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    api_key: str
    base_url: str
    embedding_model: str
    semantic_search: bool


@st.cache_resource(show_spinner=False)
def load_benchmark_index(
    path: str,
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> BenchmarkIndex:
    records = load_benchmark_records(path)
    return build_benchmark_index(
        records,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
    )


@st.cache_data(show_spinner=False)
def workbook_summary(path: str) -> tuple[int, int]:
    records = load_benchmark_records(path)
    return len(records), len({record.sheet for record in records})


def require_access() -> None:
    """Apply optional password protection when APP_PASSWORD exists in Streamlit secrets."""
    try:
        configured_password = str(st.secrets.get("APP_PASSWORD", ""))
    except (FileNotFoundError, KeyError):
        configured_password = ""
    if not configured_password or st.session_state.get("authenticated"):
        return

    st.title(APP_TITLE)
    st.info("This deployment is password protected.")
    entered = st.text_input("Password", type="password")
    if st.button("Enter", type="primary"):
        if hmac.compare_digest(entered, configured_password):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


def render_model_sidebar(key_prefix: str, *, allow_semantic_search: bool = False) -> ModelSettings:
    with st.sidebar:
        st.header("Model settings")
        provider = st.selectbox("Provider", list(MODEL_DEFAULTS), key=f"{key_prefix}_provider")
        model = st.text_input("Model", value=MODEL_DEFAULTS[provider], key=f"{key_prefix}_model")
        api_key_input = st.text_input(
            "API key",
            type="password",
            key=f"{key_prefix}_api_key",
            help=(
                "Held in session memory only. Leave blank to use the deployment environment variable "
                f"{PROVIDER_ENV_KEYS[provider]}."
            ),
        )
        api_key = resolve_api_key(provider, api_key_input)
        base_url = ""
        if provider == "OpenAI-compatible":
            base_url = st.text_input(
                "Base URL or chat-completions endpoint",
                placeholder="https://api-public.ai.tech.gov.sg/platform/models",
                key=f"{key_prefix}_base_url",
            )
        semantic_search = False
        embedding_model = DEFAULT_EMBEDDING_MODEL
        if allow_semantic_search and provider in {"OpenAI", "OpenAI-compatible"}:
            semantic_search = st.toggle(
                "Use semantic retrieval",
                value=True,
                key=f"{key_prefix}_semantic",
                help="Uses the same API key for an in-memory embedding index. BM25 remains available when off.",
            )
            if semantic_search:
                embedding_model = st.text_input(
                    "Embedding model", value=DEFAULT_EMBEDDING_MODEL, key=f"{key_prefix}_embedding"
                )
        st.caption("No key? The official-source and BM25 retrieval tools still work in retrieval-only mode.")
    return ModelSettings(provider, model, api_key, base_url, embedding_model, semantic_search)


def get_benchmark_index(settings: ModelSettings) -> BenchmarkIndex:
    provider = settings.provider if settings.semantic_search else ""
    api_key = settings.api_key if settings.semantic_search else ""
    return load_benchmark_index(
        str(DATA_PATH), provider, api_key, settings.base_url, settings.embedding_model
    )


def initialize_chat_state(prefix: str) -> None:
    defaults = {
        f"{prefix}_messages": [],
        f"{prefix}_steps": [],
        f"{prefix}_matches": [],
        f"{prefix}_sources": [],
        f"{prefix}_safety_flags": [],
        f"{prefix}_mode": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_chat_state(prefix: str) -> None:
    for suffix in ("messages", "steps", "matches", "sources", "safety_flags", "mode"):
        st.session_state.pop(f"{prefix}_{suffix}", None)
    initialize_chat_state(prefix)


def render_chat_messages(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.info("Ask a question to start this use case.")
        return
    for message in messages:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(message.get("content", ""))


def render_agent_trace(steps: list[AgentStep], inferred_mode: str = "") -> None:
    with st.expander("Agent workflow trace", expanded=False):
        if not steps:
            st.write("Run the use case to see the planner, selected tools, answer composer, and quality evaluator.")
            return
        if inferred_mode:
            st.caption(f"Workflow route: {inferred_mode}")
        for index, step in enumerate(steps, start=1):
            st.markdown(f"**{index}. {step.name}** · `{step.kind}` · `{step.status}`")
            st.markdown(step.output)


def match_rows(matches: list[tuple[BenchmarkRecord, float]]) -> pd.DataFrame:
    rows = []
    for record, score in matches:
        lower, upper = estimate_amounts(record)
        rows.append(
            {
                "retrieval_score": score,
                "benchmark": first_matching_field(
                    record, ("description", "drg_description", "ccs", "ward_type", "note")
                ),
                "lower_sgd": lower,
                "upper_sgd": upper,
                "workbook_sheet": record.sheet,
                "source_row": record.row_number,
            }
        )
    return pd.DataFrame(rows)


def render_match_table(matches: list[tuple[BenchmarkRecord, float]]) -> None:
    if not matches:
        st.info("No benchmark rows matched this query.")
        return
    st.dataframe(match_rows(matches), hide_index=True, width="stretch")
    st.caption("Retrieval scores are internal ranking values, not percentages or clinical confidence scores.")


def render_match_chart(matches: list[tuple[BenchmarkRecord, float]]) -> None:
    frame = match_rows(matches)
    if frame.empty:
        return
    chart = frame.dropna(subset=["lower_sgd", "upper_sgd"]).head(8).copy()
    if chart.empty:
        st.caption("The matched rows do not expose a lower/upper amount pair that can be charted.")
        return
    chart["benchmark"] = chart["benchmark"].fillna(chart["workbook_sheet"]).str.slice(0, 70)
    st.bar_chart(chart.set_index("benchmark")[["lower_sgd", "upper_sgd"]])
    st.caption("Chart values are reference ranges from matched workbook rows, not predicted bills.")


def render_sources(sources: list[OfficialSource]) -> None:
    with st.expander("Official sources used", expanded=False):
        if not sources:
            st.write("No source was selected for this turn.")
            return
        for source in sources:
            st.markdown(f"**[{source.title}]({source.url})** — {source.agency}")
            st.write(source.summary)
            st.caption(f"Curated source record last reviewed: {source.last_reviewed}")


def render_safety_notice() -> None:
    with st.expander("Safety, privacy, and scope", expanded=False):
        st.write(
            "This educational prototype does not diagnose, prescribe treatment, determine insurance coverage, "
            "or guarantee costs. Do not enter names, identification numbers, contact details, or medical record numbers."
        )
        st.write(
            "For a life-threatening emergency in Singapore, call 995. Discuss clinical decisions and itemised fee "
            "estimates with a licensed clinician and the relevant provider."
        )
