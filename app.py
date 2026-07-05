from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.benchmark_rag import (
    BenchmarkIndex,
    BenchmarkRecord,
    DEFAULT_EMBEDDING_MODEL,
    build_benchmark_index,
    estimate_amounts,
    first_matching_field,
    load_benchmark_records,
    search_benchmark_records,
)
from multi_agent_workflow import (
    AgentStep,
    MODEL_DEFAULTS,
    PROVIDER_ENV_KEYS,
    LLMClient,
    infer_workflow_mode,
    resolve_api_key,
    run_agent_workflow,
)


APP_TITLE = "CareCost Navigator"
DATA_PATH = Path("data/feebenchmarks.xlsx")
MOH_SOURCE_URL = "https://www.moh.gov.sg/managing-expenses/bills-and-fee-benchmarks/hospital-bills-and-fee-benchmarks/"

@st.cache_resource(show_spinner=False)
def load_benchmark_index(
    path: str,
    provider: str,
    api_key: str,
    base_url: str,
    embedding_model: str,
) -> BenchmarkIndex:
    records = load_benchmark_records(path)
    return build_benchmark_index(
        records,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
    )


def render_match_table(matches: list[tuple[BenchmarkRecord, float]]) -> None:
    if not matches:
        st.info("No benchmark rows matched this query yet.")
        return
    rows = []
    for record, score in matches:
        lower, upper = estimate_amounts(record)
        rows.append(
            {
                "score": score,
                "sheet": record.sheet,
                "row": record.row_number,
                "description": first_matching_field(record, ("description", "drg_description", "ccs", "ward_type", "note")),
                "lower_estimate": lower,
                "upper_estimate": upper,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def initialize_session_state() -> None:
    defaults = {
        "agent_steps": [],
        "last_matches": [],
        "latest_answer": "",
        "messages": [],
        "follow_up_questions": [],
        "inferred_mode": "Auto",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation() -> None:
    st.session_state.agent_steps = []
    st.session_state.last_matches = []
    st.session_state.latest_answer = ""
    st.session_state.messages = []
    st.session_state.follow_up_questions = []
    st.session_state.inferred_mode = "Auto"


def build_retrieval_query(messages: list[dict[str, str]], question: str) -> str:
    user_turns = [message["content"] for message in messages if message.get("role") == "user"]
    if not user_turns:
        return question
    return "\n".join([*user_turns[-3:], question])


def render_chat_messages(messages: list[dict[str, str]]) -> None:
    if not messages:
        st.info("Ask a question to start.")
        return
    for message in messages:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(message.get("content", ""))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    initialize_session_state()
    st.title(APP_TITLE)
    st.caption("A Streamlit prototype for condition-to-procedure guidance and Singapore MOH fee benchmark search.")

    with st.sidebar:
        st.header("Model")
        provider = st.selectbox("Provider", list(MODEL_DEFAULTS), index=0)
        model = st.text_input("Model", value=MODEL_DEFAULTS[provider])
        api_key_input = st.text_input(
            "API key",
            type="password",
            help=f"Kept only in Streamlit session memory; not written to disk. Leave blank to use {PROVIDER_ENV_KEYS[provider]} if it is set.",
        )
        api_key = resolve_api_key(provider, api_key_input)
        base_url = ""
        if provider == "OpenAI-compatible":
            base_url = st.text_input(
                "Base URL or chat completions endpoint",
                placeholder="https://api-public.ai.tech.gov.sg/platform/models",
                help="For GovTech AI Platform, use the same base_url you would pass to OpenAI(...).",
            )
        embedding_model = DEFAULT_EMBEDDING_MODEL
        if provider in {"OpenAI", "OpenAI-compatible"}:
            embedding_model = st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
        st.divider()
        st.header("Data")
        st.write(f"Workbook: `{DATA_PATH}`")
        st.link_button("MOH fee benchmarks source", MOH_SOURCE_URL)
        st.divider()
        if st.button("Clear chat"):
            clear_conversation()

    if not DATA_PATH.exists():
        st.error(f"Missing workbook: {DATA_PATH}")
        return

    benchmark_index = load_benchmark_index(str(DATA_PATH), provider, api_key, base_url, embedding_model)
    st.success(
        f"Loaded {len(benchmark_index.records):,} searchable rows/notes "
        f"and {len(benchmark_index.documents):,} RAG chunks from `{DATA_PATH}`."
    )
    st.caption(f"Retriever: {benchmark_index.retrieval_backend}")
    if benchmark_index.retrieval_note:
        st.warning(benchmark_index.retrieval_note)

    question = st.chat_input("Describe symptoms, a diagnosis, procedure, ward type, or benchmark question...")

    if question:
        previous_messages = list(st.session_state.messages)
        retrieval_query = build_retrieval_query(previous_messages, question)
        mode = infer_workflow_mode(question, previous_messages)
        matches = search_benchmark_records(benchmark_index, retrieval_query, mode)
        st.session_state.last_matches = matches
        client = LLMClient(provider=provider, api_key=api_key, model=model, base_url=base_url)
        with st.spinner("Running the routed agent workflow..."):
            try:
                result = run_agent_workflow(client, mode, question, matches, previous_messages)
                answer = result.answer
                steps = result.steps
                st.session_state.follow_up_questions = result.follow_up_questions
                st.session_state.inferred_mode = result.inferred_mode
            except Exception as exc:
                answer = f"Model call failed: {exc}"
                steps = [AgentStep("System", answer)]
                st.session_state.follow_up_questions = []
                st.session_state.inferred_mode = mode
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.latest_answer = answer
        st.session_state.agent_steps = steps

    st.subheader("Chat")
    render_chat_messages(st.session_state.messages)

    with st.expander("Agent Trace", expanded=False):
        if not st.session_state.agent_steps:
            st.write("Ask a question to see the sequential agent workflow.")
        else:
            st.caption(f"Auto-routed workflow: {st.session_state.inferred_mode}")
        for step in st.session_state.agent_steps:
            st.markdown(f"**{step.name}**")
            st.markdown(step.output)

    with st.expander("Matched Benchmark Rows", expanded=False):
        render_match_table(st.session_state.last_matches)

    with st.expander("Safety and scope"):
        st.write(
            "This prototype is educational. It does not diagnose, prescribe treatment, or guarantee costs. "
            "Clinical decisions should be discussed with a licensed medical professional."
        )


if __name__ == "__main__":
    main()
