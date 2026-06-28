from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from benchmark_rag import (
    BenchmarkIndex,
    BenchmarkRecord,
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
    resolve_api_key,
    run_agent_workflow,
)


APP_TITLE = "CareCost Navigator"
DATA_PATH = Path("data/feebenchmarks.xlsx")
MOH_SOURCE_URL = "https://www.moh.gov.sg/managing-expenses/bills-and-fee-benchmarks/hospital-bills-and-fee-benchmarks/"

@st.cache_data(show_spinner=False)
def load_benchmark_index(path: str) -> BenchmarkIndex:
    records = load_benchmark_records(path)
    return build_benchmark_index(records)


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


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
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
        st.divider()
        st.header("Data")
        st.write(f"Workbook: `{DATA_PATH}`")
        st.link_button("MOH fee benchmarks source", MOH_SOURCE_URL)
        st.caption("CrewAI note: current CrewAI docs require Python >=3.10 and <3.14, so deploy with Python 3.12.")

    if not DATA_PATH.exists():
        st.error(f"Missing workbook: {DATA_PATH}")
        return

    benchmark_index = load_benchmark_index(str(DATA_PATH))
    st.success(f"Loaded {len(benchmark_index.records):,} searchable rows/notes from `{DATA_PATH}`.")

    mode = st.radio(
        "Workflow",
        ["Condition to procedures", "Procedure cost estimate", "Both"],
        horizontal=True,
    )
    question = st.chat_input("Describe symptoms, a diagnosis, procedure, ward type, or benchmark question...")

    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []
    if "last_matches" not in st.session_state:
        st.session_state.last_matches = []
    if "latest_answer" not in st.session_state:
        st.session_state.latest_answer = ""

    if question:
        matches = search_benchmark_records(benchmark_index, question, mode)
        st.session_state.last_matches = matches
        client = LLMClient(provider=provider, api_key=api_key, model=model, base_url=base_url)
        with st.spinner("Running orchestrator, specialist, benchmark analyst, and evaluator..."):
            try:
                answer, steps = run_agent_workflow(client, mode, question, matches)
            except Exception as exc:
                answer = f"Model call failed: {exc}"
                steps = [AgentStep("System", answer)]
        st.session_state.latest_answer = answer
        st.session_state.agent_steps = steps

    with st.expander("Agent Trace", expanded=False):
        if not st.session_state.agent_steps:
            st.write("Ask a question to see the sequential agent workflow.")
        for step in st.session_state.agent_steps:
            st.markdown(f"**{step.name}**")
            st.markdown(step.output)

    st.subheader("Matched Benchmark Rows")
    render_match_table(st.session_state.last_matches)

    st.subheader("LLM Response")
    if st.session_state.latest_answer:
        st.markdown(st.session_state.latest_answer)
    else:
        st.info("Ask a question to generate a response.")

    with st.expander("Safety and scope"):
        st.write(
            "This prototype is educational. It does not diagnose, prescribe treatment, or guarantee costs. "
            "Clinical decisions should be discussed with a licensed medical professional."
        )


if __name__ == "__main__":
    main()
