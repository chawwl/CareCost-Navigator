import streamlit as st

from agentic_workflow import AgentStep, LLMClient, run_agent_workflow
from ui_components import (
    APP_TITLE,
    DATA_PATH,
    get_benchmark_index,
    render_agent_trace,
    render_match_chart,
    render_match_table,
    render_model_sidebar,
    render_safety_notice,
    render_sources,
    require_access,
)


PREFIX = "fee_explorer"

st.set_page_config(page_title=f"Fee Benchmark Explorer · {APP_TITLE}", page_icon="📊", layout="wide")
require_access()
settings = render_model_sidebar(PREFIX, allow_semantic_search=True)

st.title("Fee Benchmark Explorer")
st.write(
    "Search official MOH fee benchmark rows with a guided form, then inspect a grounded explanation, "
    "evidence table, and range visualisation."
)
st.info(
    "MOH fee benchmarks are reference ranges for routine and typical private-sector cases. They are not public-hospital "
    "benchmarks, insurance coverage decisions, or provider quotes."
)

if not DATA_PATH.exists():
    st.error(f"Missing workbook: {DATA_PATH}")
    st.stop()

with st.form("fee_search_form"):
    procedure = st.text_input("Procedure, diagnosis, or TOSP code *", placeholder="e.g. colonoscopy or exact TOSP code")
    left, right = st.columns(2)
    with left:
        setting = st.selectbox("Care setting", ["Not sure", "Day surgery / outpatient", "Inpatient", "ICU / HDU"])
    with right:
        fee_scope = st.selectbox("Fee component", ["Both hospital and doctor fees", "Hospital fees", "Doctor / professional fees"])
    detail = st.text_area(
        "Optional context",
        placeholder="Add only generic details, such as the ward type or wording from a non-identifying estimate.",
        max_chars=1_000,
    )
    submitted = st.form_submit_button("Run agentic benchmark search", type="primary")

if submitted:
    if not procedure.strip():
        st.error("Enter a procedure, diagnosis, or TOSP code.")
    else:
        question = (
            f"Estimate the fee benchmark for {procedure.strip()}. Care setting: {setting}. "
            f"Fee component: {fee_scope}. Additional context: {detail.strip() or 'none'}."
        )
        benchmark_index = get_benchmark_index(settings)
        client = LLMClient(settings.provider, settings.api_key, settings.model, settings.base_url)
        with st.spinner("Planning, retrieving benchmark rows, and evaluating the answer..."):
            try:
                result = run_agent_workflow(
                    client,
                    "Procedure cost estimate",
                    question,
                    benchmark_index,
                    st.session_state.get(f"{PREFIX}_history", []),
                )
                st.session_state[f"{PREFIX}_answer"] = result.answer
                st.session_state[f"{PREFIX}_steps"] = result.steps
                st.session_state[f"{PREFIX}_matches"] = result.matches
                st.session_state[f"{PREFIX}_sources"] = result.sources
                st.session_state[f"{PREFIX}_mode"] = result.inferred_mode
                st.session_state[f"{PREFIX}_history"] = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": result.answer},
                ]
            except Exception as exc:
                error = f"The workflow could not complete this search: {exc}"
                st.session_state[f"{PREFIX}_answer"] = error
                st.session_state[f"{PREFIX}_steps"] = [AgentStep("System", error, kind="error", status="failed")]

if st.session_state.get(f"{PREFIX}_answer"):
    st.subheader("Grounded explanation")
    st.markdown(st.session_state[f"{PREFIX}_answer"])

    table_tab, chart_tab = st.tabs(["Evidence table", "Range chart"])
    with table_tab:
        render_match_table(st.session_state.get(f"{PREFIX}_matches", []))
    with chart_tab:
        render_match_chart(st.session_state.get(f"{PREFIX}_matches", []))

render_agent_trace(
    st.session_state.get(f"{PREFIX}_steps", []),
    st.session_state.get(f"{PREFIX}_mode", ""),
)
render_sources(st.session_state.get(f"{PREFIX}_sources", []))
render_safety_notice()
