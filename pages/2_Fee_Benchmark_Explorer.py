import streamlit as st

from agentic_workflow import AgentStep, LLMClient, run_agent_workflow
from ui_components import (
    APP_TITLE,
    DATA_PATH,
    HOSPITAL_BILLS_PATH,
    get_benchmark_index,
    get_hospital_bill_index,
    hospital_bill_filter_options,
    format_hospital_option,
    create_workflow_progress,
    render_agent_trace,
    render_match_chart,
    render_match_table,
    render_hospital_bill_table,
    render_model_sidebar,
    render_safety_notice,
    render_sources,
    require_access,
)


PREFIX = "fee_explorer"

st.set_page_config(page_title=f"Fee Benchmark Explorer · {APP_TITLE}", page_icon="📊", layout="wide")
require_access()
settings = render_model_sidebar(PREFIX, allow_semantic_search=True)

# Semantic embeddings are prepared once per cached index configuration, before
# the search form is available, rather than on every form submission.
semantic_ready = settings.semantic_search and bool(settings.api_key)
try:
    if semantic_ready:
        with st.spinner("Preparing semantic retrieval…"):
            benchmark_index = get_benchmark_index(settings)
            hospital_bill_index = get_hospital_bill_index(settings)
    else:
        benchmark_index = get_benchmark_index(settings)
        hospital_bill_index = get_hospital_bill_index(settings)
except Exception as exc:
    st.error(f"Semantic retrieval could not be prepared: {exc}")
    st.stop()

st.title("Fee Benchmark Explorer")
st.write(
    "Search official MOH fee benchmark rows with a guided form, then inspect a grounded explanation, "
    "evidence table, and range visualisation."
)
st.info(
    "MOH fee benchmarks are reference ranges for routine and typical private-sector cases. They are not public-hospital "
    "benchmarks, insurance coverage decisions, or provider quotes."
)
st.caption(
    "**Hospital fees** are the facility or provider component of a bill. **Professional fees** are the separate charges "
    "for clinical services by healthcare professionals, such as the surgeon, doctor, or anaesthetist. Ask the provider "
    "for an itemised estimate because the components applicable to a procedure can vary."
)

hospital_options, ward_options = hospital_bill_filter_options(str(HOSPITAL_BILLS_PATH))

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
    stay_left, stay_right = st.columns(2)
    with stay_left:
        hospital_label = st.selectbox(
            "Hospital for bill-size data",
            ["Any hospital", *(format_hospital_option(option) for option in hospital_options)],
        )
    with stay_right:
        ward = st.selectbox("Ward type for bill-size data", ["Any ward", *ward_options])
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
        from agentic_workflow import is_symptom_based_input
        if is_symptom_based_input(procedure):
            st.warning("For symptom descriptions, use the Care Pathway Guide first. Fee Benchmark Explorer is for a known diagnosis, procedure, or TOSP code.")
            st.stop()
        hospital = hospital_label.rsplit("(", 1)[-1].rstrip(")") if hospital_label != "Any hospital" else hospital_label
        question = (
            f"Estimate the fee benchmark for {procedure.strip()}. Care setting: {setting}. "
            f"Fee component: {fee_scope}. Hospital: {hospital}. Ward type: {ward}. Additional context: {detail.strip() or 'none'}."
        )
        client = LLMClient(settings.provider, settings.api_key, settings.model, settings.base_url)
        with st.spinner("Planning, retrieving benchmark rows, and evaluating the answer..."):
            try:
                progress_update = create_workflow_progress()
                progress_update("semantic_index", "completed" if semantic_ready else "skipped")
                result = run_agent_workflow(
                    client,
                    "Procedure cost estimate",
                    question,
                    benchmark_index,
                    progress_callback=progress_update,
                    use_mesh=False,
                    hospital_bill_index=hospital_bill_index,
                    hospital_filter=hospital,
                    ward_filter=ward,
                )
                st.session_state[f"{PREFIX}_answer"] = result.answer
                st.session_state[f"{PREFIX}_steps"] = result.steps
                st.session_state[f"{PREFIX}_matches"] = result.matches
                st.session_state[f"{PREFIX}_hospital_bill_matches"] = result.hospital_bill_matches
                st.session_state[f"{PREFIX}_sources"] = result.sources
                st.session_state[f"{PREFIX}_mode"] = result.inferred_mode
            except Exception as exc:
                error = f"The workflow could not complete this search: {exc}"
                st.session_state[f"{PREFIX}_answer"] = error
                st.session_state[f"{PREFIX}_steps"] = [AgentStep("System", error, kind="error", status="failed")]

if st.session_state.get(f"{PREFIX}_answer"):
    st.subheader("Grounded explanation")
    st.markdown(st.session_state[f"{PREFIX}_answer"])

    table_tab, stay_tab, chart_tab = st.tabs(["Fee benchmark rows", "Hospital stay bill-size rows", "Range chart"])
    with table_tab:
        render_match_table(st.session_state.get(f"{PREFIX}_matches", []))
    with stay_tab:
        render_hospital_bill_table(st.session_state.get(f"{PREFIX}_hospital_bill_matches", []))
    with chart_tab:
        render_match_chart(st.session_state.get(f"{PREFIX}_matches", []))

render_agent_trace(
    st.session_state.get(f"{PREFIX}_steps", []),
    st.session_state.get(f"{PREFIX}_mode", ""),
)
render_sources(st.session_state.get(f"{PREFIX}_sources", []))
render_safety_notice()
