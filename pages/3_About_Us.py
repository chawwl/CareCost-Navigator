import streamlit as st

from agentic_workflow import load_official_sources
from ui_components import APP_TITLE, require_access


st.set_page_config(page_title=f"About Us · {APP_TITLE}", page_icon="ℹ️", layout="wide")
require_access()

st.title("About Us")
st.write(
    "CareCost Navigator is an individual AI Bootcamp capstone prototype for citizens seeking clearer, scenario-specific "
    "information about healthcare conversations and private-sector fee benchmarks in Singapore."
)

st.header("Scope and objective")
st.write(
    "The project consolidates selected official guidance and the MOH fee benchmark workbook into one educational "
    "interface. It helps users prepare better questions for a clinician or provider; it does not choose treatment, "
    "determine eligibility or coverage, or predict a final bill."
)

st.header("Two use cases")
col1, col2 = st.columns(2)
with col1:
    st.subheader("1 · Care Pathway Guide")
    st.write(
        "Conversational, non-diagnostic explanations of possible care or procedure categories, with official emergency "
        "guidance, missing-information questions, and support for follow-up turns."
    )
with col2:
    st.subheader("2 · Fee Benchmark Explorer")
    st.write(
        "Guided search over MOH benchmark records, with a grounded narrative, traceable workbook rows, a data table, "
        "and a lower/upper range chart."
    )

st.header("Official and trustworthy sources")
st.write("Source records are curated locally so the model can only cite an explicit allowlist during a workflow run.")
for source in load_official_sources():
    st.markdown(f"### [{source.title}]({source.url})")
    st.write(f"**Publisher:** {source.agency}")
    st.write(source.summary)
    st.caption(f"Source record reviewed {source.last_reviewed}")

st.header("Features")
st.markdown(
    """
- constrained plan-and-execute agentic orchestration;
- hybrid BM25 and optional embedding retrieval over row-level MOH data;
- prompt chaining through planning, tool observations, answer composition, evaluation, and bounded revision;
- visible workflow trace, source links, evidence rows, and range visualisation;
- prompt-injection heuristics, tool allowlisting, bounded inputs/history, and evidence-grounding instructions;
- optional password protection through a deployment secret.
"""
)

st.header("Limitations and privacy")
st.write(
    "Retrieval similarity does not establish clinical relevance. Source summaries may become outdated and should be "
    "checked against their linked official pages. Users should not submit personally identifiable information. API keys "
    "entered in the sidebar remain in the Streamlit session and are not written by the app."
)
