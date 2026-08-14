from pathlib import Path
import streamlit as st
from agentic_workflow import load_official_sources
from ui_components import APP_TITLE, DATA_PATH, require_access, workbook_summary

st.set_page_config(page_title=APP_TITLE, page_icon="🏥", layout="wide")
require_access()
st.title(APP_TITLE)
st.subheader("Understand possible care pathways and Singapore MOH fee benchmarks")
st.write(
    "A multi-page educational app that combines an agentic LLM workflow, local MeSH terminology retrieval, "
    "curated official guidance, and retrieval over the MOH fee benchmark workbook. Choose one of the two use cases below."
)
st.warning(
    "### **IMPORTANT NOTICE**\n\n"
    "This web application is a prototype developed for **educational purposes only**. "
    "The information provided here is **NOT intended for real-world usage** and should "
    "not be relied upon for making any decisions, especially those related to financial, "
    "legal, or healthcare matters.\n\n"
    "* **LLM Inaccuracy:** Please be aware that the LLM may generate inaccurate or incorrect information.\n"
    "* **User Responsibility:** You assume full responsibility for how you use any generated output.\n\n"
    "Always consult with **qualified professionals** for accurate and personalised advice.",
    icon="⚠️"
)
st.info("Use generic, non-identifying information only. This app is not medical advice and does not provide a bill quote.")
left, right = st.columns(2)
with left:
    st.markdown("### 1 · Care Pathway Guide")
    st.write(
        "Describe a symptom or known condition. The LLM combines safety screening, MeSH terminology expansion, "
        "and official guidance to provide a general, non-diagnostic overview and questions to discuss with a clinician."
    )
    st.page_link("pages/1_Care_Pathway_Guide.py", label="Open Care Pathway Guide", icon="🩺")
with right:
    st.markdown("### 2 · Fee Benchmark Explorer")
    st.write(
        "Enter a known procedure, diagnosis, or TOSP code. The workflow searches MOH fee-benchmark and hospital bill-size "
        "workbooks, including hospital and ward selections when provided."
    )
    st.page_link("pages/2_Fee_Benchmark_Explorer.py", label="Open Fee Benchmark Explorer", icon="📊")
st.divider()
st.subheader("What is under the hood")
if DATA_PATH.exists():
    record_count, sheet_count = workbook_summary(str(DATA_PATH))
    col1, col2, col3 = st.columns(3)
    col1.metric("Searchable benchmark records", f"{record_count:,}")
    col2.metric("Workbook sections", f"{sheet_count:,}")
    col3.metric("Curated web sources", len(load_official_sources()))
else:
    st.error(f"Missing workbook: {DATA_PATH}")
st.write(
    "The planner selects from a constrained tool allowlist. MeSH terminology retrieval uses the NLM service and returns "
    "only compact terminology matches; those terms can broaden downstream information retrieval without diagnosing the user. "
    "The workflow also performs safety screening, curated-source lookup, missing-information checks, answer evaluation, "
    "and at most one revision."
)
about, method = st.columns(2)
with about:
    st.page_link("pages/3_About_Us.py", label="About Us", icon="ℹ️")
with method:
    st.page_link("pages/4_Methodology.py", label="Methodology and flowcharts", icon="🧭")
st.caption(f"Workbook available: {'yes' if Path(DATA_PATH).exists() else 'no'}")
