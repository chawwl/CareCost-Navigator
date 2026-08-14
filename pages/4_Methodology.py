import streamlit as st

from ui_components import APP_TITLE, require_access


st.set_page_config(page_title=f"Methodology · {APP_TITLE}", page_icon="🧭", layout="wide")
require_access()

st.title("Methodology")
st.write(
    "CareCost Navigator uses an explicit stateful agent loop. It intentionally separates symptom and condition navigation "
    "from retrieval of known procedure and hospital-stay cost information."
)

st.header("Agentic data flow")
st.markdown(
    """
1. **Bound and screen input:** retain a short conversation window, cap input size, and detect prompt-injection and emergency patterns.
2. **Plan:** an LLM planner proposes a constrained JSON tool plan; the application validates the mode, allowlist, and required safety tools.
3. **Retrieve:** Care Pathway Guide uses MeSH terminology expansion and curated official sources. Fee Benchmark Explorer retrieves separately from the MOH fee-benchmark and hospital bill-size workbooks.
4. **Compose:** the model receives bounded context and retrieved evidence. It may give careful general education for a named condition, but it must not diagnose, prescribe, or make individualised treatment claims.
5. **Evaluate:** a separate pass checks emergency escalation, diagnosis boundaries, unsupported cost claims, benchmark caveats, and sources.
6. **Revise:** at most one revision is permitted. The interface reports composer and revision completion without exposing their raw internal outputs.
"""
)

st.header("Use case 1 flowchart · Care Pathway Guide")
st.graphviz_chart(
    """
digraph CarePathway {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#E8F2FF"];
  input [label="Symptom or known condition"];
  guard [label="Safety + input bounds"];
  plan [label="Constrained planner"];
  mesh [label="MeSH terminology expansion"];
  source [label="Official-source lookup"];
  missing [label="Missing-information check"];
  compose [label="General, non-diagnostic guidance"];
  eval [label="Quality evaluator"];
  revise [label="Optional one-time revision"];
  output [label="Guidance + sources + trace"];
  input -> guard -> plan;
  plan -> mesh -> compose;
  plan -> source -> compose;
  plan -> missing -> compose;
  compose -> eval;
  eval -> output [label="pass"];
  eval -> revise [label="fail"];
  revise -> output;
}
"""
)

st.header("Use case 2 flowchart · Fee Benchmark Explorer")
st.graphviz_chart(
    """
digraph FeeExplorer {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#ECF8EF"];
  form [label="Known procedure, diagnosis, or TOSP"];
  guard [label="Safety + known-term check"];
  plan [label="Constrained planner"];
  fee [label="Fee benchmark RAG"];
  stay [label="Hospital bill-size RAG"];
  filters [label="Hospital + ward filter"];
  source [label="MOH source lookup"];
  compose [label="Grounded cost explanation"];
  eval [label="Quality evaluator"];
  present [label="Text + fee/stay tables + chart"];
  form -> guard -> plan;
  plan -> fee -> compose;
  plan -> stay -> filters -> compose;
  plan -> source -> compose;
  compose -> eval -> present;
}
"""
)

st.header("Retrieval and presentation")
st.write(
    "MeSH is used in Care Pathway Guide to broaden terminology retrieval without diagnosing the user. Fee Benchmark Explorer does not use MeSH: it expects a known procedure, diagnosis, or TOSP code and directs symptom-only descriptions to Care Pathway Guide. "
    "Both cost workbooks are normalised into bounded row documents and indexed independently. BM25 is always available; an in-memory embedding index can be enabled with an OpenAI-compatible provider. Procedure/diagnosis anchors, whole-token matching, reciprocal-rank fusion, field boosts, and MMR-style diversification keep results relevant."
)

st.header("Official care-navigation sources")
st.markdown(
    """
For Care Pathway Guide requests, the curated allowlist includes MOH's:

- [Conditions](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/conditions/) directory for selected common conditions;
- [Visiting a pharmacist](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/visiting-a-pharmacist/) guidance for medication questions and common-condition support;
- [Seeking a doctor](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/seeking-a-doctor/) guidance on GP-first care and referral; and
- [When to visit the hospital for emergencies](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/visiting-the-hospital-for-emergencies/) guidance for emergency warning signs.

The source lookup boosts the page matching the user's stated need. These sources support general education and care navigation; they do not turn the app into a diagnostic service.

The curated registry also includes NUHS, SingHealth, Mount Elizabeth, and Gleneagles condition directories as **Supplementary provider education**. The app labels them separately from MOH and SCDF public guidance and does not represent a directory as condition-specific evidence unless the supplied source summary supports that claim.
"""
)

st.header("Cost-data boundaries")
st.markdown(
    """
- `feebenchmarks.xlsx` supplies MOH fee-benchmark evidence, including relevant hospital and professional-fee components.
- `hospitalbillsizes.xlsx` supplies separate actual-bill percentile rows by procedure/condition, setting, ward type, and—where available—hospital and average length of stay (ALOS).
- The hospital selector shows readable names with their workbook abbreviations. It covers the public and private institutions represented in the workbook; the abbreviation is used internally for exact filtering.
- P25, P50, and P75 figures are descriptive source-workbook percentiles. Neither data source is a quotation, insurance decision, or prediction of an individual bill.
"""
)

st.header("Prompt-injection and misuse safeguards")
st.markdown(
    """
- User messages, conversation history, retrieved rows, and drafts are marked as untrusted data in prompts.
- Planner output is parsed as JSON; unknown tools are discarded and required safety tools are restored.
- The agent has no shell, filesystem-write, credential, general-web, or arbitrary-URL tool.
- Inputs and retained history are bounded; likely override, prompt-disclosure, credential, delimiter, and jailbreak phrases are flagged.
- Cost claims must come from the retrieved fee-benchmark or hospital bill-size rows; official guidance links come from a local allowlist.
- Symptom-derived cost information carries a deterministic notice that it is not medical diagnosis, assessment, or advice.
- API keys are never inserted into prompts or written to repository files.
"""
)
st.caption("Heuristic screening and LLM evaluation reduce risk but cannot guarantee that every adversarial input or hallucination is prevented.")
