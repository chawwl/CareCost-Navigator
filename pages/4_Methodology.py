import streamlit as st

from ui_components import APP_TITLE, require_access


st.set_page_config(page_title=f"Methodology · {APP_TITLE}", page_icon="🧭", layout="wide")
require_access()

st.title("Methodology")
st.write(
    "The implementation uses an explicit stateful agent loop. One orchestrator owns the "
    "workflow state, selects constrained tools, records observations, composes an answer, and sends it through a quality gate."
)

st.header("Agentic data flow")
st.markdown(
    """
1. **Bound and screen input:** retain a short conversation window, cap input size, and detect prompt-injection and emergency patterns.
2. **Plan:** an LLM planner proposes a mode and tools as JSON. A parser enforces the mode, tool allowlist, and required safety tools; invalid plans fall back to deterministic routing.
3. **Act:** tools retrieve curated official sources, search MOH workbook records, and identify missing scenario details.
4. **Compose:** the model receives only bounded conversation context and explicit tool observations, with healthcare and grounding constraints.
5. **Evaluate:** a separate prompt checks safety, unsupported claims, source use, and benchmark caveats.
6. **Revise:** if the evaluation fails, the composer gets one bounded revision. The UI exposes every stage in the workflow trace.
"""
)

st.header("Use case 1 flowchart · Care Pathway Guide")
st.graphviz_chart(
    """
digraph CarePathway {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#E8F2FF"];
  input [label="Generic user question"];
  guard [label="Input bounds + safety check"];
  plan [label="Constrained planner"];
  source [label="Official-source lookup"];
  missing [label="Missing-information tool"];
  compose [label="Non-diagnostic answer composer"];
  eval [label="Quality evaluator"];
  revise [label="Optional one-time revision"];
  output [label="Answer + sources + trace"];
  input -> guard -> plan;
  plan -> source;
  plan -> missing;
  source -> compose;
  missing -> compose;
  guard -> compose;
  compose -> eval;
  eval -> output [label="pass"];
  eval -> revise [label="fail"];
  revise -> output;
}
""",
    width="stretch",
)

st.header("Use case 2 flowchart · Fee Benchmark Explorer")
st.graphviz_chart(
    """
digraph FeeExplorer {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#ECF8EF"];
  form [label="Procedure + setting + fee scope"];
  guard [label="Input bounds + safety check"];
  plan [label="Constrained planner"];
  retrieve [label="BM25 + optional vector retrieval"];
  rerank [label="Threshold + field boosts + MMR"];
  source [label="MOH source lookup"];
  compose [label="Grounded benchmark explanation"];
  eval [label="Quality evaluator"];
  present [label="Text + table + range chart + trace"];
  form -> guard -> plan;
  plan -> retrieve -> rerank -> compose;
  plan -> source -> compose;
  compose -> eval -> present;
}
""",
    width="stretch",
)

st.header("Retrieval and presentation")
st.write(
    "Workbook rows are normalised into LangChain Documents and split into bounded chunks. BM25 is always available. "
    "When explicitly enabled with an OpenAI or compatible key, an in-memory embedding index is added. Multi-query "
    "expansion, fetch-more-than-k retrieval, reciprocal-rank and field-aware boosts, thresholds, and MMR-style "
    "diversification produce row-level evidence. Amount fields are then presented as both a table and a chart."
)

st.header("Prompt-injection and misuse safeguards")
st.markdown(
    """
- User messages, conversation history, retrieved rows, and drafts are marked as untrusted data in prompts.
- Planner output is parsed as JSON; unknown tools are discarded and required safety tools are restored.
- The agent has no general web, filesystem, code-execution, credential, or arbitrary-URL tool.
- Inputs and retained history are bounded; likely override, prompt-disclosure, credential, delimiter, and jailbreak phrases are flagged.
- Cost claims must come from retrieved workbook rows; official guidance links come from a local allowlist.
- A quality-evaluation prompt checks emergency escalation, diagnosis language, unsupported costs, benchmark caveats, and citations.
- API keys are never inserted into prompts or written to repository files.
"""
)
st.caption("Heuristic screening reduces risk but cannot guarantee that every adversarial input or hallucination is prevented.")
