import streamlit as st
import streamlit.components.v1 as components

from agentic_workflow import HOSPITAL_STAY_QUERY_TERMS, COST_TERMS, AgentStep, LLMClient, contains_any, has_specific_anchor, run_agent_workflow
from ui_components import (
    APP_TITLE,
    clear_chat_state,
    create_workflow_progress,
    get_benchmark_index,
    get_hospital_bill_index,
    initialize_chat_state,
    render_agent_trace,
    render_chat_messages,
    render_model_sidebar,
    render_safety_notice,
    render_sources,
    require_access,
)


PREFIX = "care_pathway"

st.set_page_config(page_title=f"Care Pathway Guide · {APP_TITLE}", page_icon="🩺", layout="wide")
require_access()
initialize_chat_state(PREFIX)
settings = render_model_sidebar(PREFIX, allow_semantic_search=True)

# Build the cached retrieval indexes before the chat control is rendered.  On a
# semantic first-run this can take a little while because embeddings are made;
# keeping it outside the turn handler means later messages do not repeat that
# preparation.
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

with st.sidebar:
    st.divider()
    if st.button("Clear this conversation", key=f"{PREFIX}_clear"):
        clear_chat_state(PREFIX)
        st.rerun()

st.title("Care Pathway Guide")
st.write(
    "Get a non-diagnostic explanation of a symptom or known condition, including a general overview of common symptoms, "
    "care options a clinician may discuss, urgent warning signs, and questions to ask. Follow-up turns reuse recent conversation context."
)
st.warning("Do not enter personally identifiable or sensitive record details. Call 995 for a life-threatening emergency.")
st.markdown(
    """
    <style>
    /* Keep answer sections easy to scan without making them look like page titles. */
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4 {
        font-size: 1rem !important;
        font-weight: 700 !important;
        margin: 0.8rem 0 0.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

question = st.chat_input("Example: My doctor mentioned a colonoscopy. What should I ask before deciding?")
if question:
    previous_messages = list(st.session_state[f"{PREFIX}_messages"])
    client = LLMClient(settings.provider, settings.api_key, settings.model, settings.base_url)
    # A short follow-up such as "what about the hospital stay?" is itself a
    # cost request, even when it does not repeat the word "cost".
    wants_stay_cost = contains_any(question, HOSPITAL_STAY_QUERY_TERMS)
    wants_cost = contains_any(question, COST_TERMS) or wants_stay_cost
    workflow_mode = "Both" if wants_cost else "Condition to procedures"
    result = None
    with st.spinner("Planning, calling tools, and evaluating the answer..."):
        try:
            progress_update = create_workflow_progress()
            progress_update("semantic_index", "completed" if semantic_ready else "skipped")
            result = run_agent_workflow(
                client,
                workflow_mode,
                question,
                benchmark_index,
                conversation_history=previous_messages,
                progress_callback=progress_update,
                hospital_bill_index=hospital_bill_index if wants_stay_cost else None,
            )
            answer = result.answer
            st.session_state[f"{PREFIX}_steps"] = result.steps
            st.session_state[f"{PREFIX}_matches"] = result.matches
            st.session_state[f"{PREFIX}_sources"] = result.sources
            st.session_state[f"{PREFIX}_safety_flags"] = result.safety_flags
            st.session_state[f"{PREFIX}_mode"] = result.inferred_mode
        except Exception as exc:
            answer = f"The workflow could not complete this turn: {exc}"
            st.session_state[f"{PREFIX}_steps"] = [AgentStep("System", answer, kind="error", status="failed")]
    assistant_message: dict[str, object] = {"role": "assistant", "content": answer}
    chart_matches = result.matches if result is not None else []
    # A brief stay-cost follow-up inherits the prior procedure alternatives, so
    # the new P25/P75 stay component is added to the procedure total rather
    # than being displayed as a stand-alone stay-only estimate.
    if wants_stay_cost and not has_specific_anchor(question):
        prior_matches = next(
            (
                message["cost_matches"]
                for message in reversed(previous_messages)
                if message.get("cost_matches")
            ),
            [],
        )
        if prior_matches:
            chart_matches = prior_matches
    if wants_cost and chart_matches:
        assistant_message["cost_matches"] = chart_matches
        assistant_message["hospital_bill_matches"] = result.hospital_bill_matches
    st.session_state[f"{PREFIX}_messages"].extend([{"role": "user", "content": question}, assistant_message])
    if result is not None:
        st.session_state[f"{PREFIX}_scroll_to_reply"] = True

st.subheader("Conversation")
render_chat_messages(st.session_state[f"{PREFIX}_messages"])
if st.session_state.pop(f"{PREFIX}_scroll_to_reply", False):
    components.html(
        """
        <script>
        window.setTimeout(() => {
          const messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
          const latest = messages[messages.length - 1];
          if (latest) latest.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
        </script>
        """,
        height=0,
    )
if st.session_state[f"{PREFIX}_safety_flags"]:
    st.warning("Safety screening flagged: " + ", ".join(st.session_state[f"{PREFIX}_safety_flags"]))
render_agent_trace(st.session_state[f"{PREFIX}_steps"], st.session_state[f"{PREFIX}_mode"])
render_sources(st.session_state[f"{PREFIX}_sources"])
render_safety_notice()
