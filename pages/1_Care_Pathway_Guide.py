import streamlit as st

from agentic_workflow import AgentStep, LLMClient, run_agent_workflow
from ui_components import (
    APP_TITLE,
    clear_chat_state,
    get_benchmark_index,
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
settings = render_model_sidebar(PREFIX)

with st.sidebar:
    st.divider()
    if st.button("Clear this conversation", key=f"{PREFIX}_clear"):
        clear_chat_state(PREFIX)
        st.rerun()

st.title("Care Pathway Guide")
st.write(
    "Get a non-diagnostic explanation of possible care or procedure categories, urgent warning signs, "
    "and questions to ask a clinician. Follow-up turns reuse recent conversation context."
)
st.warning("Do not enter personally identifiable or sensitive record details. Call 995 for a life-threatening emergency.")

benchmark_index = get_benchmark_index(settings)
question = st.chat_input("Example: My doctor mentioned a colonoscopy. What should I ask before deciding?")
if question:
    previous_messages = list(st.session_state[f"{PREFIX}_messages"])
    client = LLMClient(settings.provider, settings.api_key, settings.model, settings.base_url)
    with st.spinner("Planning, calling tools, and evaluating the answer..."):
        try:
            result = run_agent_workflow(
                client,
                "Condition to procedures",
                question,
                benchmark_index,
                previous_messages,
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
    st.session_state[f"{PREFIX}_messages"].extend(
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    )

st.subheader("Conversation")
render_chat_messages(st.session_state[f"{PREFIX}_messages"])
if st.session_state[f"{PREFIX}_safety_flags"]:
    st.warning("Safety screening flagged: " + ", ".join(st.session_state[f"{PREFIX}_safety_flags"]))
render_agent_trace(st.session_state[f"{PREFIX}_steps"], st.session_state[f"{PREFIX}_mode"])
render_sources(st.session_state[f"{PREFIX}_sources"])
render_safety_notice()
