from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
import requests
import streamlit as st

from benchmark_rag import (
    BenchmarkIndex,
    BenchmarkRecord,
    build_benchmark_index,
    build_context,
    estimate_amounts,
    first_matching_field,
    load_benchmark_records,
    search_benchmark_records,
)


APP_TITLE = "CareCost Navigator"
DATA_PATH = Path("data/feebenchmarks.xlsx")
MOH_SOURCE_URL = "https://www.moh.gov.sg/managing-expenses/bills-and-fee-benchmarks/hospital-bills-and-fee-benchmarks/"

MODEL_DEFAULTS = {
    "OpenAI": "gpt-4o-mini",
    "Gemini": "gemini-1.5-flash",
    "Claude": "claude-3-5-sonnet-latest",
    "OpenAI-compatible": "gpt-4o-mini",
}

PROVIDER_ENV_KEYS = {
    "OpenAI": "OPENAI_API_KEY",
    "Gemini": "GEMINI_API_KEY",
    "Claude": "ANTHROPIC_API_KEY",
    "OpenAI-compatible": "GOVTECH_PLATFORM_API_KEY",
}


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str, base_url: str = "") -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise ValueError("Enter an API key in the sidebar first.")
        if self.provider == "Gemini":
            return self._gemini(system, user)
        if self.provider == "Claude":
            return self._anthropic(system, user)
        return self._openai_sdk(system, user)

    def _openai_sdk(self, system: str, user: str) -> str:
        kwargs: dict[str, str] = {"api_key": self.api_key}
        if self.provider == "OpenAI-compatible":
            if not self.base_url:
                raise ValueError("Enter the OpenAI-compatible base URL or chat completions endpoint.")
            kwargs["base_url"] = normalize_openai_compatible_base_url(self.base_url)

        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            timeout=90,
        )
        return response.choices[0].message.content or ""

    def _gemini(self, system: str, user: str) -> str:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        response = requests.post(
            endpoint,
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.2},
            },
            timeout=90,
        )
        payload = parse_json_response(response)
        parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    def _anthropic(self, system: str, user: str) -> str:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": 1600,
                "temperature": 0.2,
            },
            timeout=90,
        )
        payload = parse_json_response(response)
        return "".join(part.get("text", "") for part in payload.get("content", []))


def parse_json_response(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(response.text or response.reason) from exc
    if response.status_code >= 400:
        message = payload.get("error", {}).get("message") if isinstance(payload.get("error"), dict) else payload.get("error")
        raise RuntimeError(message or response.reason)
    return payload


def normalize_openai_compatible_base_url(base_url: str) -> str:
    """Accept either an OpenAI SDK base_url or a full chat completions endpoint."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    return url


def resolve_api_key(provider: str, typed_key: str) -> str:
    if typed_key:
        return typed_key
    env_key = PROVIDER_ENV_KEYS.get(provider)
    return os.environ.get(env_key, "") if env_key else ""


@st.cache_data(show_spinner=False)
def load_benchmark_index(path: str) -> BenchmarkIndex:
    records = load_benchmark_records(path)
    return build_benchmark_index(records)


def run_agent_workflow(
    client: LLMClient,
    mode: str,
    question: str,
    matches: list[tuple[BenchmarkRecord, float]],
) -> tuple[str, list[tuple[str, str]]]:
    context = build_context(matches)
    base_system = """You are CareCost Navigator, an educational Singapore healthcare assistant.
Do not diagnose. Do not provide definitive medical advice. Encourage professional clinical care.
For cost statements, use only the supplied fee benchmark rows and clearly say when data is missing or ambiguous.
Mention that actual costs vary by hospital, subsidy status, ward class, complications, implants, medications, insurance, and clinical decisions."""

    steps: list[tuple[str, str]] = []

    orchestrator = (
        "Route the user request into condition-to-procedure guidance, fee benchmarking, or both. "
        "Return a short routing summary and the key entities to inspect."
    )
    route = client.complete(base_system, f"{orchestrator}\n\nMode: {mode}\nQuestion: {question}")
    steps.append(("Orchestrator", route))

    medical_task = (
        "Act as the medical specialist agent. Explain possible procedure categories and red flags at a high level. "
        "Use cautious language and avoid diagnosis."
    )
    medical = client.complete(base_system, f"{medical_task}\n\nRouting summary:\n{route}\n\nQuestion:\n{question}")
    steps.append(("Medical Specialist", medical))

    benchmark_task = (
        "Act as the benchmark analyst agent. Use the matched rows to summarize relevant procedure, doctor fee, "
        "hospital fee, inpatient attendance, or medical-condition benchmark information."
    )
    benchmark = client.complete(
        base_system,
        f"{benchmark_task}\n\nQuestion:\n{question}\n\nMatched fee benchmark rows:\n{context}",
    )
    steps.append(("Benchmark Analyst", benchmark))

    evaluator_task = (
        "Act as the evaluator agent. Produce the final answer for the user by synthesizing the agent outputs. "
        "Use headings, concise bullets, practical next steps, and safety caveats."
    )
    final = client.complete(
        base_system,
        f"{evaluator_task}\n\nQuestion:\n{question}\n\nOrchestrator:\n{route}\n\nMedical Specialist:\n{medical}\n\nBenchmark Analyst:\n{benchmark}",
    )
    steps.append(("Evaluator", final))
    return final, steps


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
                steps = [("System", answer)]
        st.session_state.latest_answer = answer
        st.session_state.agent_steps = steps

    with st.expander("Agent Trace", expanded=False):
        if not st.session_state.agent_steps:
            st.write("Ask a question to see the sequential agent workflow.")
        for name, output in st.session_state.agent_steps:
            st.markdown(f"**{name}**")
            st.markdown(output)

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
