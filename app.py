from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


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


@dataclass
class BenchmarkRecord:
    sheet: str
    row_number: int
    fields: dict[str, str]
    searchable_text: str

    def as_context(self) -> dict[str, Any]:
        return {
            "sheet": self.sheet,
            "row_number": self.row_number,
            "fields": self.fields,
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
        endpoint = "https://api.openai.com/v1/chat/completions"
        if self.provider == "OpenAI-compatible":
            if not self.base_url:
                raise ValueError("Enter the OpenAI-compatible base URL or chat completions endpoint.")
            endpoint = normalize_openai_compatible_endpoint(self.base_url)
        return self._openai_compatible(endpoint, system, user)

    def _openai_compatible(self, endpoint: str, system: str, user: str) -> str:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            },
            timeout=90,
        )
        payload = parse_json_response(response)
        return payload.get("choices", [{}])[0].get("message", {}).get("content", "")

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


def normalize_openai_compatible_endpoint(base_url: str) -> str:
    """Accept either an OpenAI SDK base_url or a full chat completions endpoint."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def resolve_api_key(provider: str, typed_key: str) -> str:
    if typed_key:
        return typed_key
    env_key = PROVIDER_ENV_KEYS.get(provider)
    return os.environ.get(env_key, "") if env_key else ""


@st.cache_data(show_spinner=False)
def load_benchmark_records(path: str) -> list[BenchmarkRecord]:
    workbook = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    records: list[BenchmarkRecord] = []
    for sheet_name, frame in workbook.items():
        frame = frame.fillna("")
        header_index = find_header_row(frame)
        if header_index is None:
            records.extend(load_note_records(sheet_name, frame))
        else:
            records.extend(load_tabular_records(sheet_name, frame, header_index))
    return records


def find_header_row(frame: pd.DataFrame) -> int | None:
    hints = ("tosp", "description", "lower", "upper", "ward type", "drg", "ccs", "icd", "diagnosis")
    best_index: int | None = None
    best_score = 0
    for idx, row in frame.iterrows():
        text_cells = [str(cell).strip().lower() for cell in row.tolist() if str(cell).strip()]
        score = sum(any(hint in cell for cell in text_cells) for hint in hints)
        if score > best_score:
            best_index = int(idx)
            best_score = score
    return best_index if best_score >= 2 else None


def load_note_records(sheet_name: str, frame: pd.DataFrame) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for idx, row in frame.iterrows():
        text = " ".join(str(cell).strip() for cell in row.tolist() if str(cell).strip())
        if len(text) < 20:
            continue
        records.append(
            BenchmarkRecord(
                sheet=sheet_name,
                row_number=int(idx) + 1,
                fields={"note": clean_text(text)},
                searchable_text=clean_text(f"{sheet_name} {text}").lower(),
            )
        )
    return records


def load_tabular_records(sheet_name: str, frame: pd.DataFrame, header_index: int) -> list[BenchmarkRecord]:
    headers = make_headers(frame.iloc[header_index].tolist())
    records: list[BenchmarkRecord] = []
    for idx, row in frame.iloc[header_index + 1 :].iterrows():
        values = [clean_text(str(cell)) for cell in row.tolist()]
        fields = {
            headers[col_index]: value
            for col_index, value in enumerate(values)
            if col_index < len(headers) and value
        }
        if len(fields) < 2:
            continue
        row_text = " ".join(fields.values())
        records.append(
            BenchmarkRecord(
                sheet=sheet_name,
                row_number=int(idx) + 1,
                fields=fields,
                searchable_text=clean_text(f"{sheet_name} {row_text}").lower(),
            )
        )
    return records


def make_headers(values: list[Any]) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for idx, value in enumerate(values):
        header = clean_text(str(value)).lower()
        header = re.sub(r"[^a-z0-9]+", "_", header).strip("_")
        if not header:
            header = f"column_{idx + 1}"
        seen[header] = seen.get(header, 0) + 1
        if seen[header] > 1:
            header = f"{header}_{seen[header]}"
        headers.append(header)
    return headers


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def search_records(records: list[BenchmarkRecord], query: str, limit: int = 10) -> list[tuple[BenchmarkRecord, int]]:
    terms = [term for term in re.split(r"[^a-zA-Z0-9]+", query.lower()) if len(term) > 2]
    if not terms:
        return []
    scored: list[tuple[BenchmarkRecord, int]] = []
    for record in records:
        score = 0
        for term in terms:
            if term in record.searchable_text:
                score += 1
            if re.search(rf"\b{re.escape(term)}\b", record.searchable_text):
                score += 2
        if score:
            scored.append((record, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


def estimate_amounts(record: BenchmarkRecord) -> tuple[int | None, int | None]:
    numbers: list[int] = []
    for key, value in record.fields.items():
        if any(token in key for token in ("lower", "upper", "fee", "bound", "cost")):
            numbers.extend(int(num.replace(",", "")) for num in re.findall(r"\d[\d,]*", value))
    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def build_context(matches: list[tuple[BenchmarkRecord, int]]) -> str:
    rows = [record.as_context() | {"match_score": score} for record, score in matches]
    return json.dumps(rows, indent=2, ensure_ascii=False)


def run_agent_workflow(
    client: LLMClient,
    mode: str,
    question: str,
    matches: list[tuple[BenchmarkRecord, int]],
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


def render_match_table(matches: list[tuple[BenchmarkRecord, int]]) -> None:
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


def first_matching_field(record: BenchmarkRecord, names: tuple[str, ...]) -> str:
    for name in names:
        for key, value in record.fields.items():
            if name in key and value:
                return value
    return next(iter(record.fields.values()), "")


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
                help="For GovTech AI Platform, use the same base_url you would pass to OpenAI(...). The app will append /chat/completions automatically.",
            )
        st.divider()
        st.header("Data")
        st.write(f"Workbook: `{DATA_PATH}`")
        st.link_button("MOH fee benchmarks source", MOH_SOURCE_URL)
        st.caption("CrewAI note: current CrewAI docs require Python >=3.10 and <3.14, so deploy with Python 3.12.")

    if not DATA_PATH.exists():
        st.error(f"Missing workbook: {DATA_PATH}")
        return

    records = load_benchmark_records(str(DATA_PATH))
    st.success(f"Loaded {len(records):,} searchable rows/notes from `{DATA_PATH}`.")

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
        matches = search_records(records, question)
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
