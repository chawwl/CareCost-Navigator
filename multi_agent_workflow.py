from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import requests
from openai import OpenAI

from utils.benchmark_rag import BenchmarkRecord, build_context


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


@dataclass(frozen=True)
class AgentSpec:
    role: str
    goal: str
    backstory: str
    allow_delegation: bool = False


@dataclass(frozen=True)
class TaskSpec:
    description: str
    expected_output: str
    agent: AgentSpec


@dataclass(frozen=True)
class AgentStep:
    name: str
    output: str


@dataclass(frozen=True)
class WorkflowResult:
    answer: str
    steps: list[AgentStep]
    follow_up_questions: list[str]
    inferred_mode: str


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


COST_TERMS = {
    "bill",
    "bills",
    "benchmark",
    "charge",
    "charges",
    "cost",
    "costs",
    "estimate",
    "fee",
    "fees",
    "how much",
    "price",
    "prices",
}

PROCEDURE_TERMS = {
    "anaesthesia",
    "anaesthetic",
    "colonoscopy",
    "endoscopy",
    "operation",
    "procedure",
    "surgery",
    "surgical",
    "surgeon",
    "tosp",
}

CONDITION_TERMS = {
    "condition",
    "diagnosis",
    "diagnosed",
    "drg",
    "illness",
    "injury",
    "medical",
    "pain",
    "symptom",
    "symptoms",
}

SETTING_TERMS = {
    "admission",
    "admitted",
    "day surgery",
    "day procedure",
    "general ward",
    "hdu",
    "hospital stay",
    "icu",
    "inpatient",
    "outpatient",
    "ward",
    "warded",
}

FEE_TYPE_TERMS = {
    "anaesthetist",
    "anesthetist",
    "attendance",
    "doctor",
    "facility",
    "hospital",
    "surgeon",
}

GENERIC_REQUEST_TERMS = COST_TERMS | PROCEDURE_TERMS | CONDITION_TERMS | SETTING_TERMS | FEE_TYPE_TERMS | {
    "about",
    "actual",
    "average",
    "can",
    "class",
    "give",
    "help",
    "how",
    "know",
    "lower",
    "me",
    "much",
    "need",
    "please",
    "private",
    "public",
    "range",
    "singapore",
    "subsidy",
    "upper",
    "want",
    "what",
}


def infer_workflow_mode(question: str, conversation_history: list[Mapping[str, str]] | None = None) -> str:
    text = conversation_text(question, conversation_history)
    wants_cost = contains_any(text, COST_TERMS)
    has_condition = contains_any(text, CONDITION_TERMS)
    has_procedure = contains_any(text, PROCEDURE_TERMS)

    if wants_cost and (has_condition or has_procedure):
        return "Both"
    if wants_cost:
        return "Procedure cost estimate"
    if has_condition or has_procedure:
        return "Condition to procedures"
    return "Both"


def identify_missing_information(
    question: str,
    mode: str,
    matches: list[tuple[BenchmarkRecord, float]],
    conversation_history: list[Mapping[str, str]] | None = None,
) -> list[str]:
    text = conversation_text(question, conversation_history)
    missing: list[str] = []

    if not has_specific_anchor(text):
        missing.append("What diagnosis, symptom cluster, procedure name, or code should I anchor the search on?")

    wants_cost = mode in {"Procedure cost estimate", "Both"} or contains_any(text, COST_TERMS)
    if wants_cost:
        if not contains_any(text, SETTING_TERMS) and not has_ward_class(text):
            missing.append("Is this inpatient, outpatient/day surgery, ICU/HDU, and if inpatient what ward class?")
        if not contains_any(text, FEE_TYPE_TERMS):
            missing.append("Are you looking for hospital bill benchmarks, doctor fees, or both?")
        if not matches:
            missing.append("Can you provide the exact procedure/diagnosis wording used by the doctor or bill?")

    if len(missing) > 3:
        return missing[:3]
    return missing


def conversation_text(question: str, conversation_history: list[Mapping[str, str]] | None = None) -> str:
    parts = [message.get("content", "") for message in (conversation_history or [])]
    parts.append(question)
    return " ".join(parts).lower()


def contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def has_specific_anchor(text: str) -> bool:
    if re.search(r"\b[A-Z]{1,3}\d{2,4}[A-Z]?\b", text.upper()):
        return True
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    meaningful = [
        token
        for token in tokens
        if len(token) > 2 and token not in GENERIC_REQUEST_TERMS and not token.isdigit()
    ]
    return len(meaningful) >= 1


def has_ward_class(text: str) -> bool:
    return bool(
        re.search(r"\b(?:b1|b2)\b", text)
        or re.search(r"\b(?:ward|class)\s+[abc]\b", text)
        or re.search(r"\b[abc]\s+(?:ward|class)\b", text)
    )


def build_healthcare_crew() -> list[TaskSpec]:
    """Course-style CrewAI structure: Agent(role/goal/backstory) + Task(description/expected_output)."""
    orchestrator = AgentSpec(
        role="Healthcare Workflow Orchestrator",
        goal="Route each user request into condition guidance, fee benchmarking, or both.",
        backstory="You coordinate specialist agents and decide what evidence is needed before final synthesis.",
    )
    specialist = AgentSpec(
        role="Medical Specialist",
        goal="Explain likely procedure categories and urgent red flags without diagnosing.",
        backstory="You are careful, safety-oriented, and never replace a licensed clinician.",
    )
    analyst = AgentSpec(
        role="Benchmark Analyst",
        goal="Use retrieved MOH benchmark rows to summarize relevant fees and limitations.",
        backstory="You ground every cost statement in retrieved benchmark rows and flag missing data.",
    )
    evaluator = AgentSpec(
        role="Evaluator",
        goal="Synthesize the agents' work into a safe, concise final answer.",
        backstory="You check uncertainty, hallucination risk, and practical next steps before answering.",
    )
    return [
        TaskSpec(
            description="Route the request, use conversation context, identify key entities, and name missing information.",
            expected_output="A routing summary, entities to inspect, assumptions, and missing information.",
            agent=orchestrator,
        ),
        TaskSpec(
            description="Explain possible procedure categories, relevant clinical considerations, and red flags.",
            expected_output="Cautious medical guidance that does not diagnose.",
            agent=specialist,
        ),
        TaskSpec(
            description="Summarize the retrieved MOH fee benchmark rows and explain their methodology limits.",
            expected_output="Grounded benchmark summary based only on retrieved rows.",
            agent=analyst,
        ),
        TaskSpec(
            description="Combine prior outputs into the final user-facing answer with safety caveats.",
            expected_output="Structured answer with headings, bullets, next steps, and uncertainty notes.",
            agent=evaluator,
        ),
    ]


def run_agent_workflow(
    client: LLMClient,
    mode: str,
    question: str,
    matches: list[tuple[BenchmarkRecord, float]],
    conversation_history: list[Mapping[str, str]] | None = None,
) -> WorkflowResult:
    inferred_mode = mode if mode != "Auto" else infer_workflow_mode(question, conversation_history)
    context = build_context(matches)
    chat_context = format_conversation_history(conversation_history)
    follow_up_questions = identify_missing_information(question, inferred_mode, matches, conversation_history)
    missing_info_trace = format_missing_information(follow_up_questions)
    base_system = """You are CareCost Navigator, an educational Singapore healthcare assistant.
Do not diagnose. Do not provide definitive medical advice. Encourage professional clinical care.
For cost statements, use only the supplied fee benchmark rows and clearly say when data is missing or ambiguous.
Mention that actual costs vary by hospital, subsidy status, ward class, complications, implants, medications, insurance, and clinical decisions.
Use the conversation context to resolve short follow-up replies from the user."""

    tasks = build_healthcare_crew()
    selected_tasks = select_tasks(tasks, inferred_mode)
    outputs: list[AgentStep] = []

    for task in selected_tasks[:-1]:
        prior_context = format_prior_outputs(outputs)
        prompt = f"""Agent role: {task.agent.role}
Agent goal: {task.agent.goal}
Agent backstory: {task.agent.backstory}
Task: {task.description}
Expected output: {task.expected_output}

Inferred workflow: {inferred_mode}
Current user message: {question}
Conversation context:
{chat_context}

Missing information check:
{missing_info_trace}

Retrieved benchmark context:
{context}

Prior agent outputs:
{prior_context}"""
        output = client.complete(base_system, prompt)
        outputs.append(AgentStep(task.agent.role, output))

    outputs.append(AgentStep("Missing Information Check", missing_info_trace))

    evaluator_task = selected_tasks[-1]
    final_prompt = f"""Agent role: {evaluator_task.agent.role}
Agent goal: {evaluator_task.agent.goal}
Agent backstory: {evaluator_task.agent.backstory}
Task: {evaluator_task.description}
Expected output: {evaluator_task.expected_output}

Inferred workflow: {inferred_mode}
Current user message: {question}
Conversation context:
{chat_context}

Missing information check:
{missing_info_trace}

Retrieved benchmark context:
{context}

Prior agent outputs:
{format_prior_outputs(outputs)}

Produce the final answer for the user. If the missing information check contains questions, ask them under a "Follow-up questions" heading after any useful answer you can provide from the available evidence."""
    final = client.complete(base_system, final_prompt)
    final = append_follow_up_questions(final, follow_up_questions)
    outputs.append(AgentStep(evaluator_task.agent.role, final))
    return WorkflowResult(final, outputs, follow_up_questions, inferred_mode)


def select_tasks(tasks: list[TaskSpec], mode: str) -> list[TaskSpec]:
    orchestrator, specialist, analyst, evaluator = tasks
    selected = [orchestrator]
    if mode in {"Condition to procedures", "Both"}:
        selected.append(specialist)
    if mode in {"Procedure cost estimate", "Both"}:
        selected.append(analyst)
    selected.append(evaluator)
    return selected


def format_prior_outputs(outputs: list[AgentStep]) -> str:
    if not outputs:
        return "No prior agent output yet."
    return json.dumps([{step.name: step.output} for step in outputs], indent=2, ensure_ascii=False)


def format_conversation_history(conversation_history: list[Mapping[str, str]] | None) -> str:
    if not conversation_history:
        return "No prior conversation yet."
    recent_messages = conversation_history[-8:]
    return json.dumps(
        [{"role": message.get("role", ""), "content": message.get("content", "")} for message in recent_messages],
        indent=2,
        ensure_ascii=False,
    )


def format_missing_information(follow_up_questions: list[str]) -> str:
    if not follow_up_questions:
        return "No critical missing information detected for this turn."
    return "\n".join(f"- {question}" for question in follow_up_questions)


def append_follow_up_questions(answer: str, follow_up_questions: list[str]) -> str:
    if not follow_up_questions:
        return answer
    if "follow-up questions" in answer.lower():
        return answer
    questions = "\n".join(f"- {question}" for question in follow_up_questions)
    return f"{answer.rstrip()}\n\n### Follow-up questions\n{questions}"
