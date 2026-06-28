from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests
from openai import OpenAI

from benchmark_rag import BenchmarkRecord, build_context


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
            description="Route the request and identify key entities, likely workflow, and missing information.",
            expected_output="A routing summary and list of entities to inspect.",
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
) -> tuple[str, list[AgentStep]]:
    context = build_context(matches)
    base_system = """You are CareCost Navigator, an educational Singapore healthcare assistant.
Do not diagnose. Do not provide definitive medical advice. Encourage professional clinical care.
For cost statements, use only the supplied fee benchmark rows and clearly say when data is missing or ambiguous.
Mention that actual costs vary by hospital, subsidy status, ward class, complications, implants, medications, insurance, and clinical decisions."""

    tasks = build_healthcare_crew()
    outputs: list[AgentStep] = []

    for task in tasks[:-1]:
        prior_context = format_prior_outputs(outputs)
        prompt = f"""Agent role: {task.agent.role}
Agent goal: {task.agent.goal}
Agent backstory: {task.agent.backstory}
Task: {task.description}
Expected output: {task.expected_output}

Mode: {mode}
Question: {question}
Retrieved benchmark context:
{context}

Prior agent outputs:
{prior_context}"""
        output = client.complete(base_system, prompt)
        outputs.append(AgentStep(task.agent.role, output))

    evaluator_task = tasks[-1]
    final_prompt = f"""Agent role: {evaluator_task.agent.role}
Agent goal: {evaluator_task.agent.goal}
Agent backstory: {evaluator_task.agent.backstory}
Task: {evaluator_task.description}
Expected output: {evaluator_task.expected_output}

Question: {question}
Retrieved benchmark context:
{context}

Prior agent outputs:
{format_prior_outputs(outputs)}

Produce the final answer for the user."""
    final = client.complete(base_system, final_prompt)
    outputs.append(AgentStep(evaluator_task.agent.role, final))
    return final, outputs


def format_prior_outputs(outputs: list[AgentStep]) -> str:
    if not outputs:
        return "No prior agent output yet."
    return json.dumps([{step.name: step.output} for step in outputs], indent=2, ensure_ascii=False)
