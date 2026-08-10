# CareCost Navigator

CareCost Navigator is a multi-page Streamlit capstone prototype that helps people prepare for healthcare conversations and explore Singapore Ministry of Health (MOH) private-sector fee benchmark ranges. It uses an explicit agentic workflow implemented in plain Python.

For a detailed technical walkthrough, including the agent state machine and complete retrieval/reranking pipeline, see [How CareCost Navigator Works](docs/HOW_THE_APP_WORKS.md).

> Educational scope only: the app does not diagnose, prescribe, determine insurance coverage, or quote a final bill. Users should provide generic, non-personally identifiable information.

## Capstone use cases

### 1. Care Pathway Guide

A conversational guide for a symptom, diagnosis, or procedure. It:

- screens for possible emergency warning signs;
- retrieves relevant guidance from a curated allowlist of official Singapore sources;
- explains possible care or procedure categories without deciding what treatment is needed;
- identifies missing context and supports follow-up questions; and
- shows the planner, tools, composer, evaluator, and revision trace.

### 2. Fee Benchmark Explorer

A guided form for a procedure, diagnosis, or TOSP code, care setting, and fee component. It:

- searches the MOH workbook with BM25 and optional in-memory vector retrieval;
- applies query expansion, thresholds, field-aware boosts, reranking, and MMR-style diversification;
- grounds cost statements in matched workbook rows; and
- presents results as an explanation, evidence table, and lower/upper range chart.

## Agentic workflow

The workflow in `agentic_workflow.py` is a bounded plan-and-execute loop:

```text
input guard -> constrained planner -> allowlisted tools -> answer composer
                                                    -> quality evaluator
                                                    -> optional one-time revision
```

The planner proposes JSON containing a route and tool list. The application validates it, removes unknown tools, restores mandatory safety tools, and falls back to deterministic routing if the plan is malformed. The available tools are:

- `safety_check`
- `official_source_lookup`
- `benchmark_search`
- `missing_information_check`

This gives the workflow goal-directed routing, tool selection, shared state, observations, evaluation, and iteration without a multi-agent framework dependency.

## Multi-page structure

```text
app.py                              Home and use-case navigation
pages/1_Care_Pathway_Guide.py       Use case 1
pages/2_Fee_Benchmark_Explorer.py   Use case 2
pages/3_About_Us.py                 Required project documentation
pages/4_Methodology.py              Data flows, safeguards, and two flowcharts
agentic_workflow.py                 LLM clients and agentic orchestration
ui_components.py                    Shared Streamlit UI and cached index loading
utils/benchmark_rag.py              Workbook ingestion and retrieval pipeline
data/official_sources.json          Curated official-source allowlist
data/feebenchmarks.xlsx             MOH fee benchmark workbook
tests/                              Workflow and safeguard tests
```

## Official sources

The local source registry was reviewed on 10 August 2026 and links to:

- [MOH Hospital Bills and Fee Benchmarks](https://www.moh.gov.sg/managing-expenses/bills-and-fee-benchmarks/hospital-bills-and-fee-benchmarks/)
- [MOH Getting medical help](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/)
- [SCDF Emergency Medical Services](https://www.scdf.gov.sg/home/about-scdf/emergency-medical-services)

The workbook should be refreshed from the MOH page before a production deployment. Source summaries in the repository are not a substitute for checking the linked official pages.

## Prompt-injection and misuse safeguards

- User input, conversation history, retrieved records, and draft answers are marked as untrusted data in prompts.
- Input size and retained history are bounded.
- Rule-based screening flags common instruction-override, prompt-disclosure, credential, delimiter, and jailbreak patterns.
- Planner output is parsed and restricted to a fixed tool allowlist.
- The workflow has no shell, code-execution, filesystem-write, credential, general-web, or arbitrary-URL tool.
- Cost statements must use retrieved workbook rows; source links must come from the local allowlist.
- A separate evaluator checks diagnosis language, emergency escalation, benchmark caveats, unsupported claims, and source use.
- API keys are not inserted into prompts or written by the application.

These controls reduce risk but do not guarantee prevention of every adversarial input or hallucination.

## Run locally

Python 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Without an API key, both pages remain usable in retrieval-only mode. Add a key in the sidebar for LLM planning, composition, evaluation, and revision. Supported environment variables are:

```text
OPENAI_API_KEY
GEMINI_API_KEY
ANTHROPIC_API_KEY
GOVTECH_PLATFORM_API_KEY
```

For the GovTech AI Platform, select **OpenAI-compatible** and use this base URL:

```text
https://api-public.ai.tech.gov.sg/platform/models
```

Provider model names are editable in the sidebar because availability differs by account and changes over time.

## Test

The test suite uses the standard library `unittest` runner:

```powershell
python -m unittest discover -s tests -v
```

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app with `app.py` as the entry point.
3. Use Python 3.12 and install from `requirements.txt`.
4. Add model credentials in the app's Secrets settings rather than committing them.
5. Optionally set `APP_PASSWORD` to activate the built-in password gate.
6. Verify both use cases, all four pages, external source links, and the evidence table/chart on the deployed URL.

Example secrets are documented in `.streamlit/secrets.toml.example`.
