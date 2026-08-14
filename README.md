# CareCost Navigator

CareCost Navigator is a multi-page Streamlit capstone prototype for healthcare navigation and Singapore Ministry of Health (MOH) cost information. It deliberately separates a conversational, non-diagnostic Care Pathway Guide from a known-procedure Fee Benchmark Explorer.

For a detailed technical walkthrough, including the agent state machine and complete retrieval/reranking pipeline, see [How CareCost Navigator Works](docs/HOW_THE_APP_WORKS.md).

> Educational scope only: the app does not diagnose, prescribe, determine insurance coverage, or quote a final bill. Users should provide generic, non-personally identifiable information.

## Capstone use cases

### 1. Care Pathway Guide

A conversational guide for a symptom or known condition. It:

- screens for possible emergency warning signs;
- expands recognised medical terminology through MeSH strictly for retrieval, not diagnosis;
- uses the LLM to give carefully labelled general education for a named condition, including common symptoms and care options a clinician may discuss;
- retrieves relevant guidance from a curated allowlist of official Singapore sources;
- never diagnoses, prescribes, or claims that a procedure is necessary;
- identifies missing context and supports follow-up questions; and
- shows the planner, tools, composer, evaluator, and revision trace.

### 2. Fee Benchmark Explorer

A guided form for a known procedure, diagnosis, or TOSP code, care setting, and fee component. Symptom-only descriptions are directed to the Care Pathway Guide. It:

- searches `feebenchmarks.xlsx` with BM25 and optional in-memory vector retrieval;
- separately searches `hospitalbillsizes.xlsx` for hospital-stay bill-size rows;
- lets the user narrow stay-cost rows by hospital and ward type, with hospital labels shown as `Full Name (ABBREVIATION)`;
- applies query expansion, thresholds, field-aware boosts, reranking, and MMR-style diversification;
- grounds cost statements in matched workbook rows; and
- presents fee-benchmark rows, hospital stay bill-size rows, and a lower/upper range chart separately.

Hospital-stay rows can include P25, P50, and P75 total bill amounts and average length of stay (ALOS), where present in the source workbook. They are descriptive percentiles, not a bill quote or prediction.

## Agentic workflow

The workflow in `agentic_workflow.py` is a bounded plan-and-execute loop:

```text
input guard -> constrained planner -> allowlisted tools -> answer composer
                                                    -> quality evaluator
                                                    -> optional one-time revision
```

The planner proposes JSON containing a route and tool list. The application validates it, removes unknown tools, restores mandatory safety tools, and falls back to deterministic routing if the plan is malformed. The available tools are:

- `safety_check`
- `mesh_rag`
- `official_source_lookup`
- `benchmark_search`
- `hospital_bill_search` (Fee Benchmark Explorer only)
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
data/hospitalbillsizes.xlsx         MOH hospital bill-size workbook
tests/                              Workflow and safeguard tests
```

## Curated sources

The local source registry was reviewed on 14 August 2026 and links to:

- [MOH Hospital Bills and Fee Benchmarks](https://www.moh.gov.sg/managing-expenses/bills-and-fee-benchmarks/hospital-bills-and-fee-benchmarks/)
- [MOH Getting medical help](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/)
- [MOH Conditions](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/conditions/)
- [MOH Visiting a pharmacist](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/visiting-a-pharmacist/)
- [MOH Seeking a doctor](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/seeking-a-doctor/)
- [MOH When to visit the hospital for emergencies](https://www.moh.gov.sg/seeking-healthcare/getting-medical-help/visiting-the-hospital-for-emergencies/)
- [SCDF Emergency Medical Services](https://www.scdf.gov.sg/home/about-scdf/emergency-medical-services)
- [HealthHub public healthcare institutions by cluster](https://support.healthhub.sg/hc/en-us/articles/57937601611033-What-are-the-public-healthcare-institutions-under-each-cluster)
- [NUHS Find a Condition](https://www.nuhs.edu.sg/patient-care/find-a-condition)
- [SingHealth Symptoms & Medical Conditions](https://www.singhealth.com.sg/symptoms-treatments/symptoms-treatments-medical-conditions)
- [Mount Elizabeth Conditions & Diseases](https://www.mountelizabeth.com.sg/conditions-diseases)
- [Gleneagles Conditions & Diseases](https://www.gleneagles.com.sg/conditions-diseases)

MOH and SCDF entries are official public guidance. The NUHS, SingHealth, Mount Elizabeth, and Gleneagles entries are clearly labelled **Supplementary provider education** in the app; they are useful for general condition information but are not MOH policy, diagnosis, or personalised medical advice. Source summaries in the repository are not a substitute for checking the linked pages.

## Prompt-injection and misuse safeguards

- User input, conversation history, retrieved records, and draft answers are marked as untrusted data in prompts.
- Input size and retained history are bounded.
- Rule-based screening flags common instruction-override, prompt-disclosure, credential, delimiter, and jailbreak patterns.
- Planner output is parsed and restricted to a fixed tool allowlist.
- The workflow has no shell, code-execution, filesystem-write, credential, general-web, or arbitrary-URL tool.
- Cost statements must use retrieved fee-benchmark or hospital bill-size workbook rows; source links must come from the local allowlist.
- Symptom-derived cost information carries a deterministic scope notice: it is based only on the supplied description and is not diagnosis, assessment, or medical advice.
- A separate evaluator checks emergency escalation, diagnosis boundaries, unsupported costs, benchmark caveats, and source use.
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

Ensure both workbook files (`feebenchmarks.xlsx` and `hospitalbillsizes.xlsx`) are deployed with the application.

Example secrets are documented in `.streamlit/secrets.toml.example`.
