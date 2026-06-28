# CareCost Navigator

Streamlit prototype for the project idea in `Project_Ideation.md`.

## What It Does

- Searches the Singapore MOH fee benchmark workbook at `data/feebenchmarks.xlsx`.
- Runs a sequential agent workflow: orchestrator, medical specialist, benchmark analyst, evaluator.
- Supports BYO API keys for OpenAI, Gemini, Claude, and OpenAI-compatible chat completion endpoints.
- Keeps API keys in Streamlit session memory only. They are not written to project files.

## GovTech AI Platform

Choose `OpenAI-compatible` in the sidebar.

Use your platform key as the API key, and enter this as the base URL:

```text
https://api-public.ai.tech.gov.sg/platform/models
```

This matches the OpenAI SDK pattern:

```python
client = OpenAI(
    base_url="https://api-public.ai.tech.gov.sg/platform/models",
    api_key=os.environ["GOVTECH_PLATFORM_API_KEY"],
)
```

The app automatically calls `{base_url}/chat/completions`.

For local development, you can also set:

```powershell
$env:GOVTECH_PLATFORM_API_KEY="your-key"
```

Then leave the API key field blank in Streamlit.

## Run Locally

Use Python 3.12, especially if you later install CrewAI.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## CrewAI Option

The current app uses a lightweight CrewAI-style workflow in plain Python so it is easy to deploy.

CrewAI can be added after the base app is working:

```bash
pip install -r requirements-crewai.txt
```

CrewAI's current docs require Python `>=3.10 and <3.14`, so use Python 3.12 rather than the local Python 3.14 interpreter.

## Data

The app expects:

```text
data/feebenchmarks.xlsx
```

Source: <https://www.moh.gov.sg/managing-expenses/bills-and-fee-benchmarks/hospital-bills-and-fee-benchmarks/>

## Safety Notes

This is an educational prototype. It does not provide medical diagnosis, medical advice, or guaranteed cost quotes. Real costs vary by hospital, subsidy status, ward class, complications, implants, medication, insurance coverage, and clinical decisions.
