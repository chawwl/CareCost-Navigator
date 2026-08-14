from __future__ import annotations

import hmac
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agentic_workflow import (
    AgentStep,
    MODEL_DEFAULTS,
    PROVIDER_ENV_KEYS,
    OfficialSource,
    resolve_api_key,
)
from utils.mesh_rag import clear_mesh_cache, mesh_cache_info
from utils.benchmark_rag import (
    BenchmarkIndex,
    BenchmarkRecord,
    DEFAULT_EMBEDDING_MODEL,
    build_benchmark_index,
    estimate_amounts,
    first_matching_field,
    load_benchmark_records,
)


APP_TITLE = "CareCost Navigator"
DATA_PATH = Path(__file__).parent / "data" / "feebenchmarks.xlsx"
HOSPITAL_BILLS_PATH = Path(__file__).parent / "data" / "hospitalbillsizes.xlsx"
MOH_SOURCE_URL = "https://www.moh.gov.sg/managing-expenses/bills-and-fee-benchmarks/hospital-bills-and-fee-benchmarks/"
HOSPITAL_NAMES = {
    "ADMC": "Admiralty Medical Centre",
    "AH": "Alexandra Hospital",
    "CGH": "Changi General Hospital",
    "FPH": "Farrer Park Hospital",
    "GEH": "Gleneagles Hospital",
    "JMC": "Jurong Medical Centre",
    "KKH": "KK Women's and Children's Hospital",
    "KTPH": "Khoo Teck Puat Hospital",
    "MAH": "Mount Alvernia Hospital",
    "MEH": "Mount Elizabeth Hospital",
    "MNH": "Mount Elizabeth Novena Hospital",
    "NCC": "National Cancer Centre Singapore",
    "NHC": "National Heart Centre Singapore",
    "NSC": "National Skin Centre",
    "NTFGH": "Ng Teng Fong General Hospital",
    "NUH": "National University Hospital",
    "PEH": "Parkway East Hospital",
    "RH": "Raffles Hospital",
    "SGH": "Singapore General Hospital",
    "SKH": "Sengkang General Hospital",
    "SNEC": "Singapore National Eye Centre",
    "TMC": "Thomson Medical Centre",
    "TTSH": "Tan Tock Seng Hospital",
}


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    api_key: str
    base_url: str
    embedding_model: str
    semantic_search: bool


@st.cache_resource(show_spinner=False)
def load_benchmark_index(
    path: str,
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> BenchmarkIndex:
    records = load_benchmark_records(path)
    return build_benchmark_index(
        records,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
    )


@st.cache_data(show_spinner=False)
def workbook_summary(path: str) -> tuple[int, int]:
    records = load_benchmark_records(path)
    return len(records), len({record.sheet for record in records})


@st.cache_data(show_spinner=False)
def hospital_bill_filter_options(path: str) -> tuple[list[str], list[str]]:
    if not Path(path).exists():
        return [], []
    records = load_benchmark_records(path)
    hospitals = sorted({record.fields.get("hospital", "") for record in records if record.fields.get("hospital", "")})
    wards = sorted({record.fields.get("ward_type", "") for record in records if record.fields.get("ward_type", "")})
    return hospitals, wards


def format_hospital_option(abbreviation: str) -> str:
    return f"{HOSPITAL_NAMES.get(abbreviation, abbreviation)} ({abbreviation})"


def require_access() -> None:
    """Apply optional password protection when APP_PASSWORD exists in Streamlit secrets."""
    try:
        configured_password = str(st.secrets.get("APP_PASSWORD", ""))
    except (FileNotFoundError, KeyError):
        configured_password = ""
    if not configured_password or st.session_state.get("authenticated"):
        return

    st.title(APP_TITLE)
    st.info("This deployment is password protected.")
    entered = st.text_input("Password", type="password")
    if st.button("Enter", type="primary"):
        if hmac.compare_digest(entered, configured_password):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


def render_model_sidebar(key_prefix: str, *, allow_semantic_search: bool = False) -> ModelSettings:
    with st.sidebar:
        st.header("Model settings")
        provider = st.selectbox("Provider", list(MODEL_DEFAULTS), key=f"{key_prefix}_provider")
        model = st.text_input("Model", value=MODEL_DEFAULTS[provider], key=f"{key_prefix}_model")
        api_key_input = st.text_input(
            "API key",
            type="password",
            key=f"{key_prefix}_api_key",
            help=(
                "Held in session memory only. Leave blank to use the deployment environment variable "
                f"{PROVIDER_ENV_KEYS[provider]}."
            ),
        )
        api_key = resolve_api_key(provider, api_key_input)
        base_url = ""
        if provider == "OpenAI-compatible":
            base_url = st.text_input(
                "Base URL or chat-completions endpoint",
                placeholder="https://api-public.ai.tech.gov.sg/platform/models",
                key=f"{key_prefix}_base_url",
            )
        semantic_search = False
        embedding_model = DEFAULT_EMBEDDING_MODEL
        if allow_semantic_search and provider in {"OpenAI", "OpenAI-compatible"}:
            semantic_search = st.toggle(
                "Use semantic retrieval",
                value=False,
                key=f"{key_prefix}_semantic",
                help="Uses the same API key for an in-memory embedding index. BM25 remains available when off.",
            )
            if semantic_search:
                embedding_model = st.text_input(
                    "Embedding model", value=DEFAULT_EMBEDDING_MODEL, key=f"{key_prefix}_embedding"
                )
        st.caption("No key? The official-source and BM25 retrieval tools still work in retrieval-only mode.")

        with st.expander("MeSH retrieval cache", expanded=False):
            info = mesh_cache_info()
            st.caption(f"In-memory NLM query cache: {info['currsize']} cached queries")
            if st.button("Clear MeSH cache", key=f"{key_prefix}_clear_mesh_cache"):
                clear_mesh_cache()
                st.success("MeSH cache cleared.")
    return ModelSettings(provider, model, api_key, base_url, embedding_model, semantic_search)


def get_benchmark_index(settings: ModelSettings) -> BenchmarkIndex:
    provider = settings.provider if settings.semantic_search else ""
    api_key = settings.api_key if settings.semantic_search else ""
    return load_benchmark_index(
        str(DATA_PATH), provider, api_key, settings.base_url, settings.embedding_model
    )


def get_hospital_bill_index(settings: ModelSettings) -> BenchmarkIndex | None:
    if not HOSPITAL_BILLS_PATH.exists():
        return None
    provider = settings.provider if settings.semantic_search else ""
    api_key = settings.api_key if settings.semantic_search else ""
    return load_benchmark_index(
        str(HOSPITAL_BILLS_PATH), provider, api_key, settings.base_url, settings.embedding_model
    )


def initialize_chat_state(prefix: str) -> None:
    defaults = {
        f"{prefix}_messages": [],
        f"{prefix}_steps": [],
        f"{prefix}_matches": [],
        f"{prefix}_sources": [],
        f"{prefix}_safety_flags": [],
        f"{prefix}_mode": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_chat_state(prefix: str) -> None:
    for suffix in ("messages", "steps", "matches", "sources", "safety_flags", "mode"):
        st.session_state.pop(f"{prefix}_{suffix}", None)
    initialize_chat_state(prefix)


def render_chat_messages(messages: list[dict[str, object]]) -> None:
    if not messages:
        st.info("Ask a question to start this use case.")
        return
    for message in messages:
        with st.chat_message(message.get("role", "assistant")):
            content = str(message.get("content", ""))
            if message.get("cost_matches"):
                split = re.split(r"(?im)^#{2,3}\s*cost explanations\s*$", content, maxsplit=1)
                st.markdown(split[0])
                st.markdown("##### Grounded cost alternatives")
                render_cost_dashboard(
                    message["cost_matches"],  # type: ignore[arg-type]
                    message.get("hospital_bill_matches", []),  # type: ignore[arg-type]
                )
                if len(split) == 2:
                    st.markdown("### Cost explanations\n" + split[1].lstrip())
            else:
                st.markdown(content)

WORKFLOW_PROGRESS_STEPS = (
    ("semantic_index", "Preparing semantic retrieval"),
    ("planner", "Planning"),
    ("safety_check", "Safety check"),
    ("mesh_rag", "MeSH terminology search"),
    ("official_source_lookup", "Official source lookup"),
    ("benchmark_search", "Fee benchmark search"),
    ("hospital_bill_search", "Hospital stay bill-size search"),
    ("missing_information_check", "Checking for missing information"),
    ("answer_composer", "Preparing answer"),
    ("quality_evaluator", "Evaluating answer"),
    ("answer_revision", "Finalising answer"),
)


def create_workflow_progress():
    """Render all workflow stages and update them live."""
    statuses = {key: "pending" for key, _label in WORKFLOW_PROGRESS_STEPS}
    placeholder = st.empty()
    bar = st.progress(0)

    def render() -> None:
        completed = sum(
            status in {"completed", "skipped"}
            for status in statuses.values()
        )
        bar.progress(completed / len(statuses) if statuses else 0)

        rows = []
        for key, label in WORKFLOW_PROGRESS_STEPS:
            status = statuses[key]
            icon = {
                "pending": "○",
                "running": "●",
                "completed": "✓",
                "skipped": "–",
                "error": "!",
            }.get(status, "○")
            css = status

            rows.append(
                f'<div class="ccn-progress-step {css}">'
                f'<span class="ccn-progress-icon">{icon}</span>'
                f'<span>{label}</span>'
                f'</div>'
            )

        placeholder.markdown(
            """
            <style>
            .ccn-progress-step {
                display: flex;
                align-items: center;
                gap: .55rem;
                margin: .28rem 0;
                font-size: .92rem;
            }
            .ccn-progress-step.pending { opacity: .28; }
            .ccn-progress-step.running { opacity: 1; font-weight: 700; }
            .ccn-progress-step.completed { opacity: 1; }
            .ccn-progress-step.skipped { opacity: .5; }
            .ccn-progress-step.error { opacity: 1; font-weight: 700; }
            .ccn-progress-icon {
                width: 1.1rem;
                text-align: center;
                font-weight: 700;
            }
            </style>
            """ + "".join(rows),
            unsafe_allow_html=True,
        )

    def update(step_key: str, status: str) -> None:
        if step_key in statuses:
            statuses[step_key] = status
            render()

    render()
    return update

def render_agent_trace(steps: list[AgentStep], inferred_mode: str = "") -> None:
    with st.expander("Agent workflow trace", expanded=False):
        if not steps:
            st.write("Run the use case to see the planner, selected tools, answer composer, and quality evaluator.")
            return
        if inferred_mode:
            st.caption(f"Workflow route: {inferred_mode}")
        for index, step in enumerate(steps, start=1):
            st.markdown(f"**{index}. {step.name}** · `{step.kind}` · `{step.status}`")
            if step.name in {"Answer Composer", "Answer Revision"}:
                if step.status == "completed":
                    st.caption("Completed successfully.")
                elif step.status == "fallback":
                    st.caption("Model output was unavailable; a grounded retrieval-only response was used.")
                else:
                    st.caption("This stage did not complete.")
                continue
            st.markdown(step.output)


def match_rows(matches: list[tuple[BenchmarkRecord, float]]) -> pd.DataFrame:
    rows = []
    for record, score in matches:
        lower, upper = estimate_amounts(record)
        rows.append(
            {
                "relevance": score,
                "benchmark": first_matching_field(
                    record, ("description", "drg_description", "ccs", "ward_type", "note")
                ),
                "lower_sgd": lower,
                "upper_sgd": upper,
                "workbook_sheet": record.sheet,
                "source_row": record.row_number,
            }
        )
    return pd.DataFrame(rows)


def render_match_table(matches: list[tuple[BenchmarkRecord, float]]) -> None:
    if not matches:
        st.info("No benchmark rows matched this query.")
        return
    st.dataframe(match_rows(matches), hide_index=True, use_container_width=True)


def render_hospital_bill_table(matches: list[tuple[BenchmarkRecord, float]]) -> None:
    if not matches:
        st.info("No hospital bill-size rows matched the selected details.")
        return
    rows = []
    for record, score in matches:
        rows.append({
            "relevance": score,
            "procedure_or_diagnosis": first_matching_field(record, ("tosp_description", "drg_description", "description")),
            "hospital": record.fields.get("hospital", "All participating hospitals"),
            "setting": record.fields.get("setting", ""),
            "ward_type": record.fields.get("ward_type", ""),
            "average_length_of_stay_days": record.fields.get("alos", ""),
            "p25_bill_sgd": record.fields.get("p25_bill", ""),
            "p50_bill_sgd": record.fields.get("p50_bill", ""),
            "p75_bill_sgd": record.fields.get("p75_bill", ""),
            "source_sheet": record.sheet,
            "source_row": record.row_number,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_match_chart(matches: list[tuple[BenchmarkRecord, float]]) -> None:
    frame = match_rows(matches)
    if frame.empty:
        return
    chart = frame.dropna(subset=["lower_sgd", "upper_sgd"]).head(8).copy()
    if chart.empty:
        st.caption("The matched rows do not expose a lower/upper amount pair that can be charted.")
        return
    chart["benchmark"] = chart["benchmark"].fillna(chart["workbook_sheet"])
    chart["benchmark_label"] = chart["benchmark"].map(_wrap_chart_label)
    chart["range"] = chart.apply(
        lambda row: f"SGD {row['lower_sgd']:,.0f} – SGD {row['upper_sgd']:,.0f}", axis=1
    )
    labels = chart["benchmark_label"].tolist()
    range_chart = (
        alt.Chart(chart)
        .mark_bar(color="#5B8FF9")
        .encode(
            y=alt.Y(
                "benchmark_label:N",
                sort=labels,
                title=None,
                axis=alt.Axis(labelLimit=0, labelPadding=8, labelFontSize=12, labelLineHeight=15),
            ),
            x=alt.X("lower_sgd:Q", title="Fee benchmark (SGD)", axis=alt.Axis(format=",d")),
            x2="upper_sgd:Q",
            tooltip=[
                alt.Tooltip("benchmark:N", title="Benchmark"),
                alt.Tooltip("lower_sgd:Q", title="Lower (SGD)", format=",.0f"),
                alt.Tooltip("upper_sgd:Q", title="Upper (SGD)", format=",.0f"),
                alt.Tooltip("workbook_sheet:N", title="Workbook sheet"),
                alt.Tooltip("range:N", title="Range"),
            ],
        )
        .properties(height=max(260, 48 * len(chart)))
    )
    st.altair_chart(range_chart, use_container_width=True)
    st.caption("Chart values are reference ranges from matched workbook rows, not predicted bills.")


def _field_amount(value: str) -> int | None:
    match = re.search(r"\d[\d,]*", str(value))
    return int(match.group(0).replace(",", "")) if match else None


def _wrap_chart_label(value: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False, break_on_hyphens=False))


def _compact_dashboard_label(value: str) -> str:
    compact = re.split(r"\bfootnote\s*:", str(value), maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if len(compact) > 150:
        compact = compact[:147].rsplit(" ", 1)[0] + "…"
    return _wrap_chart_label(compact, width=38)


def _fee_components(record: BenchmarkRecord) -> list[tuple[str, int, int]]:
    pairs: list[tuple[int, str, int, int]] = []
    for key, value in record.fields.items():
        match = re.fullmatch(r"lower_bound(?:_(\d+))?", key)
        if not match:
            continue
        suffix = match.group(1)
        upper_key = f"upper_bound_{suffix}" if suffix else "upper_bound"
        lower, upper = _field_amount(value), _field_amount(record.fields.get(upper_key, ""))
        if lower is not None and upper is not None:
            pairs.append((int(suffix or 1), key, lower, upper))
    if not pairs:
        lower, upper = estimate_amounts(record)
        return [("Fee benchmark", lower, upper)] if lower is not None and upper is not None else []
    labels = ["Surgeon fee", "Anaesthetist fee"] if "Surg & Ana" in record.sheet else ["Hospital fee"]
    return [
        (labels[index] if index < len(labels) else f"Fee component {index + 1}", lower, upper)
        for index, (_, _, lower, upper) in enumerate(sorted(pairs))
    ]


def render_cost_dashboard(
    fee_matches: list[tuple[BenchmarkRecord, float]],
    stay_matches: list[tuple[BenchmarkRecord, float]],
) -> None:
    """Show each retrieved procedure as an alternative, component-stacked lower/upper scenario."""
    scenarios: list[dict[str, object]] = []
    if stay_matches:
        p25_values = [_field_amount(record.fields.get("p25_bill", "")) for record, _ in stay_matches]
        p75_values = [_field_amount(record.fields.get("p75_bill", "")) for record, _ in stay_matches]
        p25_values, p75_values = [value for value in p25_values if value is not None], [value for value in p75_values if value is not None]
        stay_lower, stay_upper = (min(p25_values) if p25_values else None), (max(p75_values) if p75_values else None)
    else:
        stay_lower = stay_upper = None
    for record, _ in fee_matches[:10]:
        components = _fee_components(record)
        if not components:
            continue
        scenario = first_matching_field(record, ("description", "drg_description", "ccs", "tosp")) or record.sheet
        scenario = f"{scenario} ({record.fields.get('tosp') or record.fields.get('drg') or record.sheet})"
        for label, lower, upper in components:
            scenarios.extend((
                {"scenario": scenario, "estimate": "Lower total", "component": label, "amount": lower},
                {"scenario": scenario, "estimate": "Upper total", "component": label, "amount": upper},
            ))
        if stay_lower is not None and stay_upper is not None:
            scenarios.extend((
                {"scenario": scenario, "estimate": "Lower total", "component": "Hospital stay bill (P25)", "amount": stay_lower},
                {"scenario": scenario, "estimate": "Upper total", "component": "Hospital stay bill (P75)", "amount": stay_upper},
            ))
    if not scenarios:
        st.info("No component ranges were available to build a dashboard.")
        return
    frame = pd.DataFrame(scenarios)
    scenario_order = list(dict.fromkeys(frame["scenario"]))
    scenario_ids = {scenario: str(index + 1) for index, scenario in enumerate(scenario_order)}
    frame["scenario_id"] = frame["scenario"].map(scenario_ids)
    frame["scenario_short"] = frame["scenario"].map(
        lambda value: re.sub(r"\s+", " ", re.split(r"\bfootnote\s*:", str(value), maxsplit=1, flags=re.IGNORECASE)[0]).strip()[:180]
    )
    frame["component_bound"] = frame["component"] + " · " + frame["estimate"]
    component_order = list(dict.fromkeys(frame["component_bound"]))

    def component_color(row: pd.Series) -> str:
        is_lower = row["estimate"] == "Lower total"
        component = row["component"]
        if "stay bill" in component:
            return "#9ECAE1" if is_lower else "#3182BD"
        if "surgeon" in component.lower():
            return "#A9C5F5" if is_lower else "#5B8FF9"
        if "anaesthetist" in component.lower():
            return "#BDEEDC" if is_lower else "#61DDAA"
        if "hospital" in component.lower():
            return "#C7D0DF" if is_lower else "#65789B"
        return "#C9D9F8" if is_lower else "#4E79D9"

    component_colors = [
        component_color(frame.loc[frame["component_bound"] == component].iloc[0])
        for component in component_order
    ]
    maximum_total = frame.groupby(["scenario", "estimate"])["amount"].sum().max() * 1.12
    for scenario in scenario_order:
        scenario_frame = frame[frame["scenario"] == scenario]
        st.markdown(f"#### Alternative {scenario_ids[scenario]}")
        st.write(_compact_dashboard_label(scenario).replace("\n", " "))
        base_chart = alt.Chart()
        bars = (
            base_chart
            .mark_bar()
            .encode(
                y=alt.Y("estimate:N", sort=["Upper total", "Lower total"], title=None, axis=alt.Axis(labelFontSize=12, labelPadding=10)),
                x=alt.X("sum(amount):Q", title="Combined reference amount (SGD)", scale=alt.Scale(domain=[0, maximum_total]), axis=alt.Axis(format=",d")),
                color=alt.Color("component_bound:N", legend=None, scale=alt.Scale(domain=component_order, range=component_colors)),
                tooltip=[
                    alt.Tooltip("scenario_short:N", title="Alternative"),
                    alt.Tooltip("estimate:N", title="Total type"),
                    alt.Tooltip("component:N", title="Component"),
                    alt.Tooltip("amount:Q", title="Amount (SGD)", format=",.0f"),
                ],
            )
            .properties(height=120)
        )
        total_labels = (
            base_chart
            .transform_aggregate(total_amount="sum(amount)", groupby=["estimate"])
            .transform_calculate(total_label="'SGD ' + format(datum.total_amount, ',.0f')")
            .mark_text(align="left", baseline="middle", dx=6, fontSize=11, color="#333333")
            .encode(y=alt.Y("estimate:N", sort=["Upper total", "Lower total"], axis=None), x=alt.X("total_amount:Q"), text="total_label:N")
        )
        st.altair_chart(alt.layer(bars, total_labels, data=scenario_frame), use_container_width=True)
        legend_groups: dict[str, dict[str, str]] = {}
        for component, color in zip(component_order, component_colors):
            component_name, bound = component.rsplit(" · ", 1)
            short_name = re.sub(r"\s*\(P(?:25|75)\)", "", component_name)
            legend_groups.setdefault(short_name, {})[bound] = color
        legend_columns = st.columns(4)
        for column, (name, shades) in zip(legend_columns, legend_groups.items()):
            with column:
                for bound in ("Upper total", "Lower total"):
                    if bound in shades:
                        short_bound = "Upper" if bound == "Upper total" else "Lower"
                        st.markdown(
                            f"<span style='color:{shades[bound]}; font-size:1.25rem'>■</span> {name} · {short_bound}",
                            unsafe_allow_html=True,
                        )
        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    alternatives = pd.DataFrame(
        {"alternative": [scenario_ids[scenario] for scenario in scenario_order], "procedure or condition": scenario_order}
    )
    with st.expander("Alternative procedure names", expanded=False):
        st.dataframe(alternatives, hide_index=True, use_container_width=True)
    totals = frame.groupby(["scenario", "estimate"], as_index=False)["amount"].sum().pivot(
        index="scenario", columns="estimate", values="amount"
    ).reset_index()
    st.dataframe(totals, hide_index=True, use_container_width=True)
    st.caption(
        "Each row is an alternative retrieved procedure/condition scenario. Lower totals use P25 and upper totals use P75 "
        "for retrieved hospital-stay bills. These are assembled reference amounts, not a quote or prediction; confirm which "
        "components apply and request an itemised estimate from the provider."
    )


def render_sources(sources: list[OfficialSource]) -> None:
    with st.expander("Curated sources used", expanded=False):
        if not sources:
            st.write("No source was selected for this turn.")
            return
        for source in sources:
            st.markdown(f"**[{source.title}]({source.url})** — {source.agency} · {source.source_type}")
            st.write(source.summary)
            st.caption(f"Curated source record last reviewed: {source.last_reviewed}")


def render_safety_notice() -> None:
    with st.expander("Safety, privacy, and scope", expanded=False):
        st.write(
            "This educational prototype does not diagnose, prescribe treatment, determine insurance coverage, "
            "or guarantee costs. Do not enter names, identification numbers, contact details, or medical record numbers."
        )
        st.write(
            "For a life-threatening emergency in Singapore, call 995. Discuss clinical decisions and itemised fee "
            "estimates with a licensed clinician and the relevant provider."
        )
