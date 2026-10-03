import streamlit as st

from rag.rag_engine import LocalRAG

from agents.technical_agent import (
    analyze_technical_project,
)

from agents.regulatory_agent import (
    retrieve_regulatory_evidence,
    generate_regulatory_assessment,
)

from agents.grid_agent import (
    analyze_grid_project,
)

from agents.financial_agent import (
    analyze_financial_project,
)

from agents.risk_agent import (
    analyze_project_risks,
)

from agents.project_manager import (
    synthesize_project,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SolarGrid AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# BASIC STYLING
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 14px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD RAG
# =========================================================

@st.cache_resource
def load_rag(version="rag-v6"):
    return LocalRAG()


rag = load_rag()


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_AGENT_STATES = {
    "technical": "Ready",
    "regulatory": "Ready",
    "grid": "Ready",
    "financial": "Ready",
    "risk": "Ready",
    "project_manager": "Ready",
}

for key, value in DEFAULT_AGENT_STATES.items():

    if f"agent_{key}" not in st.session_state:
        st.session_state[f"agent_{key}"] = value


if "technical_result" not in st.session_state:
    st.session_state.technical_result = None

if "regulatory_result" not in st.session_state:
    st.session_state.regulatory_result = None

if "regulatory_evidence" not in st.session_state:
    st.session_state.regulatory_evidence = []

if "grid_result" not in st.session_state:
    st.session_state.grid_result = None

if "financial_result" not in st.session_state:
    st.session_state.financial_result = None

if "risk_result" not in st.session_state:
    st.session_state.risk_result = None

if "project_manager_result" not in st.session_state:
    st.session_state.project_manager_result = None

if "workflow_complete" not in st.session_state:
    st.session_state.workflow_complete = False

if "activity_log" not in st.session_state:
    st.session_state.activity_log = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def check_groq_configuration():
    try:
        return bool(
            st.secrets.get("GROQ_API_KEY")
        )
    except Exception:
        return False


def get_project():

    return {
        "project_name": st.session_state.project_name,
        "location": st.session_state.location,
        "technology": st.session_state.technology,
        "capacity": st.session_state.capacity,
        "grid_voltage": st.session_state.grid_voltage,
    }


def reset_workflow():

    for key in DEFAULT_AGENT_STATES:

        st.session_state[
            f"agent_{key}"
        ] = "Ready"

    st.session_state.technical_result = None
    st.session_state.regulatory_result = None
    st.session_state.regulatory_evidence = []
    st.session_state.grid_result = None
    st.session_state.financial_result = None
    st.session_state.risk_result = None
    st.session_state.project_manager_result = None
    st.session_state.workflow_complete = False
    st.session_state.activity_log = []


def add_log(message):

    st.session_state.activity_log.append(
        message
    )


def render_evidence(
    evidence,
    expanded_first=True,
):

    if not evidence:

        st.warning(
            "No evidence was retrieved."
        )

        return

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        title = item.get(
            "title",
            "Unknown source",
        )

        section = item.get(
            "section",
            "General",
        )

        authority = item.get(
            "authority",
            "",
        )

        date = item.get(
            "date",
            "",
        )

        score = item.get(
            "score",
            "",
        )

        url = item.get(
            "url",
            "",
        )

        with st.expander(
            f"[S{i}] {title} — {section}",
            expanded=(
                expanded_first and i == 1
            ),
        ):

            metadata = []

            if authority:
                metadata.append(
                    authority
                )

            if date:
                metadata.append(
                    date
                )

            if score != "":
                metadata.append(
                    f"Relevance: {score}"
                )

            if metadata:

                st.caption(
                    " | ".join(metadata)
                )

            st.caption(
                "Evidence type: "
                + item.get(
                    "evidence_type",
                    "curated source summary",
                )
            )

            st.write(
                item.get(
                    "text",
                    "",
                )
            )

            if url:

                st.markdown(
                    "[📄 Open official source]"
                    f"({url})"
                )


def render_agent_card(
    icon,
    name,
    status,
):

    # Fixed-height native container keeps all cards
    # visually aligned without custom HTML.
    with st.container(
        height=150,
        border=True,
    ):

        st.markdown(
            f"### {icon}"
        )

        st.markdown(
            f"**{name}**"
        )

        if status == "Completed":

            st.success(
                "✓ Completed",
                icon="✅",
            )

        elif status == "Running":

            st.warning(
                "Running",
                icon="🔄",
            )

        elif status == "Error":

            st.error(
                "Error",
                icon="⚠️",
            )

        elif status == "Active":

            st.info(
                "Active",
                icon="📌",
            )

        else:

            st.caption(
                f"• {status}"
            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("☀️ SolarGrid AI")

    st.caption(
        "Pakistan Renewable Energy Intelligence "
        "& Preliminary Feasibility Platform"
    )

    st.divider()

    st.subheader("Project")

    st.text_input(
        "Project name",
        "Demo Solar PV Project",
        key="project_name",
    )

    st.text_input(
        "City / Project Location",
        "Lahore, Pakistan",
        key="location",
    )

    st.selectbox(
        "Technology",
        [
            "Solar PV",
            "Wind",
            "Hydropower",
            "Battery Energy Storage",
            "Hybrid Renewable Energy",
        ],
        key="technology",
    )

    st.number_input(
        "Capacity (MW)",
        min_value=0.1,
        value=50.0,
        step=1.0,
        key="capacity",
    )

    st.selectbox(
        "Grid connection voltage",
        [
            "11 kV",
            "33 kV",
            "66 kV",
            "132 kV",
            "220 kV",
            "Other",
        ],
        index=3,
        key="grid_voltage",
    )

    st.divider()

    st.subheader("System Status")

    status = rag.status()

    st.metric(
        "Indexed passages",
        status["indexed_passages"],
    )

    st.metric(
        "Indexed sources",
        status["indexed_sources"],
    )

    groq_configured = check_groq_configuration()

    if groq_configured:

        st.success(
            "🟢 Groq API configured"
        )

    else:

        st.warning(
            "🟡 Groq API key not configured"
        )

    st.caption(
        "LLM: openai/gpt-oss-120b"
    )

    # -----------------------------------------------------
    # Model assumptions / data sources
    # -----------------------------------------------------

    with st.expander(
        "📐 Model Assumptions & Data Sources"
    ):

        st.markdown(
            "#### ⚙️ Technical"
        )

        st.write(
            "**PVGIS-derived:**"
        )

        st.write(
            "• Annual PV energy"
        )

        st.write(
            "• Effective capacity factor"
        )

        st.write(
            "• Specific yield"
        )

        st.write(
            "• Equivalent full-load hours"
        )

        st.write(
            "• Location coordinates"
        )

        st.write(
            "**Fixed screening assumptions:**"
        )

        st.write(
            "• PV system loss: 14%"
        )

        st.write(
            "• Annual degradation: 0.5%"
        )

        st.write(
            "• Project life: 25 years"
        )

        st.write(
            "• Fallback capacity factor: 22%"
        )

        st.write(
            "• Fallback performance ratio: 80%"
        )

        st.markdown(
            "#### 💰 Financial"
        )

        st.write(
            "**Fixed screening assumptions:**"
        )

        st.write(
            "• Tariff: 25 PKR/kWh"
        )

        st.write(
            "• CAPEX: 180 M PKR/MW"
        )

        st.write(
            "• O&M: 2% of CAPEX/year"
        )

        st.write(
            "• Discount rate: 10%"
        )

        st.write(
            "• Project life: 25 years"
        )

        st.caption(
            "Fixed values are illustrative screening assumptions, "
            "not project-specific commercial offers or measurements."
        )

    st.divider()

    if st.button(
        "↻ Reset Project Analysis",
        key="reset_workflow",
        use_container_width=True,
    ):

        reset_workflow()

        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.title("☀️ SolarGrid AI")

st.caption(
    "Evidence-driven renewable-energy project "
    "intelligence for Pakistan"
)

st.divider()


# =========================================================
# PROJECT COMMAND CENTER
# =========================================================

st.header("Project Command Center")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Project",
        st.session_state.project_name,
    )

with c2:

    st.metric(
        "Technology",
        st.session_state.technology,
    )

with c3:

    st.metric(
        "Capacity",
        f"{st.session_state.capacity:g} MW",
    )

with c4:

    st.metric(
        "Location",
        st.session_state.location,
    )


# =========================================================
# AI WORKFLOW
# =========================================================

st.header("AI Workflow")

st.caption(
    "Sequential multi-agent preliminary project assessment"
)

workflow = [
    (
        "📥",
        "Project Intake",
        "Active",
    ),

    (
        "⚙️",
        "Technical Engineer",
        st.session_state.agent_technical,
    ),

    (
        "📚",
        "Regulatory Intelligence",
        st.session_state.agent_regulatory,
    ),

    (
        "🔌",
        "Grid Engineer",
        st.session_state.agent_grid,
    ),

    (
        "💰",
        "Financial Analyst",
        st.session_state.agent_financial,
    ),

    (
        "⚠️",
        "Risk Analyst",
        st.session_state.agent_risk,
    ),

    (
        "🧠",
        "Project Manager",
        st.session_state.agent_project_manager,
    ),
]

workflow_cols = st.columns(
    len(workflow)
)

for column, (
    icon,
    name,
    agent_status,
) in zip(
    workflow_cols,
    workflow,
):

    with column:

        render_agent_card(
            icon,
            name,
            agent_status,
        )


# =========================================================
# MAIN ACTION
# =========================================================

st.divider()

run_col, clear_col = st.columns(
    [4, 1]
)

with run_col:

    run_full_workflow = st.button(
        "🚀 Run Full SolarGrid AI Feasibility Screening",
        type="primary",
        use_container_width=True,
    )

with clear_col:

    if st.button(
        "Clear",
        use_container_width=True,
    ):

        reset_workflow()

        st.rerun()


# =========================================================
# FULL MULTI-AGENT WORKFLOW
# =========================================================

if run_full_workflow:

    if not groq_configured:

        st.error(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets before "
            "running the full workflow."
        )

    else:

        reset_workflow()

        project = get_project()

        regulatory_question = (
            f"What regulatory requirements should be "
            f"considered for a {project['capacity']} MW "
            f"{project['technology']} project in "
            f"{project['location']} with a proposed "
            f"{project['grid_voltage']} grid connection?"
        )

        with st.status(
            "Running SolarGrid AI multi-agent workflow...",
            expanded=True,
        ) as workflow_status:

            # =============================================
            # 1 — TECHNICAL
            # =============================================

            st.session_state.agent_technical = "Running"

            st.write(
                "⚙️ **1/6 — Technical Engineer**"
            )

            st.write(
                "Resolving project location and "
                "estimating PV production..."
            )

            try:

                technical_output = (
                    analyze_technical_project(
                        project=project,
                        capacity_factor_pct=22.0,
                        performance_ratio_pct=80.0,
                        annual_degradation_pct=0.5,
                        project_life_years=25,
                    )
                )

                st.session_state.technical_result = (
                    technical_output
                )

                st.session_state.agent_technical = (
                    "Completed"
                )

                add_log(
                    "Technical Engineer completed"
                )

                technical_results = (
                    technical_output.get(
                        "results",
                        {},
                    )
                )

                if technical_results.get(
                    "data_source"
                ) == (
                    "PVGIS 5.3 location-based estimate"
                ):

                    st.write(
                        "📍 PVGIS location resolved: "
                        f"{technical_results.get('location_name', project['location'])}"
                    )

                    st.write(
                        "✅ Location-based PV generation "
                        "estimate completed."
                    )

                else:

                    st.write(
                        "⚠️ PVGIS unavailable; "
                        "fallback screening assumptions used."
                    )

            except Exception as exc:

                technical_output = {
                    "answer": (
                        "Technical Agent error: "
                        f"{exc}"
                    ),
                    "results": {},
                }

                st.session_state.technical_result = (
                    technical_output
                )

                st.session_state.agent_technical = (
                    "Error"
                )

                st.write(
                    f"❌ Technical Agent error: {exc}"
                )


            # =============================================
            # 2 — REGULATORY
            # =============================================

            st.session_state.agent_regulatory = (
                "Running"
            )

            st.write(
                "📚 **2/6 — Regulatory Intelligence**"
            )

            st.write(
                "Retrieving Pakistan-specific "
                "regulatory evidence..."
            )

            try:

                retrieval = (
                    retrieve_regulatory_evidence(
                        query=regulatory_question,
                        rag=rag,
                        k=5,
                    )
                )

                regulatory_evidence = retrieval.get(
                    "evidence",
                    [],
                )

                regulatory_context = retrieval.get(
                    "context",
                    "",
                )

                st.session_state.regulatory_evidence = (
                    regulatory_evidence
                )

                if regulatory_evidence:

                    st.write(
                        f"✅ Retrieved "
                        f"{len(regulatory_evidence)} "
                        "regulatory evidence passages."
                    )

                    regulatory_answer = (
                        generate_regulatory_assessment(
                            query=regulatory_question,
                            project=project,
                            context=regulatory_context,
                            evidence=regulatory_evidence,
                            max_tokens=1800,
                        )
                    )

                    regulatory_output = {
                        "answer": regulatory_answer,
                        "evidence": regulatory_evidence,
                    }

                    st.session_state.regulatory_result = (
                        regulatory_output
                    )

                    st.session_state.agent_regulatory = (
                        "Completed"
                    )

                    add_log(
                        "Regulatory Intelligence completed"
                    )

                    st.write(
                        "✅ Regulatory assessment completed."
                    )

                else:

                    regulatory_output = {
                        "answer": (
                            "No sufficient regulatory "
                            "evidence was retrieved."
                        ),
                        "evidence": [],
                    }

                    st.session_state.regulatory_result = (
                        regulatory_output
                    )

                    st.session_state.agent_regulatory = (
                        "Error"
                    )

                    st.write(
                        "❌ No regulatory evidence found."
                    )

            except Exception as exc:

                regulatory_output = {
                    "answer": (
                        "Regulatory Agent error: "
                        f"{exc}"
                    ),
                    "evidence": [],
                }

                st.session_state.regulatory_result = (
                    regulatory_output
                )

                st.session_state.agent_regulatory = (
                    "Error"
                )

                st.write(
                    f"❌ Regulatory Agent error: {exc}"
                )


            # =============================================
            # 3 — GRID
            # =============================================

            st.session_state.agent_grid = "Running"

            st.write(
                "🔌 **3/6 — Grid Integration Engineer**"
            )

            st.write(
                "Performing preliminary grid-integration "
                "screening..."
            )

            try:

                grid_output = (
                    analyze_grid_project(
                        project=project,
                        rag=rag,
                    )
                )

                st.session_state.grid_result = (
                    grid_output
                )

                st.session_state.agent_grid = (
                    "Completed"
                )

                add_log(
                    "Grid Engineer completed"
                )

                st.write(
                    "✅ Grid screening completed."
                )

            except Exception as exc:

                grid_output = {
                    "answer": (
                        "Grid Agent error: "
                        f"{exc}"
                    ),
                    "evidence": [],
                }

                st.session_state.grid_result = (
                    grid_output
                )

                st.session_state.agent_grid = (
                    "Error"
                )

                st.write(
                    f"❌ Grid Agent error: {exc}"
                )


            # =============================================
            # 4 — FINANCIAL
            # =============================================

            st.session_state.agent_financial = (
                "Running"
            )

            st.write(
                "💰 **4/6 — Financial Analyst**"
            )

            st.write(
                "Using Technical Engineer energy "
                "results for financial screening..."
            )

            try:

                technical_results = (
                    technical_output.get(
                        "results",
                        {},
                    )
                )

                financial_output = (
                    analyze_financial_project(
                        project=project,
                        technical_results=technical_results,
                        tariff_pkr_per_kwh=25.0,
                        capex_million_pkr_per_mw=180.0,
                        opex_pct_of_capex=2.0,
                        discount_rate_pct=10.0,
                        project_life_years=25,
                    )
                )

                st.session_state.financial_result = (
                    financial_output
                )

                st.session_state.agent_financial = (
                    "Completed"
                )

                add_log(
                    "Financial Analyst completed"
                )

                st.write(
                    "✅ Financial screening completed."
                )

            except Exception as exc:

                financial_output = {
                    "answer": (
                        "Financial Agent error: "
                        f"{exc}"
                    ),
                    "results": {},
                }

                st.session_state.financial_result = (
                    financial_output
                )

                st.session_state.agent_financial = (
                    "Error"
                )

                st.write(
                    f"❌ Financial Agent error: {exc}"
                )


            # =============================================
            # 5 — RISK
            # =============================================

            st.session_state.agent_risk = "Running"

            st.write(
                "⚠️ **5/6 — Risk Analyst**"
            )

            st.write(
                "Cross-checking technical, grid, "
                "regulatory and financial outputs..."
            )

            try:

                risk_output = (
                    analyze_project_risks(
                        project=project,
                        technical_output=technical_output,
                        grid_output=grid_output,
                        regulatory_output=regulatory_output,
                        financial_output=financial_output,
                    )
                )

                st.session_state.risk_result = (
                    risk_output
                )

                st.session_state.agent_risk = (
                    "Completed"
                )

                add_log(
                    "Risk Analyst completed"
                )

                st.write(
                    "✅ Risk analysis completed."
                )

            except Exception as exc:

                risk_output = {
                    "answer": (
                        "Risk Agent error: "
                        f"{exc}"
                    ),
                }

                st.session_state.risk_result = (
                    risk_output
                )

                st.session_state.agent_risk = (
                    "Error"
                )

                st.write(
                    f"❌ Risk Agent error: {exc}"
                )


            # =============================================
            # 6 — PROJECT MANAGER
            # =============================================

            st.session_state.agent_project_manager = (
                "Running"
            )

            st.write(
                "🧠 **6/6 — Project Manager**"
            )

            st.write(
                "Synthesizing specialist-agent outputs..."
            )

            try:

                project_manager_output = (
                    synthesize_project(
                        project=project,
                        technical_output=technical_output,
                        grid_output=grid_output,
                        regulatory_output=regulatory_output,
                        financial_output=financial_output,
                        risk_output=risk_output,
                    )
                )

                st.session_state.project_manager_result = (
                    project_manager_output
                )

                st.session_state.agent_project_manager = (
                    "Completed"
                )

                add_log(
                    "Project Manager synthesis completed"
                )

                st.session_state.workflow_complete = (
                    True
                )

                st.write(
                    "✅ Integrated project assessment completed."
                )

                workflow_status.update(
                    label=(
                        "SolarGrid AI workflow completed"
                    ),
                    state="complete",
                )

            except Exception as exc:

                project_manager_output = {
                    "answer": (
                        "Project Manager error: "
                        f"{exc}"
                    ),
                }

                st.session_state.project_manager_result = (
                    project_manager_output
                )

                st.session_state.agent_project_manager = (
                    "Error"
                )

                st.session_state.workflow_complete = (
                    False
                )

                st.write(
                    f"❌ Project Manager error: {exc}"
                )

                workflow_status.update(
                    label=(
                        "SolarGrid AI workflow completed "
                        "with errors"
                    ),
                    state="error",
                )


# =========================================================
# RESULT TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Feasibility Dashboard",
        "🧠 AI Report",
        "📚 Regulatory Evidence",
        "⚙️ Technical",
        "🔎 RAG Test Lab",
        "📋 Diagnostics",
    ]
)


# =========================================================
# TAB 1 — FEASIBILITY DASHBOARD
# =========================================================

with tab1:

    st.subheader(
        "📊 Preliminary Feasibility Dashboard"
    )

    if not st.session_state.workflow_complete:

        st.info(
            "Run the full SolarGrid AI workflow "
            "to populate the project assessment."
        )

    else:

        technical_output = (
            st.session_state.technical_result
            or {}
        )

        technical_results = (
            technical_output.get(
                "results",
                {},
            )
        )

        financial_output = (
            st.session_state.financial_result
            or {}
        )

        financial_results = (
            financial_output.get(
                "results",
                {},
            )
        )

        # -------------------------------------------------
        # Main KPIs
        # -------------------------------------------------

        st.markdown(
            "### Key Results"
        )

        d1, d2, d3, d4 = st.columns(4)

        with d1:

            st.metric(
                "Year-1 Net Energy",
                f"{technical_results.get('net_first_year_gwh', 0):,.2f} GWh",
            )

        with d2:

            st.metric(
                "Lifetime Energy",
                f"{technical_results.get('lifetime_generation_gwh', 0):,.1f} GWh",
            )

        with d3:

            payback = financial_results.get(
                "simple_payback_years"
            )

            st.metric(
                "Simple Payback",
                (
                    f"{payback:.2f} yrs"
                    if payback is not None
                    else "Not achieved"
                ),
            )

        with d4:

            st.metric(
                "NPV",
                f"{financial_results.get('npv_million_pkr', 0):,.1f} M PKR",
            )

        st.divider()

        # -------------------------------------------------
        # Energy model provenance
        # -------------------------------------------------

        st.markdown(
            "### Energy Estimate Basis"
        )

        data_source = technical_results.get(
            "data_source",
            "Unknown",
        )

        if data_source == (
            "PVGIS 5.3 location-based estimate"
        ):

            st.success(
                "📍 Production estimate sourced from PVGIS 5.3"
            )

            e1, e2, e3, e4 = st.columns(4)

            with e1:

                st.metric(
                    "Energy Model",
                    "PVGIS 5.3",
                )

            with e2:

                st.metric(
                    "Location",
                    technical_results.get(
                        "location_name",
                        st.session_state.location,
                    ),
                )

            with e3:

                st.metric(
                    "Specific Yield",
                    f"{technical_results.get('specific_yield_kwh_per_kwp', 0):,.0f} kWh/kWp",
                )

            with e4:

                st.metric(
                    "Effective Capacity Factor",
                    f"{technical_results.get('capacity_factor_pct', 0):.1f}%",
                )

            st.caption(
                "PVGIS-derived values: annual energy, specific yield, "
                "equivalent full-load hours and effective capacity factor."
            )

        else:

            st.warning(
                "PVGIS was unavailable or the location could not "
                "be resolved. The Technical Engineer used fixed "
                "screening assumptions instead."
            )

        st.divider()

        # -------------------------------------------------
        # Fixed technical assumptions
        # -------------------------------------------------

        st.markdown(
            "### Fixed Technical Screening Assumptions"
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:

            st.metric(
                "System Loss",
                "14%",
            )

        with a2:

            st.metric(
                "Annual Degradation",
                "0.5%",
            )

        with a3:

            st.metric(
                "Project Life",
                "25 years",
            )

        with a4:

            st.metric(
                "Fallback CF",
                "22%",
            )

        st.caption(
            "The fallback 22% capacity factor is used only when "
            "PVGIS cannot provide a location-based estimate."
        )

        st.divider()

        # -------------------------------------------------
        # Agent completion
        # -------------------------------------------------

        st.markdown(
            "### Agent Completion"
        )

        statuses = [
            (
                "⚙️",
                "Technical",
                st.session_state.agent_technical,
            ),

            (
                "📚",
                "Regulatory",
                st.session_state.agent_regulatory,
            ),

            (
                "🔌",
                "Grid",
                st.session_state.agent_grid,
            ),

            (
                "💰",
                "Financial",
                st.session_state.agent_financial,
            ),

            (
                "⚠️",
                "Risk",
                st.session_state.agent_risk,
            ),

            (
                "🧠",
                "Project Manager",
                st.session_state.agent_project_manager,
            ),
        ]

        status_cols = st.columns(
            len(statuses)
        )

        for col, (
            icon,
            name,
            status_text,
        ) in zip(
            status_cols,
            statuses,
        ):

            with col:

                if status_text == "Completed":

                    st.success(
                        f"{icon} {name}\n\n"
                        "Completed"
                    )

                elif status_text == "Error":

                    st.error(
                        f"{icon} {name}\n\n"
                        "Error"
                    )

                else:

                    st.info(
                        f"{icon} {name}\n\n"
                        f"{status_text}"
                    )

        st.divider()

        # -------------------------------------------------
        # Financial assumptions
        # -------------------------------------------------

        st.markdown(
            "### Fixed Financial Screening Assumptions"
        )

        f1, f2, f3, f4, f5 = st.columns(5)

        with f1:

            st.metric(
                "Tariff",
                "25 PKR/kWh",
            )

        with f2:

            st.metric(
                "CAPEX",
                "180 M PKR/MW",
            )

        with f3:

            st.metric(
                "O&M",
                "2% / year",
            )

        with f4:

            st.metric(
                "Discount Rate",
                "10%",
            )

        with f5:

            st.metric(
                "Project Life",
                "25 years",
            )

        st.caption(
            "These are fixed illustrative screening assumptions. "
            "They are not project-specific commercial terms."
        )


# =========================================================
# TAB 2 — AI REPORT
# =========================================================

with tab2:

    st.subheader(
        "🧠 Integrated AI Project Assessment"
    )

    if (
        st.session_state.project_manager_result
        is None
    ):

        st.info(
            "Run the full workflow to generate "
            "the integrated AI report."
        )

    else:

        report = (
            st.session_state
            .project_manager_result
            .get(
                "answer",
                "",
            )
        )

        st.markdown(
            report
        )


# =========================================================
# TAB 3 — REGULATORY EVIDENCE
# =========================================================

with tab3:

    st.subheader(
        "📚 Regulatory Intelligence Evidence"
    )

    if st.session_state.regulatory_evidence:

        render_evidence(
            st.session_state.regulatory_evidence
        )

        st.divider()

        if st.session_state.regulatory_result:

            st.markdown(
                "### Regulatory Agent Assessment"
            )

            st.markdown(
                st.session_state
                .regulatory_result
                .get(
                    "answer",
                    "",
                )
            )

    else:

        st.info(
            "No regulatory evidence has been retrieved yet."
        )


# =========================================================
# TAB 4 — TECHNICAL
# =========================================================

with tab4:

    st.subheader(
        "⚙️ Technical Engineer Results"
    )

    if st.session_state.technical_result is None:

        st.info(
            "Run the full workflow to generate "
            "technical results."
        )

    else:

        output = (
            st.session_state.technical_result
        )

        results = output.get(
            "results",
            {},
        )

        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        t1, t2, t3, t4 = st.columns(4)

        with t1:

            st.metric(
                "Annual PV Energy",
                f"{results.get('net_first_year_gwh', 0):,.2f} GWh",
            )

        with t2:

            st.metric(
                "Full-Load Hours",
                f"{results.get('equivalent_full_load_hours', 0):,.0f} h",
            )

        with t3:

            st.metric(
                "Specific Yield",
                (
                    f"{results.get('specific_yield_kwh_per_kwp', 0):,.0f} kWh/kWp"
                    if results.get(
                        "specific_yield_kwh_per_kwp"
                    ) is not None
                    else "N/A"
                ),
            )

        with t4:

            st.metric(
                "Lifetime Energy",
                f"{results.get('lifetime_generation_gwh', 0):,.1f} GWh",
            )

        st.divider()

        # -------------------------------------------------
        # Data source
        # -------------------------------------------------

        if results.get(
            "data_source"
        ) == (
            "PVGIS 5.3 location-based estimate"
        ):

            st.success(
                "📍 Energy production is based on a PVGIS 5.3 "
                "location-specific estimate."
            )

            l1, l2, l3, l4 = st.columns(4)

            with l1:

                st.metric(
                    "Resolved Location",
                    results.get(
                        "location_name",
                        "",
                    ),
                )

            with l2:

                st.metric(
                    "Latitude",
                    f"{results.get('latitude', 0):.4f}",
                )

            with l3:

                st.metric(
                    "Longitude",
                    f"{results.get('longitude', 0):.4f}",
                )

            with l4:

                st.metric(
                    "System Loss",
                    f"{results.get('system_loss_pct', 0):.1f}%",
                )

            st.caption(
                f"PVGIS specific yield: "
                f"{results.get('specific_yield_kwh_per_kwp', 0):,.1f} "
                f"kWh/kWp/year"
            )

            st.caption(
                f"PVGIS effective capacity factor: "
                f"{results.get('capacity_factor_pct', 0):.2f}%"
            )

            st.caption(
                "PVGIS configuration: crystalline-silicon PV, "
                "free-standing fixed mounting, optimized angle, "
                "with the configured 14% system-loss assumption."
            )

        else:

            st.warning(
                results.get(
                    "location_note",
                    "Location-specific PVGIS data was not available.",
                )
            )

            st.caption(
                "Fallback assumptions: 22% capacity factor, "
                "80% performance ratio, 0.5% annual degradation, "
                "25-year project life."
            )

        st.divider()

        # -------------------------------------------------
        # AI technical assessment
        # -------------------------------------------------

        st.markdown(
            "### AI Technical Assessment"
        )

        st.markdown(
            output.get(
                "answer",
                "",
            )
        )

        st.divider()

        # -------------------------------------------------
        # Assumption ledger
        # -------------------------------------------------

        st.markdown(
            "### Technical Assumption Ledger"
        )

        if results.get(
            "data_source"
        ) == (
            "PVGIS 5.3 location-based estimate"
        ):

            st.success(
                "PVGIS-derived"
            )

            st.write(
                "• Annual PV energy"
            )

            st.write(
                "• Specific yield"
            )

            st.write(
                "• Effective capacity factor"
            )

            st.write(
                "• Equivalent full-load hours"
            )

            st.write(
                "• Project coordinates"
            )

            st.warning(
                "Fixed assumptions"
            )

            st.write(
                "• System loss = 14%"
            )

            st.write(
                "• Annual degradation = 0.5%"
            )

            st.write(
                "• Project life = 25 years"
            )

        else:

            st.warning(
                "Fallback fixed assumptions"
            )

            st.write(
                "• Capacity factor = 22%"
            )

            st.write(
                "• Performance ratio = 80%"
            )

            st.write(
                "• Annual degradation = 0.5%"
            )

            st.write(
                "• Project life = 25 years"
            )


# =========================================================
# TAB 5 — RAG TEST LAB
# =========================================================

with tab5:

    st.subheader(
        "🔎 RAG Retrieval Test Lab"
    )

    st.caption(
        "Test retrieval across different Pakistan "
        "renewable-energy topics."
    )

    test_questions = [
        (
            "What are the requirements for connecting "
            "a generation facility to the grid?"
        ),

        (
            "What is the 25 kW threshold in the "
            "2026 prosumer amendment?"
        ),

        (
            "What does Pakistan's Fast Track Solar "
            "PV framework cover?"
        ),

        (
            "What is the National Electricity Plan "
            "2023-27?"
        ),

        (
            "What information is needed before "
            "determining whether a solar project "
            "can connect to the grid?"
        ),
    ]

    selected_question = st.selectbox(
        "Select a test question",
        test_questions,
    )

    if st.button(
        "▶ Run RAG Test",
        type="primary",
        key="run_rag_test",
    ):

        with st.spinner(
            "Running local RAG retrieval..."
        ):

            results = rag.search(
                selected_question,
                k=3,
            )

        if not results:

            st.error(
                "RAG returned no evidence."
            )

        else:

            st.success(
                f"RAG retrieved {len(results)} passages."
            )

            render_evidence(
                results
            )


# =========================================================
# TAB 6 — DIAGNOSTICS
# =========================================================

with tab6:

    st.subheader(
        "📋 Diagnostics"
    )

    st.markdown(
        "### Knowledge Base"
    )

    st.json(
        status
    )

    st.markdown(
        "### Indexed Sources"
    )

    indexed_titles = sorted(
        {
            document.get(
                "title",
                "",
            )
            for document in rag.documents
            if document.get("title")
        }
    )

    for title in indexed_titles:

        st.write(
            f"✓ {title}"
        )

    st.divider()

    st.markdown(
        "### Agent Activity Log"
    )

    if st.session_state.activity_log:

        for item in st.session_state.activity_log:

            st.write(
                f"✓ {item}"
            )

    else:

        st.caption(
            "No full workflow has been executed yet."
        )

    st.divider()

    st.markdown(
        "### Agent Status"
    )

    st.json(
        {
            "technical": st.session_state.agent_technical,
            "regulatory": st.session_state.agent_regulatory,
            "grid": st.session_state.agent_grid,
            "financial": st.session_state.agent_financial,
            "risk": st.session_state.agent_risk,
            "project_manager": (
                st.session_state.agent_project_manager
            ),
        }
    )
