import streamlit as st

from rag.rag_engine import LocalRAG
from agents.regulatory_agent import (
    retrieve_regulatory_evidence,
    generate_regulatory_assessment,
)
from agents.technical_agent import (
    analyze_technical_project,
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
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 15px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD RAG
# =========================================================

@st.cache_resource
def load_rag(version="rag-v3"):
    return LocalRAG()


rag = load_rag()


# =========================================================
# SESSION STATE
# =========================================================

if "regulatory_result" not in st.session_state:
    st.session_state.regulatory_result = None

if "regulatory_evidence" not in st.session_state:
    st.session_state.regulatory_evidence = []

if "regulatory_stage" not in st.session_state:
    st.session_state.regulatory_stage = "Ready"

if "technical_result" not in st.session_state:
    st.session_state.technical_result = None


# =========================================================
# HELPERS
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


def clear_regulatory_result():
    st.session_state.regulatory_result = None
    st.session_state.regulatory_evidence = []
    st.session_state.regulatory_stage = "Ready"


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
        "Location",
        "Pakistan",
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
        "Grid Connection",
        st.session_state.grid_voltage,
    )


# =========================================================
# AI WORKFLOW
# =========================================================

st.header("AI Workflow")

st.caption(
    "Multi-agent architecture for renewable-energy "
    "project intelligence"
)

workflow = [
    ("📥", "Project Intake", "Active"),
    ("⚙️", "Technical Engineer", "Ready"),
    ("🔌", "Grid Engineer", "Planned"),
    ("💰", "Financial Analyst", "Planned"),
    (
        "📚",
        "Regulatory Intelligence",
        st.session_state.regulatory_stage,
    ),
    ("⚠️", "Risk Analyst", "Planned"),
    ("🧠", "Project Manager", "Planned"),
]

cols = st.columns(
    len(workflow)
)

for col, (
    icon,
    name,
    agent_status,
) in zip(
    cols,
    workflow,
):

    with col:

        st.info(
            f"{icon}\n\n"
            f"**{name}**\n\n"
            f"_{agent_status}_"
        )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔎 Evidence Center",
        "🤖 Regulatory AI",
        "⚙️ Technical Engineer",
        "🧪 RAG Test Lab",
        "📊 Diagnostics",
    ]
)


# =========================================================
# TAB 1 — EVIDENCE CENTER
# =========================================================

with tab1:

    st.subheader(
        "Search Pakistan Energy Evidence"
    )

    st.caption(
        "Retrieve source-grounded evidence from "
        "the SolarGrid AI knowledge base."
    )

    query = st.text_input(
        "Ask a regulatory or energy-sector question",
        placeholder=(
            "Example: What are the requirements "
            "for connecting a generation facility "
            "to the grid?"
        ),
        key="evidence_query",
    )

    if st.button(
        "🔍 Retrieve Evidence",
        type="primary",
        key="retrieve_evidence",
    ):

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the Pakistan energy "
                "knowledge base..."
            ):

                results = rag.search(
                    query,
                    k=5,
                )

            if not results:

                st.warning(
                    "No relevant evidence was found."
                )

            else:

                st.success(
                    f"Retrieved {len(results)} "
                    "relevant evidence passages."
                )

                render_evidence(
                    results
                )


# =========================================================
# TAB 2 — REGULATORY AI
# =========================================================

with tab2:

    st.subheader(
        "🤖 Regulatory Intelligence Agent"
    )

    st.caption(
        "GPT-OSS 120B analyzes retrieved "
        "Pakistan-specific regulatory evidence."
    )

    question = st.text_area(
        "Regulatory question",
        placeholder=(
            "Example: What regulatory requirements "
            "should a 50 MW solar PV project consider "
            "when connecting to the Pakistani grid?"
        ),
        height=110,
        key="regulatory_question",
    )

    st.markdown(
        "### Current Project Context"
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric(
            "Technology",
            st.session_state.technology,
        )

    with p2:
        st.metric(
            "Capacity",
            f"{st.session_state.capacity:g} MW",
        )

    with p3:
        st.metric(
            "Voltage",
            st.session_state.grid_voltage,
        )

    with p4:
        st.metric(
            "Location",
            st.session_state.location,
        )

    st.divider()

    b1, b2 = st.columns(
        [3, 1]
    )

    with b1:
        run_agent = st.button(
            "🚀 Run Regulatory Intelligence Agent",
            type="primary",
            key="run_regulatory_agent",
        )

    with b2:
        clear_result = st.button(
            "↻ Clear Result",
            key="clear_regulatory_result",
        )

    if clear_result:

        clear_regulatory_result()
        st.rerun()

    if run_agent:

        if not question.strip():

            st.warning(
                "Please enter a regulatory question."
            )

        elif not groq_configured:

            st.error(
                "GROQ_API_KEY is not configured."
            )

        else:

            project = get_project()

            st.session_state.regulatory_result = None
            st.session_state.regulatory_evidence = []

            with st.status(
                "Running Regulatory Intelligence Agent...",
                expanded=True,
            ) as workflow_status:

                st.session_state.regulatory_stage = (
                    "Retrieving evidence"
                )

                st.write(
                    "🔎 **Stage 1/3 — "
                    "Retrieving regulatory evidence...**"
                )

                retrieval = (
                    retrieve_regulatory_evidence(
                        query=question,
                        rag=rag,
                        k=5,
                    )
                )

                evidence = retrieval["evidence"]
                context = retrieval["context"]

                st.session_state.regulatory_evidence = (
                    evidence
                )

                if not evidence:

                    st.session_state.regulatory_stage = (
                        "No evidence"
                    )

                    workflow_status.update(
                        label=(
                            "No regulatory evidence found"
                        ),
                        state="error",
                    )

                else:

                    st.write(
                        f"✅ Retrieved {len(evidence)} "
                        "evidence passages."
                    )

                    st.session_state.regulatory_stage = (
                        "Analyzing evidence"
                    )

                    st.write(
                        "🧠 **Stage 2/3 — "
                        "GPT-OSS 120B analyzing evidence...**"
                    )

                    answer = (
                        generate_regulatory_assessment(
                            query=question,
                            project=project,
                            context=context,
                            evidence=evidence,
                            max_tokens=1800,
                        )
                    )

                    st.write(
                        "✅ Evidence analysis completed."
                    )

                    st.session_state.regulatory_stage = (
                        "Completed"
                    )

                    st.write(
                        "📝 **Stage 3/3 — "
                        "Preparing final assessment...**"
                    )

                    st.session_state.regulatory_result = (
                        answer
                    )

                    workflow_status.update(
                        label="Regulatory workflow completed",
                        state="complete",
                    )

            if st.session_state.regulatory_result:

                st.markdown(
                    "## Regulatory Assessment"
                )

                st.markdown(
                    st.session_state.regulatory_result
                )

                st.divider()

                st.subheader(
                    "📚 Evidence Used by the Agent"
                )

                render_evidence(
                    st.session_state.regulatory_evidence
                )

    elif st.session_state.regulatory_result:

        st.markdown(
            "## Regulatory Assessment"
        )

        st.markdown(
            st.session_state.regulatory_result
        )

        st.divider()

        st.subheader(
            "📚 Evidence Used by the Agent"
        )

        render_evidence(
            st.session_state.regulatory_evidence
        )


# =========================================================
# TAB 3 — TECHNICAL ENGINEER
# =========================================================

with tab3:

    st.subheader(
        "⚙️ Technical Engineer Agent"
    )

    st.caption(
        "Preliminary solar PV energy screening using "
        "deterministic engineering calculations."
    )

    st.info(
        "The values below are screening assumptions, "
        "not site measurements or a bankable energy-yield study."
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        capacity_factor = st.number_input(
            "Capacity Factor (%)",
            min_value=1.0,
            max_value=60.0,
            value=22.0,
            step=0.5,
            key="capacity_factor",
        )

    with a2:

        performance_ratio = st.number_input(
            "Performance Ratio (%)",
            min_value=1.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
            key="performance_ratio",
        )

    with a3:

        degradation = st.number_input(
            "Annual Degradation (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            key="degradation",
        )

    with a4:

        project_life = st.number_input(
            "Project Life (years)",
            min_value=1,
            max_value=40,
            value=25,
            step=1,
            key="project_life",
        )

    st.divider()

    st.markdown(
        "### Project Inputs"
    )

    t1, t2, t3 = st.columns(3)

    with t1:
        st.metric(
            "Technology",
            st.session_state.technology,
        )

    with t2:
        st.metric(
            "Capacity",
            f"{st.session_state.capacity:g} MW",
        )

    with t3:
        st.metric(
            "Location",
            st.session_state.location,
        )

    if st.button(
        "⚙️ Run Technical Engineer",
        type="primary",
        key="run_technical_agent",
    ):

        project = get_project()

        with st.status(
            "Running Technical Engineer Agent...",
            expanded=True,
        ) as technical_status:

            st.write(
                "⚙️ **Stage 1/2 — "
                "Running deterministic engineering calculations...**"
            )

            technical_output = (
                analyze_technical_project(
                    project=project,
                    capacity_factor_pct=capacity_factor,
                    performance_ratio_pct=performance_ratio,
                    annual_degradation_pct=degradation,
                    project_life_years=project_life,
                )
            )

            results = technical_output["results"]

            st.write(
                "✅ Engineering calculations completed."
            )

            st.write(
                "🧠 **Stage 2/2 — "
                "GPT-OSS 120B interpreting technical results...**"
            )

            st.write(
                "✅ Technical interpretation completed."
            )

            technical_status.update(
                label="Technical Engineer completed",
                state="complete",
            )

        st.session_state.technical_result = (
            technical_output
        )

    if st.session_state.technical_result:

        output = (
            st.session_state.technical_result
        )

        results = output["results"]

        st.markdown(
            "### Preliminary Technical Results"
        )

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric(
                "Gross Annual Energy",
                f"{results['gross_annual_mwh'] / 1000:,.2f} GWh",
            )

        with r2:
            st.metric(
                "Year 1 Net Energy",
                f"{results['net_first_year_gwh']:,.2f} GWh",
            )

        with r3:
            st.metric(
                "Full-Load Hours",
                f"{results['equivalent_full_load_hours']:,.0f} h",
            )

        with r4:
            st.metric(
                "25-Year Energy",
                f"{results['lifetime_generation_gwh']:,.1f} GWh",
            )

        st.divider()

        st.markdown(
            "### AI Technical Assessment"
        )

        st.markdown(
            output["answer"]
        )

        st.divider()

        st.markdown(
            "### Calculation Assumptions"
        )

        st.write(
            f"Capacity factor: "
            f"{results['capacity_factor_pct']:.1f}%"
        )

        st.write(
            f"Performance ratio: "
            f"{results['performance_ratio_pct']:.1f}%"
        )

        st.write(
            f"Annual degradation: "
            f"{results['annual_degradation_pct']:.1f}%"
        )

        st.write(
            f"Project life: "
            f"{results['project_life_years']} years"
        )


# =========================================================
# TAB 4 — RAG TEST LAB
# =========================================================

with tab4:

    st.subheader(
        "RAG Retrieval Test Lab"
    )

    st.caption(
        "Verify evidence retrieval across different "
        "Pakistan energy topics."
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
# TAB 5 — DIAGNOSTICS
# =========================================================

with tab5:

    st.subheader(
        "Knowledge Base Diagnostics"
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

    st.markdown(
        "### Indexed Evidence Passages"
    )

    for document in rag.documents:

        title = document.get(
            "title",
            "Unknown source",
        )

        section = document.get(
            "section",
            "General",
        )

        authority = document.get(
            "authority",
            "",
        )

        label = (
            f"{title} → {section}"
        )

        if authority:
            label += (
                f" | {authority}"
            )

        st.caption(
            label
        )
