import streamlit as st

from rag.rag_engine import LocalRAG
from agents.regulatory_agent import analyze_regulatory_question


st.set_page_config(
    page_title="SolarGrid AI",
    page_icon="☀️",
    layout="wide",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 15px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
    }

    .agent-status {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load RAG
# ---------------------------------------------------------

@st.cache_resource
def load_rag():
    return LocalRAG()


rag = load_rag()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.title("☀️ SolarGrid AI")

    st.caption(
        "Pakistan Renewable Energy Intelligence "
        "& Preliminary Feasibility Platform"
    )

    st.divider()

    st.subheader("Project")

    project_name = st.text_input(
        "Project name",
        "Demo Solar PV Project",
    )

    location = st.text_input(
        "Location",
        "Pakistan",
    )

    technology = st.selectbox(
        "Technology",
        [
            "Solar PV",
            "Wind",
            "Hydropower",
            "Battery Energy Storage",
            "Hybrid Renewable Energy",
        ],
    )

    capacity = st.number_input(
        "Capacity (MW)",
        min_value=0.1,
        value=50.0,
        step=1.0,
    )

    grid_voltage = st.selectbox(
        "Grid connection voltage",
        [
            "11 kV",
            "33 kV",
            "66 kV",
            "132 kV",
            "220 kV",
            "Other",
        ],
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

    # Check whether Groq secret is available.
    try:
        groq_configured = "GROQ_API_KEY" in st.secrets
    except Exception:
        groq_configured = False

    if groq_configured:
        st.success("🟢 Groq API configured")
    else:
        st.warning("🟡 Groq API key not configured")

    st.caption(
        "LLM: openai/gpt-oss-120b"
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("☀️ SolarGrid AI")

st.caption(
    "Evidence-driven renewable-energy project intelligence "
    "for Pakistan"
)

st.divider()


# ---------------------------------------------------------
# Project Command Center
# ---------------------------------------------------------

st.header("Project Command Center")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Project",
        project_name,
    )

with c2:
    st.metric(
        "Technology",
        technology,
    )

with c3:
    st.metric(
        "Capacity",
        f"{capacity:g} MW",
    )

with c4:
    st.metric(
        "Grid Connection",
        grid_voltage,
    )


# ---------------------------------------------------------
# AI Workflow
# ---------------------------------------------------------

st.header("AI Workflow")

st.caption(
    "Multi-agent architecture for renewable-energy "
    "project intelligence"
)

workflow = [
    ("📥", "Project Intake", "Ready"),
    ("⚙️", "Technical Engineer", "Ready"),
    ("🔌", "Grid Engineer", "Ready"),
    ("💰", "Financial Analyst", "Ready"),
    ("📚", "Regulatory Intelligence", "GPT-OSS + RAG"),
    ("⚠️", "Risk Analyst", "Ready"),
    ("🧠", "Project Manager", "Ready"),
]

cols = st.columns(len(workflow))

for col, (icon, name, agent_status) in zip(
    cols,
    workflow,
):

    with col:

        st.info(
            f"{icon}\n\n"
            f"**{name}**\n\n"
            f"_{agent_status}_"
        )


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔎 Evidence Center",
        "🤖 Regulatory AI",
        "🧪 RAG Test Lab",
        "📊 Diagnostics",
    ]
)


# =========================================================
# TAB 1 — Evidence Center
# =========================================================

with tab1:

    st.subheader(
        "Search Pakistan Energy Evidence"
    )

    query = st.text_input(
        "Ask a regulatory or energy-sector question",
        placeholder=(
            "Example: What are the requirements "
            "for connecting a generation facility to the grid?"
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
                "Searching the Pakistan energy knowledge base..."
            ):

                results = rag.search(
                    query,
                    k=5,
                )

            if not results:

                st.warning(
                    "No relevant evidence was found "
                    "in the current knowledge base."
                )

            else:

                st.success(
                    f"Retrieved {len(results)} relevant evidence passages."
                )

                for i, result in enumerate(
                    results,
                    start=1,
                ):

                    with st.expander(
                        f"[S{i}] {result['title']} — "
                        f"{result['section']}",
                        expanded=(i == 1),
                    ):

                        st.caption(
                            f"{result['authority']} | "
                            f"{result['date']} | "
                            f"Relevance: {result['score']}"
                        )

                        st.write(
                            result["text"]
                        )

                        if result["url"]:

                            st.markdown(
                                "[📄 Open official source]"
                                f"({result['url']})"
                            )


# =========================================================
# TAB 2 — Regulatory AI
# =========================================================

with tab2:

    st.subheader(
        "🤖 Regulatory Intelligence Agent"
    )

    st.caption(
        "GPT-OSS 120B analyzes retrieved Pakistan-specific "
        "regulatory evidence and cites the evidence used."
    )

    question = st.text_area(
        "Regulatory question",
        placeholder=(
            "Example: What regulatory requirements should "
            "a 50 MW solar PV project consider when connecting "
            "to the Pakistani grid?"
        ),
        height=100,
        key="regulatory_question",
    )

    st.markdown("### Current Project Context")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric(
            "Technology",
            technology,
        )

    with p2:
        st.metric(
            "Capacity",
            f"{capacity:g} MW",
        )

    with p3:
        st.metric(
            "Voltage",
            grid_voltage,
        )

    with p4:
        st.metric(
            "Location",
            location,
        )

    st.divider()

    if st.button(
        "🚀 Run Regulatory Intelligence Agent",
        type="primary",
        key="run_regulatory_agent",
    ):

        if not question.strip():

            st.warning(
                "Please enter a regulatory question."
            )

        elif not groq_configured:

            st.error(
                "GROQ_API_KEY is not configured. "
                "Add it to Streamlit Community Cloud Secrets "
                "before running the AI agent."
            )

        else:

            project = {
                "project_name": project_name,
                "location": location,
                "technology": technology,
                "capacity": capacity,
                "grid_voltage": grid_voltage,
            }

            # ---------------------------------------------
            # Stage 1 — Retrieval
            # ---------------------------------------------

            with st.status(
                "Running SolarGrid AI Regulatory Agent...",
                expanded=True,
            ) as agent_status:

                st.write(
                    "🔎 Retrieving relevant Pakistan regulatory evidence..."
                )

                result = analyze_regulatory_question(
                    query=question,
                    rag=rag,
                    project=project,
                )

                evidence = result["evidence"]

                st.write(
                    f"📚 Retrieved {len(evidence)} evidence passages."
                )

                st.write(
                    "🧠 Sending evidence to GPT-OSS 120B..."
                )

                st.write(
                    "📝 Generating evidence-backed regulatory assessment..."
                )

                agent_status.update(
                    label="Regulatory Agent completed",
                    state="complete",
                )

            # ---------------------------------------------
            # Result
            # ---------------------------------------------

            st.success(
                "Regulatory Intelligence Agent completed."
            )

            st.markdown("## Regulatory Assessment")

            st.markdown(
                result["answer"]
            )

            # ---------------------------------------------
            # Evidence Used
            # ---------------------------------------------

            st.divider()

            st.subheader(
                "📚 Evidence Used by the Agent"
            )

            if evidence:

                for i, item in enumerate(
                    evidence,
                    start=1,
                ):

                    with st.expander(
                        f"[S{i}] {item['title']} — "
                        f"{item['section']}",
                        expanded=(i == 1),
                    ):

                        st.caption(
                            f"{item['authority']} | "
                            f"{item['date']} | "
                            f"Relevance: {item['score']}"
                        )

                        st.write(
                            item["text"]
                        )

                        if item["url"]:

                            st.markdown(
                                "[📄 Open official source]"
                                f"({item['url']})"
                            )

            else:

                st.warning(
                    "No evidence was retrieved."
                )


# =========================================================
# TAB 3 — RAG Test Lab
# =========================================================

with tab3:

    st.subheader(
        "RAG Retrieval Test Lab"
    )

    st.write(
        "These tests allow us to verify retrieval before "
        "connecting additional AI agents."
    )

    test_questions = [
        (
            "What are the requirements for connecting "
            "a generation facility to the grid?"
        ),
        (
            "What technical code must a generation "
            "facility comply with?"
        ),
        (
            "What is the 25 kW threshold in the "
            "2026 prosumer amendment?"
        ),
        (
            "What is the National Electricity Plan 2023-27?"
        ),
        (
            "What does Pakistan's Fast Track Solar PV "
            "initiative cover?"
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

        results = rag.search(
            selected_question,
            k=3,
        )

        if not results:

            st.error(
                "RAG returned no evidence for this question."
            )

        else:

            st.success(
                f"RAG retrieved {len(results)} passages."
            )

            for i, result in enumerate(
                results,
                start=1,
            ):

                with st.expander(
                    f"[S{i}] {result['title']} — "
                    f"{result['section']}",
                    expanded=(i == 1),
                ):

                    st.write(
                        result["text"]
                    )

                    st.caption(
                        f"Relevance score: {result['score']}"
                    )

                    if result["url"]:

                        st.markdown(
                            "[📄 Open official source]"
                            f"({result['url']})"
                        )


# =========================================================
# TAB 4 — Diagnostics
# =========================================================

with tab4:

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
            document["title"]
            for document in rag.documents
            if document.get("title")
        }
    )

    for title in indexed_titles:

        st.write(
            f"✓ {title}"
        )

    st.markdown(
        "### Indexed Passages"
    )

    for document in rag.documents:

        st.caption(
            f"{document['title']} → "
            f"{document['section']}"
        )
