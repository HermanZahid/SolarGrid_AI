import streamlit as st

from rag.rag_engine import LocalRAG


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SolarGrid AI",
    page_icon="☀️",
    layout="wide",
)


# =========================================================
# GLOBAL STYLING
# =========================================================

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

    .source-box {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        background: rgba(30, 41, 59, 0.35);
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD RAG
# =========================================================

@st.cache_resource
def load_rag():
    return LocalRAG()


rag = load_rag()


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

    st.subheader("RAG Status")

    status = rag.status()

    st.metric(
        "Indexed passages",
        status["indexed_passages"],
    )

    st.metric(
        "Indexed sources",
        status["indexed_sources"],
    )


# =========================================================
# HEADER
# =========================================================

st.title("☀️ SolarGrid AI")

st.caption(
    "Evidence-driven renewable-energy project intelligence "
    "for Pakistan"
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


# =========================================================
# AI WORKFLOW
# =========================================================

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
    ("📚", "Regulatory Intelligence", "RAG Ready"),
    ("⚠️", "Risk Analyst", "Ready"),
    ("🧠", "Project Manager", "Ready"),
]

cols = st.columns(len(workflow))

for col, (icon, name, agent_status) in zip(cols, workflow):

    with col:

        st.info(
            f"{icon}\n\n"
            f"**{name}**\n\n"
            f"_{agent_status}_"
        )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔎 Evidence Center",
        "🧪 RAG Test Lab",
        "📊 Diagnostics",
    ]
)


# =========================================================
# EVIDENCE CENTER
# =========================================================

with tab1:

    st.subheader("Search Pakistan Energy Evidence")

    query = st.text_input(
        "Ask a regulatory or energy-sector question",
        placeholder=(
            "Example: What are the requirements "
            "for connecting a generation facility to the grid?"
        ),
    )

    if st.button(
        "🔍 Retrieve Evidence",
        type="primary",
    ):

        if not query.strip():

            st.warning("Please enter a question.")

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

                for i, result in enumerate(results, start=1):

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

                        st.write(result["text"])

                        if result["url"]:

                            st.markdown(
                                f"[📄 Open official source]({result['url']})"
                            )


# =========================================================
# RAG TEST LAB
# =========================================================

with tab2:

    st.subheader("RAG Retrieval Test Lab")

    st.write(
        "These tests allow us to verify retrieval before "
        "connecting the Groq LLM."
    )

    test_questions = [
        "What are the requirements for connecting a generation facility to the grid?",
        "What technical code must a generation facility comply with?",
        "What is the 25 kW threshold in the 2026 prosumer amendment?",
        "What is the National Electricity Plan 2023-27?",
        "What does Pakistan's Fast Track Solar PV initiative cover?",
    ]

    selected_question = st.selectbox(
        "Select a test question",
        test_questions,
    )

    if st.button(
        "▶ Run RAG Test",
        type="primary",
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

            for i, result in enumerate(results, start=1):

                with st.expander(
                    f"[S{i}] {result['title']} — "
                    f"{result['section']}",
                    expanded=(i == 1),
                ):

                    st.write(result["text"])

                    st.caption(
                        f"Relevance score: {result['score']}"
                    )

                    if result["url"]:

                        st.markdown(
                            f"[📄 Open official source]({result['url']})"
                        )


# =========================================================
# DIAGNOSTICS
# =========================================================

with tab3:

    st.subheader("Knowledge Base Diagnostics")

    st.json(status)

    st.markdown("### Indexed Sources")

    indexed_titles = sorted(
        {
            document["title"]
            for document in rag.documents
            if document.get("title")
        }
    )

    for title in indexed_titles:

        st.write(f"✓ {title}")

    st.markdown("### Indexed Passages")

    for document in rag.documents:

        st.caption(
            f"{document['title']} → "
            f"{document['section']}"
        )
