import streamlit as st

from rag.rag_engine import LocalRAG


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="SolarGrid AI",
    page_icon="☀️",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background: #0b1220;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(30,41,59,0.85),
            rgba(15,23,42,0.95)
        );
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        color: #aab4c5;
        font-size: 1rem;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(30,41,59,0.55);
        min-height: 120px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    .source-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(15,23,42,0.75);
        margin-bottom: 0.8rem;
    }

    .source-label {
        color: #60a5fa;
        font-weight: 700;
    }

    .source-meta {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .agent-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(30,41,59,0.55);
        text-align: center;
        min-height: 120px;
    }

    .agent-icon {
        font-size: 1.5rem;
    }

    .agent-name {
        font-weight: 600;
        margin-top: 0.4rem;
    }

    .agent-status {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD RAG
# ---------------------------------------------------------

@st.cache_resource
def load_rag():
    return LocalRAG()


rag = load_rag()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## ☀️ SolarGrid AI")

    st.caption(
        "Pakistan Renewable Energy Intelligence "
        "& Preliminary Feasibility Platform"
    )

    st.divider()

    st.markdown("### Project")

    project_name = st.text_input(
        "Project name",
        value="Demo Solar PV Project",
    )

    location = st.text_input(
        "Location",
        value="Pakistan",
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

    st.markdown("### RAG Diagnostics")

    status = rag.status()

    st.metric(
        "Indexed passages",
        status["indexed_passages"],
    )

    st.metric(
        "Indexed sources",
        status["indexed_sources"],
    )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            ☀️ SolarGrid AI
        </div>

        <div class="hero-subtitle">
            Evidence-driven renewable-energy project intelligence
            for Pakistan
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------------

st.markdown("## Project Command Center")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Project</div>
            <div class="metric-value">{project_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Technology</div>
            <div class="metric-value">{technology}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Capacity</div>
            <div class="metric-value">{capacity:g} MW</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Grid Connection</div>
            <div class="metric-value">{grid_voltage}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("")


# ---------------------------------------------------------
# AI WORKFLOW
# ---------------------------------------------------------

st.markdown("## AI Workflow")

agents = [
    ("📥", "Project Intake", "Ready"),
    ("⚙️", "Technical Engineer", "Ready"),
    ("🔌", "Grid Engineer", "Ready"),
    ("💰", "Financial Analyst", "Ready"),
    ("📚", "Regulatory Intelligence", "RAG Ready"),
    ("⚠️", "Risk Analyst", "Ready"),
    ("🧠", "Project Manager", "Ready"),
]

cols = st.columns(len(agents))

for col, (icon, name, status_text) in zip(cols, agents):

    with col:

        st.markdown(
            f"""
            <div class="agent-card">

                <div class="agent-icon">
                    {icon}
                </div>

                <div class="agent-name">
                    {name}
                </div>

                <div class="agent-status">
                    ● {status_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

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

    st.markdown("### Search Pakistan Energy Evidence")

    query = st.text_input(
        "Ask a regulatory or energy-sector question",
        placeholder=(
            "Example: What are the requirements "
            "for connecting a generation facility to the grid?"
        ),
        key="evidence_query",
    )

    search_button = st.button(
        "🔍 Retrieve Evidence",
        type="primary",
    )

    if search_button:

        if not query.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching the knowledge base..."):

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

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-label">
                                [S{i}] {result['title']}
                            </div>

                            <div class="source-meta">
                                {result['authority']}
                                &nbsp; | &nbsp;
                                {result['date']}
                                &nbsp; | &nbsp;
                                Section: {result['section']}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.write(result["text"])

                    st.caption(
                        f"Relevance score: {result['score']}"
                    )

                    if result["url"]:

                        st.markdown(
                            f"[Official source]({result['url']})"
                        )

                    st.divider()


# =========================================================
# RAG TEST LAB
# =========================================================

with tab2:

    st.markdown("### RAG Retrieval Test Lab")

    st.write(
        "Use these questions to verify that the retrieval "
        "layer is finding the correct official evidence."
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
        "Run RAG Test",
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
                    f"{result['section']}"
                ):

                    st.write(result["text"])

                    st.caption(
                        f"Score: {result['score']} | "
                        f"Authority: {result['authority']}"
                    )

                    if result["url"]:
                        st.markdown(
                            f"[Open official source]({result['url']})"
                        )


# =========================================================
# DIAGNOSTICS
# =========================================================

with tab3:

    st.markdown("### Knowledge Base Diagnostics")

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
