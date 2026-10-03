from agents.llm import generate_response

from rag.rag_engine import LocalRAG


# =========================================================
# SolarGrid AI — Grid Engineer Agent
# =========================================================

SYSTEM_PROMPT = """
You are the Grid Integration Engineer Agent of SolarGrid AI.

Your task is to perform a preliminary grid-integration screening
for a renewable-energy project in Pakistan.

IMPORTANT RULES

1. Use the supplied project inputs exactly as provided.
2. Use retrieved regulatory evidence when making regulatory claims.
3. Never invent grid capacity, fault levels, voltage limits,
   line ratings, short-circuit levels, stability results,
   protection settings, or connection approvals.
4. Never claim that a project is grid-feasible.
5. Never claim that a particular connection voltage is legally
   required unless the supplied evidence establishes it.
6. Distinguish clearly between:
   - Project Input
   - Evidence-Backed Finding
   - Engineering Investigation
   - Missing Data
7. Preliminary screening is NOT a grid study.

Use exactly these sections:

### Grid Screening

### Evidence-Backed Findings

### Engineering Assessment

### Investigation Items

### Missing Data

### Bottom Line
"""


def retrieve_grid_evidence(
    project,
    rag,
    k=5,
):
    """
    Retrieve evidence relevant to grid integration.
    """

    technology = project.get(
        "technology",
        "Solar PV",
    )

    capacity = project.get(
        "capacity",
        0,
    )

    voltage = project.get(
        "grid_voltage",
        "Not specified",
    )

    query = (
        f"grid connection requirements for a "
        f"{capacity} MW {technology} generation facility "
        f"connected at {voltage}"
    )

    context, evidence = rag.get_context(
        query,
        k=k,
    )

    return {
        "query": query,
        "context": context,
        "evidence": evidence,
    }


def analyze_grid_project(
    project,
    rag,
):
    """
    Perform preliminary grid integration screening.

    RAG supplies regulatory evidence.
    GPT-OSS interprets the evidence and project inputs.
    """

    retrieval = retrieve_grid_evidence(
        project=project,
        rag=rag,
        k=5,
    )

    evidence = retrieval["evidence"]
    context = retrieval["context"]

    if not evidence:
        return {
            "answer": (
                "The current SolarGrid AI knowledge base "
                "does not provide sufficient evidence to "
                "perform a specific grid-integration assessment."
            ),
            "evidence": [],
            "query": retrieval["query"],
        }

    project_name = project.get(
        "project_name",
        "Unnamed Project",
    )

    location = project.get(
        "location",
        "Not specified",
    )

    technology = project.get(
        "technology",
        "Solar PV",
    )

    capacity = project.get(
        "capacity",
        0,
    )

    voltage = project.get(
        "grid_voltage",
        "Not specified",
    )

    prompt = f"""
PROJECT INPUTS

Project name: {project_name}
Location: {location}
Technology: {technology}
Capacity: {capacity} MW
User-entered grid connection voltage: {voltage}

IMPORTANT:
The project fields above are user inputs.
They are not regulatory evidence.

RETRIEVED EVIDENCE

{context}

TASK

Perform a preliminary grid-integration screening.

Identify what the supplied regulatory evidence establishes.

Then relate that evidence to the project inputs without
inventing project-specific facts.

Do not assume that the user-entered voltage has been approved
or confirmed by a transmission or distribution licensee.

Do not invent load-flow, short-circuit, stability, protection,
fault-level or network-capacity results.

Where detailed studies or project information would be needed,
place those items under Investigation Items or Missing Data.

Cite regulatory findings with [S1], [S2], [S3] as appropriate.
"""

    try:

        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=1600,
        )

    except Exception as exc:

        answer = (
            "Grid screening could not be completed.\n\n"
            f"Technical error: {exc}"
        )

    return {
        "answer": answer,
        "evidence": evidence,
        "query": retrieval["query"],
    }


def get_agent_info():
    return {
        "name": "Grid Integration Engineer Agent",
        "short_name": "Grid Engineer",
        "icon": "🔌",
        "model": "openai/gpt-oss-120b",
        "retrieval": "Local TF-IDF RAG",
        "status": "ready",
        "description": (
            "Performs preliminary grid-integration "
            "screening using project inputs and "
            "Pakistan-specific regulatory evidence."
        ),
    }
