from agents.llm import generate_response


# =========================================================
# SolarGrid AI — Project Manager / Synthesis Agent
# =========================================================

SYSTEM_PROMPT = """
You are the Project Manager and Synthesis Agent of SolarGrid AI.

Your job is to combine the outputs of the Technical Engineer,
Grid Engineer, Financial Analyst, Regulatory Intelligence Agent,
and Risk Analyst into one coherent preliminary project assessment.

You are NOT a decision-maker.

You must not claim that a project is:
- approved
- feasible
- bankable
- financially viable
- grid-feasible
- legally compliant

unless the supplied agent outputs explicitly establish such a fact.

IMPORTANT RULES

1. Preserve uncertainty.
2. Do not invent missing information.
3. Do not change numerical results.
4. Distinguish calculations from assumptions.
5. Distinguish evidence-backed findings from interpretation.
6. Highlight unresolved issues.
7. Treat this as a preliminary screening assessment.

Use exactly:

### Executive Summary

### Technical Position

### Grid Position

### Regulatory Position

### Financial Position

### Key Risks

### Critical Data Gaps

### Recommended Next Investigations

### Preliminary Project Status

For Preliminary Project Status use only:

- Evidence available
- Further investigation required

Do not use a numerical score.
Do not rank the project against other projects.
Do not declare the project feasible or infeasible.
"""


def synthesize_project(
    project,
    technical_output=None,
    grid_output=None,
    regulatory_output=None,
    financial_output=None,
    risk_output=None,
):
    """
    Combine the outputs of the project agents into one
    evidence-aware preliminary assessment.
    """

    technical_output = technical_output or {}
    grid_output = grid_output or {}
    regulatory_output = regulatory_output or {}
    financial_output = financial_output or {}
    risk_output = risk_output or {}

    project_name = project.get(
        "project_name",
        "Unnamed Project",
    )

    prompt = f"""
PROJECT

Name: {project_name}
Location: {project.get("location", "Not specified")}
Technology: {project.get("technology", "Not specified")}
Capacity: {project.get("capacity", "Not specified")} MW
Grid voltage input: {project.get("grid_voltage", "Not specified")}

============================================================
TECHNICAL ENGINEER
============================================================

{technical_output.get("answer", "")}

CALCULATED TECHNICAL RESULTS

{technical_output.get("results", {})}

============================================================
GRID ENGINEER
============================================================

{grid_output.get("answer", "")}

============================================================
REGULATORY INTELLIGENCE
============================================================

{regulatory_output.get("answer", "")}

============================================================
FINANCIAL ANALYST
============================================================

{financial_output.get("answer", "")}

CALCULATED FINANCIAL RESULTS

{financial_output.get("results", {})}

============================================================
RISK ANALYST
============================================================

{risk_output.get("answer", "")}

============================================================
TASK
============================================================

Prepare a concise integrated preliminary project assessment.

Do not introduce any new facts.

Preserve the assumptions and uncertainty contained in the
individual agent outputs.

Make it clear which findings are:
- calculated;
- evidence-backed;
- assumptions;
- unresolved.

Identify the most important next investigations.

The final assessment is a preliminary screening output,
not a formal feasibility study.
"""

    try:

        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=2200,
        )

    except Exception as exc:

        answer = (
            "The Project Manager could not complete the "
            "integrated assessment.\n\n"
            f"Technical error: {exc}"
        )

    return {
        "answer": answer,
    }


def get_agent_info():
    return {
        "name": "Project Manager / Synthesis Agent",
        "short_name": "Project Manager",
        "icon": "🧠",
        "model": "openai/gpt-oss-120b",
        "status": "ready",
        "description": (
            "Synthesizes the outputs of all specialist "
            "agents into a preliminary project assessment."
        ),
    }
