from agents.llm import generate_response


# =========================================================
# SolarGrid AI — Project Manager / Synthesis Agent
# =========================================================

SYSTEM_PROMPT = """
You are the Project Manager and Synthesis Agent of SolarGrid AI.

Combine specialist-agent outputs into one concise preliminary
renewable-energy project assessment.

RULES:
- Do not invent facts.
- Do not change calculated numbers.
- Preserve uncertainty.
- Distinguish assumptions from evidence.
- Do not claim feasibility, bankability, approval, compliance,
  or grid feasibility.
- Identify unresolved issues and missing information.

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
"""


def _compact(text, max_chars=1800):
    """
    Keep only a compact portion of an upstream agent response.

    The Project Manager does not need the full verbose output
    from every specialist agent.
    """
    if not text:
        return "No output available."

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars].rsplit(" ", 1)[0]
        + "\n[Further specialist details omitted for token efficiency.]"
    )


def synthesize_project(
    project,
    technical_output=None,
    grid_output=None,
    regulatory_output=None,
    financial_output=None,
    risk_output=None,
):
    """
    Synthesize compact specialist-agent results.

    Numerical calculation dictionaries are preserved, while
    long narrative responses are shortened before being sent
    to GPT-OSS 120B.
    """

    technical_output = technical_output or {}
    grid_output = grid_output or {}
    regulatory_output = regulatory_output or {}
    financial_output = financial_output or {}
    risk_output = risk_output or {}

    project_summary = f"""
Project name: {project.get("project_name", "Unnamed Project")}
Location: {project.get("location", "Not specified")}
Technology: {project.get("technology", "Not specified")}
Capacity: {project.get("capacity", "Not specified")} MW
Grid connection input: {project.get("grid_voltage", "Not specified")}
""".strip()

    technical_results = technical_output.get(
        "results",
        {},
    )

    financial_results = financial_output.get(
        "results",
        {},
    )

    prompt = f"""
PROJECT
{project_summary}

==================================================
TECHNICAL ENGINEER
==================================================

Calculated results:
{technical_results}

Specialist assessment:
{_compact(
    technical_output.get("answer", ""),
    1200,
)}

==================================================
GRID ENGINEER
==================================================

{_compact(
    grid_output.get("answer", ""),
    1500,
)}

==================================================
REGULATORY INTELLIGENCE
==================================================

{_compact(
    regulatory_output.get("answer", ""),
    1700,
)}

==================================================
FINANCIAL ANALYST
==================================================

Calculated results:
{financial_results}

Specialist assessment:
{_compact(
    financial_output.get("answer", ""),
    1200,
)}

==================================================
RISK ANALYST
==================================================

{_compact(
    risk_output.get("answer", ""),
    1600,
)}

==================================================
TASK
==================================================

Create one concise integrated preliminary project assessment.

Use only the information supplied above.

Preserve uncertainty and assumptions.

Do not invent new technical, financial, regulatory or grid facts.

Do not change numerical results.

Identify the most important unresolved issues and next
investigations.
"""

    try:
        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=700,
        )

    except Exception as exc:
        answer = (
            "The Project Manager could not complete "
            "the integrated assessment.\n\n"
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
            "Synthesizes specialist-agent outputs into "
            "a concise preliminary project assessment."
        ),
    }
