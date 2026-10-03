from agents.llm import generate_response


# =========================================================
# SolarGrid AI — Risk Analyst Agent
# =========================================================

SYSTEM_PROMPT = """
You are the Risk Analyst Agent of SolarGrid AI.

Identify important potential risks in a preliminary
renewable-energy project assessment.

RULES:
- Do not invent facts.
- Do not invent probabilities.
- Do not invent monetary impacts.
- Do not claim project failure.
- Treat missing information as an issue requiring investigation.
- Preserve uncertainty from the specialist agents.

Classify risks as:
- Regulatory
- Grid
- Technical
- Financial
- Development / Execution
- Data / Information

Use exactly:

### Risk Overview

### Key Risks

### Risk Priorities

### Missing Information

### Bottom Line
"""


def _compact(text, max_chars=1600):
    """
    Reduce upstream agent text before sending it to the
    Risk Analyst to stay within free-tier token limits.
    """
    if not text:
        return "No output available."

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars].rsplit(" ", 1)[0]
        + "\n[Further details omitted.]"
    )


def analyze_project_risks(
    project,
    technical_output=None,
    grid_output=None,
    regulatory_output=None,
    financial_output=None,
):
    """
    Analyze risks using compact specialist-agent summaries.
    """

    technical_output = technical_output or {}
    grid_output = grid_output or {}
    regulatory_output = regulatory_output or {}
    financial_output = financial_output or {}

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

Name: {project.get("project_name", "Unnamed Project")}
Location: {project.get("location", "Not specified")}
Technology: {project.get("technology", "Not specified")}
Capacity: {project.get("capacity", "Not specified")} MW
Grid input: {project.get("grid_voltage", "Not specified")}

==================================================
TECHNICAL
==================================================

Calculated results:
{technical_results}

Assessment:
{_compact(
    technical_output.get("answer", ""),
    1200,
)}

==================================================
GRID
==================================================

{_compact(
    grid_output.get("answer", ""),
    1400,
)}

==================================================
REGULATORY
==================================================

{_compact(
    regulatory_output.get("answer", ""),
    1500,
)}

==================================================
FINANCIAL
==================================================

Calculated results:
{financial_results}

Assessment:
{_compact(
    financial_output.get("answer", ""),
    1200,
)}

==================================================
TASK
==================================================

Identify the main potential project risks.

Focus on:
- regulatory uncertainty;
- grid-connection uncertainty;
- technical assumptions;
- energy-yield assumptions;
- tariff assumptions;
- CAPEX assumptions;
- missing project information;
- execution dependencies.

Do not invent probabilities or financial losses.

Present risks as issues requiring investigation or management.
"""

    try:
        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=700,
        )

    except Exception as exc:
        answer = (
            "The Risk Analyst could not complete "
            "the assessment.\n\n"
            f"Technical error: {exc}"
        )

    return {
        "answer": answer,
    }


def get_agent_info():
    return {
        "name": "Risk Analyst Agent",
        "short_name": "Risk Analyst",
        "icon": "⚠️",
        "model": "openai/gpt-oss-120b",
        "status": "ready",
        "description": (
            "Identifies technical, grid, regulatory, "
            "financial and execution risks."
        ),
    }
