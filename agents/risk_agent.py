from agents.llm import generate_response


# =========================================================
# SolarGrid AI — Risk Analyst Agent
# =========================================================

SYSTEM_PROMPT = """
You are the Risk Analyst Agent of SolarGrid AI.

Your task is to identify and structure risks for a preliminary
renewable-energy project assessment in Pakistan.

You receive outputs from other SolarGrid AI agents.

IMPORTANT RULES

1. Do not invent facts.
2. Do not invent regulatory requirements.
3. Do not claim that a risk has definitely occurred.
4. Distinguish between:
   - Confirmed information
   - Potential risk
   - Missing information
5. Do not assign financial losses or probabilities unless
   they are explicitly supplied.
6. Do not claim project approval, feasibility or bankability.
7. Use conservative professional language.
8. A missing data item should be treated as a risk requiring
   investigation, not as proof of project failure.

Classify risks under:

- Regulatory
- Grid
- Technical
- Financial
- Development / Execution
- Data / Information

Use exactly these sections:

### Risk Overview

### Key Risks

For each important risk provide:
- Risk
- Category
- Why it matters
- Current evidence
- What needs to be investigated

### Risk Priorities

Separate issues requiring:
- Immediate investigation
- Further investigation
- Routine monitoring

### Missing Information

### Bottom Line

Do not invent risk probabilities or monetary impacts.
"""


def analyze_project_risks(
    project,
    technical_output=None,
    grid_output=None,
    regulatory_output=None,
    financial_output=None,
):
    """
    Analyze project risks using outputs from the existing agents.

    The Risk Agent does not perform new engineering or financial
    calculations. It synthesizes information and identifies
    potential risk areas.
    """

    technical_output = technical_output or {}
    grid_output = grid_output or {}
    regulatory_output = regulatory_output or {}
    financial_output = financial_output or {}

    technical_results = technical_output.get(
        "results",
        {},
    )

    technical_answer = technical_output.get(
        "answer",
        "",
    )

    grid_answer = grid_output.get(
        "answer",
        "",
    )

    regulatory_answer = regulatory_output.get(
        "answer",
        "",
    )

    financial_results = financial_output.get(
        "results",
        {},
    )

    financial_answer = financial_output.get(
        "answer",
        "",
    )

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
TECHNICAL AGENT OUTPUT
============================================================

{technical_answer}

TECHNICAL CALCULATIONS

{technical_results}

============================================================
GRID AGENT OUTPUT
============================================================

{grid_answer}

============================================================
REGULATORY AGENT OUTPUT
============================================================

{regulatory_answer}

============================================================
FINANCIAL AGENT OUTPUT
============================================================

{financial_answer}

FINANCIAL CALCULATIONS

{financial_results}

============================================================
TASK
============================================================

Identify the main potential risks revealed by the available
agent outputs.

Do not create new facts.

Pay particular attention to:

- regulatory applicability;
- grid-connection uncertainty;
- technical assumptions;
- energy-yield assumptions;
- tariff and CAPEX assumptions;
- missing project data;
- development and execution dependencies.

Do not assign probabilities.

Do not state that a project will fail.

Present risks as issues requiring investigation or management.
"""

    try:

        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=1800,
        )

    except Exception as exc:

        answer = (
            "The Risk Analyst could not complete the assessment.\n\n"
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
            "Identifies and structures technical, grid, "
            "regulatory, financial and execution risks."
        ),
    }
