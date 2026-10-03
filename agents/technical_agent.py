from agents.llm import generate_response


# =========================================================
# SolarGrid AI — Technical Engineer Agent
# =========================================================

SYSTEM_PROMPT = """
You are the Technical Engineer Agent of SolarGrid AI.

Your task is to interpret preliminary solar PV engineering
calculations for a renewable-energy project.

IMPORTANT:

1. The numerical calculations are performed by Python.
2. Do not recalculate or alter the supplied numbers.
3. Do not invent site-specific solar irradiation.
4. Do not claim bankable energy yield.
5. Clearly distinguish calculated results from assumptions.
6. Do not claim grid feasibility.
7. Do not claim land availability.
8. Do not claim equipment selection.
9. Flag important information that is still missing.
10. This is preliminary engineering screening only.

Use the following structure:

### Technical Assessment

Summarize the calculated project performance.

### Key Results

Explain the supplied numerical results.

### Assumptions

Clearly identify the assumptions used.

### Engineering Considerations

Identify factors that would need detailed engineering
assessment before a final feasibility study.

### Data Gaps

Identify important project information not currently available.

### Bottom Line

Give a concise preliminary engineering conclusion.

Do not invent numbers.
"""


def calculate_solar_screening(
    capacity_mw,
    capacity_factor_pct=22.0,
    performance_ratio_pct=80.0,
    annual_degradation_pct=0.5,
    project_life_years=25,
):
    """
    Perform deterministic preliminary solar PV calculations.

    Parameters are screening inputs and are not site measurements.
    """

    capacity_mw = float(capacity_mw)
    capacity_factor_pct = float(capacity_factor_pct)
    performance_ratio_pct = float(performance_ratio_pct)
    annual_degradation_pct = float(
        annual_degradation_pct
    )
    project_life_years = int(project_life_years)

    if capacity_mw <= 0:
        raise ValueError(
            "Capacity must be greater than zero."
        )

    if not 0 < capacity_factor_pct <= 100:
        raise ValueError(
            "Capacity factor must be between 0 and 100%."
        )

    if not 0 < performance_ratio_pct <= 100:
        raise ValueError(
            "Performance ratio must be between 0 and 100%."
        )

    if not 0 <= annual_degradation_pct < 100:
        raise ValueError(
            "Annual degradation must be between 0 and 100%."
        )

    if project_life_years <= 0:
        raise ValueError(
            "Project life must be greater than zero."
        )

    hours_per_year = 8760.0

    # Theoretical maximum annual energy.
    maximum_annual_mwh = (
        capacity_mw * hours_per_year
    )

    # Gross energy based on capacity factor.
    gross_annual_mwh = (
        maximum_annual_mwh
        * capacity_factor_pct
        / 100.0
    )

    # Net energy after applying assumed performance ratio.
    net_first_year_mwh = (
        gross_annual_mwh
        * performance_ratio_pct
        / 100.0
    )

    # Equivalent net full-load hours.
    equivalent_full_load_hours = (
        net_first_year_mwh / capacity_mw
    )

    # Build degradation profile.
    annual_generation = []

    for year in range(1, project_life_years + 1):

        degradation_factor = (
            1.0
            - annual_degradation_pct / 100.0
        ) ** (year - 1)

        generation_mwh = (
            net_first_year_mwh
            * degradation_factor
        )

        annual_generation.append(
            {
                "year": year,
                "generation_mwh": generation_mwh,
                "generation_gwh": generation_mwh / 1000.0,
            }
        )

    lifetime_generation_mwh = sum(
        item["generation_mwh"]
        for item in annual_generation
    )

    return {
        "capacity_mw": capacity_mw,
        "capacity_factor_pct": capacity_factor_pct,
        "performance_ratio_pct": performance_ratio_pct,
        "annual_degradation_pct": annual_degradation_pct,
        "project_life_years": project_life_years,
        "maximum_annual_mwh": maximum_annual_mwh,
        "gross_annual_mwh": gross_annual_mwh,
        "net_first_year_mwh": net_first_year_mwh,
        "net_first_year_gwh": (
            net_first_year_mwh / 1000.0
        ),
        "equivalent_full_load_hours": (
            equivalent_full_load_hours
        ),
        "lifetime_generation_mwh": (
            lifetime_generation_mwh
        ),
        "lifetime_generation_gwh": (
            lifetime_generation_mwh / 1000.0
        ),
        "annual_generation": annual_generation,
    }


def analyze_technical_project(
    project,
    capacity_factor_pct=22.0,
    performance_ratio_pct=80.0,
    annual_degradation_pct=0.5,
    project_life_years=25,
):
    """
    Run the Technical Engineer Agent.

    Python performs the calculations.
    GPT-OSS interprets the calculated results.
    """

    capacity = project.get(
        "capacity",
        0,
    )

    results = calculate_solar_screening(
        capacity_mw=capacity,
        capacity_factor_pct=capacity_factor_pct,
        performance_ratio_pct=performance_ratio_pct,
        annual_degradation_pct=annual_degradation_pct,
        project_life_years=project_life_years,
    )

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

    prompt = f"""
PROJECT INPUTS

Project name: {project_name}
Location: {location}
Technology: {technology}
Capacity: {capacity} MW

SCREENING ASSUMPTIONS

Capacity factor: {capacity_factor_pct:.2f}%
Performance ratio: {performance_ratio_pct:.2f}%
Annual degradation: {annual_degradation_pct:.2f}%
Project life: {project_life_years} years

CALCULATED RESULTS

Maximum theoretical annual generation:
{results["maximum_annual_mwh"]:,.1f} MWh/year

Gross annual generation:
{results["gross_annual_mwh"]:,.1f} MWh/year

First-year net generation:
{results["net_first_year_mwh"]:,.1f} MWh/year
({results["net_first_year_gwh"]:,.2f} GWh/year)

Equivalent net full-load hours:
{results["equivalent_full_load_hours"]:,.1f} hours/year

Estimated lifetime generation over:
{project_life_years} years

Lifetime generation:
{results["lifetime_generation_mwh"]:,.1f} MWh
({results["lifetime_generation_gwh"]:,.2f} GWh)

TASK

Interpret these preliminary calculations.

Do not change any supplied numerical values.

Emphasize that capacity factor and performance ratio are
screening assumptions, not measurements from the project site.

Identify the project-specific data that would be required
for a detailed energy-yield assessment.
"""

    try:

        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=1600,
        )

    except Exception as exc:

        answer = (
            "The Technical Engineer calculations were "
            "completed, but GPT-OSS could not generate "
            "the engineering interpretation.\n\n"
            f"Technical error: {exc}"
        )

    return {
        "answer": answer,
        "results": results,
    }


def get_agent_info():
    """Metadata for the multi-agent workflow."""

    return {
        "name": "Technical Engineer Agent",
        "short_name": "Technical Agent",
        "icon": "⚙️",
        "model": "openai/gpt-oss-120b",
        "calculation_engine": "Deterministic Python",
        "status": "ready",
        "description": (
            "Performs preliminary solar PV energy "
            "calculations and engineering interpretation."
        ),
    }
