from agents.llm import generate_response


# =========================================================
# SolarGrid AI — Financial Analyst Agent
# =========================================================

SYSTEM_PROMPT = """
You are the Financial Analyst Agent of SolarGrid AI.

Your task is to interpret preliminary renewable-energy
project financial calculations.

IMPORTANT

1. Financial calculations are performed by Python.
2. Do not change supplied numerical results.
3. Clearly distinguish assumptions from calculated values.
4. Do not claim project bankability.
5. Do not invent financing terms.
6. Do not invent tariffs or market prices.
7. Do not claim investment feasibility.
8. Identify missing commercial and financing information.

Use exactly:

### Financial Assessment

### Key Results

### Assumptions

### Financial Considerations

### Data Gaps

### Bottom Line
"""


def calculate_financial_screening(
    capacity_mw,
    annual_generation_gwh,
    tariff_pkr_per_kwh=25.0,
    capex_million_pkr_per_mw=180.0,
    opex_pct_of_capex=2.0,
    discount_rate_pct=10.0,
    project_life_years=25,
):
    """
    Deterministic preliminary financial screening.

    All economic inputs are assumptions for screening only.
    """

    capacity_mw = float(capacity_mw)
    annual_generation_gwh = float(
        annual_generation_gwh
    )
    tariff_pkr_per_kwh = float(
        tariff_pkr_per_kwh
    )
    capex_million_pkr_per_mw = float(
        capex_million_pkr_per_mw
    )
    opex_pct_of_capex = float(
        opex_pct_of_capex
    )
    discount_rate_pct = float(
        discount_rate_pct
    )
    project_life_years = int(
        project_life_years
    )

    if capacity_mw <= 0:
        raise ValueError(
            "Capacity must be greater than zero."
        )

    if annual_generation_gwh < 0:
        raise ValueError(
            "Annual generation cannot be negative."
        )

    if tariff_pkr_per_kwh < 0:
        raise ValueError(
            "Tariff cannot be negative."
        )

    if capex_million_pkr_per_mw < 0:
        raise ValueError(
            "CAPEX cannot be negative."
        )

    if not 0 <= opex_pct_of_capex <= 100:
        raise ValueError(
            "O&M percentage must be between 0 and 100."
        )

    if not 0 <= discount_rate_pct < 100:
        raise ValueError(
            "Discount rate must be between 0 and 100%."
        )

    if project_life_years <= 0:
        raise ValueError(
            "Project life must be greater than zero."
        )

    # -----------------------------------------------------
    # CAPEX
    # -----------------------------------------------------

    capex_million_pkr = (
        capacity_mw
        * capex_million_pkr_per_mw
    )

    # -----------------------------------------------------
    # Revenue
    #
    # 1 GWh = 1,000,000 kWh
    # -----------------------------------------------------

    annual_revenue_million_pkr = (
        annual_generation_gwh
        * 1_000_000
        * tariff_pkr_per_kwh
        / 1_000_000
    )

    # -----------------------------------------------------
    # O&M
    # -----------------------------------------------------

    annual_opex_million_pkr = (
        capex_million_pkr
        * opex_pct_of_capex
        / 100.0
    )

    # -----------------------------------------------------
    # Operating cash flow
    # -----------------------------------------------------

    annual_cashflow_million_pkr = (
        annual_revenue_million_pkr
        - annual_opex_million_pkr
    )

    # -----------------------------------------------------
    # Simple payback
    # -----------------------------------------------------

    if annual_cashflow_million_pkr > 0:

        simple_payback_years = (
            capex_million_pkr
            / annual_cashflow_million_pkr
        )

    else:

        simple_payback_years = None

    # -----------------------------------------------------
    # NPV
    # -----------------------------------------------------

    discount_rate = (
        discount_rate_pct / 100.0
    )

    npv_million_pkr = -capex_million_pkr

    for year in range(
        1,
        project_life_years + 1,
    ):

        npv_million_pkr += (
            annual_cashflow_million_pkr
            / ((1 + discount_rate) ** year)
        )

    return {
        "capacity_mw": capacity_mw,
        "annual_generation_gwh": annual_generation_gwh,
        "tariff_pkr_per_kwh": tariff_pkr_per_kwh,
        "capex_million_pkr": capex_million_pkr,
        "capex_million_pkr_per_mw": (
            capex_million_pkr_per_mw
        ),
        "opex_pct_of_capex": opex_pct_of_capex,
        "annual_opex_million_pkr": (
            annual_opex_million_pkr
        ),
        "annual_revenue_million_pkr": (
            annual_revenue_million_pkr
        ),
        "annual_cashflow_million_pkr": (
            annual_cashflow_million_pkr
        ),
        "simple_payback_years": (
            simple_payback_years
        ),
        "discount_rate_pct": discount_rate_pct,
        "project_life_years": project_life_years,
        "npv_million_pkr": npv_million_pkr,
    }


def analyze_financial_project(
    project,
    technical_results,
    tariff_pkr_per_kwh=25.0,
    capex_million_pkr_per_mw=180.0,
    opex_pct_of_capex=2.0,
    discount_rate_pct=10.0,
    project_life_years=25,
):
    """
    Run deterministic financial calculations followed
    by GPT-OSS interpretation.
    """

    capacity = project.get(
        "capacity",
        0,
    )

    annual_generation_gwh = technical_results.get(
        "net_first_year_gwh",
        0,
    )

    results = calculate_financial_screening(
        capacity_mw=capacity,
        annual_generation_gwh=annual_generation_gwh,
        tariff_pkr_per_kwh=tariff_pkr_per_kwh,
        capex_million_pkr_per_mw=(
            capex_million_pkr_per_mw
        ),
        opex_pct_of_capex=opex_pct_of_capex,
        discount_rate_pct=discount_rate_pct,
        project_life_years=project_life_years,
    )

    prompt = f"""
PROJECT

Name: {project.get("project_name", "Unnamed Project")}
Location: {project.get("location", "Not specified")}
Technology: {project.get("technology", "Solar PV")}
Capacity: {capacity} MW

TECHNICAL INPUT

First-year net generation:
{annual_generation_gwh:,.2f} GWh

FINANCIAL SCREENING ASSUMPTIONS

Tariff:
{tariff_pkr_per_kwh:,.2f} PKR/kWh

CAPEX:
{capex_million_pkr_per_mw:,.2f} million PKR/MW

O&M:
{opex_pct_of_capex:.2f}% of CAPEX per year

Discount rate:
{discount_rate_pct:.2f}%

Project life:
{project_life_years} years

CALCULATED RESULTS

Total CAPEX:
{results["capex_million_pkr"]:,.2f} million PKR

Annual revenue:
{results["annual_revenue_million_pkr"]:,.2f} million PKR

Annual O&M:
{results["annual_opex_million_pkr"]:,.2f} million PKR

Annual operating cash flow:
{results["annual_cashflow_million_pkr"]:,.2f} million PKR

Simple payback:
{
    f"{results['simple_payback_years']:.2f} years"
    if results["simple_payback_years"] is not None
    else "Not achieved"
}

NPV:
{results["npv_million_pkr"]:,.2f} million PKR

TASK

Interpret these preliminary results.

Make it clear that tariff, CAPEX, O&M and discount rate
are screening assumptions rather than project-specific
commercial terms.

Do not call the project financially viable or bankable.

Identify missing information required for a proper
investment-grade financial model.
"""

    try:

        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=1600,
        )

    except Exception as exc:

        answer = (
            "Financial calculations were completed, "
            "but GPT-OSS could not generate the "
            "financial interpretation.\n\n"
            f"Technical error: {exc}"
        )

    return {
        "answer": answer,
        "results": results,
    }


def get_agent_info():
    return {
        "name": "Financial Analyst Agent",
        "short_name": "Financial Analyst",
        "icon": "💰",
        "model": "openai/gpt-oss-120b",
        "calculation_engine": "Deterministic Python",
        "status": "ready",
        "description": (
            "Performs preliminary renewable-energy "
            "financial screening."
        ),
    }
