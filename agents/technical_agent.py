import json
import urllib.parse
import urllib.request

import streamlit as st

from agents.llm import generate_response


# =========================================================
# SolarGrid AI — Technical Engineer Agent
# =========================================================

SYSTEM_PROMPT = """
You are the Technical Engineer Agent of SolarGrid AI.

Your task is to interpret preliminary solar PV energy estimates
for a renewable-energy project.

IMPORTANT:

1. Numerical calculations are performed by Python and/or
   the PVGIS calculation service.
2. Do not recalculate or change supplied numbers.
3. Clearly distinguish PVGIS-derived estimates from assumptions.
4. Do not claim bankable energy yield.
5. Do not invent site-specific solar irradiation.
6. Do not claim grid feasibility.
7. Do not claim land availability.
8. Do not claim equipment selection.
9. Flag important information that is still missing.
10. This is preliminary engineering screening only.

When PVGIS data is available, describe the output as a
location-based PVGIS estimate.

When PVGIS is unavailable, clearly state that the system has
fallen back to a screening calculation based on assumptions.

Use exactly these sections:

### Technical Assessment

### Key Results

### Location Basis

### Assumptions

### Engineering Considerations

### Data Gaps

### Bottom Line

Do not invent numbers.
"""


# =========================================================
# Location Geocoding
# =========================================================

@st.cache_data(ttl=86400)
def geocode_location(location):
    """
    Convert a user-entered Pakistani city/location into
    latitude and longitude using Open-Meteo geocoding.

    Returns None when the location cannot be resolved.
    """

    if not location or not location.strip():
        return None

    query = urllib.parse.urlencode(
        {
            "name": location.strip(),
            "count": 1,
            "language": "en",
            "format": "json",
            "countryCode": "PK",
        }
    )

    url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + query
    )

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SolarGridAI/1.0"
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=15,
        ) as response:

            data = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

        results = data.get(
            "results",
            [],
        )

        if not results:
            return None

        result = results[0]

        latitude = result.get(
            "latitude"
        )

        longitude = result.get(
            "longitude"
        )

        if latitude is None or longitude is None:
            return None

        return {
            "name": result.get(
                "name",
                location,
            ),
            "admin1": result.get(
                "admin1",
                "",
            ),
            "country": result.get(
                "country",
                "Pakistan",
            ),
            "latitude": float(latitude),
            "longitude": float(longitude),
        }

    except Exception:
        return None


# =========================================================
# PVGIS Location-Based Production
# =========================================================

@st.cache_data(ttl=86400)
def get_pvgis_production(
    latitude,
    longitude,
    capacity_mw,
    system_loss_pct=14.0,
):
    """
    Query the official PVGIS 5.3 PVcalc API.

    PVGIS peakpower is specified in kW.

    Returns the annual PVGIS estimate and selected metadata.
    """

    capacity_kw = (
        float(capacity_mw) * 1000.0
    )

    params = {
        "lat": latitude,
        "lon": longitude,
        "peakpower": capacity_kw,
        "pvtechchoice": "crystSi",
        "mountingplace": "free",
        "loss": system_loss_pct,
        "fixed": 1,
        "optimalangles": 1,
        "usehorizon": 1,
        "outputformat": "json",
    }

    query = urllib.parse.urlencode(
        params
    )

    url = (
        "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?"
        + query
    )

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SolarGridAI/1.0"
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:

            data = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

        totals = (
            data.get("outputs", {})
            .get("totals", {})
            .get("fixed", {})
        )

        annual_energy_kwh = totals.get(
            "E_y"
        )

        if annual_energy_kwh is None:
            return None

        mounting = (
            data.get("inputs", {})
            .get("mounting_system", {})
            .get("fixed", {})
        )

        slope = (
            mounting.get("slope", {})
            .get("value")
        )

        azimuth = (
            mounting.get("azimuth", {})
            .get("value")
        )

        peak_power_kw = (
            float(capacity_mw) * 1000.0
        )

        specific_yield = (
            float(annual_energy_kwh)
            / peak_power_kw
        )

        annual_energy_gwh = (
            float(annual_energy_kwh)
            / 1_000_000.0
        )

        equivalent_full_load_hours = (
            float(annual_energy_kwh)
            / (float(capacity_mw) * 1000.0)
        )

        return {
            "annual_energy_kwh": float(
                annual_energy_kwh
            ),
            "annual_energy_gwh": annual_energy_gwh,
            "specific_yield_kwh_per_kwp": (
                specific_yield
            ),
            "equivalent_full_load_hours": (
                equivalent_full_load_hours
            ),
            "system_loss_pct": float(
                system_loss_pct
            ),
            "slope_deg": slope,
            "azimuth_deg": azimuth,
        }

    except Exception:
        return None


# =========================================================
# Fallback Screening Calculation
# =========================================================

def calculate_solar_screening(
    capacity_mw,
    capacity_factor_pct=22.0,
    performance_ratio_pct=80.0,
    annual_degradation_pct=0.5,
    project_life_years=25,
):
    """
    Fallback calculation used only if location-based PVGIS
    estimation is unavailable.
    """

    capacity_mw = float(
        capacity_mw
    )

    capacity_factor_pct = float(
        capacity_factor_pct
    )

    performance_ratio_pct = float(
        performance_ratio_pct
    )

    annual_degradation_pct = float(
        annual_degradation_pct
    )

    project_life_years = int(
        project_life_years
    )

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

    maximum_annual_mwh = (
        capacity_mw * hours_per_year
    )

    gross_annual_mwh = (
        maximum_annual_mwh
        * capacity_factor_pct
        / 100.0
    )

    net_first_year_mwh = (
        gross_annual_mwh
        * performance_ratio_pct
        / 100.0
    )

    equivalent_full_load_hours = (
        net_first_year_mwh
        / capacity_mw
    )

    annual_generation = []

    for year in range(
        1,
        project_life_years + 1,
    ):

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
                "generation_gwh": (
                    generation_mwh / 1000.0
                ),
            }
        )

    lifetime_generation_mwh = sum(
        item["generation_mwh"]
        for item in annual_generation
    )

    return {
        "capacity_mw": capacity_mw,
        "capacity_factor_pct": (
            capacity_factor_pct
        ),
        "performance_ratio_pct": (
            performance_ratio_pct
        ),
        "annual_degradation_pct": (
            annual_degradation_pct
        ),
        "project_life_years": (
            project_life_years
        ),
        "maximum_annual_mwh": (
            maximum_annual_mwh
        ),
        "gross_annual_mwh": (
            gross_annual_mwh
        ),
        "net_first_year_mwh": (
            net_first_year_mwh
        ),
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
        "annual_generation": (
            annual_generation
        ),
        "data_source": (
            "Screening assumptions"
        ),
    }


# =========================================================
# Location-Aware Technical Agent
# =========================================================

def analyze_technical_project(
    project,
    capacity_factor_pct=22.0,
    performance_ratio_pct=80.0,
    annual_degradation_pct=0.5,
    project_life_years=25,
):
    """
    Run the Technical Engineer Agent.

    For Solar PV:
        City/location
             ↓
        Geocoding
             ↓
        PVGIS
             ↓
        Location-based annual energy

    If that external lookup fails, the function safely falls
    back to the previous deterministic screening calculation.
    """

    capacity = float(
        project.get(
            "capacity",
            0,
        )
    )

    technology = project.get(
        "technology",
        "Solar PV",
    )

    location = project.get(
        "location",
        "Pakistan",
    )

    project_name = project.get(
        "project_name",
        "Unnamed Project",
    )

    # -----------------------------------------------------
    # Default fallback
    # -----------------------------------------------------

    results = None

    resolved_location = None

    location_basis = (
        "Screening assumptions"
    )

    # -----------------------------------------------------
    # Location-based Solar PV estimate
    # -----------------------------------------------------

    if (
        technology == "Solar PV"
        and location
        and location.strip().lower()
        not in {
            "pakistan",
            "not specified",
            "unknown",
        }
    ):

        resolved_location = geocode_location(
            location
        )

        if resolved_location:

            pvgis = get_pvgis_production(
                latitude=resolved_location[
                    "latitude"
                ],
                longitude=resolved_location[
                    "longitude"
                ],
                capacity_mw=capacity,
                system_loss_pct=14.0,
            )

            if pvgis:

                annual_energy_mwh = (
                    pvgis["annual_energy_kwh"]
                    / 1000.0
                )

                annual_generation = []

                for year in range(
                    1,
                    int(project_life_years) + 1,
                ):

                    degradation_factor = (
                        1.0
                        - float(
                            annual_degradation_pct
                        )
                        / 100.0
                    ) ** (year - 1)

                    generation_mwh = (
                        annual_energy_mwh
                        * degradation_factor
                    )

                    annual_generation.append(
                        {
                            "year": year,
                            "generation_mwh": (
                                generation_mwh
                            ),
                            "generation_gwh": (
                                generation_mwh
                                / 1000.0
                            ),
                        }
                    )

                lifetime_generation_mwh = sum(
                    item["generation_mwh"]
                    for item in annual_generation
                )

                effective_capacity_factor = (
                    annual_energy_mwh
                    / (
                        capacity
                        * 8760.0
                    )
                    * 100.0
                )

                results = {
                    "capacity_mw": capacity,
                    "capacity_factor_pct": (
                        effective_capacity_factor
                    ),
                    "performance_ratio_pct": None,
                    "annual_degradation_pct": (
                        float(
                            annual_degradation_pct
                        )
                    ),
                    "project_life_years": (
                        int(project_life_years)
                    ),
                    "maximum_annual_mwh": (
                        capacity * 8760.0
                    ),
                    "gross_annual_mwh": (
                        annual_energy_mwh
                    ),
                    "net_first_year_mwh": (
                        annual_energy_mwh
                    ),
                    "net_first_year_gwh": (
                        annual_energy_mwh
                        / 1000.0
                    ),
                    "equivalent_full_load_hours": (
                        pvgis[
                            "equivalent_full_load_hours"
                        ]
                    ),
                    "lifetime_generation_mwh": (
                        lifetime_generation_mwh
                    ),
                    "lifetime_generation_gwh": (
                        lifetime_generation_mwh
                        / 1000.0
                    ),
                    "annual_generation": (
                        annual_generation
                    ),
                    "data_source": (
                        "PVGIS 5.3 location-based estimate"
                    ),
                    "location_name": (
                        resolved_location[
                            "name"
                        ]
                    ),
                    "latitude": (
                        resolved_location[
                            "latitude"
                        ]
                    ),
                    "longitude": (
                        resolved_location[
                            "longitude"
                        ]
                    ),
                    "country": (
                        resolved_location[
                            "country"
                        ]
                    ),
                    "admin1": (
                        resolved_location.get(
                            "admin1",
                            "",
                        )
                    ),
                    "specific_yield_kwh_per_kwp": (
                        pvgis[
                            "specific_yield_kwh_per_kwp"
                        ]
                    ),
                    "system_loss_pct": (
                        pvgis[
                            "system_loss_pct"
                        ]
                    ),
                    "optimal_slope_deg": (
                        pvgis[
                            "slope_deg"
                        ]
                    ),
                    "optimal_azimuth_deg": (
                        pvgis[
                            "azimuth_deg"
                        ]
                    ),
                }

                location_basis = (
                    "PVGIS location-based estimate"
                )

    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    if results is None:

        results = calculate_solar_screening(
            capacity_mw=capacity,
            capacity_factor_pct=(
                capacity_factor_pct
            ),
            performance_ratio_pct=(
                performance_ratio_pct
            ),
            annual_degradation_pct=(
                annual_degradation_pct
            ),
            project_life_years=(
                project_life_years
            ),
        )

        results["location_note"] = (
            "PVGIS location lookup was unavailable "
            "or no specific city/location was supplied. "
            "The result uses screening assumptions."
        )

    else:

        results["location_note"] = (
            "Energy estimate is based on the resolved "
            "project location using PVGIS 5.3."
        )

    # -----------------------------------------------------
    # GPT-OSS interpretation
    # -----------------------------------------------------

    if results["data_source"] == (
        "PVGIS 5.3 location-based estimate"
    ):

        location_details = f"""
Resolved location:
{results["location_name"]}, {results["country"]}

Latitude:
{results["latitude"]:.5f}

Longitude:
{results["longitude"]:.5f}

PVGIS specific yield:
{results["specific_yield_kwh_per_kwp"]:,.1f}
kWh/kWp/year

PVGIS system loss assumption:
{results["system_loss_pct"]:.1f}%

PVGIS calculated annual energy:
{results["net_first_year_gwh"]:,.2f} GWh/year

Effective annual capacity factor:
{results["capacity_factor_pct"]:.2f}%

PVGIS optimal slope:
{results["optimal_slope_deg"]}

PVGIS optimal azimuth:
{results["optimal_azimuth_deg"]}
"""

        assumption_text = f"""
PVGIS location-based estimate.

Annual degradation used for the lifetime profile:
{annual_degradation_pct:.2f}%.

The PVGIS estimate uses a system-loss input of:
{results["system_loss_pct"]:.1f}%.
"""

    else:

        location_details = f"""
Location:
{location}

A location-specific PVGIS result was not available.
"""

        assumption_text = f"""
Fallback screening assumptions:

Capacity factor:
{capacity_factor_pct:.2f}%

Performance ratio:
{performance_ratio_pct:.2f}%

Annual degradation:
{annual_degradation_pct:.2f}%

Project life:
{project_life_years} years.
"""

    prompt = f"""
PROJECT INPUTS

Project name:
{project_name}

Location entered by user:
{location}

Technology:
{technology}

Capacity:
{capacity:,.2f} MW

LOCATION BASIS

{location_details}

CALCULATED RESULTS

Annual energy:
{results["net_first_year_gwh"]:,.2f} GWh/year

Equivalent full-load hours:
{results["equivalent_full_load_hours"]:,.0f} hours/year

Lifetime generation:
{results["lifetime_generation_gwh"]:,.2f} GWh

ASSUMPTIONS / METHOD

{assumption_text}

TASK

Interpret the technical results.

Clearly state whether the estimate is PVGIS-based or a
fallback screening calculation.

Do not claim that this is a bankable energy-yield assessment.

Identify important site-specific items that still require
detailed engineering, such as actual site boundary,
topography, shading/horizon conditions, equipment design,
and other project-specific inputs where applicable.

Do not invent values.
"""

    try:

        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=1400,
        )

    except Exception as exc:

        answer = (
            "The Technical Engineer calculations "
            "were completed, but GPT-OSS could not "
            "generate the interpretation.\n\n"
            f"Technical error: {exc}"
        )

    return {
        "answer": answer,
        "results": results,
    }


def get_agent_info():
    return {
        "name": "Technical Engineer Agent",
        "short_name": "Technical Agent",
        "icon": "⚙️",
        "model": "openai/gpt-oss-120b",
        "calculation_engine": (
            "PVGIS + deterministic Python"
        ),
        "data_source": (
            "PVGIS 5.3 / Open-Meteo geocoding"
        ),
        "status": "ready",
        "description": (
            "Performs location-based preliminary "
            "solar PV energy screening."
        ),
    }
