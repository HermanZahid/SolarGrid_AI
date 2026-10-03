from agents.llm import generate_response


# =========================================================
# SolarGrid AI — Regulatory Intelligence Agent
# =========================================================

SYSTEM_PROMPT = """
You are the Regulatory Intelligence Agent of SolarGrid AI.

Your job is to analyze Pakistan renewable-energy regulatory
questions using ONLY the evidence retrieved from the SolarGrid AI
knowledge base.

You are an evidence-constrained regulatory analysis agent.

============================================================
EVIDENCE DISCIPLINE
============================================================

1. NEVER invent laws, regulations, thresholds, approvals,
   technical requirements, procedures, studies, deadlines,
   licenses, tariffs, eligibility conditions, or compliance
   obligations.

2. Every important factual regulatory claim MUST be supported
   by one or more supplied evidence items.

3. Cite supporting evidence using:
   [S1], [S2], [S3], etc.

4. Treat user-entered project information as PROJECT INPUT,
   not regulatory evidence.

5. Project inputs may include:
   - project name
   - location
   - technology
   - capacity
   - grid voltage

   These values are assumptions supplied by the user for
   preliminary screening. They do NOT prove that the project
   actually has that characteristic from a regulatory perspective.

6. Never infer:
   - applicable voltage
   - point of connection
   - responsible licensee
   - approval status
   - regulatory eligibility
   - grid feasibility
   - technical compliance
   - licensing requirement
   - required study package

   solely from project capacity, technology, location, or
   user-entered assumptions.

7. Do not treat SolarGrid AI guidance, examples, recommendations,
   interpretations, or investigation suggestions as though they
   were direct legal or regulatory requirements.

8. If evidence supports only an interpretation, explicitly label
   it as PROJECT INTERPRETATION.

9. If something appears relevant but cannot be established from
   the supplied evidence, classify it as an INVESTIGATION ITEM.

10. If the information required to reach a project-specific
    conclusion is absent, classify it as MISSING EVIDENCE.

11. If the knowledge base does not contain sufficient evidence,
    explicitly state:

    "The current SolarGrid AI knowledge base does not provide
    sufficient evidence to establish this."

12. Do not claim that a project is:
    - legally compliant
    - approved
    - grid-feasible
    - financially viable
    - eligible
    - licensed

    unless the supplied evidence directly establishes that fact.

13. Do not turn examples into general rules.

14. Do not invent source content that is not present in the
    retrieved evidence.

15. This is preliminary regulatory intelligence.

    It is NOT:
    - legal advice
    - a formal regulatory determination
    - a grid study
    - a bankable feasibility study
    - a licensing opinion

============================================================
SOURCE CITATION DISCIPLINE
============================================================

Use citations immediately after the relevant factual claim.

Example:

The applicable technical standards establish requirements for
generation facilities connected to the grid. [S1]

Do NOT write citations such as [S6] when only [S1] to [S5]
were supplied.

Do NOT cite evidence that does not actually support the claim.

When several sources support the same statement, cite them together:

[S1][S2]

============================================================
RESPONSE STRUCTURE
============================================================

Use exactly these sections:

### Evidence-Backed Findings

Only findings directly supported by the supplied evidence.

### Project Interpretation

Explain how the supported findings relate to the supplied
project inputs.

Clearly distinguish interpretation from source-backed facts.

### Investigation Items

Relevant issues that cannot be established from the supplied
evidence.

Do not present these as confirmed requirements.

### Missing Evidence

Identify information or source material required for a more
specific assessment.

### Bottom Line

Give a concise summary of:

1. What the evidence establishes.
2. What remains unverified.

============================================================
STYLE
============================================================

Be professional, precise, conservative, and concise.

Do not use sensational language.

Do not overstate certainty.

Do not repeat the heading "Regulatory Assessment".

Do not create an "Evidence Sources" section unless genuinely
useful.

Prefer clear engineering/regulatory language over generic AI
language.
"""


# =========================================================
# Stage 1 — Retrieve Regulatory Evidence
# =========================================================

def retrieve_regulatory_evidence(
    query,
    rag,
    k=5,
):
    """
    Retrieve regulatory evidence from the local SolarGrid AI
    knowledge base.

    This function performs NO LLM call.

    Returns
    -------
    dict
        {
            "context": str,
            "evidence": list
        }
    """

    if not query or not query.strip():
        return {
            "context": "",
            "evidence": [],
        }

    context, evidence = rag.get_context(
        query.strip(),
        k=k,
    )

    return {
        "context": context,
        "evidence": evidence,
    }


# =========================================================
# Stage 2 — Build Project Context
# =========================================================

def build_project_context(project):
    """
    Convert project inputs into an explicit context block.

    These values are clearly marked as user-provided project inputs
    rather than regulatory evidence.
    """

    project = project or {}

    return f"""
PROJECT INPUTS

Project name:
{project.get("project_name", "")}

Location:
{project.get("location", "")}

Technology:
{project.get("technology", "")}

Capacity:
{project.get("capacity", "")} MW

Grid connection voltage:
{project.get("grid_voltage", "")}

IMPORTANT:
These values are user-provided project inputs for preliminary
screening.

They are NOT regulatory evidence.

Do not use these values as proof of legal eligibility,
approval, grid feasibility, compliance, or applicability of
a specific regulatory framework.
""".strip()


# =========================================================
# Stage 3 — Generate Regulatory Assessment
# =========================================================

def generate_regulatory_assessment(
    query,
    project,
    context,
    evidence,
    max_tokens=1800,
):
    """
    Generate an evidence-constrained regulatory assessment.

    This function performs the Groq / GPT-OSS 120B call.

    Retrieval must already have been completed before calling
    this function.
    """

    if not evidence:
        return (
            "The current SolarGrid AI knowledge base does not "
            "provide sufficient evidence to establish an answer "
            "to this question."
        )

    project_context = build_project_context(
        project
    )

    user_prompt = f"""
{project_context}

============================================================
REGULATORY QUESTION
============================================================

{query.strip()}

============================================================
RETRIEVED EVIDENCE
============================================================

{context}

============================================================
TASK
============================================================

Answer the regulatory question using ONLY the retrieved evidence.

For every important factual regulatory statement, cite the
supporting evidence immediately using [S1], [S2], [S3], etc.

Follow these rules carefully:

1. Do not use information outside the retrieved evidence.

2. Do not invent regulatory requirements.

3. Do not infer project-specific regulatory conclusions from
   the project inputs.

4. If evidence supports a statement but the application to
   the specific project requires interpretation, place that
   reasoning under "Project Interpretation".

5. If an issue may matter but the supplied evidence does not
   establish the requirement, place it under
   "Investigation Items".

6. If additional information or source documents are needed,
   place them under "Missing Evidence".

7. Never turn an investigation item into a confirmed requirement.

8. Never treat a project input as regulatory evidence.

9. Every important factual statement must have an appropriate
   citation.

10. When the evidence is insufficient, say so explicitly.

Produce the response using exactly:

### Evidence-Backed Findings

### Project Interpretation

### Investigation Items

### Missing Evidence

### Bottom Line
"""

    answer = generate_response(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )

    return answer


# =========================================================
# Backward-Compatible Complete Agent Function
# =========================================================

def analyze_regulatory_question(
    query,
    rag,
    project,
    k=5,
    max_tokens=1800,
):
    """
    Run the complete Regulatory Intelligence Agent.

    This function is kept for compatibility with the existing
    application.

    Internally it performs:

        Retrieval → Analysis

    The Streamlit UI will later call the two stages separately
    so that the visible workflow reflects the actual execution.
    """

    retrieval = retrieve_regulatory_evidence(
        query=query,
        rag=rag,
        k=k,
    )

    evidence = retrieval["evidence"]
    context = retrieval["context"]

    if not evidence:
        return {
            "answer": (
                "The current SolarGrid AI knowledge base does "
                "not provide sufficient evidence to establish "
                "an answer to this question."
            ),
            "evidence": [],
            "context": "",
        }

    try:
        answer = generate_regulatory_assessment(
            query=query,
            project=project,
            context=context,
            evidence=evidence,
            max_tokens=max_tokens,
        )

    except Exception as exc:
        answer = (
            "The Regulatory Intelligence Agent could not "
            "complete the analysis.\n\n"
            f"Technical error: {exc}"
        )

    return {
        "answer": answer,
        "evidence": evidence,
        "context": context,
    }


# =========================================================
# Agent Metadata
# =========================================================

def get_agent_info():
    """
    Metadata used later by the multi-agent orchestrator and UI.
    """

    return {
        "name": "Regulatory Intelligence Agent",
        "short_name": "Regulatory Agent",
        "icon": "📚",
        "model": "openai/gpt-oss-120b",
        "retrieval": "Local TF-IDF RAG",
        "status": "ready",
        "description": (
            "Analyzes Pakistan renewable-energy regulatory "
            "evidence using source-constrained reasoning."
        ),
    }
