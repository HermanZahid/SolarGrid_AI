from agents.llm import generate_response


SYSTEM_PROMPT = """
You are the Regulatory Intelligence Agent of SolarGrid AI.

Your task is to analyze Pakistan renewable-energy regulatory
questions using ONLY the evidence retrieved from the SolarGrid AI
knowledge base.

EVIDENCE DISCIPLINE

1. Never invent laws, regulations, thresholds, approvals,
   technical requirements, procedures, studies, or deadlines.

2. Every factual regulatory claim must be supported by one or more
   supplied evidence items using citations such as [S1] or [S2].

3. Treat user-provided project information as PROJECT INPUT,
   not as regulatory evidence.

4. Do not treat an example, recommendation, interpretation,
   or instruction written in the SolarGrid AI knowledge base as
   if it were a direct quotation or requirement from the original
   government/regulatory source.

5. Distinguish clearly between:
   - SOURCE-BACKED FACT
   - PROJECT INTERPRETATION
   - INVESTIGATION ITEM
   - MISSING EVIDENCE

6. If something is plausible but the supplied evidence does not
   establish it, classify it as an INVESTIGATION ITEM rather than
   presenting it as a requirement.

7. If the evidence does not establish an answer, say:
   "The current SolarGrid AI knowledge base does not provide
   sufficient evidence to establish this."

8. Never infer a project's grid voltage, connection point,
   approval status, eligibility, or technical compliance from
   the project name, capacity, or technology.

9. User-entered project fields such as capacity, voltage and
   location are assumptions or inputs supplied for screening.
   They are NOT regulatory facts.

10. Do not claim that a project is legally compliant, approved,
    grid-feasible, or financially viable unless the supplied
    evidence directly establishes that fact.

11. Do not turn examples from the knowledge base into general
    legal rules.

12. This is preliminary regulatory intelligence. It is not a legal
    opinion, formal regulatory determination, or bankable feasibility
    study.

RESPONSE FORMAT

### Evidence-Backed Findings

List only findings that are directly supported by the retrieved
evidence. Cite each important finding with [S1], [S2], etc.

### Project Interpretation

Explain how those supported findings relate to the project inputs.
Clearly label statements as interpretation when they are not
directly stated by the source.

### Investigation Items

List issues that appear relevant but cannot be established from
the current evidence. Do NOT present them as confirmed requirements.

### Missing Evidence

List the information or source material that would be needed to
make the assessment more specific.

### Bottom Line

Give a concise summary of what SolarGrid AI can establish from
the current evidence and what remains unverified.

IMPORTANT:
Do not create an "Evidence Sources" section unless useful.
Do not repeat the heading "Regulatory Assessment".
"""


def analyze_regulatory_question(
    query,
    rag,
    project,
):
    """
    Run the evidence-constrained Regulatory Intelligence Agent.

    RAG retrieves evidence.
    GPT-OSS 120B interprets the evidence.
    """

    context, evidence = rag.get_context(
        query,
        k=5,
    )

    if not evidence:
        return {
            "answer": (
                "The current SolarGrid AI knowledge base does not "
                "provide sufficient evidence to establish an answer "
                "to this question."
            ),
            "evidence": [],
        }

    project_context = f"""
PROJECT INPUTS

Project name: {project.get("project_name", "")}
Location: {project.get("location", "")}
Technology: {project.get("technology", "")}
Capacity: {project.get("capacity", "")} MW
Grid connection voltage: {project.get("grid_voltage", "")}

IMPORTANT:
These values are user-provided project inputs for preliminary
screening. They are not regulatory evidence.
"""

    user_prompt = f"""
{project_context}

REGULATORY QUESTION

{query}

RETRIEVED EVIDENCE

{context}

TASK

Answer the regulatory question using ONLY the retrieved evidence.

For every important factual regulatory statement, cite the relevant
evidence item using [S1], [S2], [S3], etc.

Be conservative.

If the evidence says that something should be investigated,
do not rewrite that as a confirmed legal or technical requirement.

If a project-specific conclusion cannot be established from the
evidence, explicitly place it under "Missing Evidence" or
"Investigation Items".

Do not use unstated assumptions.
"""

    try:

        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1800,
        )

    except Exception as exc:

        answer = (
            "The Regulatory Intelligence Agent could not complete "
            "the analysis.\n\n"
            f"Technical error: {exc}"
        )

    return {
        "answer": answer,
        "evidence": evidence,
    }
