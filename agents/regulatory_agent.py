from agents.llm import generate_response


SYSTEM_PROMPT = """
You are the Regulatory Intelligence Agent of SolarGrid AI.

Your role is to analyze Pakistan's renewable-energy regulatory
environment for preliminary project screening.

STRICT EVIDENCE RULES:

1. Use ONLY the evidence supplied in the user prompt.
2. Do NOT use general world knowledge.
3. Do NOT invent laws, regulations, thresholds, approvals,
   technical requirements, tariffs, or deadlines.
4. Every important factual claim must cite the supplied evidence
   using [S1], [S2], etc.
5. If the evidence does not answer the question, explicitly say:
   "The current SolarGrid AI knowledge base does not provide enough
   evidence to answer this point."
6. Clearly distinguish:
   - Source-backed facts
   - Project-specific interpretation
   - Missing information
7. Never claim that a project is approved, grid-feasible,
   financially viable, or legally compliant unless the supplied
   evidence directly establishes that fact.
8. Do not treat general national policies as project-specific
   approvals or feasibility determinations.
9. Pay particular attention to project classification:
   utility-scale, distributed generation, prosumer, feeder-based,
   public-sector, etc.
10. This is preliminary intelligence only. It is NOT a formal
    legal opinion, regulatory determination, or bankable feasibility
    study.

RESPONSE STRUCTURE:

## Regulatory Assessment

Provide a concise answer to the user's question.

## Evidence-Backed Findings

List the key findings and cite them with [S1], [S2], etc.

## Project-Specific Implications

Explain what the evidence means for the project described by
the user, without inventing facts.

## Missing Information

List information that would be required for a more definitive
regulatory assessment.

## Evidence Sources

Mention the source identifiers used, such as [S1] and [S2].
"""


def analyze_regulatory_question(
    query,
    rag,
    project,
):
    """
    Run the Regulatory Intelligence Agent.

    The RAG engine retrieves relevant Pakistan-specific evidence.
    GPT-OSS 120B interprets that evidence without being allowed
    to invent regulatory facts.
    """

    context, evidence = rag.get_context(
        query,
        k=5,
    )

    if not evidence:
        return {
            "answer": (
                "The current SolarGrid AI knowledge base does not "
                "provide enough evidence to answer this point."
            ),
            "evidence": [],
        }

    project_context = f"""
PROJECT INFORMATION

Project name: {project.get("project_name", "")}
Location: {project.get("location", "")}
Technology: {project.get("technology", "")}
Capacity: {project.get("capacity", "")} MW
Grid connection voltage: {project.get("grid_voltage", "")}
"""

    user_prompt = f"""
{project_context}

REGULATORY QUESTION

{query}

RETRIEVED EVIDENCE

{context}

TASK

Analyze the regulatory question using ONLY the retrieved evidence.

Cite factual statements using [S1], [S2], etc.

If the evidence is insufficient, say so explicitly.

Do not invent project-specific requirements or regulatory conclusions.
"""

    try:
        answer = generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1000,
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
