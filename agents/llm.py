import streamlit as st
from groq import Groq


# ---------------------------------------------------------
# SolarGrid AI — LLM Configuration
# ---------------------------------------------------------

MODEL = "openai/gpt-oss-120b"


def get_groq_client():
    """
    Create a Groq client using the API key stored in
    Streamlit Community Cloud Secrets.
    """

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception as exc:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        ) from exc

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is empty. "
            "Please configure it in Streamlit Secrets."
        )

    return Groq(
        api_key=api_key
    )


def generate_response(
    system_prompt,
    user_prompt,
    max_tokens=1800,
):
    """
    Generate a response using Groq + GPT-OSS 120B.

    Parameters
    ----------
    system_prompt : str
        Instructions defining the agent's role and behavior.

    user_prompt : str
        Task-specific prompt and retrieved evidence.

    max_tokens : int
        Maximum number of completion tokens.

    Returns
    -------
    str
        Model-generated response.

    Notes
    -----
    Groq currently documents `max_tokens` as deprecated in favor
    of `max_completion_tokens`, so this wrapper uses the newer
    parameter.
    """

    if not system_prompt or not system_prompt.strip():
        raise ValueError(
            "system_prompt cannot be empty."
        )

    if not user_prompt or not user_prompt.strip():
        raise ValueError(
            "user_prompt cannot be empty."
        )

    if not isinstance(max_tokens, int):
        raise TypeError(
            "max_tokens must be an integer."
        )

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than zero."
        )

    client = get_groq_client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_completion_tokens=max_tokens,
            reasoning_effort="medium",
        )

    except Exception as exc:
        raise RuntimeError(
            "Groq API request failed: "
            f"{exc}"
        ) from exc

    if not response.choices:
        raise RuntimeError(
            "Groq returned no completion choices."
        )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return content.strip()
