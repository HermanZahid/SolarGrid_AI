import streamlit as st
from groq import Groq


MODEL = "openai/gpt-oss-120b"


def get_groq_client():
    """Create a Groq client using Streamlit secrets."""

    if "GROQ_API_KEY" not in st.secrets:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


def generate_response(
    system_prompt,
    user_prompt,
    max_tokens=1800,
):
    """
    Send a prompt to GPT-OSS 120B through Groq.
    """

    client = get_groq_client()

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
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content
