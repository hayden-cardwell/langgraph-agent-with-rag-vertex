import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_lc_llm() -> ChatGoogleGenerativeAI:
    temperature_str = os.getenv("VERTEX_TEMPERATURE")
    temp = float(temperature_str) if temperature_str else 0.7

    return ChatGoogleGenerativeAI(
        model=os.getenv("VERTEX_MODEL", "gemini-2.5-flash"),
        project=os.getenv("GCP_PROJECT_ID"),
        location=os.getenv("GCP_LOCATION", "us-central1"),
        temperature=temp,
        max_output_tokens=int(os.getenv("VERTEX_MAX_TOKENS", "8192")),
    )
