import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
}


def get_llm_provider() -> str:
    load_dotenv(BASE_DIR / ".env", override=True)
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def get_llm_model() -> str:
    provider = get_llm_provider()
    default_model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    return os.getenv("LLM_MODEL", default_model).strip()


def get_llm_api_key(provider: str) -> str:
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        key_name = "GROQ_API_KEY or OPENAI_API_KEY"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        key_name = "OPENAI_API_KEY"

    if not api_key or api_key == "your_key_here":
        raise RuntimeError(f"Missing {key_name} in text2sql/.env")
    return api_key


def get_llm_client() -> OpenAI:
    provider = get_llm_provider()
    api_key = get_llm_api_key(provider)
    base_url = os.getenv("LLM_BASE_URL")

    if provider == "groq":
        base_url = base_url or DEFAULT_GROQ_BASE_URL

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)
