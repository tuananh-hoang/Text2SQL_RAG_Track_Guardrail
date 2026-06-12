import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
}
KEY_ENV_NAMES = {
    # fix: include the third configured key slot in provider rotation.
    "openai": ("OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3"),
    "groq": (
        "GROQ_API_KEY",
        "GROQ_API_KEY_2",
        "GROQ_API_KEY_3",
        "OPENAI_API_KEY",
        "OPENAI_API_KEY_2",
        "OPENAI_API_KEY_3",
    ),
}
RATE_LIMITED_KEY_NAMES: set[str] = set()


def get_llm_provider() -> str:
    load_dotenv(BASE_DIR / ".env", override=True)
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def get_llm_model() -> str:
    provider = get_llm_provider()
    default_model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    return os.getenv("LLM_MODEL", default_model).strip()


def get_llm_base_url(provider: str) -> str | None:
    base_url = os.getenv("LLM_BASE_URL")
    if provider == "groq":
        return base_url or DEFAULT_GROQ_BASE_URL
    return base_url


def get_llm_api_keys(provider: str) -> list[tuple[str, str]]:
    load_dotenv(BASE_DIR / ".env", override=True)
    env_names = KEY_ENV_NAMES.get(provider, KEY_ENV_NAMES["openai"])
    keys = []
    seen = set()
    for env_name in env_names:
        api_key = os.getenv(env_name)
        if not api_key or api_key == "your_key_here" or api_key in seen:
            continue
        keys.append((env_name, api_key))
        seen.add(api_key)

    if not keys:
        joined_names = " or ".join(env_names)
        raise RuntimeError(f"Missing {joined_names} in text2sql/.env")
    return keys


def get_llm_client() -> OpenAI:
    provider = get_llm_provider()
    api_key = get_llm_api_keys(provider)[0][1]
    base_url = get_llm_base_url(provider)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def build_llm_client(api_key: str, base_url: str | None) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    error_text = str(exc).lower()
    return "rate_limit" in error_text or "rate limit" in error_text or "429" in error_text


def sanitize_llm_error(exc: Exception) -> str:
    if is_rate_limit_error(exc):
        return "Provider rate limit: 429 rate_limit_exceeded"
    return str(exc)


def create_chat_completion_with_rotation(**kwargs: Any) -> Any:
    provider = get_llm_provider()
    base_url = get_llm_base_url(provider)
    api_keys = get_llm_api_keys(provider)
    active_keys = [(name, key) for name, key in api_keys if name not in RATE_LIMITED_KEY_NAMES]
    if not active_keys:
        RATE_LIMITED_KEY_NAMES.clear()
        active_keys = api_keys
    errors = []

    # fix: rotate API keys when the provider returns a 429 rate limit error.
    for index, (env_name, api_key) in enumerate(active_keys, start=1):
        client = build_llm_client(api_key, base_url)
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            errors.append(f"{env_name}: {sanitize_llm_error(exc)}")
            if not is_rate_limit_error(exc):
                raise
            RATE_LIMITED_KEY_NAMES.add(env_name)
            if index == len(active_keys):
                raise RuntimeError(
                    "All configured LLM API keys are rate limited: "
                    + " | ".join(errors)
                ) from exc

    raise RuntimeError("No LLM API key was available for request")
