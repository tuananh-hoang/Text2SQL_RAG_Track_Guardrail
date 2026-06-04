import json
import re
from pathlib import Path
from typing import Any

try:
    from text2sql.llm_client import create_chat_completion_with_rotation, get_llm_model
except ModuleNotFoundError:
    from llm_client import create_chat_completion_with_rotation, get_llm_model


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_PATH = BASE_DIR / "prompts" / "v1_baseline.txt"
REQUIRED_FIELDS = {"sql", "tables_used", "columns_used", "explanation", "confidence"}
CONFIDENCE_VALUES = {"high", "medium", "low"}


def strip_json_fence(content: str) -> str:
    stripped = content.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def parse_llm_json(content: str) -> dict[str, Any]:
    json_text = strip_json_fence(content)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        preview = json_text[:300].replace("\n", " ")
        raise ValueError(f"LLM output is not valid JSON: {preview}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM output JSON must be an object")
    return parsed


def validate_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing_fields = sorted(REQUIRED_FIELDS - set(payload))
    if missing_fields:
        raise ValueError(f"LLM output missing required fields: {', '.join(missing_fields)}")

    if not isinstance(payload["tables_used"], list):
        raise ValueError("LLM output field tables_used must be a list")
    if not isinstance(payload["columns_used"], list):
        raise ValueError("LLM output field columns_used must be a list")

    confidence = str(payload["confidence"]).lower()
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("LLM output field confidence must be high, medium, or low")

    payload["confidence"] = confidence
    payload["needs_review"] = confidence == "low"
    return payload


def load_prompt(schema_context: str) -> str:
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    return prompt_template.replace("{schema_context}", schema_context)


def generate_sql(question: str, schema_context: str) -> dict[str, Any]:
    prompt = load_prompt(schema_context)
    response = create_chat_completion_with_rotation(
        model=get_llm_model(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content")

    return validate_llm_payload(parse_llm_json(content))
