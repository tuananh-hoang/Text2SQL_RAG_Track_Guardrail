import json
from pathlib import Path
from typing import Any

from text2sql.llm_client import create_chat_completion_with_rotation, get_llm_model
from text2sql.schema.query_keyword_extractor import extract_query_keywords_rule_based
from text2sql.versions.v1_baseline import strip_json_fence


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_PATH = BASE_DIR / "prompts" / "query_decomposer.txt"
CONCEPT_TYPES = {"measure", "dimension", "filter", "time", "ranking", "aggregation", "comparison", "unknown"}
CONSTRAINT_TYPES = {"time_filter", "value_filter", "threshold", "top_k", "grouping", "unknown"}


def load_prompt(question: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{question}", question)


def parse_json(content: str) -> dict[str, Any]:
    return json.loads(strip_json_fence(content))


def normalize_decomposition(question: str, payload: dict[str, Any], raw_payload: dict[str, Any]) -> dict[str, Any]:
    retrieval_queries = payload.get("retrieval_queries") or [question]
    concepts = payload.get("schema_concepts") or []
    constraints = payload.get("constraints") or []

    normalized_concepts = []
    for item in concepts:
        concept_type = str(item.get("concept_type", "unknown"))
        if concept_type not in CONCEPT_TYPES:
            concept_type = "unknown"
        normalized_concepts.append(
            {
                "phrase": str(item.get("phrase", "")).strip(),
                "concept_type": concept_type,
                "description": str(item.get("description", "")).strip(),
            }
        )

    normalized_constraints = []
    for item in constraints:
        constraint_type = str(item.get("constraint_type", "unknown"))
        if constraint_type not in CONSTRAINT_TYPES:
            constraint_type = "unknown"
        normalized_constraints.append(
            {
                "phrase": str(item.get("phrase", "")).strip(),
                "constraint_type": constraint_type,
                "description": str(item.get("description", "")).strip(),
            }
        )

    return {
        "original_question": question,
        "retrieval_queries": [str(item).strip() for item in retrieval_queries if str(item).strip()],
        "schema_concepts": [item for item in normalized_concepts if item["phrase"]],
        "constraints": [item for item in normalized_constraints if item["phrase"]],
        "numbers": payload.get("numbers") or [],
        "date_expressions": payload.get("date_expressions") or [],
        "intent_summary": str(payload.get("intent_summary", "")).strip(),
        "raw_llm_output": raw_payload,
        "fallback_used": False,
        "fallback_reason": "",
    }


def decompose_question(question: str, model_provider: str = "configured") -> dict[str, Any]:
    try:
        prompt = load_prompt(question)
        response = create_chat_completion_with_rotation(
            model=get_llm_model(),
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty decomposition")
        raw_payload = parse_json(content)
        return normalize_decomposition(question, raw_payload, raw_payload)
    except Exception as exc:
        fallback = extract_query_keywords_rule_based(question)
        fallback["fallback_reason"] = str(exc)
        return fallback
