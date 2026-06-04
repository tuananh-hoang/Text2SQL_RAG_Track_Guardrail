import re
import unicodedata
from typing import Any


def strip_vietnamese_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def tokenize(text: str) -> list[str]:
    normalized = strip_vietnamese_accents(text).lower()
    return re.findall(r"[a-zA-Z0-9_]+", normalized)


def ngrams(tokens: list[str], max_n: int = 4) -> list[str]:
    output = []
    for n in range(1, max_n + 1):
        for index in range(0, len(tokens) - n + 1):
            output.append(" ".join(tokens[index : index + n]))
    return output


def extract_numbers(question: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?%?", question)


def extract_date_like(question: str) -> list[str]:
    patterns = [
        r"\b20\d{2}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\bquý\s*[1-4]\b",
        r"\bq[1-4]\b",
        r"\btháng\s*\d{1,2}\b",
    ]
    out: list[str] = []
    lowered = question.lower()
    for pattern in patterns:
        out.extend(re.findall(pattern, lowered, flags=re.IGNORECASE))
    return out


def extract_query_keywords_rule_based(question: str) -> dict[str, Any]:
    tokens = tokenize(question)
    grams = ngrams(tokens, max_n=4)
    retrieval_queries = []
    seen = set()
    for item in [question, *grams]:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        retrieval_queries.append(normalized)

    numbers = extract_numbers(question)
    dates = extract_date_like(question)
    return {
        "original_question": question,
        "retrieval_queries": retrieval_queries[:80],
        "schema_concepts": [
            {
                "phrase": gram,
                "concept_type": "unknown",
                "description": "Rule-based fallback keyword.",
            }
            for gram in grams[:25]
        ],
        "constraints": [
            {
                "phrase": value,
                "constraint_type": "unknown",
                "description": "Rule-based fallback extracted number/date expression.",
            }
            for value in [*numbers, *dates]
        ],
        "numbers": numbers,
        "date_expressions": dates,
        "intent_summary": "Rule-based fallback decomposition.",
        "raw_llm_output": {},
        "fallback_used": True,
        "fallback_reason": "rule_based_keyword_extractor",
    }
