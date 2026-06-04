import argparse
import json
import math
from collections import defaultdict
from typing import Any

from text2sql.schema.build_schema_vector_index import EmbeddingModel, index_path_for_mode, load_config
from text2sql.schema.paths import V2_ENTITIES_DIR
from text2sql.schema.query_decomposer import decompose_question
from text2sql.schema.query_keyword_extractor import strip_vietnamese_accents, tokenize

def load_entities(mode: str) -> list[dict[str, Any]]:
    path = V2_ENTITIES_DIR / f"schema_entities_{mode}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing schema entities: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_vector_records(mode: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    path = index_path_for_mode(config, mode)
    if not path.exists():
        raise FileNotFoundError(f"Missing vector index: {path}. Run build_schema_vector_index first.")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: str) -> str:
    return strip_vietnamese_accents(value).lower()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


def lexical_score(query: str, entity: dict[str, Any]) -> float:
    query_norm = normalize_text(query)
    text_norm = normalize_text(entity.get("text", ""))
    table_norm = normalize_text(str(entity.get("table") or entity.get("target_table") or ""))
    column_norm = normalize_text(str(entity.get("column") or entity.get("target_column") or ""))

    score = 0.0
    if query_norm and query_norm in text_norm:
        score = max(score, 1.0)
    if column_norm and column_norm in query_norm:
        score = max(score, 0.95)
    if table_norm and table_norm in query_norm:
        score = max(score, 0.85)

    query_tokens = set(tokenize(query_norm))
    text_tokens = set(tokenize(text_norm))
    name_tokens = set(tokenize(f"{table_norm} {column_norm}"))
    if query_tokens and text_tokens:
        score = max(score, len(query_tokens & text_tokens) / len(query_tokens | text_tokens))
    if query_tokens and name_tokens:
        score = max(score, 0.8 * len(query_tokens & name_tokens) / len(query_tokens | name_tokens))
    return min(score, 1.0)


def build_retrieval_strings(question: str, decomposition: dict[str, Any]) -> list[str]:
    strings = [question]
    strings.extend(decomposition.get("retrieval_queries", []))
    strings.extend(item["phrase"] for item in decomposition.get("schema_concepts", []) if item.get("phrase"))
    strings.extend(item["phrase"] for item in decomposition.get("constraints", []) if item.get("phrase"))
    out = []
    seen = set()
    for item in strings:
        text = str(item).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def vector_scores(
    retrieval_strings: list[str],
    records: list[dict[str, Any]],
    embedding_model: EmbeddingModel,
    top_k: int,
) -> dict[str, tuple[float, list[str]]]:
    scores: dict[str, tuple[float, list[str]]] = {}
    query_vectors = embedding_model.encode_many(retrieval_strings)
    for query, query_vector in zip(retrieval_strings, query_vectors):
        ranked = []
        for record in records:
            sim = (cosine(query_vector, record["embedding"]) + 1.0) / 2.0
            ranked.append((sim, record["id"]))
        for sim, entity_id in sorted(ranked, reverse=True)[:top_k]:
            current_score, reasons = scores.get(entity_id, (0.0, []))
            if sim > current_score:
                scores[entity_id] = (sim, [f"vector top match: {query}"])
            else:
                reasons.append(f"vector matched: {query}")
                scores[entity_id] = (current_score, reasons)
    return scores


def retrieve_schema(
    question: str,
    mode: str = "product_sales",
    top_k_entities: int = 20,
    top_k_columns: int = 8,
    top_k_tables: int = 5,
) -> dict[str, Any]:
    config = load_config()
    entities = load_entities(mode)
    records = load_vector_records(mode, config)
    records_by_id = {record["id"]: record for record in records}
    entities_by_id = {entity["entity_id"]: entity for entity in entities}
    decomposition = decompose_question(question)
    retrieval_strings = build_retrieval_strings(question, decomposition)
    embedding_model = EmbeddingModel(config)
    vector_by_id = vector_scores(retrieval_strings, records, embedding_model, top_k_entities)

    matched_entities = []
    for entity in entities:
        entity_id = entity["entity_id"]
        lexical = max(lexical_score(query, entity) for query in retrieval_strings)
        vector_score_value, vector_reasons = vector_by_id.get(entity_id, (0.0, []))
        boost = float(config.get("entity_type_boost", {}).get(entity["entity_type"], 0.5))
        if lexical <= 0 and vector_score_value <= 0:
            continue
        final_score = 0.50 * vector_score_value + 0.35 * lexical + 0.15 * boost
        if final_score <= 0:
            continue
        reasons = vector_reasons[:]
        if lexical > 0:
            reasons.append("lexical/alias/name match")
        matched_entities.append(
            {
                **entity,
                "final_score": round(final_score, 6),
                "vector_score": round(vector_score_value, 6),
                "lexical_score": round(lexical, 6),
                "entity_type_boost": boost,
                "reason": reasons,
            }
        )

    matched_entities = sorted(matched_entities, key=lambda item: item["final_score"], reverse=True)[:top_k_entities]

    column_agg: dict[str, dict[str, Any]] = {}
    table_agg: dict[str, dict[str, Any]] = {}
    matched_relationships = []
    for item in matched_entities:
        score = item["final_score"]
        if item["target_type"] == "column":
            schema_name = item.get("schema", "")
            key = f"{schema_name}.{item['target_table']}.{item['target_column']}"
            current = column_agg.setdefault(
                key,
                {
                    "column": f"{item['target_table']}.{item['target_column']}" if mode == "mock_relational" else item["target_column"],
                    "table": item["target_table"],
                    "score": 0.0,
                    "evidence_entity_ids": [],
                },
            )
            current["score"] = max(current["score"], score)
            current["evidence_entity_ids"].append(item["entity_id"])
            table_key = f"{schema_name}.{item['target_table']}"
            table = table_agg.setdefault(
                table_key,
                {"table": item["target_table"], "score": 0.0, "evidence_entity_ids": []},
            )
            table["score"] = max(table["score"], score * 0.95)
            table["evidence_entity_ids"].append(item["entity_id"])
        elif item["target_type"] == "table":
            schema_name = item.get("schema", "")
            table_key = f"{schema_name}.{item['target_table']}"
            table = table_agg.setdefault(
                table_key,
                {"table": item["target_table"], "score": 0.0, "evidence_entity_ids": []},
            )
            table["score"] = max(table["score"], score)
            table["evidence_entity_ids"].append(item["entity_id"])
        elif item["target_type"] == "relationship":
            matched_relationships.append(item)

    selected_columns = sorted(column_agg.values(), key=lambda item: item["score"], reverse=True)[:top_k_columns]
    selected_table_names = {item["table"] for item in selected_columns}
    selected_tables = sorted(table_agg.values(), key=lambda item: item["score"], reverse=True)
    selected_tables = [item for item in selected_tables if item["table"] in selected_table_names or len(selected_table_names) < top_k_tables]
    selected_tables = selected_tables[:top_k_tables]

    return {
        "mode": mode,
        "query_decomposition": decomposition,
        "retrieval_strings": retrieval_strings,
        "selected_tables": [item["table"] for item in selected_tables],
        "selected_columns": [item["column"] for item in selected_columns],
        "matched_entities": matched_entities,
        "column_scores": selected_columns,
        "table_scores": selected_tables,
        "matched_relationships": matched_relationships,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve selected schema entities.")
    parser.add_argument("--mode", choices=["product_sales", "mock_relational"], default="product_sales")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    result = retrieve_schema(args.question, mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
