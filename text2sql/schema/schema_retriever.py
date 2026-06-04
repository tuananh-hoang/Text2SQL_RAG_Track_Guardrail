import argparse
import json
from typing import Any

from text2sql.db.postgres_utils import create_readonly_engine
from text2sql.schema.build_schema_pgvector_index import EmbeddingModel, load_config
from text2sql.schema.paths import V2_ENTITIES_DIR
from text2sql.schema.pgvector_store import (
    DEFAULT_PGVECTOR_TABLE,
    MISSING_INDEX_MESSAGE,
    search_schema_entities_pgvector,
)
from text2sql.schema.query_decomposer import decompose_question
from text2sql.schema.query_keyword_extractor import strip_vietnamese_accents, tokenize


def load_entities(mode: str) -> list[dict[str, Any]]:
    path = V2_ENTITIES_DIR / f"schema_entities_{mode}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing schema entities: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: str) -> str:
    return strip_vietnamese_accents(value).lower()


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


def row_to_entity(row: dict[str, Any], entities_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if row["entity_id"] in entities_by_id:
        return entities_by_id[row["entity_id"]]
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and isinstance(metadata.get("entity"), dict):
        return metadata["entity"]
    return {
        "entity_id": row["entity_id"],
        "entity_type": row["entity_type"],
        "text": row["entity_text"],
        "target_type": row["target_type"],
        "target_table": row["target_table"],
        "target_column": row["target_column"],
        "schema": row.get("schema_name"),
        "table": row.get("table_name"),
        "column": row.get("column_name"),
    }


def pgvector_scores(
    retrieval_strings: list[str],
    mode: str,
    embedding_model: EmbeddingModel,
    table_name: str,
    top_k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    engine = create_readonly_engine(connect_args={"options": "-c statement_timeout=30000"})
    try:
        query_vectors = embedding_model.encode_many(retrieval_strings)
        for query, query_vector in zip(retrieval_strings, query_vectors):
            rows = search_schema_entities_pgvector(
                engine,
                query_embedding=query_vector,
                mode=mode,
                top_k=top_k,
                table_name=table_name,
            )
            for row in rows:
                entity_id = row["entity_id"]
                similarity = float(row["similarity"])
                current = scores.setdefault(
                    entity_id,
                    {
                        "vector_score": 0.0,
                        "matched_queries": [],
                        "pgvector_row": row,
                    },
                )
                if similarity > current["vector_score"]:
                    current["vector_score"] = similarity
                    current["pgvector_row"] = row
                if query not in current["matched_queries"]:
                    current["matched_queries"].append(query)
    except RuntimeError as exc:
        if "Schema pgvector index not built" in str(exc):
            raise
        raise RuntimeError(MISSING_INDEX_MESSAGE.replace("<mode>", mode)) from exc
    finally:
        engine.dispose()

    metadata = {
        "vector_store": "pgvector",
        "pgvector_table": table_name,
        "embedding_model": embedding_model.model_name,
        "embedding_dim": embedding_model.dim,
        "retrieval_query_count": len(retrieval_strings),
    }
    return scores, metadata


def retrieve_schema(
    question: str,
    mode: str = "product_sales",
    top_k_entities: int = 20,
    top_k_columns: int = 8,
    top_k_tables: int = 5,
) -> dict[str, Any]:
    config = load_config()
    table_name = config.get("pgvector_table", DEFAULT_PGVECTOR_TABLE)
    entities = load_entities(mode)
    entities_by_id = {entity["entity_id"]: entity for entity in entities}
    decomposition = decompose_question(question)
    retrieval_strings = build_retrieval_strings(question, decomposition)
    embedding_model = EmbeddingModel(config)
    vector_by_id, retrieval_metadata = pgvector_scores(
        retrieval_strings,
        mode=mode,
        embedding_model=embedding_model,
        table_name=table_name,
        top_k=top_k_entities,
    )

    matched_entities = []
    candidate_ids = set(vector_by_id)
    lexical_by_id: dict[str, float] = {}
    for entity in entities:
        lexical = max(lexical_score(query, entity) for query in retrieval_strings)
        if lexical > 0:
            candidate_ids.add(entity["entity_id"])
            lexical_by_id[entity["entity_id"]] = lexical

    for entity_id in candidate_ids:
        vector_match = vector_by_id.get(entity_id, {})
        entity = row_to_entity(vector_match["pgvector_row"], entities_by_id) if vector_match else entities_by_id[entity_id]
        lexical = lexical_by_id.get(entity_id)
        if lexical is None:
            lexical = max(lexical_score(query, entity) for query in retrieval_strings)
        vector_score_value = float(vector_match.get("vector_score", 0.0))
        boost = float(config.get("entity_type_boost", {}).get(entity["entity_type"], 0.5))
        final_score = 0.50 * vector_score_value + 0.35 * lexical + 0.15 * boost
        if final_score <= 0:
            continue
        matched_queries = vector_match.get("matched_queries", [])
        reasons = []
        if vector_score_value > 0:
            reasons.append("pgvector semantic match")
        if lexical > 0:
            reasons.append("lexical/alias/name match")
        matched_entities.append(
            {
                **entity,
                "final_score": round(final_score, 6),
                "vector_score": round(vector_score_value, 6),
                "lexical_score": round(lexical, 6),
                "entity_type_boost": boost,
                "matched_queries": matched_queries,
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

    ranked_columns = sorted(column_agg.values(), key=lambda item: item["score"], reverse=True)
    if ranked_columns:
        top_column_score = ranked_columns[0]["score"]
        score_cutoff = max(0.35, top_column_score * 0.70)
        selected_columns = [item for item in ranked_columns if item["score"] >= score_cutoff][:top_k_columns]
        if not selected_columns:
            selected_columns = ranked_columns[:1]
    else:
        selected_columns = []
    selected_table_names = {item["table"] for item in selected_columns}
    ranked_tables = sorted(table_agg.values(), key=lambda item: item["score"], reverse=True)
    if selected_table_names:
        selected_tables = [item for item in ranked_tables if item["table"] in selected_table_names][:top_k_tables]
    else:
        selected_tables = ranked_tables[:top_k_tables]

    return {
        "mode": mode,
        "query_decomposition": decomposition,
        "retrieval_strings": retrieval_strings,
        "retrieval_metadata": {
            **retrieval_metadata,
            "matched_entities_count": len(matched_entities),
        },
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
