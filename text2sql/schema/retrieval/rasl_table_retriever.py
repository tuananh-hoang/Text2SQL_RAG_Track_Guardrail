import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import text

from text2sql.db.postgres_utils import create_readonly_engine
from text2sql.schema.paths import v2_entities_path, v2_summary_path
from text2sql.schema.retrieval.build_schema_pgvector_index import EmbeddingModel, load_config
from text2sql.schema.retrieval.pgvector_store import (
    DEFAULT_PGVECTOR_TABLE,
    MISSING_INDEX_MESSAGE,
    qualified_table_name,
    vector_literal,
)
from text2sql.schema.retrieval.query_decomposer import decompose_question


RASL_ENTITY_TYPES = (
    "table_name",
    "table_alias",
    "table_description",
    "column_name",
    "column_alias",
    "column_description",
    "value_format_description",
    "relationship_description",
)


def load_summary(mode: str = "mock_relational") -> dict[str, Any]:
    path = v2_summary_path(mode)
    if not path.exists():
        raise FileNotFoundError(f"Missing schema summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_entities(mode: str = "mock_relational") -> list[dict[str, Any]]:
    path = v2_entities_path(mode)
    if not path.exists():
        raise FileNotFoundError(f"Missing schema entities: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_tables(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if "tables" in summary:
        return summary["tables"]
    return [
        {
            "schema_name": summary.get("schema_name", "public"),
            "table_name": summary["table_name"],
            "table_aliases": summary.get("table_aliases", []),
            "table_description": summary.get("table_description", ""),
            "columns": summary.get("columns", []),
        }
    ]


def flatten_text_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out = []
        for key in ("phrase", "text", "value", "date", "description"):
            if value.get(key):
                out.append(str(value[key]))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(flatten_text_items(item))
        return out
    return [str(value)]


def build_rasl_retrieval_strings(question: str, decomposition: dict[str, Any]) -> list[str]:
    strings: list[str] = [question]
    strings.extend(flatten_text_items(decomposition.get("retrieval_queries", [])))
    strings.extend(flatten_text_items(decomposition.get("schema_concepts", [])))
    strings.extend(flatten_text_items(decomposition.get("constraints", [])))
    strings.extend(flatten_text_items(decomposition.get("date_expressions", [])))

    out = []
    seen = set()
    for item in strings:
        text_value = str(item).strip()
        if text_value and text_value not in seen:
            out.append(text_value)
            seen.add(text_value)
    return out


def row_to_entity(row: dict[str, Any], entities_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if row["entity_id"] in entities_by_id:
        return dict(entities_by_id[row["entity_id"]])
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and isinstance(metadata.get("entity"), dict):
        return dict(metadata["entity"])
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


def trace_entity(entity: dict[str, Any], vector_score: float, matched_query: str) -> dict[str, Any]:
    return {
        **entity,
        "vector_score": round(float(vector_score), 6),
        "final_score": round(float(vector_score), 6),
        "lexical_score": 0.0,
        "entity_type_boost": 0.0,
        "matched_queries": [matched_query],
        "reason": ["pgvector vector-only entity-type retrieval"],
    }


def search_pgvector_by_entity_type(
    query: str,
    query_embedding: list[float],
    mode: str,
    entity_type: str,
    top_m: int,
    table_name: str,
    entities_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    stmt = text(
        f"""
        SELECT
            entity_id,
            entity_type,
            schema_name,
            table_name,
            column_name,
            target_type,
            target_table,
            target_column,
            entity_text,
            metadata,
            embedding_model,
            embedding_dim,
            embedding <=> CAST(:query_embedding AS vector) AS distance,
            1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
        FROM {qualified_table_name(table_name)}
        WHERE mode = :mode
          AND entity_type = :entity_type
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT :top_m
        """
    )
    engine = create_readonly_engine(connect_args={"options": "-c statement_timeout=30000"})
    try:
        with engine.connect() as conn:
            rows = [
                dict(row._mapping)
                for row in conn.execute(
                    stmt,
                    {
                        "query_embedding": vector_literal(query_embedding),
                        "mode": mode,
                        "entity_type": entity_type,
                        "top_m": top_m,
                    },
                )
            ]
    finally:
        engine.dispose()

    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            row["metadata"] = json.loads(metadata)
    return [trace_entity(row_to_entity(row, entities_by_id), float(row["similarity"]), query) for row in rows]


def relationship_tables(entity: dict[str, Any]) -> list[str]:
    tables = []
    for key in ("from_table", "to_table"):
        if entity.get(key):
            tables.append(str(entity[key]))
    if not tables and entity.get("target_table"):
        tables.append(str(entity["target_table"]))
    return list(dict.fromkeys(tables))


def target_tables_for_entity(entity: dict[str, Any]) -> list[str]:
    target_type = entity.get("target_type")
    if target_type in {"table", "column"} and entity.get("target_table"):
        return [str(entity["target_table"])]
    if target_type == "relationship":
        return relationship_tables(entity)
    return []


def merge_entity_match(pool: dict[str, dict[str, Any]], entity: dict[str, Any]) -> dict[str, Any]:
    entity_id = entity["entity_id"]
    current = pool.get(entity_id)
    if current is None:
        current = dict(entity)
        pool[entity_id] = current
    elif float(entity["vector_score"]) > float(current["vector_score"]):
        matched_queries = current.get("matched_queries", [])
        current.update(entity)
        current["matched_queries"] = matched_queries

    for query in entity.get("matched_queries", []):
        if query not in current["matched_queries"]:
            current["matched_queries"].append(query)
    return current


def aggregate_table_scores(matched_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table_scores: dict[str, dict[str, Any]] = {}
    for entity in matched_entities:
        for table_name in target_tables_for_entity(entity):
            current = table_scores.setdefault(
                table_name,
                {
                    "table": table_name,
                    "score": 0.0,
                    "max_vector_score": 0.0,
                    "evidence_entity_ids": [],
                    "evidence": [],
                    "score_source": "vector_score_only",
                    "score_aggregation": "max_vector_score",
                    "lexical_enabled": False,
                },
            )
            score = float(entity.get("vector_score", 0.0))
            current["max_vector_score"] = max(current["max_vector_score"], score)
            current["score"] = current["max_vector_score"]
            current["evidence_entity_ids"].append(entity["entity_id"])
            current["evidence"].append(
                {
                    "entity_id": entity["entity_id"],
                    "entity_type": entity.get("entity_type"),
                    "target_type": entity.get("target_type"),
                    "target_column": entity.get("target_column"),
                    "vector_score": round(score, 6),
                    "text": entity.get("text", ""),
                    "matched_queries": entity.get("matched_queries", []),
                }
            )

    for row in table_scores.values():
        row["score"] = round(float(row["score"]), 6)
        row["max_vector_score"] = round(float(row["max_vector_score"]), 6)
        row["evidence"] = sorted(row["evidence"], key=lambda item: item["vector_score"], reverse=True)[:10]
    return sorted(table_scores.values(), key=lambda item: item["score"], reverse=True)


def table_relationships(summary: dict[str, Any], table_names: list[str]) -> list[dict[str, Any]]:
    selected = set(table_names)
    return [
        rel
        for rel in summary.get("relationships", [])
        if rel.get("from_table") in selected or rel.get("to_table") in selected
    ]


def build_candidate_table_context(
    summary: dict[str, Any],
    candidate_table_scores: list[dict[str, Any]],
    matched_entities: list[dict[str, Any]],
    top_entities_per_table: int = 8,
) -> dict[str, Any]:
    table_score_by_name = {row["table"]: row for row in candidate_table_scores}
    entity_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in matched_entities:
        for table_name in target_tables_for_entity(entity):
            if table_name in table_score_by_name:
                entity_evidence[table_name].append(entity)

    tables = []
    for table in iter_tables(summary):
        table_name = table["table_name"]
        if table_name not in table_score_by_name:
            continue
        evidence = sorted(entity_evidence[table_name], key=lambda item: item["vector_score"], reverse=True)
        tables.append(
            {
                "table": table_name,
                "schema": table.get("schema_name", summary.get("schema_name", "public")),
                "score": table_score_by_name[table_name]["score"],
                "aliases": table.get("table_aliases", []),
                "description": table.get("table_description", ""),
                "columns": [
                    {
                        "name": column["name"],
                        "type": column.get("type", ""),
                        "description": column.get("column_description", ""),
                        "aliases": column.get("column_aliases", []),
                    }
                    for column in table.get("columns", [])
                ],
                "top_retrieved_entities": [
                    {
                        "entity_id": entity["entity_id"],
                        "entity_type": entity.get("entity_type"),
                        "target_type": entity.get("target_type"),
                        "target_column": entity.get("target_column"),
                        "vector_score": entity.get("vector_score"),
                        "text": entity.get("text", ""),
                        "matched_queries": entity.get("matched_queries", []),
                    }
                    for entity in evidence[:top_entities_per_table]
                ],
                "relationships": [
                    rel
                    for rel in summary.get("relationships", [])
                    if rel.get("from_table") == table_name or rel.get("to_table") == table_name
                ],
            }
        )

    return {
        "lexical_enabled": False,
        "tables": tables,
        "relationships": table_relationships(summary, [row["table"] for row in candidate_table_scores]),
    }


def render_candidate_table_context(candidate_context: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Candidate tables from vector-only schema entity retrieval.")
    lines.append("Important: choose only table names from this candidate list.")
    if candidate_context.get("relationships"):
        lines.append("")
        lines.append("Relationship graph among candidate tables:")
        for rel in candidate_context["relationships"]:
            lines.append(
                f"- {rel['from_table']}.{rel['from_column']} -> "
                f"{rel['to_table']}.{rel['to_column']}"
            )
        lines.append(
            "When a query needs columns from tables that are not directly connected, "
            "include the bridge table(s) shown in this relationship graph."
        )
    for table in candidate_context.get("tables", []):
        lines.append("")
        lines.append(f"TABLE {table['schema']}.{table['table']} score={table['score']}")
        if table.get("description"):
            lines.append(f"- description: {table['description']}")
        if table.get("aliases"):
            lines.append(f"- aliases: {', '.join(table['aliases'])}")
        column_names = ", ".join(f"{column['name']}:{column['type']}" for column in table.get("columns", []))
        lines.append(f"- columns: {column_names}")
        if table.get("top_retrieved_entities"):
            lines.append("- top retrieved entities:")
            for entity in table["top_retrieved_entities"]:
                lines.append(
                    "  * "
                    f"{entity['entity_type']} score={entity['vector_score']} "
                    f"target={table['table']}.{entity.get('target_column') or ''} "
                    f"text={entity.get('text', '')}"
                )
        if table.get("relationships"):
            lines.append("- relationships:")
            for rel in table["relationships"]:
                lines.append(
                    f"  * {rel['from_table']}.{rel['from_column']} -> "
                    f"{rel['to_table']}.{rel['to_column']}"
                )
    return "\n".join(lines).strip()


def retrieve_rasl_table_candidates(
    question: str,
    mode: str = "mock_relational",
    top_k_candidate_tables: int = 5,
    top_m_per_entity_type: int = 5,
) -> dict[str, Any]:
    config = load_config()
    table_name = config.get("pgvector_table", DEFAULT_PGVECTOR_TABLE)
    summary = load_summary(mode)
    entities = load_entities(mode)
    entities_by_id = {entity["entity_id"]: entity for entity in entities}
    decomposition = decompose_question(question)
    retrieval_strings = build_rasl_retrieval_strings(question, decomposition)
    embedding_model = EmbeddingModel(config)

    try:
        query_vectors = embedding_model.encode_many(retrieval_strings)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(MISSING_INDEX_MESSAGE.replace("<mode>", mode)) from exc

    pool: dict[str, dict[str, Any]] = {}
    retrieved_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieved_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query, query_vector in zip(retrieval_strings, query_vectors):
        for entity_type in RASL_ENTITY_TYPES:
            rows = search_pgvector_by_entity_type(
                query=query,
                query_embedding=query_vector,
                mode=mode,
                entity_type=entity_type,
                top_m=top_m_per_entity_type,
                table_name=table_name,
                entities_by_id=entities_by_id,
            )
            for entity in rows:
                merged = merge_entity_match(pool, entity)
                retrieved_by_query[query].append(merged)
                retrieved_by_type[entity_type].append(merged)

    matched_entities = sorted(pool.values(), key=lambda item: item["vector_score"], reverse=True)
    all_table_scores = aggregate_table_scores(matched_entities)
    candidate_table_scores = all_table_scores[:top_k_candidate_tables]
    candidate_table_context = build_candidate_table_context(summary, candidate_table_scores, matched_entities)

    return {
        "mode": mode,
        "retrieval_mode": "rasl_table_retrieval",
        "query_decomposition": decomposition,
        "retrieval_queries": retrieval_strings,
        "retrieval_strings": retrieval_strings,
        "retrieval_metadata": {
            "vector_store": "pgvector",
            "pgvector_table": table_name,
            "embedding_model": embedding_model.model_name,
            "embedding_dim": embedding_model.dim,
            "retrieval_query_count": len(retrieval_strings),
            "entity_types": list(RASL_ENTITY_TYPES),
            "top_m_per_entity_type": top_m_per_entity_type,
            "top_k_candidate_tables": top_k_candidate_tables,
            "lexical_enabled": False,
            "scoring": "vector_score_only",
        },
        "matched_entities": matched_entities,
        "retrieved_entities_by_query": {
            query: sorted(rows, key=lambda item: item["vector_score"], reverse=True)[: top_m_per_entity_type * 2]
            for query, rows in retrieved_by_query.items()
        },
        "retrieved_entities_by_entity_type": {
            entity_type: sorted(rows, key=lambda item: item["vector_score"], reverse=True)[: top_m_per_entity_type * 2]
            for entity_type, rows in retrieved_by_type.items()
        },
        "candidate_table_scores": candidate_table_scores,
        "candidate_tables": [row["table"] for row in candidate_table_scores],
        "all_table_scores": all_table_scores,
        "candidate_table_context": candidate_table_context,
        "candidate_table_context_text": render_candidate_table_context(candidate_table_context),
        "lexical_enabled": False,
    }
