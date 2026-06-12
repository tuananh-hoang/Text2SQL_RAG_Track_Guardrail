import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import text

from text2sql.app.run_v2 import generate_v2_sql
from text2sql.db.postgres_utils import create_readonly_engine
from text2sql.schema.context.selected_schema_context_builder import build_selected_schema_context
from text2sql.schema.paths import v2_summary_path
from text2sql.schema.retrieval.build_schema_pgvector_index import EmbeddingModel, load_config
from text2sql.schema.retrieval.pgvector_store import (
    DEFAULT_PGVECTOR_TABLE,
    qualified_table_name,
    vector_literal,
)
from text2sql.schema.retrieval.query_decomposer import decompose_question
from text2sql.schema.retrieval.schema_retriever import (
    build_retrieval_strings,
    lexical_score,
    load_entities,
    retrieve_schema,
)
from text2sql.sql.executor import execute_sql
from text2sql.sql.validator import validate_sql


BASE_DIR = Path(__file__).resolve().parents[1]
DEBUG_TRACE_PATH = BASE_DIR / "debug_artifacts" / "rasl_debug_traces.jsonl"
DEBUG_MODES = {"full_schema", "table_only", "table_column", "rasl_zero_shot"}
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
    return json.loads(v2_summary_path(mode).read_text(encoding="utf-8"))


def table_names(summary: dict[str, Any]) -> list[str]:
    return [table["table_name"] for table in summary.get("tables", [])]


def columns_for_tables(summary: dict[str, Any], tables: list[str]) -> list[str]:
    selected = set(tables)
    columns: list[str] = []
    for table in summary.get("tables", []):
        if table["table_name"] not in selected:
            continue
        columns.extend(f"{table['table_name']}.{column['name']}" for column in table.get("columns", []))
    return columns


def all_relationship_entities(summary: dict[str, Any], tables: list[str]) -> list[dict[str, Any]]:
    selected = set(tables)
    out = []
    for rel in summary.get("relationships", []):
        if rel["from_table"] in selected and rel["to_table"] in selected:
            out.append(
                {
                    "target_type": "relationship",
                    "from_table": rel["from_table"],
                    "from_column": rel["from_column"],
                    "to_table": rel["to_table"],
                    "to_column": rel["to_column"],
                }
            )
    return out


def write_debug_trace(record: dict[str, Any]) -> None:
    DEBUG_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def parse_pipe_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split("|") if item.strip()]


def normalize_column(value: str) -> str:
    parts = value.split(".")
    if len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    return value


def normalize_table(value: str) -> str:
    return value.split(".")[-1]


def recall(selected: list[str], gold: list[str], normalizer) -> float | str:
    gold_set = {normalizer(item) for item in gold}
    if not gold_set:
        return ""
    selected_set = {normalizer(item) for item in selected}
    return round(len(selected_set & gold_set) / len(gold_set), 4)


def rank_lookup(score_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for index, row in enumerate(score_rows, start=1):
        key = normalize_column(str(row.get("column", "")))
        out[key] = {**row, "rank": index}
    return out


def entities_by_column(entities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        table = entity.get("target_table")
        column = entity.get("target_column")
        if table and column:
            out[f"{table}.{column}"].append(entity)
    return out


def missed_column_details(
    gold_columns: list[str],
    selected_columns: list[str],
    all_column_scores: list[dict[str, Any]],
    matched_entities: list[dict[str, Any]],
    schema_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_set = {normalize_column(item) for item in selected_columns}
    score_by_column = rank_lookup(all_column_scores)
    matched_by_column = entities_by_column(matched_entities)
    schema_by_column = entities_by_column(schema_entities)
    details = []
    for column in gold_columns:
        normalized = normalize_column(column)
        if normalized in selected_set:
            continue
        score_row = score_by_column.get(normalized, {})
        matched = matched_by_column.get(normalized, [])
        details.append(
            {
                "column": normalized,
                "exists_in_schema_entities": normalized in schema_by_column,
                "rank": score_row.get("rank"),
                "score": score_row.get("score"),
                "retrieved_entity_types": sorted({item.get("entity_type") for item in matched if item.get("entity_type")}),
                "matched_retrieval_queries": sorted(
                    {
                        query
                        for item in matched
                        for query in item.get("matched_queries", [])
                        if query
                    }
                ),
            }
        )
    return details


def map_entities_by_query(matched_entities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in matched_entities:
        queries = entity.get("matched_queries") or ["<lexical_only>"]
        for query in queries:
            by_query[query].append(entity)
    return dict(by_query)


def map_entities_by_type(matched_entities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in matched_entities:
        by_type[str(entity.get("entity_type", "unknown"))].append(entity)
    return dict(by_type)


def search_pgvector_by_entity_type(
    engine,
    query: str,
    query_embedding: list[float],
    mode: str,
    entity_type: str,
    top_m: int,
    table_name: str,
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
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            row["metadata"] = json.loads(metadata)
        row["matched_query"] = query
    return rows


def row_to_debug_entity(row: dict[str, Any], entities_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entity = dict(entities_by_id.get(row["entity_id"], {}))
    if not entity:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, dict) and isinstance(metadata.get("entity"), dict):
            entity = dict(metadata["entity"])
        else:
            entity = {
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
    return entity


def retrieve_rasl_zero_shot(
    question: str,
    mode: str,
    top_k_tables: int,
    top_k_columns: int,
    top_m: int = 5,
) -> dict[str, Any]:
    config = load_config()
    table_name = config.get("pgvector_table", DEFAULT_PGVECTOR_TABLE)
    entities = load_entities(mode)
    entities_by_id = {entity["entity_id"]: entity for entity in entities}
    decomposition = decompose_question(question)
    retrieval_strings = build_retrieval_strings(question, decomposition)
    embedding_model = EmbeddingModel(config)
    query_vectors = embedding_model.encode_many(retrieval_strings)

    pool: dict[str, dict[str, Any]] = {}
    retrieved_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieved_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    engine = create_readonly_engine(connect_args={"options": "-c statement_timeout=30000"})
    try:
        for query, query_vector in zip(retrieval_strings, query_vectors):
            for entity_type in RASL_ENTITY_TYPES:
                rows = search_pgvector_by_entity_type(engine, query, query_vector, mode, entity_type, top_m, table_name)
                for row in rows:
                    entity = row_to_debug_entity(row, entities_by_id)
                    vector_score = float(row["similarity"])
                    lexical = lexical_score(query, entity)
                    boost = float(config.get("entity_type_boost", {}).get(entity["entity_type"], 0.5))
                    final_score = 0.50 * vector_score + 0.35 * lexical + 0.15 * boost
                    current = pool.get(entity["entity_id"])
                    if current is None or final_score > current["final_score"]:
                        current = {
                            **entity,
                            "final_score": round(final_score, 6),
                            "vector_score": round(vector_score, 6),
                            "lexical_score": round(lexical, 6),
                            "entity_type_boost": boost,
                            "matched_queries": [],
                            "reason": ["pgvector entity-type query"],
                        }
                        pool[entity["entity_id"]] = current
                    if query not in current["matched_queries"]:
                        current["matched_queries"].append(query)
                    retrieved_by_query[query].append(current)
                    retrieved_by_type[entity_type].append(current)
    finally:
        engine.dispose()

    matched_entities = sorted(pool.values(), key=lambda item: item["final_score"], reverse=True)
    column_agg: dict[str, dict[str, Any]] = {}
    table_agg: dict[str, dict[str, Any]] = {}
    matched_relationships = []
    for item in matched_entities:
        score = item["final_score"]
        if item["target_type"] == "column":
            key = f"{item['target_table']}.{item['target_column']}"
            current = column_agg.setdefault(
                key,
                {"column": key, "table": item["target_table"], "score": 0.0, "evidence_entity_ids": []},
            )
            current["score"] = max(current["score"], score)
            current["evidence_entity_ids"].append(item["entity_id"])
            table = table_agg.setdefault(
                item["target_table"],
                {"table": item["target_table"], "score": 0.0, "evidence_entity_ids": []},
            )
            table["score"] = max(table["score"], score * 0.95)
            table["evidence_entity_ids"].append(item["entity_id"])
        elif item["target_type"] == "table":
            table = table_agg.setdefault(
                item["target_table"],
                {"table": item["target_table"], "score": 0.0, "evidence_entity_ids": []},
            )
            table["score"] = max(table["score"], score)
            table["evidence_entity_ids"].append(item["entity_id"])
        elif item["target_type"] == "relationship":
            matched_relationships.append(item)

    all_column_scores = sorted(column_agg.values(), key=lambda item: item["score"], reverse=True)
    all_table_scores = sorted(table_agg.values(), key=lambda item: item["score"], reverse=True)
    selected_columns = all_column_scores[:top_k_columns]
    selected_tables = all_table_scores[:top_k_tables]
    return {
        "mode": mode,
        "query_decomposition": decomposition,
        "retrieval_strings": retrieval_strings,
        "retrieval_metadata": {
            "vector_store": "pgvector",
            "pgvector_table": table_name,
            "embedding_model": embedding_model.model_name,
            "embedding_dim": embedding_model.dim,
            "retrieval_query_count": len(retrieval_strings),
            "matched_entities_count": len(matched_entities),
            "rasl_top_m": top_m,
        },
        "selected_tables": [item["table"] for item in selected_tables],
        "selected_columns": [item["column"] for item in selected_columns],
        "matched_entities": matched_entities,
        "column_scores": selected_columns,
        "table_scores": selected_tables,
        "all_column_scores": all_column_scores,
        "all_table_scores": all_table_scores,
        "matched_relationships": matched_relationships,
        "retrieved_entities_by_query": {key: value[:top_m] for key, value in retrieved_by_query.items()},
        "retrieved_entities_by_entity_type": {key: value[:top_m] for key, value in retrieved_by_type.items()},
    }


def retrieve_for_debug_mode(
    question: str,
    debug_mode: str,
    top_k_tables: int = 5,
    top_k_columns: int = 8,
) -> dict[str, Any]:
    if debug_mode not in DEBUG_MODES:
        raise ValueError(f"Unknown debug mode: {debug_mode}")

    summary = load_summary("mock_relational")
    if debug_mode == "full_schema":
        selected_tables = table_names(summary)
        selected_columns = columns_for_tables(summary, selected_tables)
        return {
            "mode": "mock_relational",
            "query_decomposition": {},
            "retrieval_strings": [question],
            "retrieval_metadata": {"debug_mode": debug_mode, "retrieval_used": False},
            "selected_tables": selected_tables,
            "selected_columns": selected_columns,
            "matched_entities": [],
            "column_scores": [{"column": column, "table": column.split(".", 1)[0], "score": 1.0} for column in selected_columns],
            "table_scores": [{"table": table, "score": 1.0} for table in selected_tables],
            "all_column_scores": [{"column": column, "table": column.split(".", 1)[0], "score": 1.0} for column in selected_columns],
            "all_table_scores": [{"table": table, "score": 1.0} for table in selected_tables],
            "matched_relationships": all_relationship_entities(summary, selected_tables),
            "retrieved_entities_by_query": {},
            "retrieved_entities_by_entity_type": {},
        }

    if debug_mode == "rasl_zero_shot":
        retrieval = retrieve_rasl_zero_shot(question, "mock_relational", top_k_tables, top_k_columns)
    else:
        retrieval = retrieve_schema(
            question,
            mode="mock_relational",
            top_k_entities=40,
            top_k_columns=top_k_columns,
            top_k_tables=top_k_tables,
            use_relative_cutoff=False,
            min_score=0.0,
        )
        retrieval["retrieved_entities_by_query"] = map_entities_by_query(retrieval["matched_entities"])
        retrieval["retrieved_entities_by_entity_type"] = map_entities_by_type(retrieval["matched_entities"])

    if debug_mode == "table_only":
        selected_tables = [item["table"] for item in retrieval.get("all_table_scores", [])[:top_k_tables]]
        if not selected_tables:
            selected_tables = retrieval["selected_tables"][:top_k_tables]
        selected_columns = columns_for_tables(summary, selected_tables)
        retrieval = {
            **retrieval,
            "selected_tables": selected_tables,
            "selected_columns": selected_columns,
            "column_scores": [{"column": column, "table": column.split(".", 1)[0], "score": 1.0} for column in selected_columns],
        }
    return retrieval


def classify_debug_failure(result: dict[str, Any]) -> str:
    validation = result.get("validator_result") or {}
    execution = result.get("execution_result") or {}
    combined_error = f"{validation.get('error', '')} {execution.get('error', '')}".lower()
    if "rate limit" in combined_error or "rate_limit" in combined_error or "429" in combined_error:
        return "llm_rate_limited"
    if result.get("missed_tables"):
        return "missed_table"
    if result.get("missed_columns"):
        return "missed_column"
    sql = str(result.get("generated_sql", ""))
    if sql.strip().upper() == "NO_SQL":
        return "no_sql"
    if not validation.get("valid"):
        return "invalid_sql"
    if not execution.get("success"):
        return "execution_error"
    return "success_or_result_mismatch_unknown"


def run_debug_question(
    question: str,
    debug_mode: str,
    gold_tables: str | list[str] | None = None,
    gold_columns: str | list[str] | None = None,
    case_id: str | int | None = None,
    category: str | None = None,
    debug_context: bool = False,
    top_k_tables: int = 5,
    top_k_columns: int = 8,
) -> dict[str, Any]:
    trace_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    summary = load_summary("mock_relational")
    schema_entities = load_entities("mock_relational")
    gold_table_list = parse_pipe_list(gold_tables)
    gold_column_list = parse_pipe_list(gold_columns)

    retrieval = retrieve_for_debug_mode(question, debug_mode, top_k_tables=top_k_tables, top_k_columns=top_k_columns)
    selected_context = build_selected_schema_context(
        str(v2_summary_path("mock_relational")),
        retrieval["selected_tables"],
        retrieval["selected_columns"],
        retrieval.get("matched_relationships", []),
        mode="mock_relational",
    )
    generated_sql = ""
    generated_error = ""
    try:
        generated = generate_v2_sql(question, selected_context)
        generated_sql = generated["sql"]
    except Exception as exc:
        generated = {}
        generated_error = str(exc)

    if generated_sql.strip().upper() == "NO_SQL" or not generated_sql:
        validation = {
            "valid": False,
            "sql": generated_sql,
            "tables_from_ast": [],
            "columns_from_ast": [],
            "auto_limited": False,
            "error": generated_error or "NO_SQL",
        }
        execution = {"success": False, "data": None, "row_count": 0, "error": generated_error or "NO_SQL"}
    else:
        validation = validate_sql(generated_sql, str(v2_summary_path("mock_relational")))
        if validation["valid"]:
            execution = execute_sql(validation["sql"])
        else:
            execution = {"success": False, "data": None, "row_count": 0, "error": validation["error"]}

    selected_tables = retrieval["selected_tables"]
    selected_columns = retrieval["selected_columns"]
    missed_tables = sorted(
        {normalize_table(item) for item in gold_table_list}
        - {normalize_table(item) for item in selected_tables}
    )
    missed_columns = sorted(
        {normalize_column(item) for item in gold_column_list}
        - {normalize_column(item) for item in selected_columns}
    )
    missed_details = missed_column_details(
        gold_column_list,
        selected_columns,
        retrieval.get("all_column_scores", retrieval.get("column_scores", [])),
        retrieval.get("matched_entities", []),
        schema_entities,
    )
    result = {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "case_id": str(case_id) if case_id is not None else "",
        "category": category or "",
        "question": question,
        "debug_mode": debug_mode,
        "mode": "mock_relational",
        "query_decomposition": retrieval.get("query_decomposition", {}),
        "retrieval_strings": retrieval.get("retrieval_strings", []),
        "retrieved_entities_by_query": retrieval.get("retrieved_entities_by_query", {}),
        "retrieved_entities_by_entity_type": retrieval.get("retrieved_entities_by_entity_type", {}),
        "matched_entities": retrieval.get("matched_entities", []),
        "table_scores": retrieval.get("table_scores", []),
        "column_scores": retrieval.get("column_scores", []),
        "all_table_scores": retrieval.get("all_table_scores", []),
        "all_column_scores": retrieval.get("all_column_scores", []),
        "selected_tables": selected_tables,
        "selected_columns": selected_columns,
        "gold_tables": gold_table_list,
        "gold_columns": gold_column_list,
        "missed_tables": missed_tables,
        "missed_columns": missed_columns,
        "table_recall": recall(selected_tables, gold_table_list, normalize_table),
        "column_recall": recall(selected_columns, gold_column_list, normalize_column),
        "missed_column_details": missed_details,
        "selected_schema_context_length": len(selected_context),
        "generated_sql": validation.get("sql") if validation.get("valid") else generated_sql,
        "llm_payload": generated,
        "validator_result": validation,
        "execution_result": {
            "success": execution["success"],
            "row_count": execution["row_count"],
            "error": execution["error"],
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    result["failure_type"] = classify_debug_failure(result)
    if debug_context:
        result["selected_schema_context"] = selected_context

    # Keep the trace self-contained for root-cause debugging.
    result["schema_table_count"] = len(summary.get("tables", []))
    write_debug_trace(result)
    return result
