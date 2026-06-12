import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from text2sql.llm_client import create_chat_completion_with_rotation, get_llm_model, get_llm_provider
from text2sql.schema.context.selected_schema_context_builder import (
    build_full_table_schema_context,
    build_selected_schema_context,
)
from text2sql.schema.paths import v2_summary_path
from text2sql.schema.retrieval.rasl_table_retriever import (
    build_candidate_table_context,
    load_summary,
    retrieve_rasl_table_candidates,
)
from text2sql.schema.retrieval.schema_retriever import retrieve_schema
from text2sql.sql.ast_explainer import explain_sql_structure
from text2sql.sql.executor import execute_sql
from text2sql.sql.validator import validate_sql
from text2sql.versions.v1_baseline import parse_llm_json, validate_llm_payload


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_PATH = BASE_DIR / "prompts" / "v2_schema_retrieval.txt"
TABLE_PREDICTOR_PROMPT_PATH = BASE_DIR / "prompts" / "table_predictor.txt"
RASL_TRACE_PATH = BASE_DIR / "debug_artifacts" / "rasl_table_retrieval_traces.jsonl"
LEGACY_TRACE_PATH = BASE_DIR / "demo_logs" / "query_traces_v2.jsonl"
RETRIEVAL_MODES = {
    "rasl_table_retrieval",
    "full_schema_ablation",
    "table_column_ablation",
    "hybrid_lexical_ablation",
}


def schema_summary_path(mode: str) -> Path:
    return v2_summary_path(mode)


def add_trace_step(
    steps: list[dict[str, Any]],
    stage: str,
    status: str,
    detail: str,
    data: dict[str, Any] | None = None,
) -> None:
    steps.append(
        {
            "step": len(steps) + 1,
            "stage": stage,
            "status": status,
            "detail": detail,
            "data": data or {},
        }
    )


def dataframe_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None:
        return None
    return {
        "columns": [str(column) for column in df.columns],
        "rows": df.to_dict(orient="records"),
    }


def load_v2_prompt(selected_schema_context: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{selected_schema_context}", selected_schema_context)


def generate_v2_sql(question: str, selected_schema_context: str) -> dict[str, Any]:
    prompt = load_v2_prompt(selected_schema_context)
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


def normalize_table_name(table_name: str) -> str:
    return str(table_name).split(".")[-1].strip()


def load_table_predictor_prompt(
    question: str,
    query_decomposition: dict[str, Any],
    candidate_table_context_text: str,
) -> str:
    template = TABLE_PREDICTOR_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{question}", question)
        .replace("{query_decomposition}", json.dumps(query_decomposition, ensure_ascii=False, indent=2))
        .replace("{candidate_table_context}", candidate_table_context_text)
    )


def predict_tables_with_llm(
    question: str,
    query_decomposition: dict[str, Any],
    candidate_table_context_text: str,
    candidate_tables: list[str],
    candidate_table_context: dict[str, Any],
) -> dict[str, Any]:
    prompt = load_table_predictor_prompt(question, query_decomposition, candidate_table_context_text)
    payload = predict_tables_once(prompt)
    parsed = normalize_table_prediction(payload, candidate_tables)
    if not predicted_tables_are_connected(parsed["predicted_tables"], candidate_table_context.get("relationships", [])):
        retry_prompt = (
            prompt
            + "\n\nPrevious table prediction was disconnected in the relationship graph:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\nSelect again. Include bridge table(s) from candidate tables so predicted_tables form a connected join graph."
        )
        retry_payload = predict_tables_once(retry_prompt)
        retry_parsed = normalize_table_prediction(retry_payload, candidate_tables)
        retry_parsed["retry_reason"] = "previous_prediction_disconnected"
        retry_parsed["first_raw_output"] = payload
        retry_parsed["raw_output"] = retry_payload
        return retry_parsed
    return parsed


def predict_tables_once(prompt: str) -> dict[str, Any]:
    response = create_chat_completion_with_rotation(
        model=get_llm_model(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": prompt}],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty table prediction")
    return parse_llm_json(content)


def normalize_table_prediction(payload: dict[str, Any], candidate_tables: list[str]) -> dict[str, Any]:
    candidate_set = {normalize_table_name(table) for table in candidate_tables}
    predicted = []
    for table in payload.get("predicted_tables", []):
        normalized = normalize_table_name(str(table))
        if normalized in candidate_set and normalized not in predicted:
            predicted.append(normalized)
    if not predicted:
        raise ValueError(
            "Table predictor returned no valid candidate table. "
            f"Raw output: {json.dumps(payload, ensure_ascii=False)}"
        )
    confidence = str(payload.get("confidence", "medium")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "predicted_tables": predicted,
        "reasoning_summary": str(payload.get("reasoning_summary", "")).strip(),
        "confidence": confidence,
        "raw_output": payload,
    }


def predicted_tables_are_connected(predicted_tables: list[str], relationships: list[dict[str, Any]]) -> bool:
    selected = {normalize_table_name(table) for table in predicted_tables}
    if len(selected) <= 1:
        return True
    graph = {table: set() for table in selected}
    for rel in relationships:
        left = normalize_table_name(rel.get("from_table", ""))
        right = normalize_table_name(rel.get("to_table", ""))
        if left in selected and right in selected:
            graph[left].add(right)
            graph[right].add(left)
    start = next(iter(selected))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in graph[node] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    return seen == selected


def columns_for_tables(mode: str, tables: list[str]) -> list[str]:
    summary = load_summary(mode)
    selected = {normalize_table_name(table) for table in tables}
    columns = []
    for table in summary.get("tables", []):
        if table["table_name"] not in selected:
            continue
        columns.extend(f"{table['table_name']}.{column['name']}" for column in table.get("columns", []))
    return columns


def all_tables(mode: str) -> list[str]:
    summary = load_summary(mode)
    if "tables" in summary:
        return [table["table_name"] for table in summary["tables"]]
    return [summary["table_name"]]


def all_relationships_for_tables(mode: str, tables: list[str]) -> list[dict[str, Any]]:
    summary = load_summary(mode)
    selected = {normalize_table_name(table) for table in tables}
    return [
        rel
        for rel in summary.get("relationships", [])
        if rel.get("from_table") in selected and rel.get("to_table") in selected
    ]


def build_rasl_context(
    question: str,
    mode: str,
    debug_context: bool,
    top_k_candidate_tables: int,
    top_m_per_entity_type: int,
) -> tuple[str, dict[str, Any]]:
    retrieval = retrieve_rasl_table_candidates(
        question,
        mode=mode,
        top_k_candidate_tables=top_k_candidate_tables,
        top_m_per_entity_type=top_m_per_entity_type,
    )
    table_prediction = predict_tables_with_llm(
        question=question,
        query_decomposition=retrieval["query_decomposition"],
        candidate_table_context_text=retrieval["candidate_table_context_text"],
        candidate_tables=retrieval["candidate_tables"],
        candidate_table_context=retrieval["candidate_table_context"],
    )
    predicted_tables = table_prediction["predicted_tables"]
    context = build_full_table_schema_context(str(schema_summary_path(mode)), predicted_tables, mode=mode)
    rendered_columns = columns_for_tables(mode, predicted_tables)
    retrieval_payload = {
        **retrieval,
        "table_predictor_output": table_prediction,
        "predicted_tables": predicted_tables,
        "rendered_tables": predicted_tables,
        "selected_tables": predicted_tables,
        "selected_columns": rendered_columns,
        "column_scores": [],
        "table_scores": retrieval["candidate_table_scores"],
        "matched_relationships": all_relationships_for_tables(mode, predicted_tables),
        "selected_schema_context_length": len(context),
    }
    if debug_context:
        retrieval_payload["selected_schema_context"] = context
    return context, retrieval_payload


def build_ablation_context(
    question: str,
    mode: str,
    retrieval_mode: str,
    debug_context: bool,
    top_k_candidate_tables: int,
) -> tuple[str, dict[str, Any]]:
    if retrieval_mode == "full_schema_ablation":
        selected_tables = all_tables(mode)
        context = build_full_table_schema_context(str(schema_summary_path(mode)), selected_tables, mode=mode)
        summary = load_summary(mode)
        candidate_table_context = build_candidate_table_context(
            summary,
            [{"table": table, "score": 1.0, "evidence": []} for table in selected_tables],
            matched_entities=[],
        )
        payload = {
            "mode": mode,
            "retrieval_mode": retrieval_mode,
            "query_decomposition": {},
            "retrieval_queries": [question],
            "retrieval_strings": [question],
            "retrieval_metadata": {"retrieval_used": False, "ablation": retrieval_mode, "lexical_enabled": False},
            "matched_entities": [],
            "retrieved_entities_by_query": {},
            "retrieved_entities_by_entity_type": {},
            "candidate_table_scores": [{"table": table, "score": 1.0} for table in selected_tables],
            "candidate_tables": selected_tables,
            "candidate_table_context": candidate_table_context,
            "candidate_table_context_text": "",
            "table_predictor_output": {
                "predicted_tables": selected_tables,
                "reasoning_summary": "Full schema ablation renders all tables.",
                "confidence": "high",
            },
            "predicted_tables": selected_tables,
            "rendered_tables": selected_tables,
            "selected_tables": selected_tables,
            "selected_columns": columns_for_tables(mode, selected_tables),
            "column_scores": [],
            "table_scores": [{"table": table, "score": 1.0} for table in selected_tables],
            "matched_relationships": all_relationships_for_tables(mode, selected_tables),
            "selected_schema_context_length": len(context),
        }
        if debug_context:
            payload["selected_schema_context"] = context
        return context, payload

    retrieval = retrieve_schema(
        question,
        mode=mode,
        top_k_entities=40,
        top_k_columns=8,
        top_k_tables=top_k_candidate_tables,
        use_relative_cutoff=False,
        min_score=0.0,
    )
    context = build_selected_schema_context(
        str(schema_summary_path(mode)),
        retrieval["selected_tables"],
        retrieval["selected_columns"],
        retrieval.get("matched_relationships", []),
        mode=mode,
    )
    payload = {
        **retrieval,
        "retrieval_mode": retrieval_mode,
        "candidate_table_scores": retrieval.get("all_table_scores", retrieval.get("table_scores", []))[:top_k_candidate_tables],
        "candidate_tables": retrieval.get("selected_tables", []),
        "candidate_table_context": {},
        "candidate_table_context_text": "",
        "table_predictor_output": {},
        "predicted_tables": retrieval.get("selected_tables", []),
        "rendered_tables": retrieval.get("selected_tables", []),
        "selected_schema_context_length": len(context),
    }
    if debug_context:
        payload["selected_schema_context"] = context
    return context, payload


def write_trace(record: dict[str, Any], retrieval_mode: str) -> None:
    path = RASL_TRACE_PATH if retrieval_mode == "rasl_table_retrieval" else LEGACY_TRACE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def final_status_from_result(result: dict[str, Any]) -> str:
    if result.get("valid") and result.get("execution_success"):
        return "success"
    if result.get("error"):
        return "error"
    return "blocked"


def run_question_v2(
    question: str,
    mode: str = "mock_relational",
    debug_context: bool = False,
    retrieval_mode: str = "rasl_table_retrieval",
    top_k_candidate_tables: int = 5,
    top_m_per_entity_type: int = 5,
    debug: bool = False,
) -> dict[str, Any]:
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

    trace_id = str(uuid4())
    trace_steps: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    add_trace_step(trace_steps, "Receive question", "done", f'Question: "{question}"', {"question": question})

    try:
        if retrieval_mode == "rasl_table_retrieval":
            selected_schema_context, retrieval = build_rasl_context(
                question=question,
                mode=mode,
                debug_context=debug_context or debug,
                top_k_candidate_tables=top_k_candidate_tables,
                top_m_per_entity_type=top_m_per_entity_type,
            )
            add_trace_step(
                trace_steps,
                "LLM few-shot decomposition",
                "fallback" if retrieval["query_decomposition"].get("fallback_used") else "done",
                retrieval["query_decomposition"].get("intent_summary")
                or "Question decomposed into retrieval strings for schema retrieval.",
                {
                    "query_decomposition": retrieval["query_decomposition"],
                    "retrieval_queries": retrieval.get("retrieval_queries", []),
                    "schema_concepts": retrieval["query_decomposition"].get("schema_concepts", []),
                    "constraints": retrieval["query_decomposition"].get("constraints", []),
                    "date_expressions": retrieval["query_decomposition"].get("date_expressions", []),
                    "numbers": retrieval["query_decomposition"].get("numbers", []),
                },
            )
            add_trace_step(
                trace_steps,
                "pgvector entity retrieval by query/type",
                "done",
                f"Retrieved {len(retrieval['matched_entities'])} unique schema entities using vector-only search.",
                {
                    "retrieval_metadata": retrieval.get("retrieval_metadata", {}),
                    "retrieved_entities_by_query": retrieval.get("retrieved_entities_by_query", {}),
                    "retrieved_entities_by_entity_type": retrieval.get("retrieved_entities_by_entity_type", {}),
                    "matched_entities": retrieval.get("matched_entities", []),
                    "lexical_enabled": False,
                },
            )
            add_trace_step(
                trace_steps,
                "Aggregate to candidate tables",
                "done",
                "Aggregated vector entity evidence into candidate table scores.",
                {
                    "candidate_table_scores": retrieval.get("candidate_table_scores", []),
                    "candidate_tables": retrieval.get("candidate_tables", []),
                    "candidate_table_context": retrieval.get("candidate_table_context", {}),
                    "lexical_enabled": False,
                },
            )
            add_trace_step(
                trace_steps,
                "LLM table prediction",
                "done",
                f"Predicted tables: {', '.join(retrieval.get('predicted_tables', []))}",
                {
                    "table_predictor_output": retrieval.get("table_predictor_output", {}),
                    "predicted_tables": retrieval.get("predicted_tables", []),
                },
            )
        else:
            selected_schema_context, retrieval = build_ablation_context(
                question=question,
                mode=mode,
                retrieval_mode=retrieval_mode,
                debug_context=debug_context or debug,
                top_k_candidate_tables=top_k_candidate_tables,
            )
            add_trace_step(
                trace_steps,
                "Ablation schema retrieval",
                "done",
                f"Built schema context using {retrieval_mode}.",
                retrieval,
            )

        context_data = {
            "rendered_tables": retrieval.get("rendered_tables", retrieval.get("selected_tables", [])),
            "selected_schema_context_length": len(selected_schema_context),
        }
        if debug_context or debug:
            context_data["selected_schema_context"] = selected_schema_context
        add_trace_step(
            trace_steps,
            "Build full schema context of predicted tables",
            "done",
            f"Rendered tables: {', '.join(context_data['rendered_tables'])}; context length: {len(selected_schema_context)} characters.",
            context_data,
        )

        generated = generate_v2_sql(question, selected_schema_context)
        sql = generated["sql"]
        add_trace_step(
            trace_steps,
            "Generate SQL",
            "done",
            "LLM generated SQL from the rendered schema context.",
            {"generated_sql": sql, "llm_payload": generated},
        )
    except Exception as exc:
        result = {
            "trace_id": trace_id,
            "question": question,
            "mode": mode,
            "retrieval_mode": retrieval_mode,
            "sql": "",
            "valid": False,
            "execution_success": False,
            "row_count": 0,
            "data": None,
            "explanation": "",
            "error": str(exc),
            "trace_steps": trace_steps,
        }
        write_trace_record(result, trace_id, started_at, trace_steps, question, mode, retrieval_mode)
        return result

    if sql.strip().upper() == "NO_SQL":
        add_trace_step(
            trace_steps,
            "Return result",
            "blocked",
            "Generator returned NO_SQL; no SQL was executed.",
            {"generated_sql": sql},
        )
        result = {
            "trace_id": trace_id,
            "question": question,
            "mode": mode,
            "retrieval_mode": retrieval_mode,
            "sql": sql,
            "valid": False,
            "execution_success": False,
            "row_count": 0,
            "data": None,
            "explanation": generated.get("explanation"),
            "error": "NO_SQL",
            "trace_steps": trace_steps,
            **retrieval,
        }
        write_trace_record(result, trace_id, started_at, trace_steps, question, mode, retrieval_mode)
        return result

    ast_summary = explain_sql_structure(sql)
    add_trace_step(
        trace_steps,
        "Analyze SQL AST",
        "done" if ast_summary["success"] else "error",
        "Parsed SQL AST and summarized query structure."
        if ast_summary["success"]
        else f"AST parse error: {ast_summary['error']}",
        {
            "sql_structure_summary": ast_summary["summary_lines"],
            "sql_features": ast_summary["features"],
            "sql_structure_error": ast_summary["error"],
        },
    )

    validation = validate_sql(sql, str(schema_summary_path(mode)))
    add_trace_step(
        trace_steps,
        "Validate SQL",
        "pass" if validation["valid"] else "blocked",
        (
            "SQL is SELECT-only and uses only tables/columns in the schema allowlist."
            if validation["valid"]
            else f"Reason: {validation['error']}. SQL was not executed."
        ),
        {"validator_result": validation},
    )
    if not validation["valid"]:
        result = {
            "trace_id": trace_id,
            "question": question,
            "mode": mode,
            "retrieval_mode": retrieval_mode,
            "sql": sql,
            "valid": False,
            "execution_success": False,
            "row_count": 0,
            "data": None,
            "explanation": generated.get("explanation"),
            "error": validation["error"],
            "trace_steps": trace_steps,
            **retrieval,
        }
        write_trace_record(result, trace_id, started_at, trace_steps, question, mode, retrieval_mode)
        return result

    start = time.perf_counter()
    execution = execute_sql(validation["sql"])
    execution_time_ms = round((time.perf_counter() - start) * 1000, 2)
    add_trace_step(
        trace_steps,
        "Execute SQL",
        "pass" if execution["success"] else "error",
        (
            f"Readonly execution returned {execution['row_count']} rows in {execution_time_ms} ms."
            if execution["success"]
            else f"Execution error: {execution['error']}"
        ),
        {
            "execution_result": {
                "success": execution["success"],
                "row_count": execution["row_count"],
                "error": execution["error"],
                "execution_time_ms": execution_time_ms,
            }
        },
    )

    add_trace_step(
        trace_steps,
        "Return result",
        "success" if execution["success"] else "error",
        "Returned SQL, result table and explanation to caller."
        if execution["success"]
        else "Could not return a successful result because execution failed.",
        {"row_count": execution["row_count"]},
    )
    result = {
        "trace_id": trace_id,
        "question": question,
        "mode": mode,
        "retrieval_mode": retrieval_mode,
        "sql": validation["sql"],
        "valid": True,
        "execution_success": execution["success"],
        "row_count": execution["row_count"],
        "data": execution["data"],
        "explanation": generated.get("explanation"),
        "error": execution["error"],
        "trace_steps": trace_steps,
        **retrieval,
    }
    write_trace_record(result, trace_id, started_at, trace_steps, question, mode, retrieval_mode)
    return result


def write_trace_record(
    result: dict[str, Any],
    trace_id: str,
    started_at: str,
    trace_steps: list[dict[str, Any]],
    question: str,
    mode: str,
    retrieval_mode: str,
) -> None:
    record = {
        "trace_id": trace_id,
        "timestamp": started_at,
        "version": "v2_rasl_table_retrieval",
        "mode": mode,
        "retrieval_mode": retrieval_mode,
        "question": question,
        "final_status": final_status_from_result(result),
        "query_decomposition": result.get("query_decomposition"),
        "retrieval_queries": result.get("retrieval_queries", result.get("retrieval_strings")),
        "retrieved_entities_by_query": result.get("retrieved_entities_by_query"),
        "retrieved_entities_by_entity_type": result.get("retrieved_entities_by_entity_type"),
        "candidate_table_scores": result.get("candidate_table_scores"),
        "candidate_table_context": result.get("candidate_table_context"),
        "table_predictor_output": result.get("table_predictor_output"),
        "predicted_tables": result.get("predicted_tables"),
        "rendered_tables": result.get("rendered_tables"),
        "context_length": result.get("selected_schema_context_length"),
        "generated_sql": result.get("sql"),
        "validator_result": next(
            (step["data"].get("validator_result") for step in trace_steps if step["stage"] == "Validate SQL"),
            None,
        ),
        "execution_result": next(
            (step["data"].get("execution_result") for step in trace_steps if step["stage"] == "Execute SQL"),
            None,
        ),
        "retrieval_metadata": result.get("retrieval_metadata"),
        "matched_entities": result.get("matched_entities"),
        "selected_tables": result.get("selected_tables"),
        "selected_columns": result.get("selected_columns"),
        "steps": trace_steps,
    }
    write_trace(record, retrieval_mode)


def print_cli_result(result: dict[str, Any]) -> None:
    payload = dict(result)
    payload["data"] = dataframe_payload(result.get("data"))
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V2 RASL table retrieval Text-to-SQL pipeline.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=["mock_relational", "product_sales"], default="mock_relational")
    parser.add_argument("--retrieval-mode", choices=sorted(RETRIEVAL_MODES), default="rasl_table_retrieval")
    parser.add_argument("--top-k-candidate-tables", type=int, default=5)
    parser.add_argument("--top-m-per-entity-type", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-context", action="store_true")
    args = parser.parse_args()
    result = run_question_v2(
        args.question,
        mode=args.mode,
        debug_context=args.debug_context,
        retrieval_mode=args.retrieval_mode,
        top_k_candidate_tables=args.top_k_candidate_tables,
        top_m_per_entity_type=args.top_m_per_entity_type,
        debug=args.debug,
    )
    print_cli_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
