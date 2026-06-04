import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from text2sql.llm_client import create_chat_completion_with_rotation, get_llm_model, get_llm_provider
from text2sql.schema.context.selected_schema_context_builder import build_selected_schema_context
from text2sql.schema.paths import v2_summary_path
from text2sql.schema.retrieval.schema_retriever import retrieve_schema
from text2sql.sql.ast_explainer import explain_sql_structure
from text2sql.sql.executor import execute_sql
from text2sql.sql.validator import validate_sql
from text2sql.versions.v1_baseline import parse_llm_json, validate_llm_payload


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_PATH = BASE_DIR / "prompts" / "v2_schema_retrieval.txt"
LOG_PATH = BASE_DIR / "demo_logs" / "query_traces_v2.jsonl"


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


def write_trace(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def final_status_from_result(result: dict[str, Any]) -> str:
    if result.get("valid") and result.get("execution_success"):
        return "success"
    if result.get("error"):
        return "error"
    return "blocked"


def run_question_v2(
    question: str,
    mode: str = "product_sales",
    debug_context: bool = False,
) -> dict[str, Any]:
    trace_id = str(uuid4())
    trace_steps: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    add_trace_step(trace_steps, "Nhận câu hỏi", "done", f'Câu hỏi: "{question}"', {"question": question})

    try:
        retrieval = retrieve_schema(question, mode=mode)
    except RuntimeError as exc:
        add_trace_step(
            trace_steps,
            "Retrieve schema entities",
            "error",
            str(exc),
            {
                "vector_store": "pgvector",
                "mode": mode,
                "build_index_command": f"python -m text2sql.schema.retrieval.build_schema_pgvector_index --mode {mode} --rebuild",
            },
        )
        result = {
            "trace_id": trace_id,
            "question": question,
            "mode": mode,
            "sql": "",
            "valid": False,
            "execution_success": False,
            "row_count": 0,
            "data": None,
            "explanation": "",
            "error": str(exc),
            "trace_steps": trace_steps,
            "query_decomposition": {},
            "matched_entities": [],
            "column_scores": [],
            "table_scores": [],
            "selected_tables": [],
            "selected_columns": [],
            "retrieval_metadata": {"vector_store": "pgvector"},
        }
        write_trace_record(result, trace_id, started_at, trace_steps, question, mode)
        return result
    decomposition = retrieval["query_decomposition"]
    add_trace_step(
        trace_steps,
        "Decompose question bằng lightweight LLM",
        "fallback" if decomposition.get("fallback_used") else "done",
        decomposition.get("intent_summary") or "Đã phân tích câu hỏi thành retrieval queries/schema concepts.",
        {
            "query_decomposition": decomposition,
            "fallback_used": decomposition.get("fallback_used", False),
            "fallback_reason": decomposition.get("fallback_reason", ""),
        },
    )

    add_trace_step(
        trace_steps,
        "Retrieve schema entities",
        "done",
        f"Matched {len(retrieval['matched_entities'])} schema entities.",
        {
            **retrieval.get("retrieval_metadata", {}),
            "matched_entities": retrieval["matched_entities"],
        },
    )
    add_trace_step(
        trace_steps,
        "Aggregate entity scores thành selected table/column",
        "done",
        (
            f"Selected tables: {', '.join(retrieval['selected_tables']) or 'none'}; "
            f"selected columns: {', '.join(retrieval['selected_columns']) or 'none'}."
        ),
        {
            "selected_tables": retrieval["selected_tables"],
            "selected_columns": retrieval["selected_columns"],
            "column_scores": retrieval["column_scores"],
            "table_scores": retrieval["table_scores"],
            "matched_relationships": retrieval["matched_relationships"],
        },
    )

    summary_path = schema_summary_path(mode)
    selected_schema_context = build_selected_schema_context(
        str(summary_path),
        retrieval["selected_tables"],
        retrieval["selected_columns"],
        retrieval["matched_relationships"],
        mode=mode,
    )
    context_data = {
        "selected_schema_context_length": len(selected_schema_context),
    }
    if debug_context:
        context_data["selected_schema_context"] = selected_schema_context
    add_trace_step(
        trace_steps,
        "Build selected schema context",
        "done",
        f"Selected schema context length: {len(selected_schema_context)} characters.",
        context_data,
    )

    generated = generate_v2_sql(question, selected_schema_context)
    sql = generated["sql"]
    add_trace_step(
        trace_steps,
        "Sinh SQL",
        "done",
        "LLM đã sinh SQL từ selected schema context.",
        {"generated_sql": sql, "llm_payload": generated},
    )

    if sql.strip().upper() == "NO_SQL":
        result = {
            "trace_id": trace_id,
            "question": question,
            "mode": mode,
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
        write_trace_record(result, trace_id, started_at, trace_steps, question, mode)
        return result

    ast_summary = explain_sql_structure(sql)
    add_trace_step(
        trace_steps,
        "Phân tích cấu trúc SQL bằng AST",
        "done" if ast_summary["success"] else "error",
        "Đã phân tích SQL bằng AST." if ast_summary["success"] else f"AST parse error: {ast_summary['error']}",
        {
            "sql_structure_summary": ast_summary["summary_lines"],
            "sql_features": ast_summary["features"],
            "sql_structure_error": ast_summary["error"],
        },
    )

    validation = validate_sql(sql, str(summary_path))
    add_trace_step(
        trace_steps,
        "Validate SQL",
        "pass" if validation["valid"] else "blocked",
        (
            "SQL là SELECT và chỉ dùng bảng/cột trong allowlist."
            if validation["valid"]
            else f"Lý do: {validation['error']}. SQL không được execute."
        ),
        {"validator_result": validation},
    )
    if not validation["valid"]:
        result = {
            "trace_id": trace_id,
            "question": question,
            "mode": mode,
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
        write_trace_record(result, trace_id, started_at, trace_steps, question, mode)
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
        "Đã trả kết quả V2." if execution["success"] else "Không thể trả kết quả do lỗi execute.",
        {"row_count": execution["row_count"]},
    )
    result = {
        "trace_id": trace_id,
        "question": question,
        "mode": mode,
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
    write_trace_record(result, trace_id, started_at, trace_steps, question, mode)
    return result


def write_trace_record(
    result: dict[str, Any],
    trace_id: str,
    started_at: str,
    trace_steps: list[dict[str, Any]],
    question: str,
    mode: str,
) -> None:
    record = {
        "trace_id": trace_id,
        "timestamp": started_at,
        "version": "v2_schema_retrieval",
        "mode": mode,
        "question": question,
        "final_status": final_status_from_result(result),
        "query_decomposition": result.get("query_decomposition"),
        "fallback_used": (result.get("query_decomposition") or {}).get("fallback_used", False),
        "retrieval_metadata": result.get("retrieval_metadata"),
        "vector_store": (result.get("retrieval_metadata") or {}).get("vector_store"),
        "embedding_model": (result.get("retrieval_metadata") or {}).get("embedding_model"),
        "embedding_dim": (result.get("retrieval_metadata") or {}).get("embedding_dim"),
        "pgvector_table": (result.get("retrieval_metadata") or {}).get("pgvector_table"),
        "matched_entities": result.get("matched_entities"),
        "column_scores": result.get("column_scores"),
        "table_scores": result.get("table_scores"),
        "selected_tables": result.get("selected_tables"),
        "selected_columns": result.get("selected_columns"),
        "generated_sql": result.get("sql"),
        "validator_result": next(
            (step["data"].get("validator_result") for step in trace_steps if step["stage"] == "Validate SQL"),
            None,
        ),
        "execution_result": next(
            (step["data"].get("execution_result") for step in trace_steps if step["stage"] == "Execute SQL"),
            None,
        ),
        "steps": trace_steps,
    }
    write_trace(record)


def print_cli_result(result: dict[str, Any]) -> None:
    payload = dict(result)
    payload["data"] = dataframe_payload(result.get("data"))
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V2 schema retrieval Text-to-SQL pipeline.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=["product_sales", "mock_relational"], default="product_sales")
    parser.add_argument("--debug-context", action="store_true")
    args = parser.parse_args()
    result = run_question_v2(args.question, mode=args.mode, debug_context=args.debug_context)
    print_cli_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
