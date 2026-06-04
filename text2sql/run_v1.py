import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from schema.schema_context_builder import build_schema_context
from sql.ast_explainer import explain_sql_structure
from sql.executor import execute_sql
from sql.repair import repair_sql
from sql.validator import validate_sql
from versions.v1_baseline import generate_sql


BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema" / "artifacts" / "v1" / "schema_summary.json"
DEMO_QUESTIONS = [
    "Tong doanh thu la bao nhieu?",
    "Doanh thu theo tung khu vuc?",
    "Top 5 san pham co doanh thu cao nhat?",
    "Co bao nhieu don hang?",
    "Xoa bang product_sales di",
]


def add_trace_step(
    trace_steps: list[dict[str, Any]],
    stage: str,
    status: str,
    detail: str,
    data: dict[str, Any] | None = None,
) -> None:
    trace_steps.append(
        {
            "step": len(trace_steps) + 1,
            "stage": stage,
            "status": status,
            "detail": detail,
            "data": data or {},
        }
    )


def schema_trace_summary() -> dict[str, Any]:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        return {
            "table_name": schema.get("table_name", "unknown"),
            "column_count": schema.get("column_count", len(schema.get("columns", []))),
        }
    except Exception:
        return {"table_name": "unknown", "column_count": 0}


def add_sql_structure_trace(trace_steps: list[dict[str, Any]], sql: str) -> dict[str, Any]:
    explanation = explain_sql_structure(sql)
    add_trace_step(
        trace_steps,
        "Phân tích cấu trúc SQL",
        "done" if explanation["success"] else "error",
        (
            "Đã phân tích SQL bằng AST để mô tả cấu trúc truy vấn."
            if explanation["success"]
            else f"Không parse được SQL bằng AST: {explanation['error']}"
        ),
        {
            "sql_structure_summary": explanation["summary_lines"],
            "sql_features": explanation["features"],
            "sql_structure_error": explanation["error"],
        },
    )
    return explanation


def print_dataframe(df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        print("[RESULT]       (empty)")
        return
    print("[RESULT]")
    print(df.to_string(index=False))


def print_run_result(result: dict[str, Any]) -> None:
    print(f"[QUESTION]     {result['question']}")
    print(f"[SQL]          {result.get('sql') or ''}")
    print(f"[VALID]        {str(result.get('valid', False)).lower()}")
    print(f"[AUTO_LIMITED] {str(result.get('auto_limited', False)).lower()}")
    print(f"[ROWS]         {result.get('row_count', 0)}")
    print_dataframe(result.get("data"))
    print(f"[EXPLANATION]  {result.get('explanation') or result.get('error') or ''}")


def run_question(question: str, schema_context: str) -> dict[str, Any]:
    trace_steps: list[dict[str, Any]] = []
    add_trace_step(
        trace_steps,
        "Nhận câu hỏi",
        "done",
        f'Câu hỏi: "{question}"',
        {"question": question},
    )

    schema_info = schema_trace_summary()
    add_trace_step(
        trace_steps,
        "Chuẩn bị schema context",
        "done",
        (
            f"Đã load schema của bảng {schema_info['table_name']} gồm "
            f"{schema_info['column_count']} cột. Schema context có aliases tiếng Việt "
            "và value evidence."
        ),
        {
            "table_name": schema_info["table_name"],
            "column_count": schema_info["column_count"],
            "schema_context_chars": len(schema_context),
        },
    )

    generated = generate_sql(question, schema_context)
    sql = generated["sql"]
    add_trace_step(
        trace_steps,
        "Sinh SQL bằng LLM",
        "done",
        "LLM đã sinh SQL từ câu hỏi và schema context.",
        {"generated_sql": sql},
    )

    if sql.strip().upper() == "NO_SQL":
        add_trace_step(
            trace_steps,
            "Trả kết quả",
            "blocked",
            "LLM trả về NO_SQL nên hệ thống không gửi SQL xuống PostgreSQL.",
            {"validator_error": "NO_SQL"},
        )
        return {
            "question": question,
            "sql": sql,
            "valid": False,
            "auto_limited": False,
            "row_count": 0,
            "data": None,
            "explanation": "Khong lien quan du lieu",
            "trace_steps": trace_steps,
        }

    add_sql_structure_trace(trace_steps, sql)

    validation = validate_sql(sql, str(SCHEMA_PATH))
    validation_detail = (
        "SQL là SELECT, chỉ dùng bảng/cột trong allowlist, không có lệnh nguy hiểm."
        if validation["valid"]
        else f"Lý do: {validation['error']}\nSQL không được execute."
    )
    add_trace_step(
        trace_steps,
        "Kiểm tra SQL an toàn",
        "pass" if validation["valid"] else "blocked",
        validation_detail,
        {
            "tables_from_ast": validation["tables_from_ast"],
            "columns_from_ast": validation["columns_from_ast"],
            "auto_limited": validation["auto_limited"],
            "validator_error": validation["error"],
        },
    )
    if not validation["valid"]:
        return {
            "question": question,
            "sql": sql,
            "valid": False,
            "auto_limited": False,
            "row_count": 0,
            "data": None,
            "error": validation["error"],
            "explanation": generated.get("explanation"),
            "trace_steps": trace_steps,
        }

    current_sql = validation["sql"]
    start = time.perf_counter()
    execution = execute_sql(current_sql)
    execution_time_ms = round((time.perf_counter() - start) * 1000, 2)
    add_trace_step(
        trace_steps,
        "Thực thi SQL",
        "pass" if execution["success"] else "error",
        (
            f"Chạy bằng readonly user. Trả về {execution['row_count']} dòng "
            f"trong {execution_time_ms} ms."
        )
        if execution["success"]
        else f"Lỗi khi execute: {execution['error']}",
        {
            "row_count": execution["row_count"],
            "execution_time_ms": execution_time_ms,
            "execution_error": None if execution["success"] else execution["error"],
        },
    )
    if execution["success"]:
        add_trace_step(
            trace_steps,
            "Trả kết quả",
            "success",
            "Đã trả SQL, bảng kết quả và explanation về UI.",
            {"row_count": execution["row_count"]},
        )
        return {
            "question": question,
            "sql": current_sql,
            "valid": True,
            "auto_limited": validation["auto_limited"],
            "row_count": execution["row_count"],
            "data": execution["data"],
            "explanation": generated.get("explanation"),
            "trace_steps": trace_steps,
        }

    error_msg = execution["error"]
    for attempt in range(2):
        repaired = repair_sql(question, current_sql, error_msg, schema_context)
        if not repaired.get("success"):
            add_trace_step(
                trace_steps,
                "Repair SQL",
                "failed",
                f"Thử sửa SQL lần {attempt + 1}. Lỗi repair: {repaired.get('error', 'Repair failed.')}",
                {"repair_attempt": attempt + 1},
            )
            break

        add_trace_step(
            trace_steps,
            "Repair SQL",
            "attempt",
            f"Thử sửa SQL lần {attempt + 1}. SQL sau repair:",
            {
                "repair_attempt": attempt + 1,
                "repaired_sql": repaired["sql"],
            },
        )
        repaired_validation = validate_sql(repaired["sql"], str(SCHEMA_PATH))
        repaired_validation_detail = (
            "SQL sau repair là SELECT, chỉ dùng bảng/cột trong allowlist, không có lệnh nguy hiểm."
            if repaired_validation["valid"]
            else f"Lý do: {repaired_validation['error']}\nSQL sau repair không được execute."
        )
        add_trace_step(
            trace_steps,
            "Kiểm tra SQL sau repair",
            "pass" if repaired_validation["valid"] else "blocked",
            repaired_validation_detail,
            {
                "tables_from_ast": repaired_validation["tables_from_ast"],
                "columns_from_ast": repaired_validation["columns_from_ast"],
                "auto_limited": repaired_validation["auto_limited"],
                "validator_error": repaired_validation["error"],
            },
        )
        if not repaired_validation["valid"]:
            current_sql = repaired["sql"]
            error_msg = repaired_validation["error"]
            continue

        current_sql = repaired_validation["sql"]
        start = time.perf_counter()
        execution = execute_sql(current_sql)
        execution_time_ms = round((time.perf_counter() - start) * 1000, 2)
        add_trace_step(
            trace_steps,
            "Thực thi SQL sau repair",
            "pass" if execution["success"] else "error",
            (
                f"Chạy bằng readonly user. Trả về {execution['row_count']} dòng "
                f"trong {execution_time_ms} ms."
            )
            if execution["success"]
            else f"Lỗi khi execute: {execution['error']}",
            {
                "row_count": execution["row_count"],
                "execution_time_ms": execution_time_ms,
                "execution_error": None if execution["success"] else execution["error"],
            },
        )
        if execution["success"]:
            add_trace_step(
                trace_steps,
                "Trả kết quả",
                "success",
                "Đã trả SQL sau repair, bảng kết quả và explanation về UI.",
                {"row_count": execution["row_count"]},
            )
            return {
                "question": question,
                "sql": current_sql,
                "valid": True,
                "auto_limited": repaired_validation["auto_limited"],
                "row_count": execution["row_count"],
                "data": execution["data"],
                "explanation": repaired.get("explanation"),
                "trace_steps": trace_steps,
            }
        error_msg = execution["error"]

    add_trace_step(
        trace_steps,
        "Trả kết quả",
        "error",
        "Không thể trả kết quả do lỗi ở bước trước.",
        {"execution_error": error_msg},
    )
    return {
        "question": question,
        "sql": current_sql,
        "valid": True,
        "auto_limited": validation["auto_limited"],
        "row_count": 0,
        "data": None,
        "error": "Khong the xu ly cau hoi nay",
        "explanation": error_msg,
        "trace_steps": trace_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL V1 baseline")
    parser.add_argument("--demo", action="store_true", help="Run the five manual test questions")
    return parser.parse_args()


def main() -> int:
    if not SCHEMA_PATH.exists():
        print(f"Schema summary not found: {SCHEMA_PATH}")
        print("Run: python -m text2sql.schema.generate_schema_summary --mode product_sales")
        return 1

    args = parse_args()
    schema_context = build_schema_context(str(SCHEMA_PATH))
    questions = DEMO_QUESTIONS if args.demo else [input("Question: ").strip()]

    for index, question in enumerate(questions):
        if index:
            print()
        result = run_question(question, schema_context)
        print_run_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
