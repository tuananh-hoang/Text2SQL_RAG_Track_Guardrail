import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from schema.schema_context_builder import build_schema_context
from sql_executor import execute_sql
from sql_repair import repair_sql
from sql_validator import validate_sql
from versions.v1_baseline import generate_sql


BASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = BASE_DIR / "schema" / "schema_summary.json"
DEMO_QUESTIONS = [
    "Tổng doanh thu là bao nhiêu?",
    "Doanh thu theo từng khu vực?",
    "Top 5 sản phẩm có doanh thu cao nhất?",
    "Có bao nhiêu đơn hàng?",
    "Xóa bảng product_sales đi",
]


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
    generated = generate_sql(question, schema_context)
    sql = generated["sql"]

    if sql.strip().upper() == "NO_SQL":
        return {
            "question": question,
            "sql": sql,
            "valid": False,
            "auto_limited": False,
            "row_count": 0,
            "data": None,
            "explanation": "Không liên quan dữ liệu",
        }

    validation = validate_sql(sql, str(SCHEMA_PATH))
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
        }

    current_sql = validation["sql"]
    execution = execute_sql(current_sql)
    if execution["success"]:
        return {
            "question": question,
            "sql": current_sql,
            "valid": True,
            "auto_limited": validation["auto_limited"],
            "row_count": execution["row_count"],
            "data": execution["data"],
            "explanation": generated.get("explanation"),
        }

    error_msg = execution["error"]
    for _ in range(2):
        repaired = repair_sql(question, current_sql, error_msg, schema_context)
        if not repaired.get("success"):
            break

        repaired_validation = validate_sql(repaired["sql"], str(SCHEMA_PATH))
        if not repaired_validation["valid"]:
            current_sql = repaired["sql"]
            error_msg = repaired_validation["error"]
            continue

        current_sql = repaired_validation["sql"]
        execution = execute_sql(current_sql)
        if execution["success"]:
            return {
                "question": question,
                "sql": current_sql,
                "valid": True,
                "auto_limited": repaired_validation["auto_limited"],
                "row_count": execution["row_count"],
                "data": execution["data"],
                "explanation": repaired.get("explanation"),
            }
        error_msg = execution["error"]

    return {
        "question": question,
        "sql": current_sql,
        "valid": True,
        "auto_limited": validation["auto_limited"],
        "row_count": 0,
        "data": None,
        "error": "Không thể xử lý câu hỏi này",
        "explanation": error_msg,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL V1 baseline")
    parser.add_argument("--demo", action="store_true", help="Run the five manual test questions")
    return parser.parse_args()


def main() -> int:
    if not SCHEMA_PATH.exists():
        print(f"Schema summary not found: {SCHEMA_PATH}")
        print("Run: python schema/generate_schema_summary.py")
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
