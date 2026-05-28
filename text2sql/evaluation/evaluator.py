import csv
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from schema.schema_context_builder import build_schema_context
from sql_executor import execute_sql
from sql_repair import repair_sql
from sql_validator import validate_sql
from versions.v1_baseline import generate_sql


SCHEMA_PATH = BASE_DIR / "schema" / "schema_summary.json"
TEST_CASES_PATH = BASE_DIR / "data" / "test_cases_v1.csv"
RESULTS_PATH = BASE_DIR / "evaluation" / "results_v1.csv"
MAX_REPAIR_ATTEMPTS = 2


def get_readonly_url() -> str:
    load_dotenv(BASE_DIR / ".env")
    db_url = os.getenv("DB_READONLY_URL")
    if not db_url:
        raise RuntimeError("Missing DB_READONLY_URL in text2sql/.env")
    return db_url


def execute_gold_sql(sql: str) -> dict[str, Any]:
    engine = create_engine(
        get_readonly_url(),
        connect_args={"options": "-c statement_timeout=30000"},
    )
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(sql.strip().rstrip(";"), conn)
        return {"success": True, "data": df, "error": None}
    except Exception as exc:
        return {"success": False, "data": None, "error": str(exc)}
    finally:
        engine.dispose()


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).lower() for column in normalized.columns]

    for column in normalized.select_dtypes(include="number").columns:
        normalized[column] = normalized[column].round(2)

    if len(normalized.columns) > 0:
        normalized = normalized.sort_values(
            by=list(normalized.columns),
            kind="mergesort",
            na_position="last",
        )

    return normalized.reset_index(drop=True)


def dataframes_match(generated: pd.DataFrame, gold: pd.DataFrame) -> bool:
    generated_normalized = normalize_dataframe(generated)
    gold_normalized = normalize_dataframe(gold)
    if list(generated_normalized.columns) != list(gold_normalized.columns):
        return False
    return generated_normalized.equals(gold_normalized)


def evaluate_execute_case(case: dict[str, str], schema_context: str) -> dict[str, Any]:
    repair_attempted = False
    generated_sql = ""
    error = ""

    try:
        generated = generate_sql(case["question"], schema_context)
        generated_sql = generated["sql"]
    except Exception as exc:
        return {
            "generated_sql": generated_sql,
            "is_valid": False,
            "execution_success": False,
            "is_correct": False,
            "error": str(exc),
            "repair_attempted": False,
        }

    validation = validate_sql(generated_sql, str(SCHEMA_PATH))
    is_valid = validation["valid"]
    if not is_valid:
        return {
            "generated_sql": generated_sql,
            "is_valid": False,
            "execution_success": False,
            "is_correct": False,
            "error": validation["error"],
            "repair_attempted": False,
        }

    current_sql = validation["sql"]
    execution = execute_sql(current_sql)
    for _ in range(MAX_REPAIR_ATTEMPTS):
        if execution["success"]:
            break
        repair_attempted = True
        repaired = repair_sql(case["question"], current_sql, execution["error"], schema_context)
        if not repaired.get("success"):
            error = repaired.get("error", "Repair failed after 2 attempts")
            break

        repaired_validation = validate_sql(repaired["sql"], str(SCHEMA_PATH))
        if not repaired_validation["valid"]:
            current_sql = repaired["sql"]
            execution = {"success": False, "data": None, "error": repaired_validation["error"]}
            continue

        current_sql = repaired_validation["sql"]
        generated_sql = current_sql
        execution = execute_sql(current_sql)

    if not execution["success"]:
        return {
            "generated_sql": generated_sql,
            "is_valid": is_valid,
            "execution_success": False,
            "is_correct": False,
            "error": error or execution["error"],
            "repair_attempted": repair_attempted,
        }

    gold_execution = execute_gold_sql(case["gold_sql"])
    if not gold_execution["success"]:
        return {
            "generated_sql": generated_sql,
            "is_valid": is_valid,
            "execution_success": True,
            "is_correct": False,
            "error": f"Gold SQL failed: {gold_execution['error']}",
            "repair_attempted": repair_attempted,
        }

    is_correct = dataframes_match(execution["data"], gold_execution["data"])
    return {
        "generated_sql": generated_sql,
        "is_valid": is_valid,
        "execution_success": True,
        "is_correct": is_correct,
        "error": "" if is_correct else "Result mismatch",
        "repair_attempted": repair_attempted,
    }


def evaluate_block_case(case: dict[str, str], schema_context: str) -> dict[str, Any]:
    generated_sql = ""
    try:
        generated = generate_sql(case["question"], schema_context)
        generated_sql = generated["sql"]
    except Exception as exc:
        return {
            "generated_sql": generated_sql,
            "is_valid": False,
            "execution_success": False,
            "is_correct": True,
            "error": str(exc),
            "repair_attempted": False,
        }

    if generated_sql.strip().upper() == "NO_SQL":
        return {
            "generated_sql": generated_sql,
            "is_valid": False,
            "execution_success": False,
            "is_correct": True,
            "error": "Blocked as NO_SQL",
            "repair_attempted": False,
        }

    validation = validate_sql(generated_sql, str(SCHEMA_PATH))
    blocked = not validation["valid"]
    return {
        "generated_sql": generated_sql,
        "is_valid": validation["valid"],
        "execution_success": False,
        "is_correct": blocked,
        "error": validation["error"] if blocked else "Unsafe intent was not blocked",
        "repair_attempted": False,
    }


def read_test_cases() -> list[dict[str, str]]:
    with TEST_CASES_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_results(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "category",
        "question",
        "gold_sql",
        "generated_sql",
        "is_valid",
        "execution_success",
        "is_correct",
        "error",
        "repair_attempted",
    ]
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], execute_total: int, unsafe_total: int) -> None:
    execute_rows = [row for row in rows if row["expected_behavior"] == "execute"]
    unsafe_rows = [row for row in rows if row["expected_behavior"] == "block"]

    sql_valid = sum(bool(row["is_valid"]) for row in execute_rows)
    execution_success = sum(bool(row["execution_success"]) for row in execute_rows)
    execution_accuracy = sum(bool(row["is_correct"]) for row in execute_rows)
    unsafe_blocked = sum(bool(row["is_correct"]) for row in unsafe_rows)

    def pct(value: int, total: int) -> str:
        return f"{(value / total * 100):.1f}%" if total else "0.0%"

    print("=== V1 Evaluation Summary ===")
    print(f"Total test cases   : {len(rows)}")
    print(f"SQL valid rate     : {sql_valid}/{execute_total} ({pct(sql_valid, execute_total)})")
    print(f"Execution success  : {execution_success}/{execute_total} ({pct(execution_success, execute_total)})")
    print(f"Execution accuracy : {execution_accuracy}/{execute_total} ({pct(execution_accuracy, execute_total)})")
    print(f"Unsafe block rate  : {unsafe_blocked}/{unsafe_total}  ({pct(unsafe_blocked, unsafe_total)})")


def main() -> int:
    if not SCHEMA_PATH.exists():
        print(f"Schema summary not found: {SCHEMA_PATH}")
        print("Run: python schema/generate_schema_summary.py")
        return 1

    schema_context = build_schema_context(str(SCHEMA_PATH))
    test_cases = read_test_cases()
    results = []

    for case in test_cases:
        if case["expected_behavior"] == "execute":
            metrics = evaluate_execute_case(case, schema_context)
        else:
            metrics = evaluate_block_case(case, schema_context)

        results.append(
            {
                **case,
                **metrics,
            }
        )

    write_results(
        [
            {key: value for key, value in row.items() if key != "expected_behavior"}
            for row in results
        ]
    )
    print_summary(
        results,
        execute_total=sum(row["expected_behavior"] == "execute" for row in results),
        unsafe_total=sum(row["expected_behavior"] == "block" for row in results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
