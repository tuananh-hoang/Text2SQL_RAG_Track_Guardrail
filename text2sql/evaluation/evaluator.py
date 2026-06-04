import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import sqlglot
from sqlglot import expressions as exp
from dotenv import load_dotenv
from sqlalchemy import create_engine


BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from schema.context.schema_context_builder import build_schema_context
from sql.executor import execute_sql
from sql.repair import repair_sql
from sql.validator import validate_sql
from versions.v1_baseline import generate_sql


SCHEMA_PATH = BASE_DIR / "schema" / "artifacts" / "v1" / "schema_summary.json"
TEST_CASES_PATH = BASE_DIR / "data" / "test_cases_v1.csv"
RESULTS_ROOT = BASE_DIR / "evaluation" / "results"
RESULTS_DIR = RESULTS_ROOT / "v1"
RESULTS_PATH = RESULTS_DIR / "results_v1.csv"
ERROR_ANALYSIS_PATH = RESULTS_DIR / "error_analysis_v1.csv"
V2_RESULTS_PATH = BASE_DIR / "evaluation" / "results" / "v2" / "results_v2_schema_retrieval.csv"
V2_ERROR_ANALYSIS_PATH = BASE_DIR / "evaluation" / "results" / "v2" / "error_analysis_v2_schema_retrieval.csv"
MAX_REPAIR_ATTEMPTS = 2
FAILURE_TYPES = {
    "correct",
    "alias_mismatch_only",
    "wrong_aggregation",
    "wrong_column",
    "wrong_filter",
    "wrong_date_format",
    "schema_linking_fail",
    "should_block_but_ran",
    "should_run_but_blocked",
    "no_sql_wrong",
    "extra_columns",
    "empty_result_suspicious",
    "unknown",
}


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


def strict_dataframes_match(generated: pd.DataFrame, gold: pd.DataFrame) -> bool:
    generated_normalized = normalize_dataframe(generated)
    gold_normalized = normalize_dataframe(gold)
    if list(generated_normalized.columns) != list(gold_normalized.columns):
        return False
    return generated_normalized.equals(gold_normalized)


def dataframes_match_by_position(generated: pd.DataFrame, gold: pd.DataFrame) -> bool:
    if generated.shape != gold.shape:
        return False

    generated_normalized = normalize_dataframe(generated)
    gold_normalized = normalize_dataframe(gold)
    generated_normalized.columns = list(range(len(generated_normalized.columns)))
    gold_normalized.columns = list(range(len(gold_normalized.columns)))
    return generated_normalized.equals(gold_normalized)


def dataframes_match_gold_subset(generated: pd.DataFrame, gold: pd.DataFrame) -> bool:
    generated_columns = {str(column).lower(): column for column in generated.columns}
    selected_columns = []
    for gold_column in gold.columns:
        generated_column = generated_columns.get(str(gold_column).lower())
        if generated_column is None:
            return False
        selected_columns.append(generated_column)

    generated_subset = generated[selected_columns].copy()
    generated_subset.columns = gold.columns
    return strict_dataframes_match(generated_subset, gold)


def compare_dataframes(generated: pd.DataFrame, gold: pd.DataFrame) -> dict[str, Any]:
    strict_correct = strict_dataframes_match(generated, gold)
    if strict_correct:
        return {
            "strict_correct": True,
            "relaxed_correct": True,
            "match_type": "strict",
        }

    # fix: report execution-value correctness separately from output alias exactness.
    if dataframes_match_by_position(generated, gold):
        return {
            "strict_correct": False,
            "relaxed_correct": True,
            "match_type": "alias_only",
        }

    if dataframes_match_gold_subset(generated, gold):
        return {
            "strict_correct": False,
            "relaxed_correct": True,
            "match_type": "gold_subset",
        }

    return {
        "strict_correct": False,
        "relaxed_correct": False,
        "match_type": "mismatch",
    }


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
            "strict_correct": False,
            "relaxed_correct": False,
            "match_type": "generation_error",
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
            "strict_correct": False,
            "relaxed_correct": False,
            "match_type": "invalid_sql",
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
            "strict_correct": False,
            "relaxed_correct": False,
            "match_type": "execution_error",
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
            "strict_correct": False,
            "relaxed_correct": False,
            "match_type": "gold_error",
            "error": f"Gold SQL failed: {gold_execution['error']}",
            "repair_attempted": repair_attempted,
        }

    comparison = compare_dataframes(execution["data"], gold_execution["data"])
    is_correct = comparison["relaxed_correct"]
    return {
        "generated_sql": generated_sql,
        "is_valid": is_valid,
        "execution_success": True,
        "is_correct": is_correct,
        **comparison,
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
            "is_correct": False,
            "strict_correct": False,
            "relaxed_correct": False,
            "match_type": "generation_error",
            "error": str(exc),
            "repair_attempted": False,
        }

    if generated_sql.strip().upper() == "NO_SQL":
        return {
            "generated_sql": generated_sql,
            "is_valid": False,
            "execution_success": False,
            "is_correct": True,
            "strict_correct": True,
            "relaxed_correct": True,
            "match_type": "unsafe_blocked_no_sql",
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
        "strict_correct": blocked,
        "relaxed_correct": blocked,
        "match_type": "unsafe_blocked_by_validator" if blocked else "unsafe_not_blocked",
        "error": validation["error"] if blocked else "Unsafe intent was not blocked",
        "repair_attempted": False,
    }


def read_test_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "difficulty",
        "question",
        "gold_sql",
        "generated_sql",
        "is_valid",
        "execution_success",
        "is_correct",
        "strict_correct",
        "relaxed_correct",
        "match_type",
        "error",
        "repair_attempted",
        "failure_target",
        "query_decomposition",
        "fallback_used",
        "selected_tables",
        "selected_columns",
        "matched_entities",
        "selected_schema_context_length",
        "table_recall",
        "column_recall",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = {field: row.get(field, "") for field in fieldnames}
            output_row["error"] = sanitize_error(output_row["error"])
            writer.writerow(output_row)


def sanitize_error(error: Any) -> str:
    if not error:
        return ""
    text = str(error)
    text = re.sub(r"organization `[^`]+`", "organization `<redacted>`", text)
    if "rate_limit_exceeded" in text or "Error code: 429" in text:
        return "Provider rate limit: 429 rate_limit_exceeded"
    return text


def has_group_by(sql: str) -> bool:
    return "GROUP BY" in sql.upper()


def has_aggregate(sql: str) -> bool:
    normalized = sql.upper()
    return any(token in normalized for token in ("SUM(", "AVG(", "COUNT(", "MIN(", "MAX("))


def has_where(sql: str) -> bool:
    return "WHERE" in sql.upper()


def select_clause(sql: str) -> str:
    match = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def quoted_select_columns(sql: str) -> set[str]:
    return set(re.findall(r'"([^"]+)"', select_clause(sql)))


def select_has_aggregate(sql: str) -> bool:
    return has_aggregate(select_clause(sql))


def classify_failure(row: dict[str, Any]) -> tuple[str, str]:
    match_type = str(row.get("match_type", ""))
    generated_sql = str(row.get("generated_sql", ""))
    gold_sql = str(row.get("gold_sql", ""))
    error = str(row.get("error", ""))
    category = str(row.get("category", ""))
    expected_behavior = str(row.get("expected_behavior", ""))

    if bool(row.get("is_correct")):
        if match_type == "alias_only":
            return "alias_mismatch_only", "Kết quả đúng theo vị trí/giá trị nhưng alias output khác gold SQL."
        if match_type == "gold_subset":
            return "extra_columns", "Generated SQL trả thêm cột ngoài tập cột gold yêu cầu."
        return "correct", "SQL đạt tiêu chí benchmark cho case này."

    if expected_behavior == "block":
        if match_type == "generation_error":
            return "unknown", "Không phân loại được vì lỗi xảy ra ở bước gọi LLM/API."
        return "should_block_but_ran", "Case phải bị block nhưng SQL không bị chặn đúng cách."

    if generated_sql.strip().upper() == "NO_SQL":
        return "no_sql_wrong", "Model trả NO_SQL cho câu hỏi đáng lẽ phải sinh SQL."

    if match_type == "generation_error":
        return "unknown", "Không phân loại được vì lỗi xảy ra ở bước gọi LLM/API."

    if match_type == "invalid_sql":
        if "Column not in schema" in error:
            return "wrong_column", error
        if "Table not in schema" in error:
            return "schema_linking_fail", error
        if "NO_SQL" in error:
            return "no_sql_wrong", "Validator nhận NO_SQL cho execute case."
        return "should_run_but_blocked", f"SQL hợp lệ theo intent nhưng bị block hoặc parse fail: {error}"

    if "SELECT *" in generated_sql.upper():
        return "extra_columns", "Generated SQL dùng SELECT * hoặc trả dư cột."

    gold_select_columns = quoted_select_columns(gold_sql)
    generated_select_columns = quoted_select_columns(generated_sql)
    missing_columns = gold_select_columns - generated_select_columns
    extra_columns = generated_select_columns - gold_select_columns

    if "date" in category or '"Order_Date"' in gold_sql:
        if '"Order_Date"' in gold_sql and '"Order_Date"' not in generated_sql:
            return "wrong_date_format", "Gold SQL cần date filter nhưng generated SQL thiếu Order_Date."
        if ">=" in gold_sql and "<" in gold_sql and not (">=" in generated_sql and "<" in generated_sql):
            return "wrong_date_format", "Generated SQL không dùng half-open date range như gold SQL."

    if category in {"aggregate", "group_by", "top_k_aggregate"}:
        if select_has_aggregate(gold_sql) and not select_has_aggregate(generated_sql):
            return "wrong_aggregation", "Generated SQL thiếu aggregate ở SELECT output."
        if has_aggregate(gold_sql) != has_aggregate(generated_sql):
            return "wrong_aggregation", "Generated SQL thiếu hoặc thừa aggregate so với gold SQL."
        if has_group_by(gold_sql) != has_group_by(generated_sql):
            return "wrong_aggregation", "Generated SQL thiếu hoặc thừa GROUP BY so với entity-level gold SQL."
        if "SUM(" in gold_sql.upper() and "SUM(" not in generated_sql.upper():
            return "wrong_aggregation", "Generated SQL dùng sai aggregate function hoặc sai metric."
        if "AVG(" in gold_sql.upper() and "AVG(" not in generated_sql.upper():
            return "wrong_aggregation", "Generated SQL dùng sai aggregate function hoặc sai metric."
        if "COUNT(" in gold_sql.upper() and "COUNT(" not in generated_sql.upper():
            return "wrong_aggregation", "Generated SQL dùng sai aggregate function hoặc sai metric."

    if missing_columns:
        return "wrong_column", f"Generated SQL thiếu cột output gold yêu cầu: {', '.join(sorted(missing_columns))}."

    if extra_columns:
        return "extra_columns", f"Generated SQL trả dư cột output: {', '.join(sorted(extra_columns))}."

    if has_where(gold_sql) != has_where(generated_sql):
        return "wrong_filter", "Generated SQL thiếu hoặc thừa WHERE so với gold SQL."

    if "Column not in schema" in error:
        return "wrong_column", error

    return "unknown", "Không phân loại được bằng rule tự động, cần review thủ công."


def build_error_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analysis = []
    for row in rows:
        failure_type, diagnosis = classify_failure(row)
        if failure_type not in FAILURE_TYPES:
            failure_type = "unknown"
            diagnosis = "Failure type ngoài taxonomy, cần review thủ công."
        analysis.append(
            {
                "id": row.get("id", ""),
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "question": row.get("question", ""),
                "gold_sql": row.get("gold_sql", ""),
                "generated_sql": row.get("generated_sql", ""),
                "is_correct": row.get("is_correct", False),
                "failure_type": failure_type,
                "diagnosis": diagnosis,
            }
        )
    return analysis


def write_error_analysis(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "difficulty",
        "question",
        "gold_sql",
        "generated_sql",
        "is_correct",
        "failure_type",
        "diagnosis",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    execute_rows = [row for row in rows if row["expected_behavior"] == "execute"]
    unsafe_rows = [row for row in rows if row["expected_behavior"] == "block"]
    return {
        "total": len(rows),
        "execute_total": len(execute_rows),
        "unsafe_total": len(unsafe_rows),
        "sql_valid": sum(bool(row["is_valid"]) for row in execute_rows),
        "execution_success": sum(bool(row["execution_success"]) for row in execute_rows),
        "strict_accuracy": sum(bool(row.get("strict_correct", row["is_correct"])) for row in execute_rows),
        "relaxed_accuracy": sum(bool(row.get("relaxed_correct", row["is_correct"])) for row in execute_rows),
        "unsafe_blocked": sum(bool(row["is_correct"]) for row in unsafe_rows),
    }


def print_summary(rows: list[dict[str, Any]], error_analysis: list[dict[str, Any]], label: str = "V1.1") -> None:
    summary = summarize_rows(rows)
    execute_rows = [row for row in rows if row["expected_behavior"] == "execute"]
    unsafe_rows = [row for row in rows if row["expected_behavior"] == "block"]

    def pct(value: int, total: int) -> str:
        return f"{(value / total * 100):.1f}%" if total else "0.0%"

    print(f"=== {label} Evaluation Summary ===")
    print(f"Total test cases   : {len(rows)}")
    print(f"SQL valid rate     : {summary['sql_valid']}/{summary['execute_total']} ({pct(summary['sql_valid'], summary['execute_total'])})")
    print(f"Execution success  : {summary['execution_success']}/{summary['execute_total']} ({pct(summary['execution_success'], summary['execute_total'])})")
    print(f"Strict accuracy    : {summary['strict_accuracy']}/{summary['execute_total']} ({pct(summary['strict_accuracy'], summary['execute_total'])})")
    print(f"Relaxed accuracy   : {summary['relaxed_accuracy']}/{summary['execute_total']} ({pct(summary['relaxed_accuracy'], summary['execute_total'])})")
    print(f"Unsafe block rate  : {summary['unsafe_blocked']}/{summary['unsafe_total']}  ({pct(summary['unsafe_blocked'], summary['unsafe_total'])})")

    print("\n=== Summary by Category ===")
    for category in sorted({row["category"] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        category_summary = summarize_rows(category_rows)
        if category_summary["execute_total"]:
            print(
                f"{category}: "
                f"strict {category_summary['strict_accuracy']}/{category_summary['execute_total']} "
                f"({pct(category_summary['strict_accuracy'], category_summary['execute_total'])}), "
                f"relaxed {category_summary['relaxed_accuracy']}/{category_summary['execute_total']} "
                f"({pct(category_summary['relaxed_accuracy'], category_summary['execute_total'])})"
            )
        else:
            print(
                f"{category}: "
                f"blocked {category_summary['unsafe_blocked']}/{category_summary['unsafe_total']} "
                f"({pct(category_summary['unsafe_blocked'], category_summary['unsafe_total'])})"
            )

    failure_counts = Counter(
        row["failure_type"]
        for row in error_analysis
        if row["failure_type"] != "correct"
    )
    print("\n=== Top Failure Types ===")
    if failure_counts:
        for failure_type, count in failure_counts.most_common(10):
            print(f"{failure_type}: {count}")
    else:
        print("No failures")

    recall_rows = [
        row
        for row in rows
        if row.get("table_recall") not in ("", None) or row.get("column_recall") not in ("", None)
    ]
    if recall_rows:
        table_values = [float(row["table_recall"]) for row in recall_rows if row.get("table_recall") not in ("", None)]
        column_values = [float(row["column_recall"]) for row in recall_rows if row.get("column_recall") not in ("", None)]
        context_lengths = [
            int(row["selected_schema_context_length"])
            for row in recall_rows
            if str(row.get("selected_schema_context_length", "")).isdigit()
        ]
        print("\n=== Schema Retrieval Diagnostics ===")
        if table_values:
            print(f"Avg table recall   : {sum(table_values) / len(table_values):.3f}")
        if column_values:
            print(f"Avg column recall  : {sum(column_values) / len(column_values):.3f}")
        if context_lengths:
            print(f"Avg context length : {sum(context_lengths) / len(context_lengths):.0f} chars")


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == BASE_DIR.name:
        return BASE_DIR.parent / path
    return BASE_DIR / path


def names_from_sql(sql: str) -> tuple[set[str], set[str]]:
    try:
        ast = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return set(), set()
    tables = {table.name for table in ast.find_all(exp.Table) if table.name}
    columns = {column.name for column in ast.find_all(exp.Column) if column.name and column.name != "*"}
    return tables, columns


def recall(selected: set[str], gold: set[str]) -> float | str:
    if not gold:
        return ""
    return round(len(selected & gold) / len(gold), 4)


def evaluate_execute_case_v2(case: dict[str, str], mode: str) -> dict[str, Any]:
    from text2sql.app.run_v2 import run_question_v2

    result = run_question_v2(case["question"], mode=mode)
    generated_sql = result.get("sql", "")
    if not result.get("valid"):
        return {
            "generated_sql": generated_sql,
            "is_valid": False,
            "execution_success": False,
            "is_correct": False,
            "strict_correct": False,
            "relaxed_correct": False,
            "match_type": "invalid_sql",
            "error": result.get("error", "Invalid SQL"),
            "repair_attempted": False,
            **v2_trace_fields(result, case),
        }

    if not result.get("execution_success"):
        return {
            "generated_sql": generated_sql,
            "is_valid": True,
            "execution_success": False,
            "is_correct": False,
            "strict_correct": False,
            "relaxed_correct": False,
            "match_type": "execution_error",
            "error": result.get("error", "Execution error"),
            "repair_attempted": False,
            **v2_trace_fields(result, case),
        }

    if not case.get("gold_sql") or case["gold_sql"].strip().upper() in {"NO_SQL", "BLOCK"}:
        return {
            "generated_sql": generated_sql,
            "is_valid": True,
            "execution_success": True,
            "is_correct": False,
            "strict_correct": False,
            "relaxed_correct": False,
            "match_type": "no_gold_sql",
            "error": "No executable gold SQL for V2 comparison",
            "repair_attempted": False,
            **v2_trace_fields(result, case),
        }

    gold_execution = execute_gold_sql(case["gold_sql"])
    if not gold_execution["success"]:
        return {
            "generated_sql": generated_sql,
            "is_valid": True,
            "execution_success": True,
            "is_correct": False,
            "strict_correct": False,
            "relaxed_correct": False,
            "match_type": "gold_error",
            "error": f"Gold SQL failed: {gold_execution['error']}",
            "repair_attempted": False,
            **v2_trace_fields(result, case),
        }

    comparison = compare_dataframes(result["data"], gold_execution["data"])
    return {
        "generated_sql": generated_sql,
        "is_valid": True,
        "execution_success": True,
        "is_correct": comparison["relaxed_correct"],
        **comparison,
        "error": "" if comparison["relaxed_correct"] else "Result mismatch",
        "repair_attempted": False,
        **v2_trace_fields(result, case),
    }


def evaluate_block_case_v2(case: dict[str, str], mode: str) -> dict[str, Any]:
    from text2sql.app.run_v2 import run_question_v2

    result = run_question_v2(case["question"], mode=mode)
    blocked = str(result.get("sql", "")).strip().upper() == "NO_SQL" or not result.get("valid")
    return {
        "generated_sql": result.get("sql", ""),
        "is_valid": bool(result.get("valid")),
        "execution_success": False,
        "is_correct": blocked,
        "strict_correct": blocked,
        "relaxed_correct": blocked,
        "match_type": "blocked" if blocked else "should_block_but_ran",
        "error": result.get("error", "") if blocked else "Block case was executed",
        "repair_attempted": False,
        **v2_trace_fields(result, case),
    }


def v2_trace_fields(result: dict[str, Any], case: dict[str, str]) -> dict[str, Any]:
    selected_tables = set(result.get("selected_tables", []))
    selected_columns_raw = result.get("selected_columns", [])
    selected_columns = {str(item).split(".")[-1] for item in selected_columns_raw}
    gold_tables, gold_columns = names_from_sql(case.get("gold_sql", ""))
    selected_context_length = ""
    for step in result.get("trace_steps", []):
        if step.get("stage") == "Build selected schema context":
            selected_context_length = step.get("data", {}).get("selected_schema_context_length", "")
            break
    return {
        "query_decomposition": json.dumps(result.get("query_decomposition", {}), ensure_ascii=False),
        "fallback_used": (result.get("query_decomposition") or {}).get("fallback_used", False),
        "selected_tables": json.dumps(result.get("selected_tables", []), ensure_ascii=False),
        "selected_columns": json.dumps(result.get("selected_columns", []), ensure_ascii=False),
        "matched_entities": json.dumps(result.get("matched_entities", [])[:10], ensure_ascii=False),
        "selected_schema_context_length": selected_context_length,
        "table_recall": recall(selected_tables, gold_tables),
        "column_recall": recall(selected_columns, gold_columns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL test cases.")
    parser.add_argument("--test-file", default=str(TEST_CASES_PATH), help="CSV test file.")
    parser.add_argument("--output", default=str(RESULTS_PATH), help="Results CSV output path.")
    parser.add_argument(
        "--error-analysis",
        default=str(ERROR_ANALYSIS_PATH),
        help="Error analysis CSV output path.",
    )
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--mode", choices=["product_sales", "mock_relational"], default="product_sales")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_cases_path = resolve_path(args.test_file)
    if args.version == "v2" and args.output == str(RESULTS_PATH):
        results_path = V2_RESULTS_PATH
    else:
        results_path = resolve_path(args.output)
    if args.version == "v2" and args.error_analysis == str(ERROR_ANALYSIS_PATH):
        error_analysis_path = V2_ERROR_ANALYSIS_PATH
    else:
        error_analysis_path = resolve_path(args.error_analysis)

    if args.version == "v1" and not SCHEMA_PATH.exists():
        print(f"Schema summary not found: {SCHEMA_PATH}")
        print("Run: python -m text2sql.schema.summary.generate_schema_summary --mode product_sales")
        return 1

    test_cases = read_test_cases(test_cases_path)
    results = []

    schema_context = build_schema_context(str(SCHEMA_PATH)) if args.version == "v1" else ""
    for case in test_cases:
        if args.version == "v2":
            if case["expected_behavior"] == "execute":
                metrics = evaluate_execute_case_v2(case, args.mode)
            else:
                metrics = evaluate_block_case_v2(case, args.mode)
        else:
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

    output_rows = [
        {key: value for key, value in row.items() if key != "expected_behavior"}
        for row in results
    ]
    write_results(output_rows, results_path)
    error_analysis = build_error_analysis(results)
    write_error_analysis(error_analysis, error_analysis_path)
    print_summary(results, error_analysis, label="V2 Schema Retrieval" if args.version == "v2" else "V1.1")
    print(f"\nResults written to        : {results_path}")
    print(f"Error analysis written to : {error_analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
