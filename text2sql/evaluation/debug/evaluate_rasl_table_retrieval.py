import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from text2sql.app.run_v2 import run_question_v2
from text2sql.evaluation.evaluator import compare_dataframes, execute_gold_sql
from text2sql.schema.paths import v2_summary_path
from text2sql.sql.executor import execute_sql
from text2sql.sql.validator import validate_sql


BASE_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = BASE_DIR.parent
DEFAULT_INPUT = BASE_DIR / "data" / "debug" / "mock_relational_debug_30.csv"
DEFAULT_OUTPUT = BASE_DIR / "evaluation" / "results" / "v2" / "rasl_table_retrieval_report.csv"
DEFAULT_TRACE_COPY = BASE_DIR / "evaluation" / "results" / "v2" / "rasl_table_retrieval_report.jsonl"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == BASE_DIR.name:
        return ROOT_DIR / path
    return BASE_DIR / path


def parse_pipe_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split("|") if item.strip()]


def normalize_table(value: str) -> str:
    return str(value).split(".")[-1]


def recall(selected: list[str], gold: list[str]) -> float | str:
    gold_set = {normalize_table(item) for item in gold}
    if not gold_set:
        return ""
    selected_set = {normalize_table(item) for item in selected}
    return round(len(selected_set & gold_set) / len(gold_set), 4)


def precision(selected: list[str], gold: list[str]) -> float | str:
    selected_set = {normalize_table(item) for item in selected}
    if not selected_set:
        return ""
    gold_set = {normalize_table(item) for item in gold}
    return round(len(selected_set & gold_set) / len(selected_set), 4)


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_gold_sql(cases: list[dict[str, str]]) -> None:
    schema_path = v2_summary_path("mock_relational")
    errors = []
    for case in cases:
        gold_sql = case.get("gold_sql", "")
        if "product_sales" in gold_sql:
            errors.append(f"case {case.get('id')}: gold_sql uses product_sales")
            continue
        validation = validate_sql(gold_sql, str(schema_path))
        if not validation["valid"]:
            errors.append(f"case {case.get('id')}: gold_sql invalid: {validation['error']}")
            continue
        execution = execute_sql(validation["sql"])
        if not execution["success"]:
            errors.append(f"case {case.get('id')}: gold_sql execution failed: {execution['error']}")
    if errors:
        raise RuntimeError("Gold SQL validation failed:\n" + "\n".join(errors))


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def is_rate_limited_error_text(error_text: str) -> bool:
    normalized = str(error_text or "").lower()
    return (
        "rate limit" in normalized
        or "rate_limit" in normalized
        or "rate limits" in normalized
        or "429" in normalized
    )


def classify_failure(
    result: dict[str, Any],
    candidate_recall: float | str,
    predicted_recall: float | str,
    comparison: dict[str, Any] | None,
) -> str:
    error_text = str(result.get("error", "")).lower()
    # fix: keep provider rate limits separate from generator/retrieval failures.
    if is_rate_limited_error_text(error_text):
        return "rate_limited"
    if "connection error" in error_text:
        return "sql_generation_error"
    decomposition = result.get("query_decomposition") or {}
    if decomposition.get("fallback_used"):
        return "query_decomposition_fail"
    if not result.get("matched_entities"):
        return "vector_entity_retrieval_fail"
    if candidate_recall != "" and float(candidate_recall) < 1.0:
        return "table_ranking_fail"
    if predicted_recall != "" and float(predicted_recall) < 1.0:
        return "table_prediction_fail"
    if not result.get("sql"):
        return "sql_generation_error"
    if str(result.get("sql", "")).strip().upper() == "NO_SQL":
        return "sql_generation_error"
    if not result.get("valid"):
        return "sql_validation_fail"
    if not result.get("execution_success"):
        return "execution_error"
    if comparison and not comparison.get("relaxed_correct"):
        return "generator_semantic_fail"
    return "correct"


def evaluate_case(case: dict[str, str], debug_context: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    result = run_question_v2(
        case["question"],
        mode="mock_relational",
        retrieval_mode="rasl_table_retrieval",
        debug_context=debug_context,
        debug=debug_context,
    )
    gold_tables = parse_pipe_list(case.get("gold_tables"))
    candidate_tables = result.get("candidate_tables", [])
    predicted_tables = result.get("predicted_tables", [])
    candidate_recall = recall(candidate_tables, gold_tables)
    predicted_recall = recall(predicted_tables, gold_tables)
    predicted_precision = precision(predicted_tables, gold_tables)

    comparison = None
    gold_error = ""
    if result.get("execution_success") and case.get("gold_sql"):
        gold_execution = execute_gold_sql(case["gold_sql"])
        if gold_execution["success"]:
            comparison = compare_dataframes(result["data"], gold_execution["data"])
        else:
            gold_error = gold_execution["error"]

    failure_type = classify_failure(result, candidate_recall, predicted_recall, comparison)
    is_rate_limited = failure_type == "rate_limited"
    row = {
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "difficulty": case.get("difficulty", ""),
        "question": case.get("question", ""),
        "gold_tables": case.get("gold_tables", ""),
        "gold_sql": case.get("gold_sql", ""),
        "candidate_tables": "|".join(candidate_tables),
        "predicted_tables": "|".join(predicted_tables),
        "rendered_tables": "|".join(result.get("rendered_tables", [])),
        "table_recall_at_k_before_prediction": candidate_recall,
        "predicted_table_recall": predicted_recall,
        "predicted_table_precision": predicted_precision,
        "sql_valid": bool(result.get("valid")),
        "execution_success": bool(result.get("execution_success")),
        "result_correct": bool(comparison and comparison.get("relaxed_correct")),
        "match_type": comparison.get("match_type") if comparison else "",
        "context_length": result.get("selected_schema_context_length", ""),
        "failure_type": failure_type,
        "is_rate_limited": is_rate_limited,
        "error": result.get("error", "") or gold_error,
        "generated_sql": result.get("sql", ""),
        "query_decomposition": safe_json(result.get("query_decomposition", {})),
        "retrieval_queries": safe_json(result.get("retrieval_queries", [])),
        "candidate_table_scores": safe_json(result.get("candidate_table_scores", [])),
        "table_predictor_output": safe_json(result.get("table_predictor_output", {})),
        "retrieved_entities_by_query": safe_json(result.get("retrieved_entities_by_query", {})),
        "retrieved_entities_by_entity_type": safe_json(result.get("retrieved_entities_by_entity_type", {})),
    }
    trace = {
        **result,
        "case_id": case.get("id", ""),
        "gold_tables": gold_tables,
        "candidate_recall": candidate_recall,
        "predicted_recall": predicted_recall,
        "predicted_precision": predicted_precision,
        "comparison": comparison,
        "failure_type": failure_type,
        "is_rate_limited": is_rate_limited,
    }
    return row, trace


def write_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "difficulty",
        "question",
        "gold_sql",
        "gold_tables",
        "candidate_tables",
        "predicted_tables",
        "rendered_tables",
        "table_recall_at_k_before_prediction",
        "predicted_table_recall",
        "predicted_table_precision",
        "sql_valid",
        "execution_success",
        "result_correct",
        "match_type",
        "context_length",
        "failure_type",
        "is_rate_limited",
        "error",
        "generated_sql",
        "query_decomposition",
        "retrieval_queries",
        "candidate_table_scores",
        "table_predictor_output",
        "retrieved_entities_by_query",
        "retrieved_entities_by_entity_type",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_traces(traces: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")


def summarize(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    valid = sum(1 for row in rows if row["sql_valid"])
    executed = sum(1 for row in rows if row["execution_success"])
    correct = sum(1 for row in rows if row["result_correct"])
    rate_limited = sum(1 for row in rows if row.get("is_rate_limited"))
    effective_rows = [row for row in rows if not row.get("is_rate_limited")]
    effective_total = len(effective_rows)
    valid_effective = sum(1 for row in effective_rows if row["sql_valid"])
    executed_effective = sum(1 for row in effective_rows if row["execution_success"])
    correct_effective = sum(1 for row in effective_rows if row["result_correct"])
    failure_counts = Counter(row["failure_type"] for row in rows)
    candidate_recall_values = [
        float(row["table_recall_at_k_before_prediction"])
        for row in rows
        if row["table_recall_at_k_before_prediction"] != ""
    ]
    predicted_recall_values = [
        float(row["predicted_table_recall"])
        for row in rows
        if row["predicted_table_recall"] != ""
    ]
    predicted_precision_values = [
        float(row["predicted_table_precision"])
        for row in rows
        if row["predicted_table_precision"] != ""
    ]

    lines = [
        "=== RASL Table Retrieval Evaluation ===",
        f"Total cases                    : {total}",
        f"Avg table recall@k before LLM  : {sum(candidate_recall_values) / len(candidate_recall_values):.3f}"
        if candidate_recall_values
        else "Avg table recall@k before LLM  : n/a",
        f"Avg predicted table recall     : {sum(predicted_recall_values) / len(predicted_recall_values):.3f}"
        if predicted_recall_values
        else "Avg predicted table recall     : n/a",
        f"Avg predicted table precision  : {sum(predicted_precision_values) / len(predicted_precision_values):.3f}"
        if predicted_precision_values
        else "Avg predicted table precision  : n/a",
        f"SQL valid rate                 : {valid}/{total}",
        f"Execution success rate         : {executed}/{total}",
        f"Result correctness             : {correct}/{total}",
        f"Rate limited cases             : {rate_limited}/{total}",
        "",
        "Metrics excluding rate limit:",
        f"Effective total                : {effective_total}",
        f"SQL valid excluding rate limit : {valid_effective}/{effective_total}"
        if effective_total
        else "SQL valid excluding rate limit : n/a",
        f"Execution excluding rate limit : {executed_effective}/{effective_total}"
        if effective_total
        else "Execution excluding rate limit : n/a",
        f"Correct excluding rate limit   : {correct_effective}/{effective_total}"
        if effective_total
        else "Correct excluding rate limit   : n/a",
        "",
        "Failure distribution:",
    ]
    for key, count in failure_counts.most_common():
        lines.append(f"- {key}: {count}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V2 RASL table retrieval main pipeline.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--trace-output", default=str(DEFAULT_TRACE_COPY))
    parser.add_argument("--debug-context", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Alias for --debug-context.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    trace_output_path = resolve_path(args.trace_output)
    cases = read_cases(input_path)
    validate_gold_sql(cases)

    rows = []
    traces = []
    for case in cases:
        row, trace = evaluate_case(case, debug_context=args.debug_context or args.debug)
        rows.append(row)
        traces.append(trace)
        write_report(rows, output_path)
        write_traces(traces, trace_output_path)

    write_report(rows, output_path)
    write_traces(traces, trace_output_path)
    print(summarize(rows))
    print(f"\nReport written to      : {output_path}")
    print(f"Trace JSONL written to : {trace_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
