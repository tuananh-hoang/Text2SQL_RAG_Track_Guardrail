import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import sqlglot
from sqlglot import expressions as exp


BASE_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = BASE_DIR.parent
DEFAULT_INPUT = BASE_DIR / "evaluation" / "results" / "v2" / "rasl_table_retrieval_report.csv"
DEFAULT_TRACES = BASE_DIR / "debug_artifacts" / "rasl_table_retrieval_traces.jsonl"
DEFAULT_OUTPUT_CSV = BASE_DIR / "evaluation" / "results" / "v2" / "rasl_table_retrieval_error_analysis.csv"
DEFAULT_OUTPUT_MD = BASE_DIR / "evaluation" / "results" / "v2" / "rasl_table_retrieval_failure_report.md"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == BASE_DIR.name:
        return ROOT_DIR / path
    return BASE_DIR / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_traces(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    traces: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = str(record.get("case_id") or record.get("id") or "")
            if case_id:
                traces[case_id] = record
    return traces


def parse_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def unique(values: Iterable[Any]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def sql_text(node: exp.Expression | None) -> str:
    if node is None:
        return ""
    try:
        return node.sql(dialect="postgres")
    except Exception:
        return str(node)


def column_name(column: exp.Column) -> str:
    table = column.table
    name = column.name
    return f"{table}.{name}" if table else name


def extract_aliases(ast: exp.Expression) -> list[str]:
    aliases = []
    for table in ast.find_all(exp.Table):
        if table.alias:
            aliases.append(f"{table.name} AS {table.alias}")
    return unique(aliases)


def extract_sql_features(sql: str) -> dict[str, Any]:
    if not sql or sql.strip().upper() == "NO_SQL":
        return {"parse_success": False, "parse_error": "empty_or_no_sql"}
    try:
        ast = sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:
        return {"parse_success": False, "parse_error": str(exc)}

    cte_names = {cte.alias_or_name for cte in ast.find_all(exp.CTE) if cte.alias_or_name}
    tables = unique(
        table.name
        for table in ast.find_all(exp.Table)
        if table.name and table.name not in cte_names
    )
    columns = unique(column_name(column) for column in ast.find_all(exp.Column))
    selected_columns = unique(sql_text(item) for item in (ast.expressions or []))
    where_columns = unique(
        column_name(column)
        for where in ast.find_all(exp.Where)
        for column in where.find_all(exp.Column)
    )
    group_by_columns = unique(
        column_name(column)
        for group in ast.find_all(exp.Group)
        for column in group.find_all(exp.Column)
    )
    order_by_columns = unique(
        column_name(column)
        for order in ast.find_all(exp.Order)
        for column in order.find_all(exp.Column)
    )
    joins = list(ast.find_all(exp.Join))
    join_tables = unique(join.this.name for join in joins if isinstance(join.this, exp.Table))
    join_conditions = unique(sql_text(join.args.get("on")) for join in joins if join.args.get("on") is not None)
    aggregate_functions = unique(sql_text(agg) for agg in ast.find_all(exp.AggFunc))
    window_functions = unique(sql_text(window) for window in ast.find_all(exp.Window))
    all_sql_lower = sql.lower()

    return {
        "parse_success": True,
        "tables_used": tables,
        "columns_used": columns,
        "selected_columns": selected_columns,
        "where_columns": where_columns,
        "group_by_columns": group_by_columns,
        "order_by_columns": order_by_columns,
        "join_tables": join_tables,
        "join_conditions": join_conditions,
        "aliases_used": extract_aliases(ast),
        "aggregate_functions": aggregate_functions,
        "has_group_by": any(ast.find_all(exp.Group)),
        "has_order_by": any(ast.find_all(exp.Order)),
        "has_limit": any(ast.find_all(exp.Limit)),
        "has_cte": bool(cte_names),
        "has_subquery": any(ast.find_all(exp.Subquery)),
        "has_window_function": bool(window_functions),
        "window_functions": window_functions,
        "has_having": any(ast.find_all(exp.Having)),
        "date_filter_present": "order_date" in all_sql_lower or bool(re.search(r"'20\d{2}-\d{2}-\d{2}'", sql)),
        "numeric_threshold_present": bool(re.search(r"(?<![A-Za-z_])\d+(?:\.\d+)?", sql)),
    }


def feature_diff(gold: dict[str, Any], generated: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    missing: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    if not gold.get("parse_success") or not generated.get("parse_success"):
        return missing, extra

    list_keys = [
        "tables_used",
        "columns_used",
        "group_by_columns",
        "where_columns",
        "join_tables",
        "aggregate_functions",
        "window_functions",
    ]
    bool_keys = [
        "has_group_by",
        "has_order_by",
        "has_limit",
        "has_cte",
        "has_subquery",
        "has_window_function",
        "has_having",
        "date_filter_present",
    ]
    for key in list_keys:
        gold_set = set(gold.get(key, []))
        gen_set = set(generated.get(key, []))
        lost = sorted(gold_set - gen_set)
        added = sorted(gen_set - gold_set)
        if lost:
            missing[key] = lost
        if added:
            extra[key] = added
    for key in bool_keys:
        if gold.get(key) and not generated.get(key):
            missing[key] = True
        if generated.get(key) and not gold.get(key):
            extra[key] = True
    return missing, extra


def is_rate_limited(row: dict[str, str]) -> bool:
    error_text = str(row.get("error", "")).lower()
    return (
        as_bool(row.get("is_rate_limited"))
        or row.get("failure_type") == "rate_limited"
        or "rate limit" in error_text
        or "rate_limit" in error_text
        or "429" in error_text
    )


def classify_execution_root_cause(row: dict[str, str], generated_features: dict[str, Any]) -> tuple[str, str]:
    error = str(row.get("error", ""))
    lower = error.lower()
    generated_sql = str(row.get("generated_sql", ""))
    if (
        "perhaps you meant to reference" in lower
        and re.search(r"\b[A-Za-z]\w*\.[A-Z][A-Za-z0-9_]*\b", generated_sql)
    ):
        return "wrong_quote_identifier", "Quote PostgreSQL mixed-case identifiers, e.g. T1.\"Product_ID\" instead of T1.Product_ID."
    if "undefinedcolumn" in lower or "column" in lower and "does not exist" in lower:
        return "wrong_column_name", "Use schema context/AST validation to ensure referenced columns exist in the table alias scope."
    if "undefinedtable" in lower or "relation" in lower and "does not exist" in lower:
        return "wrong_table_name", "Ensure generator uses predicted table names with schema prefix and valid aliases."
    if "missing from-clause entry" in lower or "invalid reference to from-clause" in lower:
        return "wrong_table_alias", "Check alias declaration/use consistency before execution."
    if "operator does not exist" in lower or "cannot be matched" in lower or "invalid input syntax" in lower:
        return "type_error", "Add SQL type-aware validation or prompt examples for date/numeric/text operations."
    if any(token in lower for token in ["date", "timestamp", "extract", "to_char", "interval"]):
        return "date_function_error", "Check PostgreSQL date expression generation against Order_Date type."
    if generated_features.get("join_tables") and not generated_features.get("join_conditions"):
        return "wrong_join_condition", "Validate joins include ON conditions matching FK relationships."
    return "unknown_execution_error", "Inspect PostgreSQL error and generated SQL manually before changing generator."


def classify_semantic_root_cause(
    row: dict[str, str],
    gold_features: dict[str, Any],
    generated_features: dict[str, Any],
    missing_features: dict[str, Any],
) -> tuple[str, str]:
    question = str(row.get("question", "")).lower()
    generated_sql = str(row.get("generated_sql", "")).lower()

    if "profit margin" in question and not (
        "sum" in generated_sql and "profit" in generated_sql and "revenue" in generated_sql
    ):
        return "semantic_wrong_profit_margin", "Define business metric formula: SUM(Profit) / NULLIF(SUM(Revenue), 0)."
    if any(token in question for token in ["80%", "10%", "contribution", "dong gop", "đóng góp"]):
        return "semantic_wrong_contribution", "Add explicit contribution/cumulative-analysis planning before SQL generation."
    if any(token in question for token in ["2023", "2024", "yoy", "tang truong", "muc tang", "tăng trưởng", "mức tăng"]):
        if missing_features.get("date_filter_present") or "2023" not in generated_sql or "2024" not in generated_sql:
            return "semantic_wrong_yoy", "Add intent handling for year-over-year comparison with separate year aggregates."
    if missing_features.get("date_filter_present"):
        return "semantic_wrong_date_filter", "Ensure time constraints from decomposition are reflected in generated SQL."
    if gold_features.get("has_window_function") and not generated_features.get("has_window_function"):
        return "semantic_wrong_window", "Use window functions for per-group ranking, lag/lead, running total, or cumulative contribution."
    if gold_features.get("has_group_by") and not generated_features.get("has_group_by"):
        return "semantic_wrong_grain", "Force entity-level questions to aggregate at entity grain before ranking/filtering."
    if gold_features.get("aggregate_functions") and not generated_features.get("aggregate_functions"):
        return "semantic_wrong_aggregation", "Require SUM/AVG/STDDEV/etc. for measure questions instead of raw row-level metrics."
    if missing_features.get("join_tables") or missing_features.get("tables_used"):
        return "semantic_wrong_join", "Use relationship graph to build the required join path across predicted tables."
    if missing_features.get("selected_columns"):
        return "semantic_wrong_projection", "Make selected output columns match the entity/metric/time requested by the question."
    if gold_features.get("has_subquery") and not generated_features.get("has_subquery"):
        return "semantic_wrong_nested_aggregation", "Plan nested aggregation explicitly for comparisons against group/global averages."
    return "result_mismatch_unknown", "Compare gold and generated AST manually to decide whether prompt, skeleton planner, or checker is needed."


def analyze_row(row: dict[str, str], trace: dict[str, Any] | None) -> dict[str, Any]:
    gold_features = extract_sql_features(row.get("gold_sql", ""))
    generated_features = extract_sql_features(row.get("generated_sql", ""))
    missing_features, extra_features = feature_diff(gold_features, generated_features)
    failure_type = row.get("failure_type") or ""
    postgres_error = row.get("error", "") if failure_type == "execution_error" else ""

    if is_rate_limited(row):
        root_cause = "provider_rate_limited"
        fix_direction = "Exclude from semantic metrics; rely on API key rotation or rerun after quota resets."
        failure_type = "rate_limited"
    elif failure_type == "execution_error":
        root_cause, fix_direction = classify_execution_root_cause(row, generated_features)
    elif failure_type == "generator_semantic_fail":
        root_cause, fix_direction = classify_semantic_root_cause(
            row,
            gold_features,
            generated_features,
            missing_features,
        )
    elif failure_type == "sql_generation_error":
        root_cause = "llm_generation_failed"
        fix_direction = "Inspect provider response/error before changing SQL prompt."
    elif failure_type in {"table_ranking_fail", "table_prediction_fail"}:
        root_cause = failure_type
        fix_direction = "Log and inspect retrieval/table prediction evidence before touching generator."
    elif failure_type == "correct":
        root_cause = "correct"
        fix_direction = ""
    else:
        root_cause = failure_type or "unknown"
        fix_direction = "Inspect case manually."

    selected_schema_context = ""
    if trace:
        selected_schema_context = str(
            next(
                (
                    step.get("data", {}).get("selected_schema_context", "")
                    for step in trace.get("steps", [])
                    if step.get("stage") == "Build full schema context of predicted tables"
                ),
                "",
            )
        )

    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "category": row.get("category", ""),
        "difficulty": row.get("difficulty", ""),
        "gold_sql": row.get("gold_sql", ""),
        "generated_sql": row.get("generated_sql", ""),
        "predicted_tables": row.get("predicted_tables", ""),
        "candidate_table_scores": row.get("candidate_table_scores", ""),
        "selected_schema_context_length": row.get("context_length", ""),
        "selected_schema_context": selected_schema_context,
        "sql_valid": row.get("sql_valid", ""),
        "execution_success": row.get("execution_success", ""),
        "postgres_error": postgres_error,
        "result_correct": row.get("result_correct", ""),
        "failure_type": failure_type,
        "root_cause": root_cause,
        "fix_direction": fix_direction,
        "tables_referenced_by_generated_sql": "|".join(generated_features.get("tables_used", [])),
        "columns_referenced_by_generated_sql": "|".join(generated_features.get("columns_used", [])),
        "aliases_used_in_generated_sql": "|".join(generated_features.get("aliases_used", [])),
        "gold_features": json.dumps(gold_features, ensure_ascii=False),
        "generated_features": json.dumps(generated_features, ensure_ascii=False),
        "missing_features": json.dumps(missing_features, ensure_ascii=False),
        "extra_features": json.dumps(extra_features, ensure_ascii=False),
    }


def write_analysis_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "category",
        "difficulty",
        "gold_sql",
        "generated_sql",
        "predicted_tables",
        "candidate_table_scores",
        "selected_schema_context_length",
        "selected_schema_context",
        "sql_valid",
        "execution_success",
        "postgres_error",
        "result_correct",
        "failure_type",
        "root_cause",
        "fix_direction",
        "tables_referenced_by_generated_sql",
        "columns_referenced_by_generated_sql",
        "aliases_used_in_generated_sql",
        "gold_features",
        "generated_features",
        "missing_features",
        "extra_features",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def metric_line(label: str, numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"- {label}: n/a"
    return f"- {label}: {numerator}/{denominator} ({numerator / denominator:.1%})"


def write_markdown_report(source_rows: list[dict[str, str]], analysis_rows: list[dict[str, Any]], output_path: Path) -> None:
    total = len(source_rows)
    rate_limited_count = sum(1 for row in analysis_rows if row["failure_type"] == "rate_limited")
    effective_rows = [row for row in source_rows if not is_rate_limited(row)]
    effective_total = len(effective_rows)

    raw_valid = sum(1 for row in source_rows if as_bool(row.get("sql_valid")))
    raw_exec = sum(1 for row in source_rows if as_bool(row.get("execution_success")))
    raw_correct = sum(1 for row in source_rows if as_bool(row.get("result_correct")))
    eff_valid = sum(1 for row in effective_rows if as_bool(row.get("sql_valid")))
    eff_exec = sum(1 for row in effective_rows if as_bool(row.get("execution_success")))
    eff_correct = sum(1 for row in effective_rows if as_bool(row.get("result_correct")))

    failure_counts = Counter(row["failure_type"] for row in analysis_rows)
    root_counts = Counter(row["root_cause"] for row in analysis_rows if row["root_cause"] != "correct")
    execution_rows = [row for row in analysis_rows if row["failure_type"] == "execution_error"]
    semantic_rows = [row for row in analysis_rows if row["failure_type"] == "generator_semantic_fail"]
    important_rows = [row for row in analysis_rows if row["failure_type"] not in {"correct", "rate_limited"}][:5]

    lines = [
        "# RASL Table Retrieval Failure Report",
        "",
        "## 1. Metric Overview",
        "",
        "Raw metrics:",
        metric_line("SQL valid raw", raw_valid, total),
        metric_line("Execution success raw", raw_exec, total),
        metric_line("Result correctness raw", raw_correct, total),
        f"- Rate limited cases: {rate_limited_count}/{total}",
        "",
        "Metrics excluding rate limit:",
        f"- Effective total: {effective_total}",
        metric_line("SQL valid excluding rate limit", eff_valid, effective_total),
        metric_line("Execution success excluding rate limit", eff_exec, effective_total),
        metric_line("Correctness excluding rate limit", eff_correct, effective_total),
        "",
        "## 2. Failure Distribution",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in failure_counts.most_common())

    lines.extend(["", "## 3. Execution Error Analysis", ""])
    if execution_rows:
        exec_counts = Counter(row["root_cause"] for row in execution_rows)
        lines.extend(f"- {name}: {count}" for name, count in exec_counts.most_common())
        lines.append("")
        for row in execution_rows:
            lines.extend(
                [
                    f"### Case {row['id']}: {row['root_cause']}",
                    "",
                    f"- Question: {row['question']}",
                    f"- Postgres error: `{row['postgres_error']}`",
                    f"- Fix direction: {row['fix_direction']}",
                    "",
                    "Generated SQL:",
                    "```sql",
                    row["generated_sql"],
                    "```",
                    "",
                ]
            )
    else:
        lines.append("- No execution errors.")

    lines.extend(["", "## 4. Semantic Failure Analysis", ""])
    if semantic_rows:
        semantic_counts = Counter(row["root_cause"] for row in semantic_rows)
        lines.extend(f"- {name}: {count}" for name, count in semantic_counts.most_common())
        lines.append("")
        for row in semantic_rows:
            lines.extend(
                [
                    f"### Case {row['id']}: {row['root_cause']}",
                    "",
                    f"- Question: {row['question']}",
                    f"- Missing AST features: `{row['missing_features']}`",
                    f"- Fix direction: {row['fix_direction']}",
                    "",
                    "Gold SQL:",
                    "```sql",
                    row["gold_sql"],
                    "```",
                    "",
                    "Generated SQL:",
                    "```sql",
                    row["generated_sql"],
                    "```",
                    "",
                ]
            )
    else:
        lines.append("- No semantic failures.")

    lines.extend(["", "## 5. Top Repeated Root Causes", ""])
    if root_counts:
        lines.extend(f"- {name}: {count}" for name, count in root_counts.most_common())
    else:
        lines.append("- No repeated failure root cause.")

    lines.extend(["", "## 6. Five Important Failed Cases", ""])
    for row in important_rows:
        lines.extend(
            [
                f"### Case {row['id']}: {row['root_cause']}",
                "",
                f"- Question: {row['question']}",
                f"- Failure type: {row['failure_type']}",
                f"- Fix direction: {row['fix_direction']}",
                "",
            ]
        )
    if not important_rows:
        lines.append("- No non-rate-limited failed case found.")

    lines.extend(
        [
            "",
            "## 7. Recommended Next Fix",
            "",
            "Do not change retrieval or calibration from this report alone. The next fix should target the most repeated non-rate-limited root cause above. If semantic failures dominate, add a lightweight SQL skeleton planner or semantic checker before prompt expansion.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RASL table retrieval failures.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--traces", default=str(DEFAULT_TRACES))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    traces_path = resolve_path(args.traces)
    output_csv = resolve_path(args.output_csv)
    output_md = resolve_path(args.output_md)

    rows = read_csv(input_path)
    traces = read_traces(traces_path)
    analysis_rows = [
        analyze_row(row, traces.get(str(row.get("id", ""))))
        for row in rows
    ]
    write_analysis_csv(analysis_rows, output_csv)
    write_markdown_report(rows, analysis_rows, output_md)

    failure_counts = Counter(row["failure_type"] for row in analysis_rows)
    root_counts = Counter(row["root_cause"] for row in analysis_rows if row["root_cause"] != "correct")
    print("=== RASL Failure Analysis ===")
    print(f"Cases analyzed: {len(analysis_rows)}")
    print("Failure distribution:")
    for name, count in failure_counts.most_common():
        print(f"- {name}: {count}")
    print("Root cause distribution:")
    for name, count in root_counts.most_common():
        print(f"- {name}: {count}")
    print(f"\nCSV written to     : {output_csv}")
    print(f"Markdown written to: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
