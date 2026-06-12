import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from text2sql.app.run_v2_debug import DEBUG_MODES, parse_pipe_list, run_debug_question
from text2sql.schema.paths import v2_summary_path
from text2sql.sql.executor import execute_sql
from text2sql.sql.validator import validate_sql


BASE_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = BASE_DIR.parent
DEFAULT_INPUT = BASE_DIR / "data" / "debug" / "mock_relational_debug_30.csv"
DEFAULT_OUTPUT = BASE_DIR / "evaluation" / "results" / "v2" / "rasl_debug_modes_report.csv"
ORDER_DATE_REPORT = BASE_DIR / "evaluation" / "results" / "v2" / "order_date_debug_report.md"
TRACE_PATH = BASE_DIR / "debug_artifacts" / "rasl_debug_traces.jsonl"
DEBUG_MODE_ORDER = ["full_schema", "table_only", "table_column", "rasl_zero_shot"]
ORDER_DATE_FOCUS_IDS = {"1", "5", "6", "11", "26"}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == BASE_DIR.name:
        return ROOT_DIR / path
    return BASE_DIR / path


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_gold_sql(cases: list[dict[str, str]]) -> None:
    schema_path = v2_summary_path("mock_relational")
    errors = []
    for case in cases:
        gold_sql = case["gold_sql"]
        if "product_sales" in gold_sql:
            errors.append(f"case {case['id']}: gold_sql uses product_sales")
            continue
        validation = validate_sql(gold_sql, str(schema_path))
        if not validation["valid"]:
            errors.append(f"case {case['id']}: gold_sql invalid: {validation['error']}")
            continue
        execution = execute_sql(validation["sql"])
        if not execution["success"]:
            errors.append(f"case {case['id']}: gold_sql execution failed: {execution['error']}")
    if errors:
        raise RuntimeError("Gold SQL validation failed:\n" + "\n".join(errors))


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def write_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "category",
        "debug_mode",
        "gold_tables",
        "gold_columns",
        "selected_tables",
        "selected_columns",
        "missed_tables",
        "missed_columns",
        "table_recall",
        "column_recall",
        "context_length",
        "sql_valid",
        "execution_success",
        "failure_type",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_existing_report(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_existing_traces(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    traces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            traces.append(json.loads(line))
    return traces


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "success"}


def recall_value(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def summarize(rows: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("=== RASL Debug Modes Summary ===")
    for mode in DEBUG_MODE_ORDER:
        mode_rows = [row for row in rows if row["debug_mode"] == mode]
        if not mode_rows:
            continue
        table_recalls = [recall_value(row["table_recall"]) for row in mode_rows]
        column_recalls = [recall_value(row["column_recall"]) for row in mode_rows]
        table_recalls = [value for value in table_recalls if value is not None]
        column_recalls = [value for value in column_recalls if value is not None]
        valid = sum(1 for row in mode_rows if boolish(row["sql_valid"]))
        executed = sum(1 for row in mode_rows if boolish(row["execution_success"]))
        avg_context = sum(int(row["context_length"]) for row in mode_rows) / len(mode_rows)
        failures = Counter(row["failure_type"] for row in mode_rows)
        lines.append("")
        lines.append(f"[{mode}]")
        lines.append(f"avg table_recall      : {sum(table_recalls) / len(table_recalls):.3f}")
        lines.append(f"avg column_recall     : {sum(column_recalls) / len(column_recalls):.3f}")
        lines.append(f"SQL valid rate        : {valid}/{len(mode_rows)}")
        lines.append(f"execution success     : {executed}/{len(mode_rows)}")
        lines.append(f"avg context length    : {avg_context:.0f} chars")
        lines.append("failure distribution  : " + ", ".join(f"{key}={count}" for key, count in failures.most_common()))

    missed_columns = Counter()
    missed_tables = Counter()
    for row in rows:
        missed_columns.update(parse_pipe_list(row.get("missed_columns", "")))
        missed_tables.update(parse_pipe_list(row.get("missed_tables", "")))
    lines.append("")
    lines.append("Top missed columns:")
    for column, count in missed_columns.most_common(10):
        lines.append(f"- {column}: {count}")
    lines.append("Top missed tables:")
    for table, count in missed_tables.most_common(10):
        lines.append(f"- {table}: {count}")
    lines.extend(preliminary_conclusion(rows))
    return "\n".join(lines)


def preliminary_conclusion(rows: list[dict[str, Any]]) -> list[str]:
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row["debug_mode"]].append(row)

    def avg_column_recall(mode: str) -> float:
        values = [float(row["column_recall"]) for row in by_mode[mode] if row["column_recall"] != ""]
        return sum(values) / len(values) if values else 0.0

    def success_rate(mode: str) -> float:
        values = by_mode[mode]
        return sum(1 for row in values if row["execution_success"]) / len(values) if values else 0.0

    lines = ["", "Preliminary conclusion:"]
    if success_rate("full_schema") < 0.5:
        lines.append("=> full_schema vẫn sai nhiều: SQL generator/prompt/semantic reasoning là bottleneck lớn.")
    elif success_rate("table_only") < success_rate("full_schema") - 0.2:
        lines.append("=> full_schema tốt hơn table_only rõ rệt: table retrieval là bottleneck.")
    elif avg_column_recall("table_column") < avg_column_recall("table_only") - 0.2:
        lines.append("=> table_only giữ đủ schema hơn table_column: column selection/pruning là bottleneck.")
    elif avg_column_recall("rasl_zero_shot") > avg_column_recall("table_column") + 0.15:
        lines.append("=> rasl_zero_shot giữ recall tốt hơn table_column: retrieval pool/pruning hiện tại chưa giống RASL zero-shot.")
    else:
        lines.append("=> Chưa có một bottleneck đơn lẻ; cần đọc trace theo từng case để tách generator và retrieval.")

    context_has_gold_but_sql_fails = [
        row
        for row in rows
        if not row["missed_tables"]
        and not row["missed_columns"]
        and row["failure_type"] in {"invalid_sql", "execution_error", "no_sql"}
    ]
    if context_has_gold_but_sql_fails:
        lines.append("=> Có case context đủ gold schema nhưng SQL vẫn lỗi: generator semantic fail, cần semantic intent/checker sau.")
    return lines


def order_date_entity_status(trace: dict[str, Any]) -> dict[str, Any]:
    all_scores = trace.get("all_column_scores", [])
    selected_columns = trace.get("selected_columns", [])
    generated_sql = trace.get("generated_sql", "")
    matched_entities = trace.get("matched_entities", [])
    order_date_scores = [
        {**row, "rank": index}
        for index, row in enumerate(all_scores, start=1)
        if str(row.get("column", "")).endswith("orders.Order_Date")
    ]
    order_date_entities = [
        entity
        for entity in matched_entities
        if entity.get("target_table") == "orders" and entity.get("target_column") == "Order_Date"
    ]
    return {
        "selected": "orders.Order_Date" in selected_columns,
        "score_rows": order_date_scores,
        "pgvector_retrieved": bool(order_date_entities),
        "entity_types": sorted({entity.get("entity_type") for entity in order_date_entities if entity.get("entity_type")}),
        "matched_queries": sorted(
            {
                query
                for entity in order_date_entities
                for query in entity.get("matched_queries", [])
                if query
            }
        ),
        "generator_used": '"Order_Date"' in generated_sql or "Order_Date" in generated_sql,
    }


def write_order_date_report(traces: list[dict[str, Any]], output_path: Path = ORDER_DATE_REPORT) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    focus = [trace for trace in traces if str(trace.get("case_id")) in ORDER_DATE_FOCUS_IDS]
    lines = [
        "# Order_Date Debug Report",
        "",
        "Scope: mock_relational only. Product_sales is not used in V2 debug retrieval/evaluation.",
        "",
    ]
    by_case_mode = {(str(trace["case_id"]), trace["debug_mode"]): trace for trace in focus}
    for case_id in sorted(ORDER_DATE_FOCUS_IDS, key=int):
        case_traces = [trace for trace in focus if str(trace["case_id"]) == case_id]
        if not case_traces:
            continue
        question = case_traces[0]["question"]
        lines.append(f"## Case {case_id}")
        lines.append("")
        lines.append(question)
        lines.append("")
        for mode in DEBUG_MODE_ORDER:
            trace = by_case_mode.get((case_id, mode))
            if not trace:
                continue
            status = order_date_entity_status(trace)
            score_text = "not ranked"
            if status["score_rows"]:
                top = status["score_rows"][0]
                score_text = f"rank {top.get('rank')}, score {top.get('score')}"
            decomposition = trace.get("query_decomposition") or {}
            date_expressions = decomposition.get("date_expressions", [])
            retrieval_strings = trace.get("retrieval_strings", [])
            lines.append(
                f"- {mode}: selected={status['selected']}; pgvector_retrieved={status['pgvector_retrieved']}; "
                f"column_score={score_text}; context_rendered={status['selected']}; "
                f"generator_used={status['generator_used']}; sql_valid={trace.get('validator_result', {}).get('valid')}; "
                f"execute={trace.get('execution_result', {}).get('success')}"
            )
            lines.append(f"  - date_expressions: {date_expressions}")
            lines.append(f"  - retrieval_strings: {retrieval_strings[:8]}")
            lines.append(f"  - retrieved_entity_types: {status['entity_types']}")
            lines.append(f"  - matched_queries_for_Order_Date: {status['matched_queries'][:8]}")
        lines.append("")

    lines.append("## Answers")
    lines.append("")
    for mode in DEBUG_MODE_ORDER:
        selected_count = sum(1 for trace in focus if trace["debug_mode"] == mode and order_date_entity_status(trace)["selected"])
        retrieved_count = sum(1 for trace in focus if trace["debug_mode"] == mode and order_date_entity_status(trace)["pgvector_retrieved"])
        lines.append(f"- {mode}: keeps Order_Date in {selected_count}/{len(ORDER_DATE_FOCUS_IDS)} focus cases; pgvector retrieves it in {retrieved_count}/{len(ORDER_DATE_FOCUS_IDS)} cases.")
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("- pgvector_retrieved=true but selected=false means aggregation/column selection dropped Order_Date.")
    lines.append("- selected=true but generator_used=false means context has Order_Date, but SQL generator ignored it.")
    lines.append("- pgvector_retrieved=false means query decomposition/vector retrieval did not surface Order_Date entities.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V2 RASL debug retrieval modes on mock_relational.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug-context", action="store_true")
    parser.add_argument("--top-k-tables", type=int, default=5)
    parser.add_argument("--top-k-columns", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    cases = read_cases(input_path)
    validate_gold_sql(cases)
    output_rows: list[dict[str, Any]] = read_existing_report(output_path) if args.resume else []
    traces: list[dict[str, Any]] = read_existing_traces(TRACE_PATH) if args.resume else []
    completed = {(str(row.get("id")), str(row.get("debug_mode"))) for row in output_rows}

    if not args.resume and TRACE_PATH.exists():
        TRACE_PATH.unlink()

    for case in cases:
        for mode in DEBUG_MODE_ORDER:
            if (str(case["id"]), mode) in completed:
                continue
            result = run_debug_question(
                case["question"],
                mode,
                gold_tables=case.get("gold_tables"),
                gold_columns=case.get("gold_columns"),
                case_id=case["id"],
                category=case["category"],
                debug_context=args.debug_context,
                top_k_tables=args.top_k_tables,
                top_k_columns=args.top_k_columns,
            )
            traces.append(result)
            output_rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "category": case["category"],
                    "debug_mode": mode,
                    "gold_tables": case.get("gold_tables", ""),
                    "gold_columns": case.get("gold_columns", ""),
                    "selected_tables": "|".join(result["selected_tables"]),
                    "selected_columns": "|".join(result["selected_columns"]),
                    "missed_tables": "|".join(result["missed_tables"]),
                    "missed_columns": "|".join(result["missed_columns"]),
                    "table_recall": result["table_recall"] if "table_recall" in result else "",
                    "column_recall": result["column_recall"] if "column_recall" in result else "",
                    "context_length": result["selected_schema_context_length"],
                    "sql_valid": bool(result["validator_result"].get("valid")),
                    "execution_success": bool(result["execution_result"].get("success")),
                    "failure_type": result["failure_type"],
                }
            )
            output_rows[-1]["table_recall"] = output_rows[-1]["table_recall"] or ""
            output_rows[-1]["column_recall"] = output_rows[-1]["column_recall"] or ""
            output_rows[-1]["table_recall"] = result.get("table_recall", "")
            output_rows[-1]["column_recall"] = result.get("column_recall", "")
            write_report(output_rows, output_path)
            completed.add((str(case["id"]), mode))

    # Fill recalls after all other fields are stable.
    for row, trace in zip(output_rows, traces):
        row["table_recall"] = trace.get("table_recall", "")
        row["column_recall"] = trace.get("column_recall", "")

    write_report(output_rows, output_path)
    write_order_date_report(traces)
    summary = summarize(output_rows)
    print(summary)
    print(f"\nReport written to          : {output_path}")
    print(f"Trace JSONL written to     : {TRACE_PATH}")
    print(f"Order_Date report written  : {ORDER_DATE_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
