import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import expressions as exp

from text2sql.app.run_v2 import predict_tables_with_llm
from text2sql.schema.retrieval.rasl_table_retriever import (
    build_candidate_table_context,
    load_entities,
    load_summary,
    render_candidate_table_context,
    retrieve_rasl_table_candidates,
)


BASE_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = BASE_DIR.parent
DEFAULT_INPUT = BASE_DIR / "data" / "debug" / "mock_relational_debug_30.csv"
DEFAULT_TRACES = BASE_DIR / "debug_artifacts" / "rasl_table_retrieval_traces.jsonl"
DEFAULT_OUTPUT_CSV = BASE_DIR / "evaluation" / "results" / "v2" / "table_retrieval_debug_ablation.csv"
DEFAULT_OUTPUT_MD = BASE_DIR / "evaluation" / "results" / "v2" / "table_retrieval_debug_report.md"
EXPECTED_MOCK_TABLES = {"customers", "locations", "orders", "order_items", "products"}
TABLE_ROLES = {
    "order_items": ["fact_table"],
    "orders": ["bridge_table", "time_table"],
    "products": ["dimension_table"],
    "locations": ["dimension_table"],
    "customers": ["dimension_table"],
}


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


def parse_pipe_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def normalize_table_name(table: str) -> str:
    text = str(table or "").strip().strip(";")
    text = re.split(r"\s+AS\s+", text, flags=re.IGNORECASE)[0].strip()
    text = text.split()[0] if " " in text else text
    text = text.replace('"', "")
    if "." in text:
        text = text.split(".")[-1]
    return text.strip()


def normalize_tables(tables: list[str]) -> list[str]:
    out = []
    seen = set()
    for table in tables:
        normalized = normalize_table_name(table)
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def physical_tables_from_sql(sql: str) -> list[str]:
    if not sql:
        return []
    try:
        ast = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return []
    cte_names = {cte.alias_or_name for cte in ast.find_all(exp.CTE) if cte.alias_or_name}
    tables = []
    for table in ast.find_all(exp.Table):
        name = normalize_table_name(table.name)
        if name and name not in cte_names and name not in tables:
            tables.append(name)
    return tables


def recall(selected: list[str], gold: list[str]) -> float | str:
    gold_set = set(normalize_tables(gold))
    if not gold_set:
        return ""
    selected_set = set(normalize_tables(selected))
    return round(len(selected_set & gold_set) / len(gold_set), 4)


def precision(selected: list[str], gold: list[str]) -> float | str:
    selected_set = set(normalize_tables(selected))
    if not selected_set:
        return "" if not gold else 0.0
    gold_set = set(normalize_tables(gold))
    return round(len(selected_set & gold_set) / len(selected_set), 4)


def missed(selected: list[str], gold: list[str]) -> list[str]:
    return sorted(set(normalize_tables(gold)) - set(normalize_tables(selected)))


def extra(selected: list[str], gold: list[str]) -> list[str]:
    return sorted(set(normalize_tables(selected)) - set(normalize_tables(gold)))


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def load_trace_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    by_case: dict[str, dict[str, Any]] = {}
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
                by_case[case_id] = record
    return by_case


def entities_for_table(entities: list[dict[str, Any]], table: str) -> list[dict[str, Any]]:
    out = []
    for entity in entities:
        if normalize_table_name(str(entity.get("target_table", ""))) == table:
            out.append(entity)
        elif normalize_table_name(str(entity.get("from_table", ""))) == table:
            out.append(entity)
        elif normalize_table_name(str(entity.get("to_table", ""))) == table:
            out.append(entity)
    return out


def retrieved_entities_for_table(matched_entities: list[dict[str, Any]], table: str) -> list[dict[str, Any]]:
    return entities_for_table(matched_entities, table)


def table_score_lookup(table_scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize_table_name(str(row.get("table", ""))): row for row in table_scores}


def missed_table_evidence(
    missed_tables: list[str],
    entities: list[dict[str, Any]],
    matched_entities: list[dict[str, Any]],
    all_table_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scores = table_score_lookup(all_table_scores)
    rank_by_table = {
        normalize_table_name(str(row.get("table", ""))): index + 1
        for index, row in enumerate(all_table_scores)
    }
    evidence = []
    for table in missed_tables:
        schema_entities = entities_for_table(entities, table)
        retrieved = retrieved_entities_for_table(matched_entities, table)
        score_row = scores.get(table, {})
        retrieval_queries = sorted(
            {
                query
                for entity in retrieved
                for query in entity.get("matched_queries", [])
                if query
            }
        )
        entity_types = sorted({str(entity.get("entity_type", "")) for entity in retrieved if entity.get("entity_type")})
        evidence.append(
            {
                "missed_table": table,
                "roles": TABLE_ROLES.get(table, []),
                "exists_in_schema_entities": bool(schema_entities),
                "retrieved_entity_count": len(retrieved),
                "retrieved_entities_for_that_table": [
                    {
                        "entity_id": entity.get("entity_id"),
                        "entity_type": entity.get("entity_type"),
                        "target_column": entity.get("target_column"),
                        "vector_score": entity.get("vector_score"),
                        "text": entity.get("text"),
                    }
                    for entity in sorted(retrieved, key=lambda item: float(item.get("vector_score", 0.0)), reverse=True)[:8]
                ],
                "max_entity_score_for_table": score_row.get("max_vector_score", ""),
                "table_score_rank": rank_by_table.get(table, ""),
                "table_score": score_row.get("score", ""),
                "retrieval_queries_that_hit_this_table": retrieval_queries,
                "entity_types_that_hit_this_table": entity_types,
            }
        )
    return evidence


def classify_stage(
    missed_before_llm: list[str],
    missed_after_llm: list[str],
    candidate_tables: list[str],
    gold_tables: list[str],
) -> str:
    if missed_before_llm:
        return "retrieval_candidate_miss"
    if not missed_before_llm and missed_after_llm:
        return "table_predictor_dropped"
    if not missed_before_llm and not missed_after_llm:
        return "correct_table_selection"
    if set(normalize_tables(candidate_tables)) >= set(normalize_tables(gold_tables)):
        return "metric_or_normalization_bug"
    return "unknown"


def suspected_root_cause(
    failure_stage: str,
    evidence: list[dict[str, Any]],
) -> str:
    if failure_stage == "correct_table_selection":
        return "none"
    if failure_stage == "table_predictor_dropped":
        return "table_predictor_drop_fail"
    if failure_stage == "metric_or_normalization_bug":
        return "metric_or_normalization_bug"
    if failure_stage != "retrieval_candidate_miss":
        return "unknown"
    if any(not item["exists_in_schema_entities"] for item in evidence):
        return "schema_entity_missing"
    if any(item["retrieved_entity_count"] == 0 for item in evidence):
        return "vector_entity_retrieval_miss"
    if any(item["table_score_rank"] and int(item["table_score_rank"]) > 5 for item in evidence):
        return "table_score_aggregation_fail"
    return "candidate_table_filter_fail"


def all_table_score_rows(all_tables: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "table": table,
            "score": 1.0,
            "max_vector_score": 1.0,
            "evidence_entity_ids": [],
            "evidence": [],
            "score_source": "oracle_all_tables",
            "score_aggregation": "oracle",
            "lexical_enabled": False,
        }
        for table in all_tables
    ]


def predict_tables_safe(
    question: str,
    query_decomposition: dict[str, Any],
    candidate_tables: list[str],
    candidate_context: dict[str, Any],
) -> tuple[list[str], dict[str, Any], str]:
    try:
        payload = predict_tables_with_llm(
            question=question,
            query_decomposition=query_decomposition,
            candidate_table_context_text=render_candidate_table_context(candidate_context),
            candidate_tables=candidate_tables,
            candidate_table_context=candidate_context,
        )
        return payload.get("predicted_tables", []), payload, ""
    except Exception as exc:
        return [], {}, str(exc)


def build_debug_row(
    case: dict[str, str],
    ablation_mode: str,
    all_tables: list[str],
    entities: list[dict[str, Any]],
    candidate_tables: list[str],
    predicted_tables: list[str],
    table_scores: list[dict[str, Any]],
    matched_entities: list[dict[str, Any]],
    query_decomposition: dict[str, Any],
    sqlglot_tables_raw: list[str],
    gold_tables_final: list[str],
    gold_table_mismatch_warning: bool,
    table_predictor_output: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    missed_before = missed(candidate_tables, gold_tables_final)
    missed_after = missed(predicted_tables, gold_tables_final)
    extra_before = extra(candidate_tables, gold_tables_final)
    extra_after = extra(predicted_tables, gold_tables_final)
    failure_stage = classify_stage(missed_before, missed_after, candidate_tables, gold_tables_final)
    evidence = missed_table_evidence(
        sorted(set(missed_before) | set(missed_after)),
        entities,
        matched_entities,
        table_scores,
    )
    root_cause = suspected_root_cause(failure_stage, evidence)
    if error and "rate limit" in error.lower() or error and "429" in error:
        root_cause = "provider_rate_limited"

    return {
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "ablation_mode": ablation_mode,
        "gold_sql": case.get("gold_sql", ""),
        "gold_tables_raw": case.get("gold_tables", ""),
        "gold_tables": "|".join(gold_tables_final),
        "gold_tables_normalized": "|".join(gold_tables_final),
        "sqlglot_tables_raw": "|".join(sqlglot_tables_raw),
        "gold_table_mismatch_warning": gold_table_mismatch_warning,
        "candidate_tables_raw": "|".join(candidate_tables),
        "candidate_tables": "|".join(normalize_tables(candidate_tables)),
        "candidate_tables_before_llm": "|".join(normalize_tables(candidate_tables)),
        "candidate_tables_normalized": "|".join(normalize_tables(candidate_tables)),
        "predicted_tables_raw": "|".join(predicted_tables),
        "predicted_tables": "|".join(normalize_tables(predicted_tables)),
        "predicted_tables_after_llm": "|".join(normalize_tables(predicted_tables)),
        "predicted_tables_normalized": "|".join(normalize_tables(predicted_tables)),
        "all_available_tables": "|".join(all_tables),
        "table_scores_before_llm": safe_json(table_scores),
        "missed_tables_before_llm": "|".join(missed_before),
        "missed_tables_after_llm": "|".join(missed_after),
        "extra_tables_before_llm": "|".join(extra_before),
        "extra_tables_after_llm": "|".join(extra_after),
        "table_recall_before_llm": recall(candidate_tables, gold_tables_final),
        "predicted_table_recall": recall(predicted_tables, gold_tables_final),
        "predicted_table_precision": precision(predicted_tables, gold_tables_final),
        "failure_stage": failure_stage,
        "suspected_root_cause": root_cause,
        "missed_table_evidence": safe_json(evidence),
        "query_decomposition": safe_json(query_decomposition),
        "retrieval_queries": safe_json(query_decomposition.get("retrieval_queries", [])),
        "table_predictor_output": safe_json(table_predictor_output or {}),
        "error": error,
        "top_k_candidate_tables": 5,
        "min_table_score": 0.0,
        "max_tables_for_prediction": len(candidate_tables),
        "number_of_candidate_tables_before_filter": len(table_scores),
        "number_of_candidate_tables_after_filter": len(candidate_tables),
    }


def analyze_case(
    case: dict[str, str],
    summary: dict[str, Any],
    entities: list[dict[str, Any]],
    debug_all_tables_candidate: bool,
) -> list[dict[str, Any]]:
    all_tables = [table["table_name"] for table in summary.get("tables", [])]
    gold_tables_from_csv = normalize_tables(parse_pipe_list(case.get("gold_tables")))
    sqlglot_tables_raw = physical_tables_from_sql(case.get("gold_sql", ""))
    gold_tables_final = gold_tables_from_csv or sqlglot_tables_raw
    gold_table_mismatch_warning = bool(
        gold_tables_from_csv
        and sqlglot_tables_raw
        and set(gold_tables_from_csv) != set(sqlglot_tables_raw)
    )

    rows: list[dict[str, Any]] = []
    try:
        retrieval = retrieve_rasl_table_candidates(
            case["question"],
            mode="mock_relational",
            top_k_candidate_tables=5,
            top_m_per_entity_type=5,
        )
    except Exception as exc:
        return [
            build_debug_row(
                case,
                ablation_mode=mode,
                all_tables=all_tables,
                entities=entities,
                candidate_tables=[],
                predicted_tables=[],
                table_scores=[],
                matched_entities=[],
                query_decomposition={},
                sqlglot_tables_raw=sqlglot_tables_raw,
                gold_tables_final=gold_tables_final,
                gold_table_mismatch_warning=gold_table_mismatch_warning,
                error=str(exc),
            )
            for mode in ["oracle_all_tables_candidate", "vector_retrieval_candidate", "no_llm_top5_tables"]
        ]

    all_scores = retrieval.get("all_table_scores", [])
    vector_scores = retrieval.get("candidate_table_scores", [])
    vector_candidates = retrieval.get("candidate_tables", [])
    if debug_all_tables_candidate:
        vector_scores = all_table_score_rows(all_tables)
        vector_candidates = all_tables
    matched_entities = retrieval.get("matched_entities", [])
    query_decomposition = retrieval.get("query_decomposition", {})

    oracle_scores = all_table_score_rows(all_tables)
    oracle_context = build_candidate_table_context(summary, oracle_scores, matched_entities)
    oracle_predicted, oracle_output, oracle_error = predict_tables_safe(
        case["question"],
        query_decomposition,
        all_tables,
        oracle_context,
    )
    rows.append(
        build_debug_row(
            case,
            ablation_mode="oracle_all_tables_candidate",
            all_tables=all_tables,
            entities=entities,
            candidate_tables=all_tables,
            predicted_tables=oracle_predicted,
            table_scores=oracle_scores,
            matched_entities=matched_entities,
            query_decomposition=query_decomposition,
            sqlglot_tables_raw=sqlglot_tables_raw,
            gold_tables_final=gold_tables_final,
            gold_table_mismatch_warning=gold_table_mismatch_warning,
            table_predictor_output=oracle_output,
            error=oracle_error,
        )
    )

    vector_context = build_candidate_table_context(summary, vector_scores, matched_entities)
    vector_predicted, vector_output, vector_error = predict_tables_safe(
        case["question"],
        query_decomposition,
        vector_candidates,
        vector_context,
    )
    rows.append(
        build_debug_row(
            case,
            ablation_mode="vector_retrieval_candidate",
            all_tables=all_tables,
            entities=entities,
            candidate_tables=vector_candidates,
            predicted_tables=vector_predicted,
            table_scores=all_scores,
            matched_entities=matched_entities,
            query_decomposition=query_decomposition,
            sqlglot_tables_raw=sqlglot_tables_raw,
            gold_tables_final=gold_tables_final,
            gold_table_mismatch_warning=gold_table_mismatch_warning,
            table_predictor_output=vector_output,
            error=vector_error,
        )
    )

    rows.append(
        build_debug_row(
            case,
            ablation_mode="no_llm_top5_tables",
            all_tables=all_tables,
            entities=entities,
            candidate_tables=vector_candidates,
            predicted_tables=vector_candidates,
            table_scores=all_scores,
            matched_entities=matched_entities,
            query_decomposition=query_decomposition,
            sqlglot_tables_raw=sqlglot_tables_raw,
            gold_tables_final=gold_tables_final,
            gold_table_mismatch_warning=gold_table_mismatch_warning,
            table_predictor_output={"predicted_tables": vector_candidates, "reasoning_summary": "Bypass LLM table predictor."},
            error="",
        )
    )
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "ablation_mode",
        "gold_sql",
        "gold_tables_raw",
        "gold_tables",
        "gold_tables_normalized",
        "sqlglot_tables_raw",
        "gold_table_mismatch_warning",
        "candidate_tables_raw",
        "candidate_tables",
        "candidate_tables_before_llm",
        "candidate_tables_normalized",
        "predicted_tables_raw",
        "predicted_tables",
        "predicted_tables_after_llm",
        "predicted_tables_normalized",
        "all_available_tables",
        "table_scores_before_llm",
        "missed_tables_before_llm",
        "missed_tables_after_llm",
        "extra_tables_before_llm",
        "extra_tables_after_llm",
        "table_recall_before_llm",
        "predicted_table_recall",
        "predicted_table_precision",
        "failure_stage",
        "suspected_root_cause",
        "missed_table_evidence",
        "query_decomposition",
        "retrieval_queries",
        "table_predictor_output",
        "error",
        "top_k_candidate_tables",
        "min_table_score",
        "max_tables_for_prediction",
        "number_of_candidate_tables_before_filter",
        "number_of_candidate_tables_after_filter",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def rows_for_mode(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["ablation_mode"] == mode]


def metric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value == "":
            continue
        values.append(float(value))
    return values


def split_tables(value: str) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    vector_rows = rows_for_mode(rows, "vector_retrieval_candidate")
    oracle_rows = rows_for_mode(rows, "oracle_all_tables_candidate")
    no_llm_rows = rows_for_mode(rows, "no_llm_top5_tables")
    all_tables = vector_rows[0]["all_available_tables"] if vector_rows else ""
    modes = [
        ("oracle_all_tables_candidate", oracle_rows),
        ("vector_retrieval_candidate", vector_rows),
        ("no_llm_top5_tables", no_llm_rows),
    ]

    missed_table_counter: Counter[str] = Counter()
    missed_role_counter: Counter[str] = Counter()
    for row in vector_rows:
        for table in split_tables(row["missed_tables_before_llm"]) + split_tables(row["missed_tables_after_llm"]):
            missed_table_counter[table] += 1
            for role in TABLE_ROLES.get(table, ["unknown"]):
                missed_role_counter[role] += 1

    stage_counts = Counter(row["failure_stage"] for row in vector_rows)
    recall_lt_1 = [row for row in vector_rows if row["table_recall_before_llm"] != "" and float(row["table_recall_before_llm"]) < 1.0]
    predictor_drop = [row for row in vector_rows if row["failure_stage"] == "table_predictor_dropped"]
    metric_bug = [row for row in vector_rows if row["failure_stage"] == "metric_or_normalization_bug"]

    lines = [
        "# Table Retrieval Debug Report",
        "",
        "## 1. Overview",
        "",
        f"- Total cases: {len(vector_rows)}",
        f"- Avg table recall before LLM: {average(metric_values(vector_rows, 'table_recall_before_llm')):.3f}",
        f"- Avg predicted table recall: {average(metric_values(vector_rows, 'predicted_table_recall')):.3f}",
        f"- Avg predicted table precision: {average(metric_values(vector_rows, 'predicted_table_precision')):.3f}",
        "",
        "## 2. Sanity Check",
        "",
        f"- all_available_tables: `{all_tables}`",
        "- expected tables: `customers|locations|orders|order_items|products`",
        "- top_k_candidate_tables: `5`",
        "- min_table_score: `0.0`",
        "- candidate pool can contain all 5 tables: `true`",
        "",
        "## 3. Cases With Recall < 1",
        "",
    ]
    if recall_lt_1 or predictor_drop:
        for row in vector_rows:
            if row["failure_stage"] == "correct_table_selection":
                continue
            lines.extend(
                [
                    f"### Case {row['id']}",
                    "",
                    f"- Question: {row['question']}",
                    f"- gold_tables: `{row['gold_tables']}`",
                    f"- candidate_tables_before_llm: `{row['candidate_tables_before_llm']}`",
                    f"- predicted_tables_after_llm: `{row['predicted_tables_after_llm']}`",
                    f"- missed_before_llm: `{row['missed_tables_before_llm']}`",
                    f"- missed_after_llm: `{row['missed_tables_after_llm']}`",
                    f"- failure_stage: `{row['failure_stage']}`",
                    f"- suspected_root_cause: `{row['suspected_root_cause']}`",
                    "",
                ]
            )
    else:
        lines.append("- No vector retrieval cases with table recall < 1.")

    lines.extend(["", "## 4. Missed Table Distribution", ""])
    if missed_table_counter:
        lines.extend(f"- {table}: {count}" for table, count in missed_table_counter.most_common())
    else:
        lines.append("- No missed tables in vector_retrieval_candidate mode.")
    lines.append("")
    if missed_role_counter:
        lines.extend(f"- role {role}: {count}" for role, count in missed_role_counter.most_common())
    else:
        lines.append("- No missed table roles.")

    lines.extend(["", "## 5. Before vs After LLM", ""])
    lines.extend(
        [
            f"- retriever miss from start: {sum(1 for row in vector_rows if row['failure_stage'] == 'retrieval_candidate_miss')}",
            f"- LLM predictor dropped gold table: {len(predictor_drop)}",
            f"- possible metric/normalization bugs: {len(metric_bug)}",
            "",
        ]
    )

    lines.extend(["## 6. Ablation Result", ""])
    for mode, mode_rows in modes:
        lines.extend(
            [
                f"### {mode}",
                "",
                f"- Avg table_recall_before_llm: {average(metric_values(mode_rows, 'table_recall_before_llm')):.3f}",
                f"- Avg predicted_table_recall: {average(metric_values(mode_rows, 'predicted_table_recall')):.3f}",
                f"- Avg predicted_table_precision: {average(metric_values(mode_rows, 'predicted_table_precision')):.3f}",
                f"- Failure stages: {dict(Counter(row['failure_stage'] for row in mode_rows))}",
                "",
            ]
        )

    most_common_stage = stage_counts.most_common(1)[0][0] if stage_counts else "unknown"
    conclusion = (
        "The main observed issue is not vector candidate retrieval if vector_retrieval_candidate recall is 1.0. "
        "If table predictor recall is lower than candidate recall, the issue is in LLM table prediction."
    )
    if most_common_stage == "correct_table_selection":
        conclusion = "Table retrieval and table prediction are correct on this run; previous low recall was likely metric contamination from failed/rate-limited rows."
    elif most_common_stage == "table_predictor_dropped":
        conclusion = "Candidate retrieval keeps the gold tables, but the LLM table predictor drops at least one required table."
    elif most_common_stage == "retrieval_candidate_miss":
        conclusion = "Vector candidate retrieval misses gold tables before the LLM table predictor."

    lines.extend(
        [
            "## 7. Conclusion",
            "",
            f"- Most common failure_stage: `{most_common_stage}`",
            f"- Conclusion: {conclusion}",
            "",
            "## 8. Recommended Next Fix",
            "",
            "Do not change SQL generator from this report. If predictor drop dominates, adjust table predictor prompt or add deterministic bridge/time/fact table retention. If retrieval_candidate_miss dominates, inspect entity retrieval and table score aggregation. If metric contamination dominates, update evaluator to exclude provider failures from table recall.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug RASL table retrieval and table prediction.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--traces", default=str(DEFAULT_TRACES))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--debug-all-tables-candidate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    traces_path = resolve_path(args.traces)
    output_csv = resolve_path(args.output_csv)
    output_md = resolve_path(args.output_md)
    cases = read_cases(input_path)
    summary = load_summary("mock_relational")
    entities = load_entities("mock_relational")
    all_tables = [table["table_name"] for table in summary.get("tables", [])]
    if set(all_tables) != EXPECTED_MOCK_TABLES:
        raise RuntimeError(f"Unexpected mock_relational tables: {all_tables}")

    # Trace loading is useful for compatibility with prior debug artifacts, but this script recomputes retrieval.
    load_trace_records(traces_path)

    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(analyze_case(case, summary, entities, args.debug_all_tables_candidate))
        write_csv(rows, output_csv)
        write_markdown(rows, output_md)

    write_csv(rows, output_csv)
    write_markdown(rows, output_md)

    vector_rows = rows_for_mode(rows, "vector_retrieval_candidate")
    stage_counts = Counter(row["failure_stage"] for row in vector_rows)
    missed_tables = Counter(
        table
        for row in vector_rows
        for table in split_tables(row["missed_tables_before_llm"]) + split_tables(row["missed_tables_after_llm"])
    )
    print("=== Table Retrieval Debug ===")
    print(f"Total cases                         : {len(vector_rows)}")
    print(f"Avg table recall before LLM         : {average(metric_values(vector_rows, 'table_recall_before_llm')):.3f}")
    print(f"Avg predicted table recall          : {average(metric_values(vector_rows, 'predicted_table_recall')):.3f}")
    print(f"Avg predicted table precision       : {average(metric_values(vector_rows, 'predicted_table_precision')):.3f}")
    print(f"Cases recall < 1 before LLM         : {sum(1 for row in vector_rows if row['table_recall_before_llm'] != '' and float(row['table_recall_before_llm']) < 1.0)}")
    print(f"Cases where LLM dropped gold table  : {stage_counts.get('table_predictor_dropped', 0)}")
    print(f"Possible normalization bugs         : {stage_counts.get('metric_or_normalization_bug', 0)}")
    print(f"Most missed table                   : {missed_tables.most_common(1)[0][0] if missed_tables else 'none'}")
    print(f"Most common failure_stage           : {stage_counts.most_common(1)[0][0] if stage_counts else 'none'}")
    print(f"\nCSV written to                      : {output_csv}")
    print(f"Markdown written to                 : {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
