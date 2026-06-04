import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from text2sql.schema.paths import V1_SCHEMA_SUMMARY_PATH, V2_ENTITIES_DIR, V2_SUMMARIES_DIR, v2_entities_path


REQUIRED_ENTITY_FIELDS = {
    "entity_id",
    "entity_type",
    "text",
    "target_type",
    "target_table",
    "target_column",
}
VALUE_EVIDENCE_FIELDS = {
    "all_values",
    "sample_values",
    "min",
    "max",
    "avg",
    "date_format",
}


def load_summary(mode: str) -> dict[str, Any]:
    path = V2_SUMMARIES_DIR / f"schema_summary_{mode}.json"
    if not path.exists() and mode == "product_sales":
        path = V1_SCHEMA_SUMMARY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing schema summary for mode={mode}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_tables(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if "tables" in summary:
        return summary["tables"]
    return [
        {
            "schema_name": summary.get("schema_name", "public"),
            "table_name": summary["table_name"],
            "row_count": summary.get("row_count"),
            "column_count": summary.get("column_count", len(summary.get("columns", []))),
            "table_aliases": summary.get("table_aliases", []),
            "table_description": summary.get("table_description", f"Table {summary['table_name']}."),
            "columns": summary.get("columns", []),
        }
    ]


def entity(
    entity_id: str,
    entity_type: str,
    text: str,
    target_type: str,
    target_table: str,
    target_column: str | None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "text": str(text).strip(),
        "target_type": target_type,
        "target_table": target_table,
        "target_column": target_column,
    }
    payload.update(extra)
    return payload


def entity_prefix(entity_type: str, schema_name: str, table_name: str, column_name: str | None = None) -> str:
    if column_name:
        return f"{entity_type}.{schema_name}.{table_name}.{column_name}"
    return f"{entity_type}.{schema_name}.{table_name}"


def build_column_description(table_name: str, column: dict[str, Any]) -> str:
    description = column.get("column_description")
    if description:
        return description
    aliases = column.get("column_aliases", [])
    alias_text = f" Aliases: {', '.join(aliases)}." if aliases else ""
    return f"{column['name']} is a {column.get('type', 'unknown')} column in table {table_name}.{alias_text}"


def build_value_format_text(column: dict[str, Any]) -> str:
    parts = []
    if column.get("all_values"):
        parts.append("Possible values: " + ", ".join(map(str, column["all_values"])))
    if column.get("sample_values"):
        parts.append("Sample values: " + ", ".join(map(str, column["sample_values"])))
    if column.get("min") is not None or column.get("max") is not None:
        parts.append(f"Range: {column.get('min')} to {column.get('max')}")
    if column.get("avg") is not None:
        parts.append(f"Average: {column.get('avg')}")
    if column.get("date_format"):
        parts.append(f"Date format: {column.get('date_format')}")
    return ". ".join(parts).strip(".") + "."


def build_relationship_text(rel: dict[str, Any]) -> str:
    return (
        f"{rel['from_table']} joins {rel['to_table']} using "
        f"{rel['from_column']} = {rel['to_column']}."
    )


def build_entities(summary: dict[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for table in iter_tables(summary):
        schema_name = table.get("schema_name") or summary.get("schema_name") or "public"
        table_name = table["table_name"]
        table_extra = {
            "schema": schema_name,
            "table": table_name,
            "column": None,
        }
        entities.append(
            entity(
                entity_prefix("table_name", schema_name, table_name),
                "table_name",
                table_name,
                "table",
                table_name,
                None,
                **table_extra,
            )
        )
        if table.get("table_aliases"):
            entities.append(
                entity(
                    entity_prefix("table_alias", schema_name, table_name),
                    "table_alias",
                    "; ".join(table["table_aliases"]),
                    "table",
                    table_name,
                    None,
                    **table_extra,
                )
            )
        entities.append(
            entity(
                entity_prefix("table_description", schema_name, table_name),
                "table_description",
                table.get("table_description") or f"Table {table_name}.",
                "table",
                table_name,
                None,
                **table_extra,
            )
        )

        for column in table.get("columns", []):
            column_name = column["name"]
            column_extra = {
                "schema": schema_name,
                "table": table_name,
                "column": column_name,
            }
            entities.append(
                entity(
                    entity_prefix("column_name", schema_name, table_name, column_name),
                    "column_name",
                    column_name,
                    "column",
                    table_name,
                    column_name,
                    **column_extra,
                )
            )
            if column.get("column_aliases"):
                entities.append(
                    entity(
                        entity_prefix("column_alias", schema_name, table_name, column_name),
                        "column_alias",
                        "; ".join(column["column_aliases"]),
                        "column",
                        table_name,
                        column_name,
                        **column_extra,
                    )
                )
            entities.append(
                entity(
                    entity_prefix("column_description", schema_name, table_name, column_name),
                    "column_description",
                    build_column_description(table_name, column),
                    "column",
                    table_name,
                    column_name,
                    **column_extra,
                )
            )
            if any(key in column for key in VALUE_EVIDENCE_FIELDS):
                entities.append(
                    entity(
                        entity_prefix("value_format", schema_name, table_name, column_name),
                        "value_format_description",
                        build_value_format_text(column),
                        "column",
                        table_name,
                        column_name,
                        **column_extra,
                    )
                )

    for rel in summary.get("relationships", []):
        schema_name = rel.get("from_schema") or summary.get("schema_name") or "public"
        from_table = rel["from_table"]
        from_column = rel["from_column"]
        to_table = rel["to_table"]
        to_column = rel["to_column"]
        entities.append(
            entity(
                f"relationship.{schema_name}.{from_table}.{from_column}.{to_table}.{to_column}",
                "relationship_description",
                build_relationship_text(rel),
                "relationship",
                from_table,
                from_column,
                schema=schema_name,
                from_table=from_table,
                from_column=from_column,
                to_table=to_table,
                to_column=to_column,
                source=rel.get("source", "unknown"),
            )
        )

    validate_schema_entities(summary, entities)
    return entities


def validate_schema_entities(summary: dict[str, Any], entities: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    by_type_target: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    relationship_entity_ids = set()

    for ent in entities:
        missing = REQUIRED_ENTITY_FIELDS - set(ent)
        if missing:
            errors.append(f"{ent.get('entity_id', '<unknown>')} missing required fields: {sorted(missing)}")
        if not str(ent.get("text", "")).strip():
            errors.append(f"{ent.get('entity_id', '<unknown>')} has empty text")
        by_type_target[(ent.get("entity_type"), ent.get("target_table"), ent.get("target_column"))].append(ent)
        if ent.get("entity_type") == "relationship_description":
            relationship_entity_ids.add(ent.get("entity_id"))

    for table in iter_tables(summary):
        table_name = table["table_name"]
        for column in table.get("columns", []):
            column_name = column["name"]
            if not by_type_target.get(("column_name", table_name, column_name)):
                errors.append(f"{table_name}.{column_name} missing column_name entity")
            if not by_type_target.get(("column_description", table_name, column_name)):
                errors.append(f"{table_name}.{column_name} missing column_description entity")
            if column.get("column_aliases") and not by_type_target.get(("column_alias", table_name, column_name)):
                errors.append(f"{table_name}.{column_name} missing column_alias entity")
            if any(key in column for key in VALUE_EVIDENCE_FIELDS) and not by_type_target.get(
                ("value_format_description", table_name, column_name)
            ):
                errors.append(f"{table_name}.{column_name} missing value_format_description entity")

    for rel in summary.get("relationships", []):
        schema_name = rel.get("from_schema") or summary.get("schema_name") or "public"
        expected_id = (
            f"relationship.{schema_name}.{rel['from_table']}.{rel['from_column']}."
            f"{rel['to_table']}.{rel['to_column']}"
        )
        if expected_id not in relationship_entity_ids:
            errors.append(f"Relationship missing entity: {expected_id}")

    if errors:
        raise ValueError("Schema entity correctness check failed:\n- " + "\n- ".join(errors))


def output_path_for_mode(mode: str) -> Path:
    return v2_entities_path(mode)


def write_entities(mode: str, entities: list[dict[str, Any]]) -> Path:
    V2_ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for_mode(mode)
    output_path.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_schema_entities(mode: str) -> list[dict[str, Any]]:
    summary = load_summary(mode)
    entities = build_entities(summary)
    output_path = write_entities(mode, entities)
    counts = Counter(entity["entity_type"] for entity in entities)
    print(f"Schema entities saved to {output_path}")
    print("=== Entity counts ===")
    for entity_type, count in sorted(counts.items()):
        print(f"{entity_type.ljust(28)}: {count}")
    return entities


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RASL-style schema entities.")
    parser.add_argument("--mode", choices=["product_sales", "mock_relational"], default="product_sales")
    args = parser.parse_args()
    build_schema_entities(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
