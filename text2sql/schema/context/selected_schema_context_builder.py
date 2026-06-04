import json
from pathlib import Path
from typing import Any


NUMERIC_TYPES = {"integer", "bigint", "smallint", "numeric", "real", "double precision", "float"}
DATE_TYPES = {"date", "timestamp", "timestamp without time zone", "timestamp with time zone", "timestamptz"}


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def quote_sql_value(value: Any) -> str:
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def load_schema(schema_summary_path: str) -> dict[str, Any]:
    return json.loads(Path(schema_summary_path).read_text(encoding="utf-8"))


def iter_tables(schema: dict[str, Any]) -> list[dict[str, Any]]:
    if "tables" in schema:
        return schema["tables"]
    return [
        {
            "schema_name": schema.get("schema_name", "public"),
            "table_name": schema["table_name"],
            "table_aliases": schema.get("table_aliases", []),
            "table_description": schema.get("table_description", ""),
            "columns": schema.get("columns", []),
        }
    ]


def render_column_evidence(column: dict[str, Any]) -> str:
    parts = []
    if column.get("column_description"):
        parts.append(column["column_description"])
    if column.get("column_aliases"):
        parts.append("aliases: " + ", ".join(column["column_aliases"]))
    if column.get("all_values"):
        parts.append("values: " + ", ".join(quote_sql_value(value) for value in column["all_values"]))
    elif column.get("sample_values"):
        parts.append("sample: " + ", ".join(quote_sql_value(value) for value in column["sample_values"]))
    if column.get("min") is not None or column.get("max") is not None:
        parts.append(f"range: {column.get('min')} to {column.get('max')}")
    if column.get("avg") is not None:
        parts.append(f"avg: {column.get('avg')}")
    if column.get("date_format"):
        parts.append(f"date_format: {column['date_format']}")
    return "; ".join(parts)


def normalize_selected_column(selected: str, mode: str) -> tuple[str | None, str]:
    if mode == "mock_relational" and "." in selected:
        table_name, column_name = selected.split(".", 1)
        return table_name, column_name
    return None, selected


def direct_relationships(schema: dict[str, Any], selected_table_set: set[str], matched_relationships: list[dict] | None) -> list[dict[str, Any]]:
    relationships = []
    seen = set()
    for rel in schema.get("relationships", []):
        if rel["from_table"] in selected_table_set and rel["to_table"] in selected_table_set:
            key = (rel["from_table"], rel["from_column"], rel["to_table"], rel["to_column"])
            relationships.append(rel)
            seen.add(key)
    for rel in matched_relationships or []:
        from_table = rel.get("from_table")
        to_table = rel.get("to_table")
        from_column = rel.get("from_column")
        to_column = rel.get("to_column")
        if not from_table or not to_table:
            continue
        key = (from_table, from_column, to_table, to_column)
        if key not in seen:
            relationships.append(
                {
                    "from_table": from_table,
                    "from_column": from_column,
                    "to_table": to_table,
                    "to_column": to_column,
                }
            )
    return relationships


def build_selected_schema_context(
    schema_summary_path: str,
    selected_tables: list[str],
    selected_columns: list[str],
    matched_relationships: list[dict] | None = None,
    mode: str = "product_sales",
) -> str:
    schema = load_schema(schema_summary_path)
    tables = iter_tables(schema)
    selected_table_set = set(selected_tables)
    selected_column_pairs = {normalize_selected_column(column, mode) for column in selected_columns}
    selected_table_set.update(table for table, _ in selected_column_pairs if table)
    relationships = direct_relationships(schema, selected_table_set, matched_relationships)
    relationship_columns = {
        (rel["from_table"], rel["from_column"]) for rel in relationships
    } | {
        (rel["to_table"], rel["to_column"]) for rel in relationships
    }
    lines: list[str] = []

    for table in tables:
        table_name = table["table_name"]
        table_selected_columns = []
        for column in table.get("columns", []):
            column_name = column["name"]
            if (
                (None, column_name) in selected_column_pairs
                or (table_name, column_name) in selected_column_pairs
                or (table_name, column_name) in relationship_columns
            ):
                table_selected_columns.append(column)

        if table_name not in selected_table_set and not table_selected_columns:
            continue
        if not table_selected_columns:
            # Keep table visible with all ID/FK columns if relationship-only retrieval selected it.
            table_selected_columns = [column for column in table.get("columns", []) if column.get("is_id")]

        if table.get("table_description"):
            lines.append(f"-- Table {table_name}: {table['table_description']}")
        if table.get("table_aliases"):
            lines.append(f"-- Table aliases: {', '.join(table['table_aliases'])}")
        rendered_table_name = f"{table.get('schema_name', 'mock_relational')}.{table_name}" if mode == "mock_relational" else table_name
        lines.append(f"CREATE TABLE {rendered_table_name} (")
        for index, column in enumerate(table_selected_columns):
            comma = "," if index < len(table_selected_columns) - 1 else ""
            evidence = render_column_evidence(column)
            line = f"    {quote_identifier(column['name'])} {column['type'].upper()}{comma}"
            if evidence:
                line += f" -- {evidence}"
            lines.append(line)
        lines.append(");")
        lines.append("")
        selected_table_set.add(table_name)

    if relationships:
        lines.append("-- Relationships:")
        for rel in relationships:
            lines.append(
                f"-- {rel['from_table']}.{rel['from_column']} = {rel['to_table']}.{rel['to_column']}"
            )

    if not lines:
        raise ValueError("No selected schema context could be built from selected tables/columns")
    return "\n".join(lines).strip()
