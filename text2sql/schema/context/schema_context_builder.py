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


def is_numeric_type(data_type: str) -> bool:
    return data_type.lower() in NUMERIC_TYPES


def is_date_type(data_type: str) -> bool:
    normalized = data_type.lower()
    return normalized in DATE_TYPES or normalized.startswith("timestamp")


def render_column_comment(column: dict[str, Any]) -> str:
    data_type = column["type"].lower()
    if column.get("all_values"):
        values = ", ".join(quote_sql_value(value) for value in column["all_values"])
        return f"values: {values}"

    if column.get("sample_values"):
        values = ", ".join(quote_sql_value(value) for value in column["sample_values"])
        return f"sample: {values}"

    if is_numeric_type(data_type) and column.get("min") is not None and column.get("max") is not None:
        avg = column.get("avg")
        if avg is not None:
            return f"range: {column['min']} to {column['max']}, avg: {avg}"
        return f"range: {column['min']} to {column['max']}"

    if is_date_type(data_type) and column.get("min") is not None and column.get("max") is not None:
        return f"range: {quote_sql_value(column['min'])} to {quote_sql_value(column['max'])}"

    return ""


def build_schema_context(schema_path: str) -> str:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    table_name = schema["table_name"]
    visible_columns = [column for column in schema["columns"] if not column.get("is_id")]

    if not visible_columns:
        raise ValueError("Schema summary has no visible columns")

    quoted_names = [quote_identifier(column["name"]) for column in visible_columns]
    max_name_len = max(len(name) for name in quoted_names)
    max_type_len = max(len(column["type"]) for column in visible_columns)

    # fix: expose table grain so top-k entity questions aggregate rows correctly.
    lines = [
        "-- Table grain:",
        "-- product_sales is order-line level; each Product_Name can appear in many rows.",
        '-- Product-level revenue means SUM("Revenue") GROUP BY "Product_Name".',
        "-- Dimension-level metrics by Region, Category, Sub_Category, City, or Product_Name require GROUP BY.",
        "",
        f"CREATE TABLE {table_name} (",
    ]
    for index, column in enumerate(visible_columns):
        quoted_name = quote_identifier(column["name"])
        comma = "," if index < len(visible_columns) - 1 else ""
        type_with_comma = f"{column['type']}{comma}"
        line = f"    {quoted_name.ljust(max_name_len)} {type_with_comma.ljust(max_type_len + 1)}"
        comment = render_column_comment(column)
        if comment:
            line = f"{line.ljust(8 + max_name_len + 1 + max_type_len + 1)} -- {comment}"
        lines.append(line)
    lines.append(");")

    alias_lines = []
    for column in visible_columns:
        aliases = column.get("column_aliases", [])
        if aliases:
            alias_lines.append(f"-- {quote_identifier(column['name']).ljust(max_name_len)} = {', '.join(aliases)}")

    if alias_lines:
        lines.append("")
        lines.append("-- Column aliases (Vietnamese):")
        lines.extend(alias_lines)

    return "\n".join(lines)
