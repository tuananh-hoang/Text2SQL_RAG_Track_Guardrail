# OWASP Defense in Depth - 3 lop bao ve:
# Lop 1 - Parameterized queries:
#   KHONG ap dung duoc vi Text-to-SQL bat buoc
#   execute dynamic SQL. Day la accepted risk.
#   Giam thieu bang Lop 2 va Lop 3.
# Lop 2 - Allowlist validation: implemented o day.
# Lop 3 - Least privilege readonly user:
#   implemented in sql/executor.py + docker-compose.yml

# PICARD limitation:
# V1 dung post-hoc validation (validate sau khi LLM
# sinh xong toan bo SQL). PICARD de xuat constrained
# decoding (validate tung token trong qua trinh sinh)
# nhung khong kha thi voi OpenAI API.
# Huong cai thien V2/V3: dung open-source LLM de
# implement constrained decoding theo PICARD.

import json
from pathlib import Path
from typing import Any, Iterable

import sqlglot
from sqlglot import expressions as exp


DANGEROUS_EXPRESSION_NAMES = (
    "Drop",
    "Delete",
    "Update",
    "Insert",
    "Alter",
    "AlterTable",
    "Truncate",
    "TruncateTable",
    "Create",
    "Command",
)

PROHIBITED_SYSTEM_EXPRESSION_NAMES = (
    "CurrentCatalog",
    "CurrentDatabase",
    "CurrentSchema",
    "CurrentUser",
    "CurrentVersion",
)

PROHIBITED_SYSTEM_FUNCTION_NAMES = {
    "current_database",
    "current_schema",
    "current_setting",
    "current_user",
    "inet_client_addr",
    "inet_client_port",
    "inet_server_addr",
    "inet_server_port",
    "pg_backend_pid",
    "pg_conf_load_time",
    "pg_database_size",
    "pg_postmaster_start_time",
    "pg_sleep",
    "session_user",
    "user",
    "version",
}


def iter_ast_nodes(ast: exp.Expression) -> Iterable[exp.Expression]:
    for item in ast.walk():
        yield item[0] if isinstance(item, tuple) else item


def load_allowlist(schema_path: str) -> tuple[set[str], set[str]]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if "tables" in schema:
        allowed_tables = {table["table_name"] for table in schema["tables"]}
        allowed_columns = {
            column["name"]
            for table in schema["tables"]
            for column in table.get("columns", [])
        }
    else:
        allowed_tables = {schema["table_name"]}
        allowed_columns = {column["name"] for column in schema["columns"]}
    return allowed_tables, allowed_columns


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def expression_classes(names: Iterable[str]) -> tuple[type[exp.Expression], ...]:
    classes = []
    for name in names:
        cls = getattr(exp, name, None)
        if cls is not None:
            classes.append(cls)
    return tuple(classes)


def cte_aliases(ast: exp.Expression) -> set[str]:
    return {
        cte.alias_or_name
        for cte in ast.find_all(exp.CTE)
        if cte.alias_or_name
    }


def table_names_from_ast(ast: exp.Expression) -> list[str]:
    aliases = cte_aliases(ast)
    return unique_preserve_order(
        table.name
        for table in ast.find_all(exp.Table)
        if table.name and table.name not in aliases
    )


def has_prohibited_system_call(ast: exp.Expression) -> bool:
    system_classes = expression_classes(PROHIBITED_SYSTEM_EXPRESSION_NAMES)
    for node in iter_ast_nodes(ast):
        if system_classes and isinstance(node, system_classes):
            return True
        if isinstance(node, exp.Anonymous):
            function_name = str(node.name or node.this or "").lower()
            if function_name in PROHIBITED_SYSTEM_FUNCTION_NAMES:
                return True
    return False


def has_aggregate(ast: exp.Expression) -> bool:
    aggregate_class = getattr(exp, "AggFunc", None)
    if aggregate_class is None:
        aggregate_class = expression_classes(("Count", "Sum", "Avg", "Min", "Max"))
    else:
        aggregate_class = (aggregate_class,)
    return any(isinstance(node, aggregate_class) for node in iter_ast_nodes(ast))


def has_outer_limit(ast: exp.Expression) -> bool:
    return ast.args.get("limit") is not None


def add_limit(sql: str, ast: exp.Expression, limit: int = 100) -> str:
    try:
        limited_ast = ast.copy()
        limited_ast.set("limit", exp.Limit(expression=exp.Literal.number(limit)))
        return limited_ast.sql(dialect="postgres")
    except Exception:
        return f"{sql.strip().rstrip(';')} LIMIT {limit}"


def validate_sql(sql: str, schema_path: str) -> dict[str, Any]:
    if sql.strip().upper() == "NO_SQL":
        return {
            "valid": False,
            "sql": sql,
            "tables_from_ast": [],
            "columns_from_ast": [],
            "auto_limited": False,
            "error": "NO_SQL",
        }

    try:
        statements = sqlglot.parse(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return {
            "valid": False,
            "sql": sql,
            "tables_from_ast": [],
            "columns_from_ast": [],
            "auto_limited": False,
            "error": "SQL syntax error",
        }

    statements = [statement for statement in statements if statement is not None]
    if len(statements) != 1:
        return {
            "valid": False,
            "sql": sql,
            "tables_from_ast": [],
            "columns_from_ast": [],
            "auto_limited": False,
            "error": "Only one SQL statement allowed",
        }

    ast = statements[0]
    if not isinstance(ast, exp.Select):
        return {
            "valid": False,
            "sql": sql,
            "tables_from_ast": [],
            "columns_from_ast": [],
            "auto_limited": False,
            "error": "Only SELECT allowed",
        }

    dangerous_classes = expression_classes(DANGEROUS_EXPRESSION_NAMES)
    if dangerous_classes:
        for node in iter_ast_nodes(ast):
            if isinstance(node, dangerous_classes):
                return {
                    "valid": False,
                    "sql": sql,
                    "tables_from_ast": [],
                    "columns_from_ast": [],
                    "auto_limited": False,
                    "error": "Dangerous statement detected",
                }

    if has_prohibited_system_call(ast):
        return {
            "valid": False,
            "sql": sql,
            "tables_from_ast": [],
            "columns_from_ast": [],
            "auto_limited": False,
            "error": "Database metadata/system function not allowed",
        }

    allowed_tables, allowed_columns = load_allowlist(schema_path)
    tables_from_ast = table_names_from_ast(ast)
    alias_names = {
        alias.alias
        for alias in ast.find_all(exp.Alias)
        if alias.alias
    }

    raw_columns = unique_preserve_order(
        column.name
        for column in ast.find_all(exp.Column)
        if column.name and column.name != "*"
    )
    columns_from_ast = [column for column in raw_columns if column in allowed_columns]

    # fix: metadata-only SELECTs have no table nodes, so require business data access.
    if not tables_from_ast:
        return {
            "valid": False,
            "sql": sql,
            "tables_from_ast": tables_from_ast,
            "columns_from_ast": columns_from_ast,
            "auto_limited": False,
            "error": "Query must reference an allowed data table",
        }

    for table in tables_from_ast:
        if table not in allowed_tables:
            return {
                "valid": False,
                "sql": sql,
                "tables_from_ast": tables_from_ast,
                "columns_from_ast": columns_from_ast,
                "auto_limited": False,
                "error": "Table not in schema",
            }

    for column in raw_columns:
        if column not in allowed_columns and column not in alias_names:
            return {
                "valid": False,
                "sql": sql,
                "tables_from_ast": tables_from_ast,
                "columns_from_ast": columns_from_ast,
                "auto_limited": False,
                "error": f"Column not in schema: {column}",
            }

    auto_limited = False
    validated_sql = sql.strip().rstrip(";")
    if not has_outer_limit(ast) and not has_aggregate(ast):
        validated_sql = add_limit(validated_sql, ast, limit=100)
        auto_limited = True

    return {
        "valid": True,
        "sql": validated_sql,
        "tables_from_ast": tables_from_ast,
        "columns_from_ast": columns_from_ast,
        "auto_limited": auto_limited,
        "error": None,
    }
