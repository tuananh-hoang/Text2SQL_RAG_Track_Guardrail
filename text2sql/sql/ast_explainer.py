from typing import Any, Iterable

import sqlglot
from sqlglot import expressions as exp


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen = set()
    ordered = []
    for value in values:
        if value is None:
            continue
        key = str(value)
        if key not in seen:
            ordered.append(value)
            seen.add(key)
    return ordered


def sql_text(node: exp.Expression | None) -> str:
    if node is None:
        return ""
    try:
        return node.sql(dialect="postgres")
    except Exception:
        return str(node)


def strip_quotes(value: str) -> str:
    return value.replace('"', "")


def cte_aliases(ast: exp.Expression) -> list[str]:
    return unique_preserve_order(
        cte.alias_or_name
        for cte in ast.find_all(exp.CTE)
        if cte.alias_or_name
    )


def real_table_names(ast: exp.Expression) -> list[str]:
    aliases = set(cte_aliases(ast))
    return unique_preserve_order(
        table.name
        for table in ast.find_all(exp.Table)
        if table.name and table.name not in aliases
    )


def select_expressions(select_ast: exp.Expression) -> list[str]:
    expressions = getattr(select_ast, "expressions", []) or []
    return unique_preserve_order(strip_quotes(sql_text(expression)) for expression in expressions)


def where_filters(ast: exp.Expression) -> list[str]:
    return unique_preserve_order(strip_quotes(sql_text(where.this)) for where in ast.find_all(exp.Where))


def group_by_expressions(ast: exp.Expression) -> list[str]:
    values: list[str] = []
    for group in ast.find_all(exp.Group):
        values.extend(strip_quotes(sql_text(expression)) for expression in (group.expressions or []))
    return unique_preserve_order(values)


def having_expressions(ast: exp.Expression) -> list[str]:
    return unique_preserve_order(strip_quotes(sql_text(having.this)) for having in ast.find_all(exp.Having))


def order_by_expressions(ast: exp.Expression) -> list[str]:
    values: list[str] = []
    for order in ast.find_all(exp.Order):
        values.extend(strip_quotes(sql_text(expression)) for expression in (order.expressions or []))
    return unique_preserve_order(values)


def limit_value(ast: exp.Expression) -> int | None:
    limits = list(ast.find_all(exp.Limit))
    if not limits:
        return None
    limit = limits[0]
    expression = limit.args.get("expression")
    if expression is None:
        return None
    try:
        return int(expression.name)
    except Exception:
        try:
            return int(sql_text(expression))
        except Exception:
            return None


def aggregate_expressions(ast: exp.Expression) -> list[str]:
    return unique_preserve_order(strip_quotes(sql_text(agg)) for agg in ast.find_all(exp.AggFunc))


def window_expressions(ast: exp.Expression) -> list[str]:
    return unique_preserve_order(strip_quotes(sql_text(window)) for window in ast.find_all(exp.Window))


def join_expressions(ast: exp.Expression) -> list[str]:
    return unique_preserve_order(strip_quotes(sql_text(join)) for join in ast.find_all(exp.Join))


def direct_join_expressions(ast: exp.Expression) -> list[str]:
    return unique_preserve_order(strip_quotes(sql_text(join)) for join in (ast.args.get("joins") or []))


def main_from_text(ast: exp.Expression) -> str:
    from_expr = ast.args.get("from_")
    if from_expr is None:
        return ""
    return strip_quotes(sql_text(from_expr.this))


def describe_select_part(prefix: str, ast: exp.Expression) -> list[str]:
    lines = []
    tables = real_table_names(ast)
    filters = where_filters(ast)
    groups = group_by_expressions(ast)
    havings = having_expressions(ast)
    aggs = aggregate_expressions(ast)

    pieces = []
    if tables:
        pieces.append("đọc dữ liệu từ " + ", ".join(tables))
    if filters:
        pieces.append("lọc " + " AND ".join(filters))
    if groups:
        pieces.append("GROUP BY " + ", ".join(groups))
    if aggs:
        pieces.append("tính " + ", ".join(aggs))
    if havings:
        pieces.append("HAVING " + " AND ".join(havings))

    if pieces:
        lines.append(f"{prefix}: " + ", ".join(pieces) + ".")
    return lines


def cte_summary_lines(ast: exp.Expression) -> list[str]:
    lines = []
    ctes = list(ast.find_all(exp.CTE))
    names = [cte.alias_or_name for cte in ctes if cte.alias_or_name]
    if names:
        lines.append(f"Có {len(names)} CTE: {', '.join(names)}.")
    for cte in ctes:
        name = cte.alias_or_name or "unknown"
        lines.extend(describe_select_part(f"CTE {name}", cte.this))
    return lines


def main_query_summary_lines(ast: exp.Expression) -> list[str]:
    lines = []
    selected = select_expressions(ast)
    from_text = main_from_text(ast)
    joins = direct_join_expressions(ast)
    where = ast.args.get("where")
    group = ast.args.get("group")
    having = ast.args.get("having")
    order = ast.args.get("order")
    limit = ast.args.get("limit")
    filters = [strip_quotes(sql_text(where.this))] if where is not None else []
    groups = (
        unique_preserve_order(strip_quotes(sql_text(expression)) for expression in (group.expressions or []))
        if group is not None
        else []
    )
    havings = [strip_quotes(sql_text(having.this))] if having is not None else []
    aggs = unique_preserve_order(strip_quotes(sql_text(agg)) for expression in (ast.expressions or []) for agg in expression.find_all(exp.AggFunc))
    windows = unique_preserve_order(strip_quotes(sql_text(window)) for expression in (ast.expressions or []) for window in expression.find_all(exp.Window))
    orders = (
        unique_preserve_order(strip_quotes(sql_text(expression)) for expression in (order.expressions or []))
        if order is not None
        else []
    )
    limit_num = None
    if limit is not None:
        expression = limit.args.get("expression")
        if expression is not None:
            try:
                limit_num = int(expression.name)
            except Exception:
                limit_num = None

    if selected:
        lines.append("Main query SELECT: " + ", ".join(selected) + ".")
    if from_text:
        lines.append("Main query FROM: " + from_text + ".")
    if joins:
        lines.append("Main query JOIN: " + "; ".join(joins) + ".")
    if filters:
        lines.append("WHERE filters: " + " AND ".join(filters) + ".")
    if groups:
        lines.append("GROUP BY: " + ", ".join(groups) + ".")
    if havings:
        lines.append("HAVING: " + " AND ".join(havings) + ".")
    if aggs:
        lines.append("Aggregate functions: " + ", ".join(aggs) + ".")
    if windows:
        lines.append("Window functions: " + ", ".join(windows) + ".")
    if orders:
        lines.append("ORDER BY: " + ", ".join(orders) + ".")
    if limit_num is not None:
        lines.append(f"LIMIT {limit_num}.")

    if not lines:
        lines.append("SQL parse được nhưng không nhận diện được thành phần truy vấn chính.")
    return lines


def explain_sql_structure(sql: str) -> dict[str, Any]:
    try:
        ast = sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:
        return {
            "success": False,
            "summary_lines": [],
            "features": {},
            "error": str(exc),
        }

    try:
        cte_names = cte_aliases(ast)
        features = {
            "has_cte": bool(cte_names),
            "cte_names": cte_names,
            "real_tables": real_table_names(ast),
            "select_expressions": select_expressions(ast),
            "joins": join_expressions(ast),
            "where_filters": where_filters(ast),
            "group_by": group_by_expressions(ast),
            "having": having_expressions(ast),
            "order_by": order_by_expressions(ast),
            "limit": limit_value(ast),
            "aggregates": aggregate_expressions(ast),
            "window_functions": window_expressions(ast),
        }

        summary_lines = []
        summary_lines.extend(cte_summary_lines(ast))
        summary_lines.extend(main_query_summary_lines(ast))

        return {
            "success": True,
            "summary_lines": summary_lines,
            "features": features,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "summary_lines": [],
            "features": {},
            "error": str(exc),
        }
