import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from text2sql.db.postgres_utils import create_admin_engine, qualified_name, quote_identifier
from text2sql.schema.paths import V1_SCHEMA_SUMMARY_PATH, V2_SUMMARIES_DIR

NUMERIC_TYPES = {
    "integer",
    "bigint",
    "smallint",
    "numeric",
    "real",
    "double precision",
    "float",
}
TEXT_TYPES = {"text", "character varying", "varchar", "character"}
DATE_TYPES = {"date", "timestamp", "timestamp without time zone", "timestamp with time zone", "timestamptz"}

TABLE_ALIASES = {
    "product_sales": ["bán hàng sản phẩm", "doanh số sản phẩm"],
    "orders": ["đơn hàng", "order"],
    "order_items": ["chi tiết đơn hàng", "dòng đơn hàng", "mặt hàng trong đơn"],
    "products": ["sản phẩm", "hàng hóa"],
    "customers": ["khách hàng"],
    "locations": ["địa điểm", "khu vực", "vị trí"],
}

TABLE_DESCRIPTIONS = {
    "product_sales": "Single denormalized sales table at order-line grain.",
    "orders": "Orders table with order date, customer and location references.",
    "order_items": "Fact table at order item grain with quantity, price, revenue and profit.",
    "products": "Products table with product name, category and sub category.",
    "customers": "Customers table with customer names.",
    "locations": "Locations table with city, state, region and country.",
}

COLUMN_ALIASES = {
    "revenue": ["doanh thu", "tiền"],
    "profit": ["lợi nhuận", "lãi"],
    "product_name": ["sản phẩm", "tên sản phẩm"],
    "category": ["danh mục", "loại", "nhóm"],
    "sub_category": ["danh mục con", "nhóm con"],
    "region": ["khu vực", "vùng", "miền"],
    "state": ["bang", "tiểu bang"],
    "city": ["thành phố"],
    "country": ["quốc gia", "nước"],
    "customer_name": ["khách hàng", "tên khách hàng"],
    "order_date": ["ngày", "ngày đặt", "tháng", "năm"],
    "quantity": ["số lượng", "sl"],
    "unit_price": ["giá", "đơn giá"],
    "order_id": ["mã đơn hàng", "id đơn hàng"],
    "customer_id": ["mã khách hàng", "id khách hàng"],
    "product_id": ["mã sản phẩm", "id sản phẩm"],
    "location_id": ["mã địa điểm", "id địa điểm"],
    "order_item_id": ["mã dòng đơn hàng", "id dòng đơn hàng"],
}


def normalize_type(data_type: str) -> str:
    return data_type.lower()


def is_numeric_type(data_type: str) -> bool:
    return normalize_type(data_type) in NUMERIC_TYPES


def is_date_type(data_type: str) -> bool:
    normalized = normalize_type(data_type)
    return normalized in DATE_TYPES or normalized.startswith("timestamp")


def is_text_type(data_type: str) -> bool:
    return normalize_type(data_type) in TEXT_TYPES


def round_number(value: Any) -> float | int | None:
    if value is None:
        return None
    rounded = round(float(value), 2)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def serialize_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def describe_column(column_name: str, data_type: str) -> str:
    normalized = column_name.lower()
    if normalized in {"revenue", "profit", "quantity", "unit_price"}:
        return f"{column_name} is a numeric business measure."
    if normalized.endswith("_id") or normalized == "order_id":
        return f"{column_name} is an identifier column."
    if normalized == "order_date":
        return "Order_Date is the date of the order."
    if normalized in {"region", "state", "city", "country"}:
        return f"{column_name} is a geographic dimension."
    if normalized in {"category", "sub_category", "product_name"}:
        return f"{column_name} describes product attributes."
    if normalized == "customer_name":
        return "Customer_Name identifies the customer name."
    return f"{column_name} is a {data_type} column."


def get_row_count(conn, schema_name: str, table_name: str) -> int:
    return int(
        conn.execute(
            text(f"SELECT COUNT(*) FROM {qualified_name(schema_name, table_name)}")
        ).scalar_one()
    )


def get_distinct_count(conn, schema_name: str, table_name: str, column_name: str) -> int:
    query = text(
        f"SELECT COUNT(DISTINCT {quote_identifier(column_name)}) "
        f"FROM {qualified_name(schema_name, table_name)}"
    )
    return int(conn.execute(query).scalar_one())


def get_distinct_values(
    conn,
    schema_name: str,
    table_name: str,
    column_name: str,
    limit: int,
) -> list[Any]:
    query = text(
        f"SELECT DISTINCT {quote_identifier(column_name)} AS value "
        f"FROM {qualified_name(schema_name, table_name)} "
        f"WHERE {quote_identifier(column_name)} IS NOT NULL "
        f"ORDER BY {quote_identifier(column_name)} "
        f"LIMIT :limit"
    )
    return [serialize_value(row.value) for row in conn.execute(query, {"limit": limit})]


def get_columns(conn, schema_name: str, table_name: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            text(
                """
                SELECT column_name, data_type, is_nullable, ordinal_position
                FROM information_schema.columns
                WHERE table_schema = :schema_name
                  AND table_name = :table_name
                ORDER BY ordinal_position
                """
            ),
            {"schema_name": schema_name, "table_name": table_name},
        ).mappings()
    ]


def build_column_summary(
    conn,
    schema_name: str,
    table_name: str,
    column: dict[str, Any],
    row_count: int,
) -> dict[str, Any]:
    name = column["column_name"]
    data_type = normalize_type(column["data_type"])
    summary: dict[str, Any] = {
        "name": name,
        "type": data_type,
        "nullable": column["is_nullable"] == "YES",
        "is_id": name.lower().endswith("_id") or name.lower() == "order_id",
        "column_description": describe_column(name, data_type),
    }

    aliases = COLUMN_ALIASES.get(name.lower())
    if aliases:
        summary["column_aliases"] = aliases

    quoted_column = quote_identifier(name)
    quoted_table = qualified_name(schema_name, table_name)

    if is_numeric_type(data_type):
        distinct_count = get_distinct_count(conn, schema_name, table_name, name)
        summary["distinct_count"] = distinct_count
        if not summary["is_id"]:
            stats = conn.execute(
                text(
                    f"SELECT MIN({quoted_column}) AS min_value, "
                    f"MAX({quoted_column}) AS max_value, "
                    f"AVG({quoted_column}) AS avg_value "
                    f"FROM {quoted_table}"
                )
            ).mappings().one()
            summary["min"] = round_number(stats["min_value"])
            summary["max"] = round_number(stats["max_value"])
            summary["avg"] = round_number(stats["avg_value"])
        return summary

    if is_date_type(data_type):
        stats = conn.execute(
            text(
                f"SELECT MIN({quoted_column}) AS min_value, "
                f"MAX({quoted_column}) AS max_value "
                f"FROM {quoted_table}"
            )
        ).mappings().one()
        summary["min"] = serialize_value(stats["min_value"])
        summary["max"] = serialize_value(stats["max_value"])
        summary["date_format"] = "YYYY-MM-DD"
        return summary

    if is_text_type(data_type):
        distinct_count = get_distinct_count(conn, schema_name, table_name, name)
        summary["distinct_count"] = distinct_count
        values = get_distinct_values(
            conn,
            schema_name,
            table_name,
            name,
            20 if distinct_count <= 20 else 5,
        )
        if distinct_count <= 20:
            summary["all_values"] = values
        else:
            summary["sample_values"] = values
        return summary

    distinct_count = get_distinct_count(conn, schema_name, table_name, name)
    summary["distinct_count"] = distinct_count
    summary["sample_values"] = get_distinct_values(conn, schema_name, table_name, name, 5)
    return summary


def build_table_summary(conn, schema_name: str, table_name: str) -> dict[str, Any]:
    row_count = get_row_count(conn, schema_name, table_name)
    columns = get_columns(conn, schema_name, table_name)
    return {
        "schema_name": schema_name,
        "table_name": table_name,
        "row_count": row_count,
        "column_count": len(columns),
        "table_aliases": TABLE_ALIASES.get(table_name, []),
        "table_description": TABLE_DESCRIPTIONS.get(table_name, f"Table {table_name}."),
        "columns": [
            build_column_summary(conn, schema_name, table_name, column, row_count)
            for column in columns
        ],
    }


def get_relationships(conn, schema_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        text(
            """
            SELECT
                kcu.table_schema AS from_schema,
                kcu.table_name AS from_table,
                kcu.column_name AS from_column,
                ccu.table_schema AS to_schema,
                ccu.table_name AS to_table,
                ccu.column_name AS to_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND kcu.table_schema = :schema_name
            ORDER BY kcu.table_name, kcu.column_name
            """
        ),
        {"schema_name": schema_name},
    ).mappings()
    return [dict(row) | {"source": "foreign_key"} for row in rows]


def list_schema_tables(conn, schema_name: str) -> list[str]:
    return [
        row.table_name
        for row in conn.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema_name
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ),
            {"schema_name": schema_name},
        )
    ]


def build_product_sales_summary(conn) -> dict[str, Any]:
    table_summary = build_table_summary(conn, "public", "product_sales")
    summary = {
        "database_name": "text2sql_db",
        "schema_name": "public",
        "table_name": "product_sales",
        "row_count": table_summary["row_count"],
        "column_count": table_summary["column_count"],
        "table_aliases": table_summary["table_aliases"],
        "table_description": table_summary["table_description"],
        "columns": table_summary["columns"],
        "relationships": [],
    }
    return summary


def build_mock_relational_summary(conn) -> dict[str, Any]:
    schema_name = "mock_relational"
    tables = list_schema_tables(conn, schema_name)
    if not tables:
        raise RuntimeError("Schema mock_relational has no tables. Run python -m text2sql.db.build_relational_mock --rebuild")
    return {
        "database_name": "text2sql_db",
        "schema_name": schema_name,
        "tables": [build_table_summary(conn, schema_name, table_name) for table_name in tables],
        "relationships": get_relationships(conn, schema_name),
    }


def output_path_for_mode(mode: str) -> Path:
    return V2_SUMMARIES_DIR / f"schema_summary_{mode}.json"


def write_summary(mode: str, summary: dict[str, Any]) -> Path:
    V2_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for_mode(mode)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if mode == "product_sales":
        V1_SCHEMA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        V1_SCHEMA_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def generate_summary(mode: str) -> Path:
    engine = create_admin_engine()
    with engine.connect() as conn:
        if mode == "product_sales":
            summary = build_product_sales_summary(conn)
        elif mode == "mock_relational":
            summary = build_mock_relational_summary(conn)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    engine.dispose()
    output_path = write_summary(mode, summary)
    print(f"Schema summary saved to {output_path}")
    if mode == "product_sales":
        print(f"V1 schema summary saved to {V1_SCHEMA_SUMMARY_PATH}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PostgreSQL schema summary.")
    parser.add_argument(
        "--mode",
        choices=["product_sales", "mock_relational"],
        default="product_sales",
        help="Schema summary mode.",
    )
    args = parser.parse_args()
    generate_summary(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
