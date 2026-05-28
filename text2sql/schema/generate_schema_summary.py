import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCHEMA_DIR / "schema_summary.json"
TABLE_NAME = "product_sales"

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

COLUMN_ALIASES = {
    "revenue": ["doanh thu", "tiền"],
    "region": ["khu vực", "vùng", "miền"],
    "category": ["danh mục", "loại", "nhóm"],
    "product_name": ["sản phẩm", "tên sản phẩm"],
    "order_date": ["ngày", "ngày đặt"],
    "quantity": ["số lượng", "sl"],
    "unit_price": ["giá", "đơn giá"],
    "order_id": ["mã đơn hàng", "id đơn hàng"],
    "customer_name": ["khách hàng", "tên khách hàng"],
    "city": ["thành phố"],
    "state": ["bang", "tiểu bang"],
    "country": ["quốc gia", "nước"],
    "sub_category": ["danh mục con", "nhóm con"],
    "profit": ["lợi nhuận", "lãi"],
}


def get_admin_url() -> str:
    load_dotenv(BASE_DIR / ".env")
    db_url = os.getenv("DB_ADMIN_URL")
    if not db_url:
        raise RuntimeError("Missing DB_ADMIN_URL in text2sql/.env")
    return db_url


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


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


def get_distinct_count(conn, table_name: str, column_name: str) -> int:
    query = text(
        f"SELECT COUNT(DISTINCT {quote_identifier(column_name)}) "
        f"FROM {quote_identifier(table_name)}"
    )
    return int(conn.execute(query).scalar_one())


def get_distinct_values(conn, table_name: str, column_name: str, limit: int) -> list[Any]:
    query = text(
        f"SELECT DISTINCT {quote_identifier(column_name)} AS value "
        f"FROM {quote_identifier(table_name)} "
        f"WHERE {quote_identifier(column_name)} IS NOT NULL "
        f"ORDER BY {quote_identifier(column_name)} "
        f"LIMIT :limit"
    )
    return [serialize_value(row.value) for row in conn.execute(query, {"limit": limit})]


def build_column_summary(conn, table_name: str, column: dict[str, Any], row_count: int) -> dict[str, Any]:
    name = column["column_name"]
    data_type = normalize_type(column["data_type"])
    summary: dict[str, Any] = {
        "name": name,
        "type": data_type,
        "nullable": column["is_nullable"] == "YES",
        "is_id": False,
    }

    aliases = COLUMN_ALIASES.get(name.lower())
    if aliases:
        summary["column_aliases"] = aliases

    quoted_table = quote_identifier(table_name)
    quoted_column = quote_identifier(name)

    if is_numeric_type(data_type):
        distinct_count = get_distinct_count(conn, table_name, name)
        summary["distinct_count"] = distinct_count
        if distinct_count == row_count:
            summary["is_id"] = True
            return summary

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
        distinct_count = get_distinct_count(conn, table_name, name)
        summary["distinct_count"] = distinct_count
        values = get_distinct_values(conn, table_name, name, 20 if distinct_count <= 20 else 5)
        if distinct_count <= 20:
            summary["all_values"] = values
        else:
            summary["sample_values"] = values
        return summary

    distinct_count = get_distinct_count(conn, table_name, name)
    summary["distinct_count"] = distinct_count
    summary["sample_values"] = get_distinct_values(conn, table_name, name, 5)
    return summary


def main() -> int:
    engine = create_engine(get_admin_url())
    with engine.connect() as conn:
        row_count = int(
            conn.execute(text(f"SELECT COUNT(*) FROM {quote_identifier(TABLE_NAME)}")).scalar_one()
        )
        columns = [
            dict(row)
            for row in conn.execute(
                text(
                    """
                    SELECT column_name, data_type, is_nullable, ordinal_position
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = :table_name
                    ORDER BY ordinal_position
                    """
                ),
                {"table_name": TABLE_NAME},
            ).mappings()
        ]

        schema_summary = {
            "table_name": TABLE_NAME,
            "row_count": row_count,
            "column_count": len(columns),
            "columns": [
                build_column_summary(conn, TABLE_NAME, column, row_count)
                for column in columns
            ],
        }

    OUTPUT_PATH.write_text(
        json.dumps(schema_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Schema summary saved to schema/schema_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
