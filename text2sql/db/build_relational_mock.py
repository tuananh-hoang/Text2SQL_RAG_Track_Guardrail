import argparse
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import make_url

from text2sql.db.postgres_utils import create_admin_engine, get_readonly_url, quote_identifier


MOCK_SCHEMA = "mock_relational"
SOURCE_TABLE = "product_sales"


def qname(schema_name: str, table_name: str) -> str:
    return f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"


def ensure_product_sales_exists(conn) -> None:
    exists = conn.execute(
        text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = :table_name
            )
            """
        ),
        {"table_name": SOURCE_TABLE},
    ).scalar_one()
    if not exists:
        raise RuntimeError("Table public.product_sales does not exist. Run import_csv.py first.")


def create_schema_and_tables(conn, rebuild: bool) -> None:
    if rebuild:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {quote_identifier(MOCK_SCHEMA)} CASCADE"))
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(MOCK_SCHEMA)}"))

    for table_name in ["order_items", "orders", "locations", "products", "customers"]:
        conn.execute(text(f"DROP TABLE IF EXISTS {qname(MOCK_SCHEMA, table_name)} CASCADE"))

    conn.execute(
        text(
            f"""
            CREATE TABLE {qname(MOCK_SCHEMA, "customers")} (
                "Customer_ID" INTEGER PRIMARY KEY,
                "Customer_Name" TEXT NOT NULL UNIQUE
            )
            """
        )
    )
    conn.execute(
        text(
            f"""
            CREATE TABLE {qname(MOCK_SCHEMA, "products")} (
                "Product_ID" INTEGER PRIMARY KEY,
                "Product_Name" TEXT NOT NULL,
                "Category" TEXT,
                "Sub_Category" TEXT,
                UNIQUE ("Product_Name", "Category", "Sub_Category")
            )
            """
        )
    )
    conn.execute(
        text(
            f"""
            CREATE TABLE {qname(MOCK_SCHEMA, "locations")} (
                "Location_ID" INTEGER PRIMARY KEY,
                "City" TEXT,
                "State" TEXT,
                "Region" TEXT,
                "Country" TEXT,
                UNIQUE ("City", "State", "Region", "Country")
            )
            """
        )
    )
    conn.execute(
        text(
            f"""
            CREATE TABLE {qname(MOCK_SCHEMA, "orders")} (
                "Order_ID" INTEGER PRIMARY KEY,
                "Order_Date" DATE,
                "Customer_ID" INTEGER NOT NULL REFERENCES {qname(MOCK_SCHEMA, "customers")}("Customer_ID"),
                "Location_ID" INTEGER NOT NULL REFERENCES {qname(MOCK_SCHEMA, "locations")}("Location_ID")
            )
            """
        )
    )
    conn.execute(
        text(
            f"""
            CREATE TABLE {qname(MOCK_SCHEMA, "order_items")} (
                "Order_Item_ID" INTEGER PRIMARY KEY,
                "Order_ID" INTEGER NOT NULL REFERENCES {qname(MOCK_SCHEMA, "orders")}("Order_ID"),
                "Product_ID" INTEGER NOT NULL REFERENCES {qname(MOCK_SCHEMA, "products")}("Product_ID"),
                "Quantity" INTEGER,
                "Unit_Price" NUMERIC,
                "Revenue" NUMERIC,
                "Profit" NUMERIC
            )
            """
        )
    )


def with_surrogate_id(df: pd.DataFrame, id_column: str, sort_columns: list[str]) -> pd.DataFrame:
    out = df.drop_duplicates().sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    out.insert(0, id_column, range(1, len(out) + 1))
    return out


def load_product_sales(engine) -> pd.DataFrame:
    df = pd.read_sql_query(f"SELECT * FROM {quote_identifier(SOURCE_TABLE)}", engine)
    if df.empty:
        raise RuntimeError("public.product_sales is empty; cannot build mock relational schema.")
    return df


def build_tables(source: pd.DataFrame) -> dict[str, pd.DataFrame]:
    customers = with_surrogate_id(
        source[["Customer_Name"]],
        "Customer_ID",
        ["Customer_Name"],
    )
    products = with_surrogate_id(
        source[["Product_Name", "Category", "Sub_Category"]],
        "Product_ID",
        ["Product_Name", "Category", "Sub_Category"],
    )
    locations = with_surrogate_id(
        source[["City", "State", "Region", "Country"]],
        "Location_ID",
        ["Country", "Region", "State", "City"],
    )

    enriched = source.merge(customers, on="Customer_Name", how="left")
    enriched = enriched.merge(products, on=["Product_Name", "Category", "Sub_Category"], how="left")
    enriched = enriched.merge(locations, on=["City", "State", "Region", "Country"], how="left")

    orders = (
        enriched[["Order_ID", "Order_Date", "Customer_ID", "Location_ID"]]
        .sort_values(["Order_ID", "Order_Date", "Customer_ID", "Location_ID"], kind="mergesort")
        .drop_duplicates(subset=["Order_ID"], keep="first")
        .reset_index(drop=True)
    )
    orders["Order_ID"] = orders["Order_ID"].astype(int)

    order_items = enriched[
        ["Order_ID", "Product_ID", "Quantity", "Unit_Price", "Revenue", "Profit"]
    ].copy()
    order_items.insert(0, "Order_Item_ID", range(1, len(order_items) + 1))
    order_items["Order_ID"] = order_items["Order_ID"].astype(int)

    return {
        "customers": customers,
        "products": products,
        "locations": locations,
        "orders": orders,
        "order_items": order_items,
    }


def write_tables(engine, tables: dict[str, pd.DataFrame]) -> None:
    for table_name in ["customers", "products", "locations", "orders", "order_items"]:
        tables[table_name].to_sql(
            table_name,
            engine,
            schema=MOCK_SCHEMA,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )


def grant_readonly_access(conn) -> None:
    readonly_user = make_url(get_readonly_url()).username
    if not readonly_user:
        return
    role = quote_identifier(readonly_user)
    conn.execute(text(f"GRANT USAGE ON SCHEMA {quote_identifier(MOCK_SCHEMA)} TO {role}"))
    conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA {quote_identifier(MOCK_SCHEMA)} TO {role}"))


def db_scalar(conn, sql: str) -> Any:
    return conn.execute(text(sql)).scalar_one()


def print_summary(conn) -> None:
    product_count = db_scalar(conn, f"SELECT COUNT(*) FROM {quote_identifier(SOURCE_TABLE)}")
    item_count = db_scalar(conn, f"SELECT COUNT(*) FROM {qname(MOCK_SCHEMA, 'order_items')}")
    source_revenue = db_scalar(conn, f"SELECT SUM({quote_identifier('Revenue')}) FROM {quote_identifier(SOURCE_TABLE)}")
    mock_revenue = db_scalar(conn, f"SELECT SUM({quote_identifier('Revenue')}) FROM {qname(MOCK_SCHEMA, 'order_items')}")
    source_profit = db_scalar(conn, f"SELECT SUM({quote_identifier('Profit')}) FROM {quote_identifier(SOURCE_TABLE)}")
    mock_profit = db_scalar(conn, f"SELECT SUM({quote_identifier('Profit')}) FROM {qname(MOCK_SCHEMA, 'order_items')}")

    print("=== Mock relational schema summary ===")
    print(f"product_sales row count              : {product_count}")
    print(f"order_items row count                : {item_count}")
    for table_name in ["customers", "products", "locations", "orders"]:
        count = db_scalar(conn, f"SELECT COUNT(*) FROM {qname(MOCK_SCHEMA, table_name)}")
        print(f"{table_name} count".ljust(36) + f": {count}")
    print(f"SUM Revenue product_sales            : {float(source_revenue):.2f}")
    print(f"SUM Revenue mock_relational          : {float(mock_revenue):.2f}")
    print(f"SUM Profit product_sales             : {float(source_profit):.2f}")
    print(f"SUM Profit mock_relational           : {float(mock_profit):.2f}")


def build_mock_relational(rebuild: bool) -> None:
    engine = create_admin_engine()
    with engine.begin() as conn:
        ensure_product_sales_exists(conn)
        create_schema_and_tables(conn, rebuild=rebuild)

    source = load_product_sales(engine)
    tables = build_tables(source)
    write_tables(engine, tables)

    with engine.begin() as conn:
        grant_readonly_access(conn)
        print_summary(conn)
    engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build mock_relational schema from public.product_sales.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and rebuild mock_relational schema.")
    args = parser.parse_args()
    build_mock_relational(rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
