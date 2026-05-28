import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy import types as sqltypes


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "product_sales.csv"
TABLE_NAME = "product_sales"


def get_admin_url() -> str:
    load_dotenv(BASE_DIR / ".env")
    db_url = os.getenv("DB_ADMIN_URL")
    if not db_url:
        raise RuntimeError("Missing DB_ADMIN_URL in text2sql/.env")
    return db_url


def fail_missing_csv() -> None:
    print(f"CSV not found: {CSV_PATH}")
    print()
    print("Manual download fallback:")
    print("1. Open https://www.kaggle.com/datasets/yashyennewar/product-sales-dataset-2023-2024")
    print("2. Download the dataset zip from Kaggle.")
    print("3. Extract the CSV file.")
    print(f"4. Rename/place it as: {CSV_PATH}")


def is_date_like_column(column_name: str) -> bool:
    lower_name = column_name.lower()
    return "date" in lower_name or "time" in lower_name


def read_csv_with_inferred_types() -> tuple[pd.DataFrame, dict[str, object]]:
    raw_columns = list(pd.read_csv(CSV_PATH, nrows=0).columns)
    stripped_columns = [column.strip() for column in raw_columns]
    date_columns = [
        raw_column
        for raw_column, clean_column in zip(raw_columns, stripped_columns, strict=True)
        if is_date_like_column(clean_column)
    ]

    read_kwargs: dict[str, object] = {}
    if date_columns:
        read_kwargs["parse_dates"] = date_columns
        read_kwargs["date_format"] = "%m-%d-%y"

    df = pd.read_csv(CSV_PATH, **read_kwargs)
    df.columns = stripped_columns

    dtype_map: dict[str, object] = {}
    for column in df.columns:
        if is_date_like_column(column):
            converted = pd.to_datetime(df[column], errors="coerce")
            if "date" in column.lower():
                df[column] = converted.dt.date
                dtype_map[column] = sqltypes.Date()
            else:
                df[column] = converted
                dtype_map[column] = sqltypes.DateTime()
            continue

        as_text = df[column].astype("string").str.strip()
        non_empty = as_text.notna() & (as_text != "")
        numeric_values = pd.to_numeric(as_text, errors="coerce")
        numeric_failures = non_empty & numeric_values.isna()

        if not numeric_failures.any():
            non_null_numeric = numeric_values.dropna()
            is_integer = bool(
                not non_null_numeric.empty
                and ((non_null_numeric % 1) == 0).all()
            )
            if is_integer:
                df[column] = numeric_values.astype("Int64")
                dtype_map[column] = sqltypes.Integer()
            else:
                df[column] = numeric_values.astype("float64")
                dtype_map[column] = sqltypes.Float(precision=53)
        else:
            df[column] = df[column].astype("string")
            dtype_map[column] = sqltypes.Text()

    return df, dtype_map


def main() -> int:
    if not CSV_PATH.exists():
        fail_missing_csv()
        return 1

    df, dtype_map = read_csv_with_inferred_types()

    engine = create_engine(get_admin_url())
    df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
        chunksize=1000,
        dtype=dtype_map,
    )

    with engine.begin() as conn:
        conn.execute(text(f"GRANT SELECT ON TABLE {TABLE_NAME} TO readonly_user"))
        row_count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()

    print(f"Imported {row_count} rows into {TABLE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
