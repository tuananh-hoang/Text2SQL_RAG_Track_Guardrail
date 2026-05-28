from pathlib import Path
import os
import sqlite3

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
TABLE_NAME = "product_sales"


def resolve_db_path() -> Path:
    load_dotenv(BASE_DIR / ".env")
    db_path = Path(os.getenv("DB_PATH", "product_sales.db"))
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path
    return db_path


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def format_sample(value: object) -> str:
    if value is None:
        return "NULL"
    return str(value)


def main() -> int:
    db_path = resolve_db_path()
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run: python import_csv.py")
        return 1

    conn = sqlite3.connect(db_path)
    try:
        columns = conn.execute(f"PRAGMA table_info({quote_identifier(TABLE_NAME)})").fetchall()
        if not columns:
            print(f"Table not found: {TABLE_NAME}")
            print("Run: python import_csv.py")
            return 1

        row_count = conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(TABLE_NAME)}").fetchone()[0]

        print(f"Database: {db_path}")
        print(f"Table: {TABLE_NAME}")
        print(f"Rows: {row_count}")
        print()
        print("Columns:")

        for _, name, data_type, _, _, _ in columns:
            samples = conn.execute(
                (
                    f"SELECT DISTINCT {quote_identifier(name)} "
                    f"FROM {quote_identifier(TABLE_NAME)} "
                    f"WHERE {quote_identifier(name)} IS NOT NULL "
                    f"LIMIT 3"
                )
            ).fetchall()
            sample_text = ", ".join(format_sample(row[0]) for row in samples) or "(no non-null samples)"
            print(f"- {name} ({data_type}): {sample_text}")

        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
