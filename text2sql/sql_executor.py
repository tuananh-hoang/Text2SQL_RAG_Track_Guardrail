import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


BASE_DIR = Path(__file__).resolve().parent
MAX_RESULT_ROWS = 500


def get_readonly_url() -> str:
    load_dotenv(BASE_DIR / ".env")
    db_url = os.getenv("DB_READONLY_URL")
    if not db_url:
        raise RuntimeError("Missing DB_READONLY_URL in text2sql/.env")
    return db_url


def strip_sql_terminator(sql: str) -> str:
    return sql.strip().rstrip(";")


def limited_query(sql: str) -> str:
    return f"SELECT * FROM ({strip_sql_terminator(sql)}) AS text2sql_result LIMIT {MAX_RESULT_ROWS}"


def execute_sql(sql: str) -> dict[str, Any]:
    engine = create_engine(
        get_readonly_url(),
        connect_args={"options": "-c statement_timeout=30000"},
    )

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(limited_query(sql), conn)
        return {
            "success": True,
            "data": df,
            "row_count": len(df),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "data": None,
            "row_count": 0,
            "error": str(exc),
        }
    finally:
        engine.dispose()
