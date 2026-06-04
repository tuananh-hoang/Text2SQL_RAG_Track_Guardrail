import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine


BASE_DIR = Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    load_dotenv(BASE_DIR / ".env", override=True)


def get_db_url(env_name: str) -> str:
    load_project_env()
    db_url = os.getenv(env_name)
    if not db_url:
        raise RuntimeError(f"Missing {env_name} in text2sql/.env")
    return db_url


def get_admin_url() -> str:
    return get_db_url("DB_ADMIN_URL")


def get_readonly_url() -> str:
    return get_db_url("DB_READONLY_URL")


def create_admin_engine(**kwargs: Any):
    return create_engine(get_admin_url(), **kwargs)


def create_readonly_engine(**kwargs: Any):
    return create_engine(get_readonly_url(), **kwargs)


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def qualified_name(schema_name: str | None, table_name: str) -> str:
    if schema_name:
        return f"{quote_identifier(schema_name)}.{quote_identifier(table_name)}"
    return quote_identifier(table_name)
