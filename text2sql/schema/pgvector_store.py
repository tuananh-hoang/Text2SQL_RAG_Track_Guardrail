import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

from text2sql.db.postgres_utils import get_readonly_url, quote_identifier


DEFAULT_PGVECTOR_TABLE = "schema_retrieval.schema_entity_embeddings"
MISSING_INDEX_MESSAGE = (
    "Schema pgvector index not built. Run python -m "
    "text2sql.schema.build_schema_pgvector_index --mode <mode> --rebuild"
)


def split_table_name(table_name: str) -> tuple[str, str]:
    parts = table_name.split(".")
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError("pgvector table name must be schema.table")
    return parts[0].strip(), parts[1].strip()


def qualified_table_name(table_name: str) -> str:
    schema_name, raw_table_name = split_table_name(table_name)
    return f"{quote_identifier(schema_name)}.{quote_identifier(raw_table_name)}"


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.10f}" for value in vector) + "]"


def ensure_pgvector_extension(engine: Engine) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "pgvector extension is not installed in PostgreSQL container. "
            "Please use a PostgreSQL image with pgvector support, e.g. pgvector/pgvector:pg15."
        ) from exc


def ensure_schema_retrieval_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS schema_retrieval"))


def grant_readonly_access(engine: Engine, table_name: str = DEFAULT_PGVECTOR_TABLE) -> None:
    readonly_user = make_url(get_readonly_url()).username
    if not readonly_user:
        return
    schema_name, _ = split_table_name(table_name)
    with engine.begin() as conn:
        conn.execute(text(f"GRANT USAGE ON SCHEMA {quote_identifier(schema_name)} TO {quote_identifier(readonly_user)}"))
        conn.execute(text(f"GRANT SELECT ON {qualified_table_name(table_name)} TO {quote_identifier(readonly_user)}"))


def create_embedding_table(engine: Engine, embedding_dim: int, table_name: str = DEFAULT_PGVECTOR_TABLE) -> None:
    ensure_pgvector_extension(engine)
    ensure_schema_retrieval_schema(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {qualified_table_name(table_name)} (
                    id BIGSERIAL PRIMARY KEY,
                    mode TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    schema_name TEXT,
                    table_name TEXT,
                    column_name TEXT,
                    target_type TEXT,
                    target_table TEXT,
                    target_column TEXT,
                    entity_text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({embedding_dim}) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(mode, entity_id, embedding_model)
                )
                """
            )
        )
    grant_readonly_access(engine, table_name=table_name)


def recreate_embedding_table(
    engine: Engine,
    embedding_dim: int,
    table_name: str = DEFAULT_PGVECTOR_TABLE,
) -> None:
    ensure_pgvector_extension(engine)
    ensure_schema_retrieval_schema(engine)
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {qualified_table_name(table_name)} CASCADE"))
    create_embedding_table(engine, embedding_dim=embedding_dim, table_name=table_name)


def create_vector_index(engine: Engine, table_name: str = DEFAULT_PGVECTOR_TABLE) -> tuple[str, list[str]]:
    warnings: list[str] = []
    qtable = qualified_table_name(table_name)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_schema_entity_embeddings_hnsw
                    ON {qtable}
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            )
        return "hnsw", warnings
    except SQLAlchemyError as exc:
        warnings.append(f"HNSW index failed, fallback to IVFFLAT: {exc}")

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_schema_entity_embeddings_ivfflat
                    ON {qtable}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
            )
        return "ivfflat", warnings
    except SQLAlchemyError as exc:
        warnings.append(f"IVFFLAT index failed; exact search will still work: {exc}")
        return "none", warnings


def upsert_schema_entity_embeddings(
    engine: Engine,
    rows: list[dict[str, Any]],
    table_name: str = DEFAULT_PGVECTOR_TABLE,
) -> None:
    if not rows:
        return
    stmt = text(
        f"""
        INSERT INTO {qualified_table_name(table_name)} (
            mode,
            entity_id,
            entity_type,
            schema_name,
            table_name,
            column_name,
            target_type,
            target_table,
            target_column,
            entity_text,
            metadata,
            embedding,
            embedding_model,
            embedding_dim
        )
        VALUES (
            :mode,
            :entity_id,
            :entity_type,
            :schema_name,
            :table_name,
            :column_name,
            :target_type,
            :target_table,
            :target_column,
            :entity_text,
            CAST(:metadata AS JSONB),
            CAST(:embedding AS vector),
            :embedding_model,
            :embedding_dim
        )
        ON CONFLICT (mode, entity_id, embedding_model)
        DO UPDATE SET
            entity_type = EXCLUDED.entity_type,
            schema_name = EXCLUDED.schema_name,
            table_name = EXCLUDED.table_name,
            column_name = EXCLUDED.column_name,
            target_type = EXCLUDED.target_type,
            target_table = EXCLUDED.target_table,
            target_column = EXCLUDED.target_column,
            entity_text = EXCLUDED.entity_text,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            embedding_dim = EXCLUDED.embedding_dim,
            created_at = NOW()
        """
    )
    payload = []
    for row in rows:
        payload.append(
            {
                **row,
                "metadata": json.dumps(row.get("metadata", {}), ensure_ascii=False),
                "embedding": vector_literal(row["embedding"]),
            }
        )
    with engine.begin() as conn:
        conn.execute(stmt, payload)


def search_schema_entities_pgvector(
    engine: Engine,
    query_embedding: list[float],
    mode: str,
    top_k: int = 20,
    table_name: str = DEFAULT_PGVECTOR_TABLE,
) -> list[dict[str, Any]]:
    stmt = text(
        f"""
        SELECT
            entity_id,
            entity_type,
            schema_name,
            table_name,
            column_name,
            target_type,
            target_table,
            target_column,
            entity_text,
            metadata,
            embedding_model,
            embedding_dim,
            embedding <=> CAST(:query_embedding AS vector) AS distance,
            1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
        FROM {qualified_table_name(table_name)}
        WHERE mode = :mode
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT :top_k
        """
    )
    try:
        with engine.connect() as conn:
            result = conn.execute(
                stmt,
                {
                    "query_embedding": vector_literal(query_embedding),
                    "mode": mode,
                    "top_k": top_k,
                },
            )
            rows = [dict(row._mapping) for row in result]
    except SQLAlchemyError as exc:
        raise RuntimeError(MISSING_INDEX_MESSAGE.replace("<mode>", mode)) from exc

    if not rows:
        raise RuntimeError(MISSING_INDEX_MESSAGE.replace("<mode>", mode))

    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            row["metadata"] = json.loads(metadata)
    return rows
