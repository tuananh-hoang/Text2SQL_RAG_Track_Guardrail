import argparse
import json
from pathlib import Path
from typing import Any

from text2sql.db.postgres_utils import create_admin_engine
from text2sql.schema.paths import V2_ENTITIES_DIR
from text2sql.schema.retrieval.pgvector_store import (
    DEFAULT_PGVECTOR_TABLE,
    create_embedding_table,
    create_vector_index,
    delete_schema_entity_embeddings,
    get_existing_embedding_dim,
    recreate_embedding_table,
    upsert_schema_entity_embeddings,
)


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config" / "schema_retrieval_config.json"
RESOLVED_CONFIG_PATH = BASE_DIR / "config" / "schema_retrieval_config.resolved.json"
_SENTENCE_TRANSFORMER_CACHE: dict[str, Any] = {}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class EmbeddingModel:
    def __init__(self, config: dict[str, Any]):
        if config.get("embedding_provider") != "sentence_transformers":
            raise RuntimeError("V2 pgvector retrieval requires embedding_provider=sentence_transformers")
        self.provider = "sentence_transformers"
        self.model_name = ""
        self.dim = 0
        self._model = self._load_model(config)
        self.dim = len(self.encode_one("test"))

    def _load_model(self, config: dict[str, Any]):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for V2 pgvector retrieval. "
                "Run: pip install -r requirements.txt"
            ) from exc

        candidates = [
            config.get("embedding_model_name"),
            config.get("fallback_embedding_model_name"),
        ]
        errors = []
        for model_name in [name for name in candidates if name]:
            try:
                if model_name in _SENTENCE_TRANSFORMER_CACHE:
                    self.model_name = model_name
                    return _SENTENCE_TRANSFORMER_CACHE[model_name]
                model = SentenceTransformer(model_name)
                _SENTENCE_TRANSFORMER_CACHE[model_name] = model
                self.model_name = model_name
                return model
            except Exception as exc:  # model download/load failures differ by backend
                errors.append(f"{model_name}: {exc}")
        raise RuntimeError("Could not load any sentence-transformers embedding model. " + " | ".join(errors))

    def encode_one(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in vector]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]


def load_entities(mode: str) -> list[dict[str, Any]]:
    path = V2_ENTITIES_DIR / f"schema_entities_{mode}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing schema entities: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def entity_to_row(
    mode: str,
    entity: dict[str, Any],
    embedding: list[float],
    embedding_model: EmbeddingModel,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "schema_name": entity.get("schema"),
        "table_name": entity.get("table") or entity.get("from_table") or entity.get("target_table"),
        "column_name": entity.get("column") or entity.get("from_column") or entity.get("target_column"),
        "target_type": entity["target_type"],
        "target_table": entity["target_table"],
        "target_column": entity["target_column"],
        "entity_text": entity["text"],
        "metadata": {"entity": entity},
        "embedding": embedding,
        "embedding_model": embedding_model.model_name,
        "embedding_dim": embedding_model.dim,
    }


def write_resolved_config(config: dict[str, Any], embedding_model: EmbeddingModel, table_name: str, mode: str) -> None:
    resolved = dict(config)
    resolved["embedding_provider"] = embedding_model.provider
    resolved["embedding_model_name_resolved"] = embedding_model.model_name
    resolved["embedding_dim"] = embedding_model.dim
    resolved["vector_store"] = "pgvector"
    resolved["pgvector_table"] = table_name
    resolved["last_indexed_mode"] = mode
    RESOLVED_CONFIG_PATH.write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")


def build_pgvector_index(mode: str, rebuild: bool = False) -> dict[str, Any]:
    config = load_config()
    table_name = config.get("pgvector_table", DEFAULT_PGVECTOR_TABLE)
    entities = load_entities(mode)
    embedding_model = EmbeddingModel(config)
    vectors = embedding_model.encode_many([entity["text"] for entity in entities])
    rows = [
        entity_to_row(mode, entity, vector, embedding_model)
        for entity, vector in zip(entities, vectors)
    ]

    engine = create_admin_engine()
    try:
        if rebuild:
            existing_dim = get_existing_embedding_dim(engine, table_name=table_name)
            if existing_dim is not None and existing_dim != embedding_model.dim:
                recreate_embedding_table(engine, embedding_dim=embedding_model.dim, table_name=table_name)
            else:
                create_embedding_table(engine, embedding_dim=embedding_model.dim, table_name=table_name)
                delete_schema_entity_embeddings(
                    engine,
                    mode=mode,
                    embedding_model=embedding_model.model_name,
                    table_name=table_name,
                )
        else:
            create_embedding_table(engine, embedding_dim=embedding_model.dim, table_name=table_name)
        upsert_schema_entity_embeddings(engine, rows, table_name=table_name)
        index_type, warnings = create_vector_index(engine, table_name=table_name)
    finally:
        engine.dispose()

    write_resolved_config(config, embedding_model, table_name, mode)
    summary = {
        "mode": mode,
        "embedding_model": embedding_model.model_name,
        "embedding_provider": embedding_model.provider,
        "embedding_dim": embedding_model.dim,
        "entities_loaded": len(entities),
        "embeddings_upserted": len(rows),
        "pgvector_table": table_name,
        "index_created": index_type,
        "warnings": warnings,
    }
    print("=== Schema pgvector index summary ===")
    for key, value in summary.items():
        print(f"{key.ljust(24)}: {value}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build schema entity pgvector index.")
    parser.add_argument("--mode", choices=["product_sales", "mock_relational"], default="product_sales")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate pgvector table.")
    args = parser.parse_args()
    build_pgvector_index(mode=args.mode, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
