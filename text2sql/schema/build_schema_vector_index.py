import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from text2sql.schema.paths import V2_ENTITIES_DIR


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "schema_retrieval_config.json"
RESOLVED_CONFIG_PATH = BASE_DIR / "config" / "schema_retrieval_config.resolved.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


class EmbeddingModel:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider = "local_hash"
        self.model_name = "local_hash_384"
        self.dim = 384
        self._model = None
        self._load_sentence_transformer_if_available()

    def _load_sentence_transformer_if_available(self) -> None:
        requested_provider = self.config.get("embedding_provider")
        if requested_provider != "sentence_transformers":
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model_name = self.config.get("embedding_model_name")
            try:
                self._model = SentenceTransformer(model_name)
                self.provider = "sentence_transformers"
                self.model_name = model_name
            except Exception:
                fallback_name = self.config.get("fallback_embedding_model_name")
                self._model = SentenceTransformer(fallback_name)
                self.provider = "sentence_transformers"
                self.model_name = fallback_name
            self.dim = len(self.encode_one("dimension probe"))
        except Exception:
            self.provider = "local_hash"
            self.model_name = "local_hash_384"
            self.dim = 384

    def encode_one(self, text: str) -> list[float]:
        if self._model is not None:
            vector = self._model.encode([text], normalize_embeddings=True)[0]
            return [float(value) for value in vector]
        return local_hash_embedding(text, dim=self.dim)

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [[float(value) for value in vector] for vector in vectors]
        return [self.encode_one(text) for text in texts]


def local_hash_embedding(text: str, dim: int = 384) -> list[float]:
    vector = [0.0] * dim
    tokens = tokenize(text)
    grams = tokens[:]
    grams.extend(" ".join(tokens[index : index + 2]) for index in range(max(0, len(tokens) - 1)))
    for gram in grams:
        digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def load_entities(mode: str) -> list[dict[str, Any]]:
    path = V2_ENTITIES_DIR / f"schema_entities_{mode}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing schema entities: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def persist_dir(config: dict[str, Any]) -> Path:
    raw = Path(config["persist_dir"])
    if raw.is_absolute():
        return raw
    if raw.parts and raw.parts[0] == "text2sql":
        return BASE_DIR.parent / raw
    return BASE_DIR / raw


def index_path_for_mode(config: dict[str, Any], mode: str) -> Path:
    collection = config["collections"][mode]
    return persist_dir(config) / f"{collection}.json"


def build_vector_index(mode: str) -> dict[str, Any]:
    config = load_config()
    entities = load_entities(mode)
    embedding_model = EmbeddingModel(config)
    documents = [entity["text"] for entity in entities]
    vectors = embedding_model.encode_many(documents)

    records = []
    for entity, vector in zip(entities, vectors):
        records.append(
            {
                "id": entity["entity_id"],
                "document": entity["text"],
                "embedding": vector,
                "metadata": {
                    "entity_type": entity["entity_type"],
                    "schema": entity.get("schema"),
                    "table": entity.get("table") or entity.get("from_table") or entity.get("target_table"),
                    "column": entity.get("column") or entity.get("from_column") or entity.get("target_column"),
                    "target_type": entity["target_type"],
                    "target_table": entity["target_table"],
                    "target_column": entity["target_column"],
                },
                "entity": entity,
            }
        )

    output_path = index_path_for_mode(config, mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    resolved = dict(config)
    resolved["embedding_provider_resolved"] = embedding_model.provider
    resolved["embedding_model_name_resolved"] = embedding_model.model_name
    resolved["embedding_dim"] = embedding_model.dim
    resolved["vector_db_resolved"] = "simple_json"
    resolved["last_indexed_mode"] = mode
    RESOLVED_CONFIG_PATH.write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "mode": mode,
        "embedding_model": embedding_model.model_name,
        "embedding_provider": embedding_model.provider,
        "embedding_dim": embedding_model.dim,
        "entities_indexed": len(records),
        "vector_store_path": str(output_path),
    }
    print("=== Schema vector index summary ===")
    for key, value in summary.items():
        print(f"{key.ljust(24)}: {value}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local schema entity vector index.")
    parser.add_argument("--mode", choices=["product_sales", "mock_relational"], default="product_sales")
    args = parser.parse_args()
    build_vector_index(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
