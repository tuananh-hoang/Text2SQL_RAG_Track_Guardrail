from text2sql.schema.build_schema_pgvector_index import build_pgvector_index, main


def build_vector_index(mode: str) -> dict:
    return build_pgvector_index(mode=mode, rebuild=False)


if __name__ == "__main__":
    raise SystemExit(main())
