# Text2SQL RAG Track Guardrail

Baseline Text-to-SQL pipeline for the Product Sales dataset using PostgreSQL,
schema evidence, LLM SQL generation, AST-based SQL validation, readonly
execution, repair, and evaluation.

## Main Flow

1. Import CSV data into PostgreSQL.
2. Generate schema summary and schema context.
3. Generate SQL from Vietnamese questions.
4. Validate SQL with allowlisted tables/columns.
5. Execute SQL with a readonly database user.
6. Evaluate generated SQL against gold SQL test cases.

## Quick Start

```powershell
cd text2sql
docker compose up -d
python import_csv.py
python schema/generate_schema_summary.py
python run_v1.py --demo
python evaluation/evaluator.py
```

Create `text2sql/.env` locally before running. Do not commit secrets.

## V2 Schema Retrieval With pgvector

V2 stores schema entity embeddings in PostgreSQL using pgvector. The Docker
compose file uses `pgvector/pgvector:pg15`; if your local database was created
from a plain PostgreSQL image and extension creation fails, recreate the
container with a pgvector-enabled image.

```powershell
docker compose -f text2sql/docker-compose.yml up -d postgres postgres_init

python -m text2sql.db.build_relational_mock --rebuild

python -m text2sql.schema.generate_schema_summary --mode product_sales
python -m text2sql.schema.generate_schema_summary --mode mock_relational

python -m text2sql.schema.build_schema_entities --mode product_sales
python -m text2sql.schema.build_schema_entities --mode mock_relational

python -m text2sql.schema.build_schema_pgvector_index --mode product_sales --rebuild
python -m text2sql.schema.build_schema_pgvector_index --mode mock_relational --rebuild

python -m text2sql.app.run_v2 --mode product_sales --question "Top 5 sản phẩm có doanh thu cao nhất"
python -m text2sql.app.run_v2 --mode mock_relational --question "Top 5 sản phẩm có doanh thu cao nhất"
```

Useful checks:

```sql
SELECT extname FROM pg_extension WHERE extname = 'vector';
SELECT COUNT(*) FROM schema_retrieval.schema_entity_embeddings;
```
