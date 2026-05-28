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
