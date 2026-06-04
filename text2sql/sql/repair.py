from pathlib import Path
from typing import Any

from llm_client import create_chat_completion_with_rotation, get_llm_model
from versions.v1_baseline import parse_llm_json, validate_llm_payload


BASE_DIR = Path(__file__).resolve().parents[1]
MAX_REPAIR_ATTEMPTS = 2

REPAIR_PROMPT_TEMPLATE = """
SQL sau đây bị lỗi khi chạy trên PostgreSQL.

Schema:
{schema_context}

Câu hỏi gốc: {question}
SQL lỗi: {sql}
Error: {error}

Sửa SQL. Vẫn phải tuân thủ toàn bộ quy tắc:
- Chỉ SELECT
- Chỉ dùng bảng/cột trong schema
- Cú pháp PostgreSQL
- Giữ double quotes quanh tên cột đúng như schema

Trả về JSON không có markdown:
{{
  "sql": "...",
  "tables_used": [...],
  "columns_used": [...],
  "explanation": "...",
  "confidence": "high|medium|low"
}}
""".strip()


def repair_sql(
    question: str,
    failed_sql: str,
    error_msg: str,
    schema_context: str,
) -> dict[str, Any]:
    prompt = REPAIR_PROMPT_TEMPLATE.format(
        schema_context=schema_context,
        question=question,
        sql=failed_sql,
        error=error_msg,
    )

    last_error = None
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        try:
            response = create_chat_completion_with_rotation(
                model=get_llm_model(),
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Sửa SQL lỗi ở trên và chỉ trả JSON hợp lệ."},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty content")

            repaired = validate_llm_payload(parse_llm_json(content))
            repaired["success"] = True
            repaired["attempts"] = attempt
            return repaired
        except Exception as exc:
            last_error = str(exc)

    return {
        "success": False,
        "error": "Repair failed after 2 attempts",
        "last_error": last_error,
        "attempts": MAX_REPAIR_ATTEMPTS,
    }
