import argparse
import json
import time
import uuid
from datetime import date, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from llm_client import get_llm_model, get_llm_provider
from run_v1 import SCHEMA_PATH, add_sql_structure_trace, add_trace_step, run_question
from schema.schema_context_builder import build_schema_context
from sql.executor import execute_sql
from sql.validator import validate_sql


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEMO_LOG_PATH = BASE_DIR / "demo_logs" / "query_traces.jsonl"


HTML_PAGE = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Text-to-SQL V1</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #18202f;
      --muted: #647085;
      --line: #d9e0ea;
      --accent: #087f8c;
      --accent-dark: #075f69;
      --warn: #b76e00;
      --danger: #b42318;
      --good: #217a45;
      --code: #111827;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }

    header {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }

    main {
      display: grid;
      grid-template-columns: minmax(320px, 440px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      min-height: calc(100vh - 58px);
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }

    .panel-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .panel-head h2 {
      margin: 0;
      font-size: 14px;
      font-weight: 700;
    }

    .panel-body { padding: 16px; }

    .stack { display: grid; gap: 12px; }

    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }

    textarea {
      width: 100%;
      min-height: 130px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      color: var(--ink);
      font: inherit;
      line-height: 1.45;
      outline: none;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(8, 127, 140, 0.12);
    }

    button {
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      background: #eef3f7;
      color: var(--ink);
    }

    button.primary {
      background: var(--accent);
      color: #ffffff;
    }

    button.primary:hover { background: var(--accent-dark); }
    button:disabled { cursor: wait; opacity: 0.65; }

    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .demo-grid {
      display: grid;
      gap: 8px;
    }

    .demo-grid button {
      text-align: left;
      font-weight: 600;
      background: #ffffff;
      border-color: var(--line);
    }

    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      background: #eef3f7;
      border-radius: 8px;
      padding: 3px;
    }

    .tab {
      border-radius: 6px;
      background: transparent;
      padding: 8px 10px;
    }

    .tab.active {
      background: #ffffff;
      box-shadow: 0 1px 3px rgba(24, 32, 47, 0.12);
    }

    .meta {
      color: var(--muted);
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }

    .badge {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      background: #ffffff;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }

    .badge.good { color: var(--good); border-color: rgba(33, 122, 69, 0.35); }
    .badge.bad { color: var(--danger); border-color: rgba(180, 35, 24, 0.35); }
    .badge.warn { color: var(--warn); border-color: rgba(183, 110, 0, 0.35); }

    pre {
      margin: 0;
      overflow: auto;
      background: var(--code);
      color: #eef7ff;
      border-radius: 8px;
      padding: 14px;
      line-height: 1.45;
      max-height: 260px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }

    th {
      position: sticky;
      top: 0;
      background: #f1f5f9;
      font-size: 12px;
      color: var(--muted);
      z-index: 1;
    }

    .table-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: auto;
      max-height: calc(100vh - 330px);
      min-height: 180px;
    }

    .schema-panel { grid-column: 1 / -1; }

    .reference-grid {
      display: grid;
      grid-template-columns: minmax(320px, 0.95fr) minmax(320px, 1.05fr);
      gap: 16px;
    }

    .reference-block {
      display: grid;
      gap: 10px;
      min-width: 0;
    }

    .schema-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }

    .schema-summary strong {
      color: var(--ink);
      font-size: 14px;
    }

    .role {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 650;
      color: var(--ink);
      background: #f8fafc;
      white-space: nowrap;
    }

    .empty {
      min-height: 180px;
      display: grid;
      place-items: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfcfe;
    }

    .error {
      color: var(--danger);
      background: #fff4f2;
      border: 1px solid rgba(180, 35, 24, 0.25);
      border-radius: 8px;
      padding: 12px;
      white-space: pre-wrap;
    }

    .hidden { display: none; }

    @media (max-width: 880px) {
      main { grid-template-columns: 1fr; }
      .table-wrap { max-height: 420px; }
      .reference-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Text-to-SQL V1</h1>
    <div id="status" class="meta"></div>
  </header>

  <main>
    <section class="panel">
      <div class="panel-head">
        <h2>Input</h2>
      </div>
      <div class="panel-body stack">
        <div class="tabs">
          <button id="tab-question" class="tab active" type="button">Question</button>
          <button id="tab-sql" class="tab" type="button">Manual SQL</button>
        </div>

        <div id="question-mode" class="stack">
          <label for="question">Vietnamese question</label>
          <textarea id="question">Doanh thu theo từng khu vực?</textarea>
          <div class="button-row">
            <button id="run-question" class="primary" type="button">Run</button>
            <button id="clear-question" type="button">Clear</button>
          </div>
          <label>Demo questions</label>
          <div class="demo-grid" id="demo-questions"></div>
        </div>

        <div id="sql-mode" class="stack hidden">
          <label for="manual-sql">SQL</label>
          <textarea id="manual-sql">SELECT "Region", SUM("Revenue") AS total_revenue FROM product_sales GROUP BY "Region" ORDER BY total_revenue DESC</textarea>
          <div class="button-row">
            <button id="run-sql" class="primary" type="button">Validate & Execute</button>
            <button id="clear-sql" type="button">Clear</button>
          </div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Output</h2>
        <div id="badges" class="meta"></div>
      </div>
      <div class="panel-body stack">
        <div id="error" class="error hidden"></div>
        <label>SQL</label>
        <pre id="sql-output">No query executed yet.</pre>
        <label>Explanation</label>
        <div id="explanation" class="empty">No explanation yet.</div>
        <label>Processing Flow</label>
        <div id="trace" class="empty">No flow trace yet.</div>
        <label>Result</label>
        <div id="result" class="empty">No rows yet.</div>
      </div>
    </section>

    <section class="panel schema-panel">
      <div class="panel-head">
        <h2>Schema & Table Preview</h2>
      </div>
      <div class="panel-body reference-grid">
        <div class="reference-block">
          <label>Schema Diagram</label>
          <div id="schema-diagram" class="empty">Loading schema...</div>
        </div>
        <div class="reference-block">
          <label>Table Preview</label>
          <div id="table-preview" class="empty">Loading preview...</div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const demoQuestions = [
      "Tổng doanh thu là bao nhiêu?",
      "Doanh thu theo từng khu vực?",
      "Top 5 sản phẩm có doanh thu cao nhất?",
      "Có bao nhiêu đơn hàng?",
      "Xóa bảng product_sales đi"
    ];

    const els = {
      status: document.getElementById("status"),
      badges: document.getElementById("badges"),
      error: document.getElementById("error"),
      sqlOutput: document.getElementById("sql-output"),
      explanation: document.getElementById("explanation"),
      result: document.getElementById("result"),
      trace: document.getElementById("trace"),
      question: document.getElementById("question"),
      manualSql: document.getElementById("manual-sql"),
      runQuestion: document.getElementById("run-question"),
      runSql: document.getElementById("run-sql"),
      tabQuestion: document.getElementById("tab-question"),
      tabSql: document.getElementById("tab-sql"),
      questionMode: document.getElementById("question-mode"),
      sqlMode: document.getElementById("sql-mode"),
      demoQuestions: document.getElementById("demo-questions"),
      schemaDiagram: document.getElementById("schema-diagram"),
      tablePreview: document.getElementById("table-preview")
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function setMode(mode) {
      const isQuestion = mode === "question";
      els.tabQuestion.classList.toggle("active", isQuestion);
      els.tabSql.classList.toggle("active", !isQuestion);
      els.questionMode.classList.toggle("hidden", !isQuestion);
      els.sqlMode.classList.toggle("hidden", isQuestion);
    }

    function setBusy(busy) {
      els.runQuestion.disabled = busy;
      els.runSql.disabled = busy;
      els.runQuestion.textContent = busy ? "Running..." : "Run";
      els.runSql.textContent = busy ? "Running..." : "Validate & Execute";
    }

    function renderBadges(payload) {
      const badges = [];
      if (payload.valid !== undefined) {
        badges.push(`<span class="badge ${payload.valid ? "good" : "bad"}">valid: ${payload.valid}</span>`);
      }
      if (payload.auto_limited !== undefined) {
        badges.push(`<span class="badge ${payload.auto_limited ? "warn" : ""}">auto limit: ${payload.auto_limited}</span>`);
      }
      if (payload.row_count !== undefined) {
        badges.push(`<span class="badge">rows: ${payload.row_count}</span>`);
      }
      els.badges.innerHTML = badges.join("");
    }

    function renderTable(data) {
      if (!data || !data.columns || data.columns.length === 0) {
        els.result.className = "empty";
        els.result.textContent = "No rows.";
        return;
      }

      const header = data.columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("");
      const rows = data.rows.map((row) => {
        return `<tr>${data.columns.map((col) => `<td>${escapeHtml(row[col])}</td>`).join("")}</tr>`;
      }).join("");
      els.result.className = "table-wrap";
      els.result.innerHTML = `<table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>`;
    }

    function renderGenericTable(target, data, emptyText) {
      if (!data || !data.columns || data.columns.length === 0) {
        target.className = "empty";
        target.textContent = emptyText;
        return;
      }

      const header = data.columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("");
      const rows = (data.rows || []).map((row) => {
        return `<tr>${data.columns.map((col) => `<td>${escapeHtml(row[col])}</td>`).join("")}</tr>`;
      }).join("");
      target.className = "table-wrap";
      target.innerHTML = `<table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>`;
    }

    function renderTrace(payload) {
      const steps = payload.trace_steps || [];
      if (!steps.length) {
        els.trace.className = "empty";
        els.trace.textContent = "No flow trace.";
        return;
      }
      els.trace.className = "";
      function renderStepData(step) {
        const data = step.data || {};
        const parts = [];
        if (data.generated_sql) {
          parts.push(`<div><strong>SQL sinh ra</strong><pre>${escapeHtml(data.generated_sql)}</pre></div>`);
        }
        if (data.repaired_sql) {
          parts.push(`<div><strong>SQL sau repair</strong><pre>${escapeHtml(data.repaired_sql)}</pre></div>`);
        }
        if (data.sql_structure_summary && data.sql_structure_summary.length) {
          const items = data.sql_structure_summary
            .map((line) => `<li>${escapeHtml(line)}</li>`)
            .join("");
          parts.push(`<div><strong>SQL Structure Summary</strong><ol>${items}</ol></div>`);
        }
        if (data.sql_features && Object.keys(data.sql_features).length) {
          parts.push(
            `<details><summary>SQL features (AST JSON)</summary><pre>${escapeHtml(JSON.stringify(data.sql_features, null, 2))}</pre></details>`
          );
        }
        if (data.sql_structure_error) {
          parts.push(`<div class="error">Lỗi phân tích AST: ${escapeHtml(data.sql_structure_error)}</div>`);
        }
        if (data.validator_error) {
          parts.push(`<div class="error">Lý do block: ${escapeHtml(data.validator_error)}</div>`);
        }
        if (data.execution_error) {
          parts.push(`<div class="error">Lỗi execute: ${escapeHtml(data.execution_error)}</div>`);
        }
        if (data.row_count !== undefined || data.execution_time_ms !== undefined) {
          const rowText = data.row_count !== undefined ? `Số dòng: ${escapeHtml(data.row_count)}` : "";
          const timeText = data.execution_time_ms !== undefined ? `Thời gian: ${escapeHtml(data.execution_time_ms)} ms` : "";
          parts.push(`<div>${[rowText, timeText].filter(Boolean).join(" · ")}</div>`);
        }
        return parts.join("");
      }
      els.trace.innerHTML = `<ol>${steps.map((step) => {
        const prefix = `b${escapeHtml(step.step)}: ${escapeHtml(step.stage)} (${escapeHtml(step.status)})`;
        return `<li><strong>${prefix}</strong><br><span>${escapeHtml(step.detail)}</span>${renderStepData(step)}</li>`;
      }).join("")}</ol>`;
    }

    function renderSchemaOverview(payload) {
      const columns = ["Column", "Data Type", "Sample / Range"];
      const rows = payload.columns.map((column) => ({
        "Column": column.name,
        "Data Type": column.type,
        "Sample / Range": column.evidence || column.aliases || ""
      }));
      const table = {
        columns,
        rows: rows.map((row) => ({
          "Column": row["Column"],
          "Data Type": row["Data Type"],
          "Sample / Range": row["Sample / Range"]
        }))
      };
      renderGenericTable(els.schemaDiagram, table, "No schema found.");
      els.schemaDiagram.insertAdjacentHTML(
        "afterbegin",
        `<div class="schema-summary">
          <strong>${escapeHtml(payload.table_name)}</strong>
          <span class="badge">rows: ${escapeHtml(payload.row_count)}</span>
          <span class="badge">columns: ${escapeHtml(payload.column_count)}</span>
          <span class="badge">grain: ${escapeHtml(payload.table_grain)}</span>
        </div>`
      );
      renderGenericTable(els.tablePreview, payload.preview, "No preview rows.");
    }

    function renderPayload(payload) {
      els.error.classList.toggle("hidden", !payload.error);
      els.error.textContent = payload.error || "";
      els.sqlOutput.textContent = payload.sql || "NO_SQL";
      els.explanation.className = payload.explanation ? "" : "empty";
      els.explanation.textContent = payload.explanation || "No explanation.";
      renderTrace(payload);
      renderBadges(payload);
      renderTable(payload.data);
    }

    async function postJson(path, body) {
      setBusy(true);
      try {
        const response = await fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const payload = await response.json();
        if (!response.ok) {
          payload.error = payload.error || `HTTP ${response.status}`;
        }
        renderPayload(payload);
      } catch (error) {
        renderPayload({ error: String(error), sql: "", data: null });
      } finally {
        setBusy(false);
      }
    }

    async function loadHealth() {
      const response = await fetch("/api/health");
      const payload = await response.json();
      const schemaClass = payload.schema_ready ? "good" : "bad";
      els.status.innerHTML = [
        `<span class="badge ${schemaClass}">schema: ${payload.schema_ready ? "ready" : "missing"}</span>`,
        `<span class="badge">provider: ${escapeHtml(payload.provider)}</span>`,
        `<span class="badge">model: ${escapeHtml(payload.model)}</span>`
      ].join("");
    }

    async function loadSchemaOverview() {
      try {
        const response = await fetch("/api/schema-overview");
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || `HTTP ${response.status}`);
        }
        renderSchemaOverview(payload);
      } catch (error) {
        els.schemaDiagram.className = "error";
        els.schemaDiagram.textContent = String(error);
        els.tablePreview.className = "empty";
        els.tablePreview.textContent = "Preview unavailable.";
      }
    }

    els.tabQuestion.addEventListener("click", () => setMode("question"));
    els.tabSql.addEventListener("click", () => setMode("sql"));
    els.runQuestion.addEventListener("click", () => {
      postJson("/api/query", { question: els.question.value });
    });
    els.runSql.addEventListener("click", () => {
      postJson("/api/manual-sql", { sql: els.manualSql.value });
    });
    document.getElementById("clear-question").addEventListener("click", () => {
      els.question.value = "";
      els.question.focus();
    });
    document.getElementById("clear-sql").addEventListener("click", () => {
      els.manualSql.value = "";
      els.manualSql.focus();
    });

    demoQuestions.forEach((question) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = question;
      button.addEventListener("click", () => {
        els.question.value = question;
        postJson("/api/query", { question });
      });
      els.demoQuestions.appendChild(button);
    });

    loadHealth();
    loadSchemaOverview();
  </script>
</body>
</html>
"""


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def load_schema_summary() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def infer_column_role(column: dict[str, Any]) -> str:
    data_type = str(column.get("type", "")).lower()
    name = str(column.get("name", "")).lower()
    if column.get("is_id") or name.endswith("_id") or name == "id":
        return "id"
    if "date" in data_type or "time" in data_type or "date" in name or "time" in name:
        return "time"
    if data_type in {"integer", "bigint", "smallint", "numeric", "real", "double precision", "float"}:
        return "measure"
    return "dimension"


def column_use_label(role: str) -> str:
    labels = {
        "id": "record id",
        "time": "date/time filter",
        "dimension": "group/filter/display",
        "measure": "numeric metric",
    }
    return labels.get(role, role)


def evidence_for_column(column: dict[str, Any]) -> str:
    evidence_parts = []
    if column.get("all_values"):
        values = ", ".join(str(value) for value in column["all_values"][:8])
        evidence_parts.append("values: " + values)
    elif column.get("sample_values"):
        values = ", ".join(str(value) for value in column["sample_values"][:5])
        evidence_parts.append("sample: " + values)
    elif column.get("min") is not None and column.get("max") is not None:
        range_text = f"range: {column['min']} to {column['max']}"
        if column.get("avg") is not None:
            range_text += f", avg: {column['avg']}"
        evidence_parts.append(range_text)

    return "; ".join(evidence_parts)


def preview_query(schema: dict[str, Any], limit: int = 20) -> str:
    table_name = schema["table_name"]
    columns = [column["name"] for column in schema["columns"]]
    selected_columns = ", ".join(quote_identifier(column) for column in columns)
    order_column = next(
        (column["name"] for column in schema["columns"] if column.get("is_id")),
        columns[0],
    )
    return (
        f"SELECT {selected_columns} "
        f"FROM {table_name} "
        f"ORDER BY {quote_identifier(order_column)} "
        f"LIMIT {limit}"
    )


def schema_overview_payload() -> dict[str, Any]:
    schema = load_schema_summary()
    columns = []
    for column in schema["columns"]:
        role = infer_column_role(column)
        columns.append(
            {
                "name": column["name"],
                "type": column["type"],
                "role": role,
                "use": column_use_label(role),
                "nullable": column.get("nullable"),
                "aliases": "",
                "evidence": evidence_for_column(column),
            }
        )

    preview = execute_sql(preview_query(schema, limit=20))
    return {
        "table_name": schema["table_name"],
        "row_count": schema["row_count"],
        "column_count": schema["column_count"],
        "table_grain": "order-line level",
        "columns": columns,
        "preview": dataframe_to_payload(preview["data"]) if preview["success"] else None,
        "preview_error": preview["error"],
    }


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def dataframe_to_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None:
        return None
    columns = [str(column) for column in df.columns]
    rows = []
    for record in df.to_dict(orient="records"):
        rows.append({str(key): json_safe(value) for key, value in record.items()})
    return {"columns": columns, "rows": rows}


def result_to_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload["data"] = dataframe_to_payload(payload.get("data"))
    return payload


def trace_final_status(result: dict[str, Any]) -> str:
    if result.get("valid") is False:
        return "blocked"
    if result.get("error"):
        return "error"
    return "success"


def write_demo_log(
    mode: str,
    result: dict[str, Any],
    question: str | None = None,
    input_sql: str | None = None,
) -> None:
    try:
        DEMO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "question": question,
            "input_sql": input_sql,
            "generated_sql": result.get("sql") if mode == "natural_language" else None,
            "final_status": trace_final_status(result),
            "steps": result.get("trace_steps", []),
        }
        with DEMO_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return


def run_manual_sql(sql: str) -> dict[str, Any]:
    trace_steps: list[dict[str, Any]] = []
    add_trace_step(
        trace_steps,
        "Nhận manual SQL",
        "done",
        f'SQL input: "{sql}"',
        {"input_sql": sql},
    )
    add_sql_structure_trace(trace_steps, sql)
    validation = validate_sql(sql, str(SCHEMA_PATH))
    validation_detail = (
        "SQL hợp lệ và được phép execute."
        if validation["valid"]
        else f"Lý do: {validation['error']}\nSQL không được gửi xuống PostgreSQL."
    )
    add_trace_step(
        trace_steps,
        "Kiểm tra SQL an toàn",
        "pass" if validation["valid"] else "blocked",
        validation_detail,
        {
            "tables_from_ast": validation["tables_from_ast"],
            "columns_from_ast": validation["columns_from_ast"],
            "auto_limited": validation["auto_limited"],
            "validator_error": validation["error"],
        },
    )
    payload: dict[str, Any] = {
        "question": None,
        "sql": validation["sql"],
        "valid": validation["valid"],
        "auto_limited": validation["auto_limited"],
        "row_count": 0,
        "data": None,
        "explanation": "Manual SQL validation",
        "error": validation["error"],
        "trace_steps": trace_steps,
    }

    if not validation["valid"]:
        return payload

    start = time.perf_counter()
    execution = execute_sql(validation["sql"])
    execution_time_ms = round((time.perf_counter() - start) * 1000, 2)
    add_trace_step(
        trace_steps,
        "Thực thi SQL",
        "pass" if execution["success"] else "error",
        (
            f"Chạy bằng readonly user. Trả về {execution['row_count']} dòng "
            f"trong {execution_time_ms} ms."
        )
        if execution["success"]
        else f"Lỗi khi execute: {execution['error']}",
        {
            "row_count": execution["row_count"],
            "execution_time_ms": execution_time_ms,
            "execution_error": None if execution["success"] else execution["error"],
        },
    )
    payload["row_count"] = execution["row_count"]
    payload["data"] = execution["data"]
    payload["error"] = execution["error"]
    if execution["success"]:
        payload["explanation"] = "SQL passed validator and executed with readonly user."
        add_trace_step(
            trace_steps,
            "Trả kết quả",
            "success",
            "Đã trả SQL, bảng kết quả và explanation về UI.",
            {"row_count": execution["row_count"]},
        )
    else:
        add_trace_step(
            trace_steps,
            "Trả kết quả",
            "error",
            "Không thể trả kết quả do lỗi ở bước execute.",
            {"row_count": 0, "execution_error": execution["error"]},
        )
    return payload


class Text2SQLHandler(BaseHTTPRequestHandler):
    server_version = "Text2SQLV1/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body) if body else {}

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/health":
            self.send_json(
                {
                    "schema_ready": SCHEMA_PATH.exists(),
                    "provider": get_llm_provider(),
                    "model": get_llm_model(),
                }
            )
            return

        if path == "/api/schema-overview":
            if not SCHEMA_PATH.exists():
                self.send_json(
                    {"error": "schema/schema_summary.json is missing. Run python schema/generate_schema_summary.py."},
                    status=400,
                )
                return
            self.send_json(schema_overview_payload())
            return

        self.send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if not SCHEMA_PATH.exists():
                self.send_json(
                    {"error": "schema/schema_summary.json is missing. Run python schema/generate_schema_summary.py."},
                    status=400,
                )
                return

            payload = self.read_json()
            if path == "/api/query":
                question = str(payload.get("question", "")).strip()
                if not question:
                    self.send_json({"error": "Question is empty"}, status=400)
                    return
                schema_context = build_schema_context(str(SCHEMA_PATH))
                result = run_question(question, schema_context)
                write_demo_log("natural_language", result, question=question)
                self.send_json(result_to_payload(result))
                return

            if path == "/api/manual-sql":
                sql = str(payload.get("sql", "")).strip()
                if not sql:
                    self.send_json({"error": "SQL is empty"}, status=400)
                    return
                result = run_manual_sql(sql)
                write_demo_log("manual_sql", result, input_sql=sql)
                self.send_json(result_to_payload(result))
                return

            self.send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Text-to-SQL V1 web UI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Text2SQLHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Text-to-SQL V1 UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
