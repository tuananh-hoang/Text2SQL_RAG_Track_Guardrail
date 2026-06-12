import argparse
import csv
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from text2sql.app.run_v2 import run_question_v2
from text2sql.llm_client import get_llm_model, get_llm_provider
from text2sql.schema.paths import v2_entities_path, v2_summary_path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7861
ADVANCED_CASES_PATH = BASE_DIR / "data" / "test_cases_v2_advanced.csv"
MODES = ("mock_relational", "product_sales")


HTML_PAGE = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Text-to-SQL V2</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #647086;
      --line: #d8e0ea;
      --soft: #edf2f7;
      --accent: #0b7285;
      --accent-dark: #075b6a;
      --good: #1d7a46;
      --bad: #b42318;
      --warn: #9a6700;
      --code: #101827;
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
      gap: 18px;
      padding: 0 22px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }

    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 750;
      letter-spacing: 0;
    }

    main {
      display: grid;
      grid-template-columns: minmax(330px, 430px) minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 58px);
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }

    .panel-head {
      min-height: 52px;
      padding: 13px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border-bottom: 1px solid var(--line);
    }

    .panel-head h2 {
      margin: 0;
      font-size: 14px;
      font-weight: 750;
    }

    .panel-body {
      padding: 16px;
    }

    .stack {
      display: grid;
      gap: 12px;
    }

    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    textarea {
      width: 100%;
      min-height: 146px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      color: var(--ink);
      font: inherit;
      line-height: 1.45;
      outline: none;
      background: #ffffff;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(11, 114, 133, 0.13);
    }

    button, select {
      font: inherit;
    }

    button {
      min-height: 38px;
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 9px 12px;
      font-weight: 750;
      color: var(--ink);
      background: #edf2f7;
      cursor: pointer;
    }

    button.primary {
      color: #ffffff;
      background: var(--accent);
    }

    button.primary:hover { background: var(--accent-dark); }
    button:disabled { opacity: 0.65; cursor: wait; }

    select {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      color: var(--ink);
      background: #ffffff;
    }

    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .toggle-row {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }

    .examples {
      display: grid;
      gap: 8px;
      max-height: 380px;
      overflow: auto;
      padding-right: 2px;
    }

    .examples button {
      min-height: auto;
      text-align: left;
      font-weight: 650;
      border-color: var(--line);
      background: #ffffff;
      line-height: 1.35;
    }

    .meta {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
      gap: 8px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--muted);
      background: #ffffff;
      font-size: 12px;
      white-space: nowrap;
    }

    .badge.good { color: var(--good); border-color: rgba(29, 122, 70, 0.35); }
    .badge.bad { color: var(--bad); border-color: rgba(180, 35, 24, 0.35); }
    .badge.warn { color: var(--warn); border-color: rgba(154, 103, 0, 0.35); }

    pre {
      margin: 0;
      max-height: 280px;
      overflow: auto;
      padding: 13px;
      border-radius: 8px;
      color: #eef7ff;
      background: var(--code);
      line-height: 1.45;
      font-size: 13px;
    }

    .tabs {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 3px;
      padding: 3px;
      border-radius: 8px;
      background: var(--soft);
    }

    .tab {
      min-height: 34px;
      border-radius: 6px;
      background: transparent;
      padding: 7px 8px;
      font-weight: 750;
    }

    .tab.active {
      background: #ffffff;
      box-shadow: 0 1px 3px rgba(23, 32, 51, 0.12);
    }

    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

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
      z-index: 1;
      color: var(--muted);
      background: #f1f5f9;
      font-size: 12px;
    }

    .table-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: auto;
      min-height: 180px;
      max-height: calc(100vh - 390px);
    }

    .empty {
      display: grid;
      place-items: center;
      min-height: 160px;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfcfe;
    }

    .flow {
      display: grid;
      gap: 10px;
    }

    .step {
      display: grid;
      grid-template-columns: 42px minmax(0, 1fr);
      gap: 10px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }

    .step-no {
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      border-radius: 999px;
      font-weight: 800;
      color: #ffffff;
      background: var(--accent);
    }

    .step-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 5px;
      font-weight: 750;
    }

    .step-detail {
      color: var(--muted);
      line-height: 1.45;
      overflow-wrap: anywhere;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--ink);
      background: #ffffff;
      font-size: 12px;
      font-weight: 650;
    }

    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      background: #ffffff;
    }

    .metric b {
      display: block;
      margin-bottom: 3px;
      font-size: 16px;
    }

    .metric span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }

    @media (max-width: 1050px) {
      main { grid-template-columns: 1fr; }
      .split { grid-template-columns: 1fr; }
      .tabs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Text-to-SQL V2</h1>
    <div class="meta" id="health">
      <span class="badge warn">loading</span>
    </div>
  </header>

  <main>
    <section class="panel">
      <div class="panel-head">
        <h2>Input</h2>
      </div>
      <div class="panel-body stack">
        <label for="mode">Schema mode</label>
        <select id="mode">
          <option value="mock_relational">mock_relational</option>
          <option value="product_sales">product_sales</option>
        </select>

        <label for="question">Vietnamese question</label>
        <textarea id="question">Top 5 sản phẩm có doanh thu cao nhất</textarea>

        <div class="button-row">
          <button class="primary" id="runBtn" type="button">Run V2</button>
          <button id="clearBtn" type="button">Clear</button>
          <label class="toggle-row">
            <input id="debugContext" type="checkbox" checked />
            selected context
          </label>
        </div>

        <label>Advanced examples</label>
        <div class="examples" id="examples"></div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Output</h2>
        <div class="meta" id="statusBadges">
          <span class="badge">valid: -</span>
          <span class="badge">execution: -</span>
          <span class="badge">rows: 0</span>
        </div>
      </div>
      <div class="panel-body stack">
        <div class="metric-grid">
          <div class="metric"><b id="metricTables">-</b><span>selected tables</span></div>
          <div class="metric"><b id="metricColumns">-</b><span>selected columns</span></div>
          <div class="metric"><b id="metricEntities">-</b><span>matched entities</span></div>
        </div>

        <label>Generated SQL</label>
        <pre id="sqlBox">No SQL yet.</pre>

        <div class="tabs" role="tablist">
          <button class="tab active" data-tab="result" type="button">Result</button>
          <button class="tab" data-tab="flow" type="button">Flow</button>
          <button class="tab" data-tab="retrieval" type="button">Retrieval</button>
          <button class="tab" data-tab="context" type="button">Context</button>
        </div>

        <div id="tab-result" class="tab-pane active"></div>
        <div id="tab-flow" class="tab-pane"></div>
        <div id="tab-retrieval" class="tab-pane"></div>
        <div id="tab-context" class="tab-pane"></div>
      </div>
    </section>
  </main>

  <script>
    const state = { lastPayload: null };

    function esc(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function compact(value, max = 120) {
      const text = String(value ?? "");
      return text.length > max ? text.slice(0, max - 1) + "…" : text;
    }

    function badge(text, kind = "") {
      return `<span class="badge ${kind}">${esc(text)}</span>`;
    }

    async function getJson(url) {
      const res = await fetch(url);
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || res.statusText);
      return payload;
    }

    async function postJson(url, data) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.error || res.statusText);
      return payload;
    }

    function renderHealth(payload) {
      const modes = payload.modes || {};
      const modeBadges = Object.entries(modes).map(([name, info]) => {
        const ready = info.summary_ready && info.entities_ready;
        return badge(`${name}: ${ready ? "ready" : "missing"}`, ready ? "good" : "bad");
      }).join("");
      document.getElementById("health").innerHTML =
        badge(`provider: ${payload.provider}`) +
        badge(`model: ${payload.model}`) +
        badge(`vector: pgvector`, "good") +
        modeBadges;
    }

    function renderExamples(payload) {
      const box = document.getElementById("examples");
      const items = payload.examples || [];
      box.innerHTML = items.map(item => (
        `<button type="button" data-question="${esc(item.question)}">${esc(item.id)}. ${esc(item.question)}</button>`
      )).join("");
      box.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
          document.getElementById("question").value = btn.dataset.question;
        });
      });
    }

    function renderStatus(payload) {
      const valid = !!payload.valid;
      const executed = !!payload.execution_success;
      const rowCount = payload.row_count || 0;
      document.getElementById("statusBadges").innerHTML =
        badge(`valid: ${valid}`, valid ? "good" : "bad") +
        badge(`execution: ${executed}`, executed ? "good" : "bad") +
        badge(`rows: ${rowCount}`, rowCount ? "good" : "");
      document.getElementById("metricTables").textContent = (payload.selected_tables || []).length;
      document.getElementById("metricColumns").textContent = (payload.selected_columns || []).length;
      document.getElementById("metricEntities").textContent = (payload.matched_entities || []).length;
    }

    function renderTable(data) {
      if (!data || !data.columns || !data.rows || data.rows.length === 0) {
        return `<div class="empty">No rows.</div>`;
      }
      const head = `<thead><tr>${data.columns.map(col => `<th>${esc(col)}</th>`).join("")}</tr></thead>`;
      const body = `<tbody>${data.rows.map(row => (
        `<tr>${data.columns.map(col => `<td>${esc(row[col])}</td>`).join("")}</tr>`
      )).join("")}</tbody>`;
      return `<div class="table-wrap"><table>${head}${body}</table></div>`;
    }

    function flowLabel(step) {
      const n = Number(step.step || 0);
      const labels = {
        1: "Nhận câu hỏi",
        2: "Tách intent / retrieval queries",
        3: "Retrieve schema entities",
        4: "Chọn bảng và cột",
        5: "Build selected schema context",
        6: "Sinh SQL",
        7: "Phân tích SQL bằng AST",
        8: "Validate SQL",
        9: "Execute SQL",
        10: "Trả kết quả"
      };
      return labels[n] || step.stage || `Step ${n}`;
    }

    function flowDetail(step) {
      const data = step.data || {};
      if (data.question) return `Câu hỏi: "${data.question}"`;
      if (data.query_decomposition) {
        const d = data.query_decomposition;
        return d.intent_summary || `retrieval_queries: ${(d.retrieval_queries || []).join(", ")}`;
      }
      if (data.matched_entities) {
        return `Matched ${data.matched_entities.length} schema entities bằng pgvector.`;
      }
      if (data.selected_tables || data.selected_columns) {
        return `Tables: ${(data.selected_tables || []).join(", ") || "none"}; Columns: ${(data.selected_columns || []).join(", ") || "none"}.`;
      }
      if (data.selected_schema_context_length) {
        return `Context length: ${data.selected_schema_context_length} characters.`;
      }
      if (data.generated_sql) return "LLM đã sinh SQL từ selected schema context.";
      if (data.validator_result) {
        return data.validator_result.valid
          ? "SQL pass SELECT-only và schema allowlist."
          : `Blocked: ${data.validator_result.error}`;
      }
      if (data.execution_result) {
        const r = data.execution_result;
        return r.success
          ? `Readonly execution returned ${r.row_count} rows in ${r.execution_time_ms} ms.`
          : `Execution error: ${r.error}`;
      }
      if (data.row_count !== undefined) return `Rows: ${data.row_count}`;
      return step.detail || "";
    }

    function flowExtra(step) {
      const data = step.data || {};
      if (data.generated_sql) return `<pre>${esc(data.generated_sql)}</pre>`;
      if (data.sql_structure_summary && data.sql_structure_summary.length) {
        return `<pre>${esc(data.sql_structure_summary.join("\n"))}</pre>`;
      }
      if (data.validator_result && !data.validator_result.valid) {
        return `<pre>${esc(data.validator_result.error)}</pre>`;
      }
      return "";
    }

    function renderFlow(payload) {
      const steps = payload.trace_steps || [];
      if (!steps.length) return `<div class="empty">No trace.</div>`;
      return `<div class="flow">${steps.map(step => {
        const kind = step.status === "pass" || step.status === "success" || step.status === "done" ? "good"
          : step.status === "blocked" || step.status === "error" ? "bad" : "warn";
        return `<div class="step">
          <div class="step-no">${esc(step.step)}</div>
          <div>
            <div class="step-title"><span>${esc(flowLabel(step))}</span>${badge(step.status || "-", kind)}</div>
            <div class="step-detail">${esc(flowDetail(step))}</div>
            ${flowExtra(step)}
          </div>
        </div>`;
      }).join("")}</div>`;
    }

    function renderRetrieval(payload) {
      const decomposition = payload.query_decomposition || {};
      const retrievalQueries = decomposition.retrieval_queries || [];
      const schemaConcepts = decomposition.schema_concepts || [];
      const constraints = decomposition.constraints || [];
      const dateExpressions = decomposition.date_expressions || [];
      const numbers = decomposition.numbers || [];
      const selectedTables = payload.selected_tables || [];
      const selectedColumns = payload.selected_columns || [];
      const entities = (payload.matched_entities || []).slice(0, 16);
      const meta = payload.retrieval_metadata || {};

      const chipList = (items) => (
        items && items.length
          ? items.map(item => `<span class="chip">${esc(item)}</span>`).join("")
          : badge("none", "warn")
      );

      const conceptTable = schemaConcepts.length ? `<div class="table-wrap"><table>
        <thead><tr><th>phrase</th><th>concept_type</th><th>description</th></tr></thead>
        <tbody>${schemaConcepts.map(item => `<tr>
          <td>${esc(item.phrase)}</td>
          <td>${esc(item.concept_type)}</td>
          <td>${esc(item.description)}</td>
        </tr>`).join("")}</tbody>
      </table></div>` : `<div class="empty">No schema concepts.</div>`;

      const constraintTable = constraints.length ? `<div class="table-wrap"><table>
        <thead><tr><th>phrase</th><th>constraint_type</th><th>description</th></tr></thead>
        <tbody>${constraints.map(item => `<tr>
          <td>${esc(item.phrase)}</td>
          <td>${esc(item.constraint_type)}</td>
          <td>${esc(item.description)}</td>
        </tr>`).join("")}</tbody>
      </table></div>` : `<div class="empty">No constraints.</div>`;

      const entityTable = entities.length ? `<div class="table-wrap"><table>
        <thead><tr><th>entity</th><th>type</th><th>target</th><th>score</th><th>evidence</th></tr></thead>
        <tbody>${entities.map(ent => `<tr>
          <td>${esc(ent.entity_id)}</td>
          <td>${esc(ent.entity_type)}</td>
          <td>${esc([ent.table, ent.column].filter(Boolean).join("."))}</td>
          <td>${esc(ent.final_score)}</td>
          <td>${esc(compact(ent.text, 90))}</td>
        </tr>`).join("")}</tbody>
      </table></div>` : `<div class="empty">No matched entities.</div>`;

      return `<div class="stack">
        <div class="split">
          <div class="stack">
            <label>Intent summary</label>
            <pre>${esc(decomposition.intent_summary || "")}</pre>
          </div>
          <div class="stack">
            <label>Retrieval metadata</label>
            <pre>${esc(JSON.stringify(meta, null, 2))}</pre>
          </div>
        </div>
        <div class="split">
          <div class="stack">
            <label>retrieval_queries</label>
            <div class="chips">${chipList(retrievalQueries)}</div>
          </div>
          <div class="stack">
            <label>date_expressions / numbers</label>
            <div class="chips">${chipList(dateExpressions)}${chipList(numbers)}</div>
          </div>
        </div>
        <div class="split">
          <div class="stack">
            <label>schema_concepts</label>
            ${conceptTable}
          </div>
          <div class="stack">
            <label>constraints</label>
            ${constraintTable}
          </div>
        </div>
        <div class="split">
          <div class="stack">
            <label>Selected tables</label>
            <div class="chips">${selectedTables.map(item => `<span class="chip">${esc(item)}</span>`).join("") || badge("none", "warn")}</div>
          </div>
          <div class="stack">
            <label>Selected columns</label>
            <div class="chips">${selectedColumns.map(item => `<span class="chip">${esc(item)}</span>`).join("") || badge("none", "warn")}</div>
          </div>
        </div>
        <label>Top matched entities</label>
        ${entityTable}
      </div>`;
    }

    function selectedContext(payload) {
      for (const step of payload.trace_steps || []) {
        const ctx = step.data && step.data.selected_schema_context;
        if (ctx) return ctx;
      }
      return "";
    }

    function renderPayload(payload) {
      state.lastPayload = payload;
      renderStatus(payload);
      document.getElementById("sqlBox").textContent = payload.sql || "NO_SQL / no SQL";
      document.getElementById("tab-result").innerHTML =
        payload.error ? `${renderTable(payload.data)}<pre>${esc(payload.error)}</pre>` : renderTable(payload.data);
      document.getElementById("tab-flow").innerHTML = renderFlow(payload);
      document.getElementById("tab-retrieval").innerHTML = renderRetrieval(payload);
      const ctx = selectedContext(payload);
      document.getElementById("tab-context").innerHTML = ctx ? `<pre>${esc(ctx)}</pre>` : `<div class="empty">No selected context in payload.</div>`;
    }

    async function runQuery() {
      const btn = document.getElementById("runBtn");
      const question = document.getElementById("question").value.trim();
      const mode = document.getElementById("mode").value;
      const debugContext = document.getElementById("debugContext").checked;
      if (!question) return;
      btn.disabled = true;
      btn.textContent = "Running...";
      document.getElementById("sqlBox").textContent = "Running V2 pipeline...";
      try {
        const payload = await postJson("/api/query", { question, mode, debug_context: debugContext });
        renderPayload(payload);
      } catch (err) {
        document.getElementById("statusBadges").innerHTML = badge("error", "bad");
        document.getElementById("tab-result").innerHTML = `<pre>${esc(err.message)}</pre>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Run V2";
      }
    }

    document.querySelectorAll(".tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(item => item.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach(item => item.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
      });
    });

    document.getElementById("runBtn").addEventListener("click", runQuery);
    document.getElementById("clearBtn").addEventListener("click", () => {
      document.getElementById("question").value = "";
      document.getElementById("question").focus();
    });

    getJson("/api/health").then(renderHealth).catch(err => {
      document.getElementById("health").innerHTML = badge(err.message, "bad");
    });
    getJson("/api/examples").then(renderExamples).catch(() => {
      document.getElementById("examples").innerHTML = "";
    });
  </script>
</body>
</html>
"""


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
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
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
    payload["data"] = dataframe_to_payload(result.get("data"))
    return json_safe(payload)


def mode_status(mode: str) -> dict[str, Any]:
    return {
        "summary_path": str(v2_summary_path(mode).relative_to(BASE_DIR)),
        "summary_ready": v2_summary_path(mode).exists(),
        "entities_path": str(v2_entities_path(mode).relative_to(BASE_DIR)),
        "entities_ready": v2_entities_path(mode).exists(),
    }


def examples_payload() -> dict[str, Any]:
    examples = []
    if ADVANCED_CASES_PATH.exists():
        with ADVANCED_CASES_PATH.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                examples.append(
                    {
                        "id": row.get("id", ""),
                        "category": row.get("category", ""),
                        "difficulty": row.get("difficulty", ""),
                        "question": row.get("question", ""),
                    }
                )
    if not examples:
        examples = [
            {"id": "1", "category": "smoke", "difficulty": "medium", "question": "Top 5 sản phẩm có doanh thu cao nhất"},
            {"id": "2", "category": "smoke", "difficulty": "medium", "question": "Doanh thu theo từng Region trong năm 2024"},
        ]
    return {"examples": examples}


class Text2SQLV2Handler(BaseHTTPRequestHandler):
    server_version = "Text2SQLV2/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
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
                    "provider": get_llm_provider(),
                    "model": get_llm_model(),
                    "modes": {mode: mode_status(mode) for mode in MODES},
                }
            )
            return

        if path == "/api/examples":
            self.send_json(examples_payload())
            return

        self.send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path != "/api/query":
                self.send_json({"error": "Not found"}, status=404)
                return

            payload = self.read_json()
            question = str(payload.get("question", "")).strip()
            mode = str(payload.get("mode", "mock_relational")).strip()
            debug_context = bool(payload.get("debug_context", True))

            if not question:
                self.send_json({"error": "Question is empty"}, status=400)
                return
            if mode not in MODES:
                self.send_json({"error": f"Unsupported mode: {mode}"}, status=400)
                return
            if not v2_summary_path(mode).exists():
                self.send_json(
                    {
                        "error": (
                            f"Missing summary for {mode}. Run: "
                            f"python -m text2sql.schema.summary.generate_schema_summary --mode {mode}"
                        )
                    },
                    status=400,
                )
                return

            result = run_question_v2(question, mode=mode, debug_context=debug_context)
            self.send_json(result_to_payload(result))
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Text-to-SQL V2 web UI")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Text2SQLV2Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Text-to-SQL V2 UI running at {url}")
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
