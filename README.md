
# Enterprise MCP Project — Use Case Reference

This document maps every capability in the project: which server owns it,
what it does, how it's implemented, and what a user question that
triggers it looks like. Use this as the "what did I actually build"
reference for interview prep.

---

## 1. `hr_server.py` — Employee data (CSV-backed)

**Domain owned:** everything about employee records.
**Backing store:** `data/employees.csv`, re-read from disk on every call
(mirrors how a real read replica or lightweight lookup service behaves —
no in-memory staleness, always current as of the file on disk).

| Function | What it does | Input | Output | Example trigger question |
|---|---|---|---|---|
| `get_employee(employee_id)` | Looks up a single employee by ID | `employee_id: str` (e.g. `"E002"`) | Dict with name, department, role, email — or an error dict if not found | *"Look up employee E002"* |
| `list_employees_by_department(department)` | Lists everyone in a given department, case-insensitive | `department: str` (e.g. `"Engineering"`) | List of matching employee dicts | *"Who's in the Sales department?"* |

**Resource (read-only, not a callable tool):**
`hr://employees/all` — exposes the full roster as raw CSV text. Resources
differ from tools in MCP: they're meant to be pulled in as context, not
"called" with arguments the way a tool is.

**Implementation notes:**
- Uses `FastMCP` from the official MCP Python SDK (`mcp.server.fastmcp`).
- Each function is decorated with `@mcp.tool()` — this decorator is what
  makes the function discoverable by a client's `list_tools()` call. The
  function's docstring becomes the tool's `description`, and its type
  hints become the JSON schema the LLM sees.
- No `salary` field exists in the current data model — if you want a
  "list salary by department" use case, that's a natural next function to
  add (`average_salary_by_department`), following the exact same
  `@mcp.tool()` pattern as the two above.

---

## 2. `ticketing_server.py` — Support tickets (SQLite-backed)

**Domain owned:** creating and tracking internal support tickets (a mini
ITSM/Jira-style system).
**Backing store:** `data/tickets.db`, a real SQLite database — chosen over
an in-memory dict specifically so state survives a server restart.

| Function | What it does | Input | Output | Example trigger question |
|---|---|---|---|---|
| `create_ticket(title, description, priority, created_by)` | Inserts a new ticket row | `title`, `description`, `priority` (low/medium/high/critical), `created_by` | Dict with the new `ticket_id` and all fields | *"Create a high priority ticket for a broken VPN, requested by E002"* |
| `get_ticket(ticket_id)` | Fetches one ticket by ID | `ticket_id: int` | Full ticket row as a dict, or error | *"What's the status of ticket 1?"* |
| `list_tickets(status)` | Lists tickets, optionally filtered by status | `status: str` (optional — `"open"`/`"in_progress"`/`"closed"`, or blank for all) | List of ticket dicts | *"Show me all open tickets"* |
| `update_ticket_status(ticket_id, new_status)` | Changes a ticket's status | `ticket_id`, `new_status` | Confirmation dict, or error | *"Mark ticket 1 as in progress"* |

**Implementation notes:**
- This is the project's example of tools **with side effects** —
  `create_ticket` and `update_ticket_status` actually mutate state, unlike
  the HR server's pure lookups. Worth calling out explicitly in an
  interview: MCP tools aren't just read-only queries, they can perform
  real actions.
- `_init_db()` runs once at import time (`CREATE TABLE IF NOT EXISTS`),
  so the schema is always guaranteed to exist without wiping existing
  rows on restart.
- Priority and status values are validated against a fixed set before
  insert/update — basic input validation, not just trusting whatever the
  LLM sends.

---

## 3. `knowledge_server.py` — Policy document search (RAG-backed)

**Domain owned:** answering questions against company policy docs (leave
policy, expense policy, onboarding guide).
**Backing store:** `data/company_docs/*.txt`, indexed once at server
startup.

| Function | What it does | Input | Output | Example trigger question |
|---|---|---|---|---|
| `search_knowledge_base(question, top_k)` | Retrieves the most relevant document(s) for a question | `question: str`, `top_k: int` (default 2) | List of `{document, similarity_score, content}` dicts | *"What is our leave policy?"* |

**Resource:** `knowledge://docs/list` — lists all indexed document names.

**Implementation notes (current version, TF-IDF):**
- `_build_index()` runs once at startup: reads every `.txt` file in
  `data/company_docs/`, fits a `TfidfVectorizer` across all of them, and
  keeps the resulting matrix in memory (`_doc_matrix`) — built once,
  queried many times, same as a real vector index would be.
- At query time: the question is transformed into the same TF-IDF space,
  then `cosine_similarity()` ranks every document against it, and the
  `top_k` highest-scoring documents are returned.
- This is the *same pipeline shape* as your original embeddings-based RAG
  project (chunk/doc → vectorize → store → query → cosine similarity →
  retrieve). TF-IDF is a lighter-weight stand-in for a real embedding
  model, used here to avoid a model-download dependency.
- **Planned upgrade (in progress):** swap `TfidfVectorizer` for real
  semantic embeddings via `ollama.embed(model="nomic-embed-text", ...)`.
  This changes only the "how do we turn text into a vector" step — the
  surrounding retrieval logic (build once, compare with cosine
  similarity, return top_k) stays identical.

---

## 4. `enterprise_client.py` / `ollama_enterprise_client.py` — The orchestrator

This isn't a domain server — it's the **Host/Client** in MCP terms: the
one component that knows all three servers exist and coordinates between
them and the LLM. `enterprise_client.py` uses Anthropic's API;
`ollama_enterprise_client.py` uses a local Ollama model (`qwen3:8b`) for
zero-cost development. Both share the same orchestration architecture —
only the LLM-calling code differs, since each provider has a different
request/response shape for tool calling.

| Capability | What it does | How it's implemented |
|---|---|---|
| **Dynamic tool discovery** | Finds out what tools exist without hardcoding them | `_connect_one_server()` calls `session.list_tools()` on each server at startup and builds `self.tool_routing: {tool_name: (session, server_name)}` |
| **Multi-server orchestration / routing** | Sends every discovered tool to the LLM in one combined list; when the LLM picks one, routes the call to whichever server actually owns it | `call_tool_routed()` looks up `self.tool_routing[tool_name]` — the LLM never talks to a server directly, only ever says "call this tool with these args" |
| **Conversation memory** | Lets follow-up questions like "what's the status of *that* ticket" resolve correctly across turns | `self.messages` is a list that persists on the client instance (set once in `__init__`, appended to — never reassigned — in `ask()`) |
| **Config-driven server registry** | Add/remove/disable servers by editing a YAML file, no code changes | `load_config()` reads `MCP_CONFIG_PATH` (defaulting to `servers.yaml`), parses it, filters out any server with `enabled: false`; falls back to a hardcoded list if no config file exists, but fails loudly (raises) if the file exists but is malformed YAML |
| **Audit trail** | Persists a record of every tool call — question, tool, args, server, success/failure, result — for later debugging or compliance | `log_audit_entry()` writes one row per call to `data/audit_log.db` (SQLite); `tool_args` (a dict) is serialized with `json.dumps()` since SQLite has no native dict column type |
| **Error isolation** | One broken server or failed tool call doesn't crash the whole system | Every server connection attempt and every tool call is wrapped in try/except; failures are logged and surfaced as a message, not an unhandled crash |

---

## 5. Logging & Observability — two distinct layers

This project actually has two separate logging mechanisms, serving
different purposes. They were built at different points and it's easy to
conflate them, so here's the clear split:

### Layer 1 — Per-server operational logs (console only, NOT persisted)

Each server (`hr_server.py`, `ticketing_server.py`, `knowledge_server.py`)
has its own named logger with a distinct prefix:

```python
logger = logging.getLogger("hr_server")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[HR_SERVER] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False
```

Every tool function logs its own activity, e.g.
`[HR_SERVER] get_employee called with employee_id=E002`. This is useful
for watching what's happening *inside one specific server*, in real time,
while debugging.

**Important limitation:** these logs only print to the console (stdout).
Once the terminal closes, they're gone — there's no file or database
behind them. This is genuinely a gap: if `ticketing_server.py` crashed
internally for some reason, the *client's* audit trail would only know
"the call failed," not *why* it failed internally — that detail lives
only in this ephemeral console output. This was flagged as an intentional
design gap during development, with persisting these local logs to a
file as a reasonable follow-up.

The client (`ollama_enterprise_client.py`) has its own logger in this
same category — `[ENTERPRISE_CLIENT]` — tracking discovery, routing
decisions, and tool call outcomes. Also console-only, also not persisted.

### Layer 2 — Centralized audit trail (persisted, SQLite)

Covered in section 4 above (`log_audit_entry()` / `data/audit_log.db`).
This is the durable, queryable record: one row per tool call, with the
user's question, the exact tool + arguments used, which server handled
it, success/failure, and the result — survives restarts, can be queried
after the fact.

### Why both exist (the actual interview answer)

- **Layer 1 (per-server console logs)** = local, real-time debugging of
  one service in isolation — "what is *this* server doing right now."
- **Layer 2 (audit trail)** = centralized, durable, cross-service record
  for reconstructing what happened in a full user interaction after the
  fact — "who did what, when, and did it work."

Real production systems typically have both: services log locally for
their own operators, while a central system (often something like an
audit table, or a proper observability platform) captures the
request-level story across services. This project intentionally mirrors
that split, even though Layer 1 isn't yet persisted to disk.



| If the user asks about... | It routes to... |
|---|---|
| A specific person, their department, their role/email | `hr_server.py` |
| Creating, checking, or updating a support ticket | `ticketing_server.py` |
| Leave policy, expense policy, onboarding process | `knowledge_server.py` |
| (None of the above — general chit-chat) | No tool called; the LLM answers directly from the conversation |

The LLM decides this routing itself, purely from each tool's `name` and
`description` (visible to it via `self.ollama_tools` / `self.all_tool_schemas`)
— nothing in the client hardcodes "HR questions go here."