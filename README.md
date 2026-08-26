# Enterprise MCP Demo

A multi-server MCP setup that mirrors how a real company would structure this:
each domain team (HR, Support, Docs/Knowledge) owns and runs its **own**
independent server, and a single orchestrating client discovers and routes
across all of them at runtime.

## Architecture

```
                     ┌──────────────────────────┐
                     │   enterprise_client.py     │   <- Host/Client (orchestrator)
                     │  - discovers tools          │
                     │  - routes LLM tool calls     │
                     │  - logs every call            │
                     │  - isolates failures          │
                     └───────────┬──────────────┘
                stdio  │           │  stdio        │  stdio
             ┌─────────┘           │                └─────────┐
             ▼                     ▼                          ▼
     ┌───────────────┐   ┌───────────────────┐   ┌─────────────────────┐
     │  hr_server.py   │   │ ticketing_server.py │   │ knowledge_server.py  │
     │  (CSV-backed)    │   │  (SQLite-backed)     │   │  (TF-IDF / RAG)        │
     │                  │   │                       │   │                        │
     │ get_employee     │   │ create_ticket          │   │ search_knowledge_base  │
     │ list_employees_  │   │ get_ticket             │   │                        │
     │  by_department   │   │ list_tickets           │   │                        │
     │                  │   │ update_ticket_status   │   │                        │
     └───────────────┘   └───────────────────┘   └─────────────────────┘
```

Each server is a **completely independent process** that only knows about
its own domain. None of them import or reference each other. The client is
the only component that knows all three exist -- exactly how you'd want it
in a real company, where the HR team's server and the Support team's server
are built and deployed by different teams entirely.

## Why this is "enterprise" and not a toy demo

| Concept | Where it shows up |
|---|---|
| **Dynamic tool discovery** | `_connect_one_server()` calls `session.list_tools()` -- no hardcoded tool list in the client |
| **Multi-server orchestration** | One LLM conversation, tools aggregated from 3 servers, routed transparently |
| **Tools vs. Resources** | `create_ticket` (side effect) vs. `hr://employees/all` (read-only resource) |
| **Persistence** | Ticketing uses real SQLite, not an in-memory dict -- state survives restarts |
| **Structured logging / audit trail** | Every tool call logs which server handled it, with what args, and whether it succeeded |
| **Error isolation** | A down server or failed tool call doesn't crash the client -- it's caught, logged, and surfaced gracefully |
| **RAG tie-in** | `knowledge_server.py` embeds docs (TF-IDF here, swappable for real embedding models), does cosine similarity search, and returns grounded context -- the same retrieval pattern as a production RAG pipeline |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python enterprise_client.py
```

## Example interactions

- `Look up employee E002`
- `List all employees in Engineering`
- `Create a high priority ticket for a broken VPN, requested by E002`
- `What's the status of ticket 1?`
- `What is our leave policy?`
- `How do expense reimbursements work?`

## Files

- `hr_server.py` -- employee lookups, CSV-backed
- `ticketing_server.py` -- support ticket CRUD, SQLite-backed
- `knowledge_server.py` -- RAG-style policy document search, TF-IDF-backed
- `enterprise_client.py` -- orchestrator: discovery, routing, logging, error handling
- `data/` -- employee CSV, tickets DB (created on first run), company policy docs

## Talking points for interviews

1. **"Why multiple servers instead of one big server?"**
   Mirrors org boundaries -- each team owns their domain, deploys independently,
   and can even run on separate infrastructure. The client doesn't care how
   many servers there are or what's inside them, only that they expose tools
   via the standard MCP interface.

2. **"How does the client know what tools exist?"**
   It doesn't hardcode them. On connect, it calls `list_tools()` on each
   server and builds a routing table (`tool_name -> which server owns it`)
   dynamically. Add a 4th server and the client needs zero code changes to
   use its tools.

3. **"What happens if a server goes down?"**
   `connect_all_servers()` wraps each connection attempt in a try/except --
   one server failing to start just means fewer tools are available, not a
   crash. Same for individual tool calls via `call_tool_routed()`.

4. **"How does this relate to RAG?"**
   `knowledge_server.py`'s `search_knowledge_base` tool *is* a RAG retrieval
   step, just wrapped as an MCP tool instead of being baked directly into
   the app. The LLM decides *when* to call it (e.g. only for policy
   questions), rather than every query hitting the vector store regardless
   of whether retrieval is even needed.
