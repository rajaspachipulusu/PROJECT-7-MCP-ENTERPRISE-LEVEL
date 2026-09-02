"""
Finance Server (MCP)
-----------------------
A fourth domain server, added to prove the multi-server orchestration
pattern (dynamic discovery + routing in enterprise_client.py /
ollama_enterprise_client.py) generalizes past 3 servers without any
client-side code changes -- only servers.yaml needs a new entry.

Simulates a lightweight expense-claim workflow: submit, look up, list,
and approve/reject claims. Backed by SQLite (not in-memory) so claim
state survives a server restart -- same reasoning as ticketing_server.py:
this server has real side effects (claims get created and their status
changes), so its state needs to persist, unlike the read-mostly hr_server
or the rebuild-from-source-files knowledge_server.

Config-driven logging follows the same pattern established for
knowledge_server.py: reads FINANCE_LOG_PATH from the environment (set by
the client from servers.yaml's log_path field), and adds a file handler
alongside the console handler when present.
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

LOG_PATH = os.environ.get("FINANCE_LOG_PATH")  # e.g. "logs/finance_server.log"

logger = logging.getLogger("finance_server")
logger.setLevel(logging.INFO)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("[FINANCE_SERVER] %(message)s"))
logger.addHandler(_console_handler)

if LOG_PATH:
    _log_file = Path(LOG_PATH)
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [FINANCE_SERVER] %(message)s"))
    logger.addHandler(_file_handler)

logger.propagate = False

DB_PATH = Path(__file__).parent / "data" / "finance.db"
VALID_STATUSES = {"pending", "approved", "rejected"}
VALID_CATEGORIES = {"travel", "meals", "software", "training", "equipment", "other"}

mcp = FastMCP(name="finance-server")


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expense_claims (
                claim_id     TEXT PRIMARY KEY,
                employee_id  TEXT NOT NULL,
                category     TEXT NOT NULL,
                amount       REAL NOT NULL,
                description  TEXT,
                status       TEXT NOT NULL DEFAULT 'pending',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _next_claim_id(conn: sqlite3.Connection) -> str:
    cursor = conn.execute("SELECT COUNT(*) FROM expense_claims")
    count = cursor.fetchone()[0]
    return f"EXP{count + 1:04d}"


@mcp.tool()
def submit_expense_claim(employee_id: str, category: str, amount: float, description: str = "") -> dict:
    """
    Submit a new expense claim for an employee. category must be one of:
    travel, meals, software, training, equipment, other. Returns the
    created claim including its generated claim_id and 'pending' status.
    """
    if category not in VALID_CATEGORIES:
        return {"error": f"Invalid category '{category}'. Must be one of: {sorted(VALID_CATEGORIES)}"}
    if amount <= 0:
        return {"error": "amount must be a positive number."}

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        claim_id = _next_claim_id(conn)
        conn.execute(
            """INSERT INTO expense_claims
               (claim_id, employee_id, category, amount, description, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (claim_id, employee_id, category, amount, description, now, now),
        )
        conn.commit()

    logger.info(f"submit_expense_claim: {claim_id} for {employee_id} ({category}, {amount})")
    return {
        "claim_id": claim_id, "employee_id": employee_id, "category": category,
        "amount": amount, "description": description, "status": "pending",
        "created_at": now, "updated_at": now,
    }


@mcp.tool()
def get_expense_claim(claim_id: str) -> dict:
    """Look up a single expense claim by its claim_id."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM expense_claims WHERE claim_id = ?", (claim_id,)).fetchone()

    if row is None:
        return {"error": f"No expense claim found with id '{claim_id}'."}
    return dict(row)


@mcp.tool()
def list_expense_claims(employee_id: str = "", status: str = "") -> list[dict]:
    """
    List expense claims, optionally filtered by employee_id and/or status
    (pending, approved, rejected). Leave a filter blank to not filter on it.
    """
    query = "SELECT * FROM expense_claims WHERE 1=1"
    params: list = []
    if employee_id:
        query += " AND employee_id = ?"
        params.append(employee_id)
    if status:
        if status not in VALID_STATUSES:
            return [{"error": f"Invalid status '{status}'. Must be one of: {sorted(VALID_STATUSES)}"}]
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


@mcp.tool()
def update_claim_status(claim_id: str, status: str) -> dict:
    """
    Approve or reject an expense claim. status must be 'approved' or
    'rejected'. This is a state-changing action, logged like any other
    tool call and reflected immediately in subsequent get/list calls.
    """
    if status not in {"approved", "rejected"}:
        return {"error": f"Invalid status '{status}'. Must be 'approved' or 'rejected'."}

    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "UPDATE expense_claims SET status = ?, updated_at = ? WHERE claim_id = ?",
            (status, now, claim_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            return {"error": f"No expense claim found with id '{claim_id}'."}

    logger.info(f"update_claim_status: {claim_id} -> {status}")
    return {"claim_id": claim_id, "status": status, "updated_at": now}


@mcp.resource("finance://claims/summary")
def claims_summary() -> str:
    """Read-only resource: count of claims per status, for a quick overview."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*), COALESCE(SUM(amount), 0) FROM expense_claims GROUP BY status"
        ).fetchall()

    if not rows:
        return "No expense claims on file."
    return "\n".join(f"{status}: {count} claim(s), total {total:.2f}" for status, count, total in rows)


_init_db()

if __name__ == "__main__":
    logger.info("Starting Finance MCP server (stdio transport)...")
    mcp.run()