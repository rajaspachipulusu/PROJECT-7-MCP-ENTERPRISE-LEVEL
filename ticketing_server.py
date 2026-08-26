"""
Ticketing Server (MCP)
-----------------------
Simulates an internal support/ITSM system (like a mini Jira/ServiceNow).
Backed by SQLite so state actually persists across restarts -- unlike a
toy in-memory dict, this behaves like a real service would.

Demonstrates tools WITH SIDE EFFECTS (create/update), not just read-only
lookups -- an important distinction to be able to explain in an interview:
MCP "tools" can mutate state, "resources" are meant to be read-only.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("ticketing_server")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[TICKETING_SERVER] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False

DB_PATH = Path(__file__).parent / "data" / "tickets.db"

mcp = FastMCP(name="ticketing-server")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                description TEXT,
                priority    TEXT NOT NULL DEFAULT 'medium',
                status      TEXT NOT NULL DEFAULT 'open',
                created_by  TEXT,
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.commit()


@mcp.tool()
def create_ticket(title: str, description: str, priority: str, created_by: str) -> dict:
    """
    Create a new support ticket.
    priority should be one of: 'low', 'medium', 'high', 'critical'.
    Returns the newly created ticket including its assigned ticket_id.
    """
    logger.info(f"create_ticket called: title={title!r}, priority={priority}")
    if priority not in {"low", "medium", "high", "critical"}:
        return {"error": f"Invalid priority '{priority}'. Must be low/medium/high/critical."}

    created_at = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO tickets (title, description, priority, status, created_by, created_at)
               VALUES (?, ?, ?, 'open', ?, ?)""",
            (title, description, priority, created_by, created_at),
        )
        conn.commit()
        ticket_id = cur.lastrowid

    return {
        "ticket_id": ticket_id,
        "title": title,
        "priority": priority,
        "status": "open",
        "created_by": created_by,
        "created_at": created_at,
    }


@mcp.tool()
def get_ticket(ticket_id: int) -> dict:
    """Fetch a single ticket by its ticket_id."""
    logger.info(f"get_ticket called: ticket_id={ticket_id}")
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    if row is None:
        return {"error": f"No ticket found with id {ticket_id}"}
    return dict(row)


@mcp.tool()
def list_tickets(status: str = "") -> list[dict]:
    """
    List tickets. If status is given ('open', 'in_progress', 'closed'),
    filters to that status; otherwise returns all tickets.
    """
    logger.info(f"list_tickets called: status_filter={status!r}")
    with _get_conn() as conn:
        if status:
            rows = conn.execute("SELECT * FROM tickets WHERE status = ?", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tickets").fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def update_ticket_status(ticket_id: int, new_status: str) -> dict:
    """
    Update a ticket's status. new_status should be one of:
    'open', 'in_progress', 'closed'.
    """
    logger.info(f"update_ticket_status called: ticket_id={ticket_id}, new_status={new_status}")
    if new_status not in {"open", "in_progress", "closed"}:
        return {"error": f"Invalid status '{new_status}'."}

    with _get_conn() as conn:
        cur = conn.execute(
            "UPDATE tickets SET status = ? WHERE ticket_id = ?", (new_status, ticket_id)
        )
        conn.commit()
        if cur.rowcount == 0:
            return {"error": f"No ticket found with id {ticket_id}"}

    return {"ticket_id": ticket_id, "status": new_status, "updated": True}


_init_db()

if __name__ == "__main__":
    logger.info("Starting Ticketing MCP server (stdio transport)...")
    mcp.run()
