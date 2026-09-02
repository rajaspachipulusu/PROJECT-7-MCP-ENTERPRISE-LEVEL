"""
LLMOps Summary — quick terminal observability report
---------------------------------------------------------
Reads data/audit_log.db (already being populated by
ollama_enterprise_client.py's log_audit_entry()) and prints a summary:
total calls, today's calls, success rate, per-server breakdown, most-used
tools, and recent failures.

This is deliberately the smallest useful version of an LLMOps dashboard --
no web server, no new dependencies, no schema changes. It answers three
questions any real production system needs answered: is it being used,
is it working, and what broke most recently.

Run with: python llmops_summary.py
Optional: python llmops_summary.py --db path/to/audit_log.db
"""

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"No audit log found at '{db_path}'. Run the assistant first to generate some data.")
        raise SystemExit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def print_header(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick LLMOps summary from audit_log.db")
    parser.add_argument("--db", default="data/audit_log.db", help="Path to audit_log.db")
    args = parser.parse_args()

    conn = connect(Path(args.db))

    total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    if total == 0:
        print("Audit log exists but is empty -- no tool calls have been logged yet.")
        return

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_count = conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE timestamp LIKE ?", (f"{today}%",)
    ).fetchone()[0]

    success_count = conn.execute("SELECT COUNT(*) FROM audit_log WHERE success = 1").fetchone()[0]
    fail_count = total - success_count
    success_rate = (success_count / total) * 100

    print("=" * 50)
    print("  LLMOps Summary")
    print("=" * 50)

    print_header("Overview")
    print(f"  Total tool calls (all-time): {total}")
    print(f"  Tool calls today:            {today_count}")
    print(f"  Overall success rate:        {success_rate:.1f}%  ({success_count} ok / {fail_count} failed)")

    print_header("By Server")
    rows = conn.execute(
        """
        SELECT server_name,
               COUNT(*) AS total,
               SUM(success) AS ok,
               COUNT(*) - SUM(success) AS failed
        FROM audit_log
        GROUP BY server_name
        ORDER BY total DESC
        """
    ).fetchall()
    print(f"  {'Server':<15}{'Calls':<10}{'Success':<10}{'Failed':<10}{'Rate':<8}")
    for row in rows:
        rate = (row["ok"] / row["total"]) * 100 if row["total"] else 0
        print(f"  {row['server_name']:<15}{row['total']:<10}{row['ok']:<10}{row['failed']:<10}{rate:.0f}%")

    print_header("Most-Called Tools (top 5)")
    rows = conn.execute(
        """
        SELECT tool_name, COUNT(*) AS calls
        FROM audit_log
        GROUP BY tool_name
        ORDER BY calls DESC
        LIMIT 5
        """
    ).fetchall()
    for row in rows:
        print(f"  {row['tool_name']:<30}{row['calls']} call(s)")

    print_header("Recent Failures (last 5)")
    rows = conn.execute(
        """
        SELECT timestamp, server_name, tool_name, result_or_error
        FROM audit_log
        WHERE success = 0
        ORDER BY timestamp DESC
        LIMIT 5
        """
    ).fetchall()
    if not rows:
        print("  None -- every logged tool call has succeeded so far.")
    else:
        for row in rows:
            error_preview = (row["result_or_error"] or "")[:80]
            print(f"  [{row['timestamp']}] {row['server_name']}/{row['tool_name']}")
            print(f"      {error_preview}")

    print()
    conn.close()


if __name__ == "__main__":
    main()