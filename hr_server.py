"""
HR Server (MCP)
----------------
Simulates a company's HR system as an independent MCP server.
In a real company, the HR team would own and deploy this server on their
own infrastructure, with their own auth/access controls -- other teams
never touch the underlying data directly, they only call the exposed tools.

Transport: stdio (simplest for local/dev; enterprise setups often swap
this for streamable-http so the server can run as a standalone service).
"""

import csv
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("hr_server")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[HR_SERVER] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False

CSV_PATH = Path(__file__).parent / "data" / "employees.csv"

mcp = FastMCP(name="hr-server")


def _load_employees() -> list[dict]:
    """Reload from disk each call -- in a real system this would be a DB query."""
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@mcp.tool()
def get_employee(employee_id: str) -> dict:
    """
    Look up a single employee by their employee_id.
    Returns their name, department, role, and email.
    """
    logger.info(f"get_employee called with employee_id={employee_id}")
    employees = _load_employees()
    for emp in employees:
        if emp.get("employee_id") == employee_id:
            return emp
    return {"error": f"No employee found with id {employee_id}"}


@mcp.tool()
def list_employees_by_department(department: str) -> list[dict]:
    """
    List all employees belonging to a given department (e.g. 'Engineering', 'Sales').
    Case-insensitive match.
    """
    logger.info(f"list_employees_by_department called with department={department}")
    employees = _load_employees()
    matches = [
        e for e in employees
        if e.get("department", "").strip().lower() == department.strip().lower()
    ]
    if not matches:
        return [{"error": f"No employees found in department '{department}'"}]
    return matches


@mcp.resource("hr://employees/all")
def all_employees() -> str:
    """Read-only resource exposing the full employee roster as CSV text."""
    if not CSV_PATH.exists():
        return "No employee data available."
    return CSV_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    logger.info("Starting HR MCP server (stdio transport)...")
    mcp.run()
