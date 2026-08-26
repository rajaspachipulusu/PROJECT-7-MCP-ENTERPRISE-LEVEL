"""
Enterprise MCP Client / Host -- OLLAMA (local LLM) version
--------------------------------------------------------------
Identical orchestration architecture to enterprise_client.py (discovery,
routing, logging, error isolation) -- the ONLY thing that changes is which
LLM decides which tool to call. This swaps Anthropic's hosted API for a
locally-running Ollama model (default: qwen3:8b), so you can develop and
test for free with zero API cost.

Talking point for interviews: the MCP *servers* (hr_server.py,
ticketing_server.py, knowledge_server.py) needed ZERO changes to work with
a different LLM provider. Only the client's "ask the LLM which tool to
call" logic changes -- because MCP standardizes the tool-calling interface,
but different LLM providers still have different request/response formats
for tool calling itself (Anthropic's content-block format vs Ollama/OpenAI-
style function-call format). This is a legitimate architecture pattern:
provider abstraction.

Prerequisites (on YOUR machine, not in a sandbox):
  1. Ollama installed and running: https://ollama.com
  2. Model pulled:  ollama pull qwen3:8b
  3. pip install ollama mcp==1.9.4 scikit-learn
"""

import asyncio
import json
import logging
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path

import ollama
import os
import sqlite3
import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ENTERPRISE_CLIENT] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OLLAMA_MODEL = "qwen3:8b"  # change if you're using a different local model

SERVER_REGISTRY = {
    "hr": BASE_DIR / "hr_server.py",
    "ticketing": BASE_DIR / "ticketing_server.py",
    "knowledge": BASE_DIR / "knowledge_server.py",
}


def load_config() -> dict:
    """
    Reads the YAML config path from the MCP_CONFIG_PATH environment
    variable (defaulting to servers.yaml next to this script), parses it,
    and returns {"model": ..., "servers": [enabled server dicts only]}.

    Design decisions, and WHY:
      - Missing config file -> fall back to the hardcoded SERVER_REGISTRY
        above, so the project still runs with zero setup. This is a
        reasonable default: "no config provided" isn't necessarily a
        mistake.
      - Malformed YAML (file exists but fails to parse) -> FAIL LOUDLY
        by re-raising. This is deliberately NOT swallowed into an empty
        config, because "config exists but is broken" almost always means
        someone made a typo and silently limping along with zero servers
        would hide that mistake instead of surfacing it.
      - A server entry missing the 'enabled' key defaults to enabled=True
        (fail-open for missing fields, not fail-closed) so a forgotten
        field doesn't silently disable a whole server.
    """
    config_path = os.environ.get("MCP_CONFIG_PATH", str(BASE_DIR / "servers.yaml"))

    if not Path(config_path).exists():
        logger.warning(
            f"No config file found at '{config_path}'. "
            f"Falling back to hardcoded default server list."
        )
        return {
            "model": OLLAMA_MODEL,
            "servers": [
                {"name": name, "path": str(path), "enabled": True}
                for name, path in SERVER_REGISTRY.items()
            ],
        }

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        # Fail loudly: a broken config file is a real mistake, not a
        # reason to silently run with zero servers.
        raise RuntimeError(
            f"Config file at '{config_path}' exists but could not be parsed as YAML: {exc}"
        ) from exc

    enabled_servers = [s for s in config.get("servers", []) if s.get("enabled", True)]
    skipped = [s["name"] for s in config.get("servers", []) if not s.get("enabled", True)]
    if skipped:
        logger.info(f"Skipping disabled server(s) from config: {skipped}")

    return {
        "model": config.get("model", OLLAMA_MODEL),
        "servers": enabled_servers,
    }


AUDIT_DB_PATH = BASE_DIR / "data" / "audit_log.db"


def _init_audit_db() -> None:
    """
    Create the audit_log table if it doesn't already exist.
    Called once when the client starts up. Uses "CREATE TABLE IF NOT
    EXISTS" so it's safe to call every run -- it won't wipe existing
    history, it just makes sure the table is there.
    """
    AUDIT_DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(AUDIT_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT NOT NULL,
                user_question    TEXT NOT NULL,
                tool_name        TEXT NOT NULL,
                tool_args        TEXT NOT NULL,
                server_name      TEXT NOT NULL,
                success          INTEGER NOT NULL,
                result_or_error  TEXT
            )
            """
        )
        conn.commit()


def log_audit_entry(
    user_question: str,
    tool_name: str,
    tool_args: dict,
    server_name: str,
    success: bool,
    result_or_error: str,
) -> None:
    """
    Write one row to the audit log -- one row per tool call.

    KEY DETAIL: tool_args is a Python dict, e.g. {"employee_id": "E002"}.
    SQLite columns only store text/numbers/blobs -- there's no native
    "dict" column type. So we serialize it to a JSON string with
    json.dumps() before inserting, exactly the same problem (and same
    fix) as when ticketing_server.py returns a Python dict as a text
    block over the MCP protocol -- structured data has to become text
    to cross a boundary that only understands text/columns.

    success is a Python bool, but SQLite doesn't have a native boolean
    type either -- it stores True/False as integers 1/0 automatically,
    so int(success) makes that explicit rather than relying on SQLite's
    implicit conversion.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(AUDIT_DB_PATH) as conn:
        conn.execute(
            """INSERT INTO audit_log
               (timestamp, user_question, tool_name, tool_args, server_name, success, result_or_error)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                user_question,
                tool_name,
                json.dumps(tool_args),
                server_name,
                int(success),
                result_or_error,
            ),
        )
        conn.commit()


def _mcp_schema_to_ollama_tool(tool_schema: dict) -> dict:
    """
    Convert an MCP tool schema (name/description/input_schema) into the
    OpenAI-style function-calling format that Ollama expects.
    This is the ONLY translation layer needed between MCP and Ollama.
    """
    return {
        "type": "function",
        "function": {
            "name": tool_schema["name"],
            "description": tool_schema["description"] or "",
            "parameters": tool_schema["input_schema"],
        },
    }


class EnterpriseMCPClient:
    def __init__(self, model: str):
        self.model = model
        self.exit_stack = AsyncExitStack()
        self.tool_routing: dict[str, tuple[ClientSession, str]] = {}
        self.all_tool_schemas: list[dict] = []  # MCP-native schemas
        self.ollama_tools: list[dict] = []       # Ollama/OpenAI-format schemas
        self.messages: list[dict] = []           # persistent conversation history
        self.current_question: str = ""          # tracked so call_tool_routed can log it
        _init_audit_db()

    async def connect_all_servers(self, server_configs: list[dict]) -> None:
        for server in server_configs:
            server_name = server["name"]
            script_path = BASE_DIR / server["path"]
            try:
                await self._connect_one_server(server_name, script_path)
            except Exception as exc:
                logger.error(f"Failed to connect to '{server_name}' server: {exc}")

        logger.info(
            f"Discovery complete. {len(self.all_tool_schemas)} tools available "
            f"across {len(set(s for _, s in self.tool_routing.values()))} server(s)."
        )

    async def _connect_one_server(self, server_name: str, script_path: Path) -> None:
        if not script_path.exists():
            raise FileNotFoundError(f"Server script not found: {script_path}")

        params = StdioServerParameters(command=sys.executable, args=[str(script_path)])
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        response = await session.list_tools()
        for tool in response.tools:
            self.tool_routing[tool.name] = (session, server_name)
            schema = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            self.all_tool_schemas.append(schema)
            self.ollama_tools.append(_mcp_schema_to_ollama_tool(schema))
            logger.info(f"Discovered tool '{tool.name}' on server '{server_name}'")

    async def call_tool_routed(self, tool_name: str, tool_args: dict) -> str:
        if tool_name not in self.tool_routing:
            logger.error(f"No server owns tool '{tool_name}'")
            return f"Error: tool '{tool_name}' is not available."

        session, server_name = self.tool_routing[tool_name]
        logger.info(f"Routing call '{tool_name}({tool_args})' -> server '{server_name}'")

        try:
            result = await session.call_tool(tool_name, tool_args)
            logger.info(f"Tool '{tool_name}' on '{server_name}' succeeded")
            result_text = "\n".join(
                block.text for block in result.content if hasattr(block, "text")
            )
            log_audit_entry(
                user_question=self.current_question,
                tool_name=tool_name,
                tool_args=tool_args,
                server_name=server_name,
                success=True,
                result_or_error=result_text,
            )
            return result_text
        except Exception as exc:
            logger.error(f"Tool '{tool_name}' on '{server_name}' failed: {exc}")
            error_text = f"Error calling '{tool_name}' on server '{server_name}': {exc}"
            log_audit_entry(
                user_question=self.current_question,
                tool_name=tool_name,
                tool_args=tool_args,
                server_name=server_name,
                success=False,
                result_or_error=error_text,
            )
            return error_text

    async def ask(self, user_question: str) -> str:
        """
        Send the question to the local Ollama model along with all
        discovered tools AND the full conversation history so far
        (self.messages persists across calls -- this is what gives the
        assistant memory of earlier turns, like "that ticket" referring
        to a ticket created two messages ago).
        Ollama returns tool_calls on the response message when it decides
        a tool is needed; we loop until it returns a plain text answer.
        """
        self.messages.append({"role": "user", "content": user_question})
        self.current_question = user_question

        response = ollama.chat(model=self.model, messages=self.messages, tools=self.ollama_tools)

        # Loop while the model keeps requesting tool calls
        while response.message.tool_calls:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": response.message.content or "",
                    "tool_calls": response.message.tool_calls,
                }
            )

            for call in response.message.tool_calls:
                tool_name = call.function.name
                tool_args = call.function.arguments or {}
                result_text = await self.call_tool_routed(tool_name, dict(tool_args))
                self.messages.append(
                    {
                        "role": "tool",
                        "content": result_text,
                    }
                )

            response = ollama.chat(model=self.model, messages=self.messages, tools=self.ollama_tools)

        # Persist the final assistant answer too, so future turns see it
        self.messages.append({"role": "assistant", "content": response.message.content})
        return response.message.content

    async def close(self) -> None:
        await self.exit_stack.aclose()


async def main() -> None:
    config = load_config()
    client = EnterpriseMCPClient(model=config["model"])
    try:
        await client.connect_all_servers(config["servers"])
        print(f"\nEnterprise MCP Assistant ready (local model: {client.model}).")
        connected = ", ".join(s["name"] for s in config["servers"])
        print(f"Connected servers: {connected}")
        print("Try asking things like:")
        print("  - 'Look up employee E002'")
        print("  - 'Create a high priority ticket for a broken VPN, requested by E002'")
        print("  - 'What is our leave policy?'")
        print("Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            if not user_input:
                continue
            answer = await client.ask(user_input)
            print(f"\nAssistant: {answer}\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())