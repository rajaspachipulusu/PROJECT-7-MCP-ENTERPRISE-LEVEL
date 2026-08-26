"""
Enterprise MCP Client / Host
------------------------------
This is the ORCHESTRATOR: the "Host" application from the MCP vocabulary
(Host = the AI application; Client = connects Host to servers).

Real-world analogy: this is like a company-wide AI assistant that can
talk to the HR team's server, the Support team's server, and the
Knowledge/Docs team's server -- each independently owned and deployed,
none of them aware of each other. This client is the only thing that
knows about all three.

What makes this "enterprise" rather than a toy demo:
  1. Dynamic tool discovery  -- no hardcoded tool lists; each server is
     asked "what can you do?" at connect time.
  2. Multi-server orchestration -- one LLM conversation, many backend
     servers, transparently routed.
  3. Structured logging       -- every tool call is logged with which
     server handled it and what happened (a basic audit trail).
  4. Error isolation          -- if one server is down or a tool call
     fails, the whole system doesn't crash; the failure is reported
     back gracefully and other servers keep working.

Requires an Anthropic API key in the environment as ANTHROPIC_API_KEY.
"""

import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ENTERPRISE_CLIENT] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# Registry of servers this client knows how to connect to.
# In a real company, this list might come from a config file or a
# service registry rather than being hardcoded here.
SERVER_REGISTRY = {
    "hr": BASE_DIR / "hr_server.py",
    "ticketing": BASE_DIR / "ticketing_server.py",
    "knowledge": BASE_DIR / "knowledge_server.py",
}


class EnterpriseMCPClient:
    def __init__(self):
        self.anthropic = Anthropic()
        self.exit_stack = AsyncExitStack()
        # tool_name -> (ClientSession, server_name)  -- built during discovery
        self.tool_routing: dict[str, tuple[ClientSession, str]] = {}
        self.all_tool_schemas: list[dict] = []

    async def connect_all_servers(self) -> None:
        """Connect to every server in the registry and discover its tools."""
        for server_name, script_path in SERVER_REGISTRY.items():
            try:
                await self._connect_one_server(server_name, script_path)
            except Exception as exc:
                # ERROR ISOLATION: one server failing to start doesn't
                # take down the whole client -- log it and move on.
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

        # DISCOVERY: ask the server what tools it offers, rather than
        # hardcoding tool names/schemas in the client.
        response = await session.list_tools()
        for tool in response.tools:
            self.tool_routing[tool.name] = (session, server_name)
            self.all_tool_schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
            )
            logger.info(f"Discovered tool '{tool.name}' on server '{server_name}'")

    async def call_tool_routed(self, tool_name: str, tool_args: dict) -> str:
        """Route a tool call to whichever server actually owns that tool."""
        if tool_name not in self.tool_routing:
            logger.error(f"No server owns tool '{tool_name}'")
            return f"Error: tool '{tool_name}' is not available."

        session, server_name = self.tool_routing[tool_name]
        logger.info(f"Routing call '{tool_name}({tool_args})' -> server '{server_name}'")

        try:
            result = await session.call_tool(tool_name, tool_args)
            logger.info(f"Tool '{tool_name}' on '{server_name}' succeeded")
            return "\n".join(
                block.text for block in result.content if hasattr(block, "text")
            )
        except Exception as exc:
            # ERROR ISOLATION: a failed tool call becomes a message the
            # LLM can react to, not a crash.
            logger.error(f"Tool '{tool_name}' on '{server_name}' failed: {exc}")
            return f"Error calling '{tool_name}' on server '{server_name}': {exc}"

    async def ask(self, user_question: str) -> str:
        """
        Send the user's question to the LLM along with ALL discovered
        tools (across all servers). The LLM decides which tool(s) to
        call; we route each call to the right server transparently.
        """
        messages = [{"role": "user", "content": user_question}]

        response = self.anthropic.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=messages,
            tools=self.all_tool_schemas,
        )

        while response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    result_text = await self.call_tool_routed(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
            response = self.anthropic.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=messages,
                tools=self.all_tool_schemas,
            )

        final_text = "".join(b.text for b in response.content if b.type == "text")
        return final_text

    async def close(self) -> None:
        await self.exit_stack.aclose()


async def main() -> None:
    client = EnterpriseMCPClient()
    try:
        await client.connect_all_servers()
        print("\nEnterprise MCP Assistant ready. Connected servers: hr, ticketing, knowledge")
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in your environment before running this client.")
        sys.exit(1)
    asyncio.run(main())
