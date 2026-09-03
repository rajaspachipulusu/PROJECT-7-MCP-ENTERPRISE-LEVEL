"""
Test script for prompts_server.py
-------------------------------------
Since ollama_enterprise_client.py's chat loop doesn't have a "pick a
prompt from a menu" code path (it only implements Tools discovery/
calling), this script is how you actually exercise the Prompts
primitive directly: it launches prompts_server.py as a subprocess (same
way the real client does) and calls list_prompts() / get_prompt() over
the real MCP protocol -- no Ollama required, since this server has no
embedding or chat dependency at all.

Run with: python test_prompts_server.py
"""

import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).parent / "prompts_server.py"


async def main() -> None:
    params = StdioServerParameters(command="python3", args=[str(SERVER_PATH)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- Discovery: list every prompt this server offers ---
            prompts = await session.list_prompts()
            print(f"Discovered {len(prompts.prompts)} prompt(s):\n")
            for p in prompts.prompts:
                args = [a.name for a in (p.arguments or [])]
                print(f"  - {p.name}({', '.join(args)})")
            print()

            # --- Call 1: onboard_new_hire, with arguments ---
            print("=" * 70)
            print("get_prompt: onboard_new_hire")
            print("=" * 70)
            result = await session.get_prompt(
                "onboard_new_hire",
                {"employee_name": "Priya Nair", "department": "Data Engineering"},
            )
            for msg in result.messages:
                print(f"[{msg.role}]\n{msg.content.text}\n")

            # --- Call 2: review_expense_claim, with arguments ---
            print("=" * 70)
            print("get_prompt: review_expense_claim")
            print("=" * 70)
            result = await session.get_prompt("review_expense_claim", {"claim_id": "EXP0001"})
            for msg in result.messages:
                print(f"[{msg.role}]\n{msg.content.text}\n")

            # --- Call 3: weekly_finance_digest, no arguments ---
            print("=" * 70)
            print("get_prompt: weekly_finance_digest")
            print("=" * 70)
            result = await session.get_prompt("weekly_finance_digest", {})
            for msg in result.messages:
                print(f"[{msg.role}]\n{msg.content.text}\n")


if __name__ == "__main__":
    asyncio.run(main())