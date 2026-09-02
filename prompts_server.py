"""
Prompts Server (MCP)
------------------------
A server dedicated to demonstrating MCP's third primitive: Prompts.

Quick contrast with the other two primitives, all present somewhere in
this project by now:

  TOOLS      -- the MODEL decides when to call them, mid-reasoning.
                (e.g. search_knowledge_base, submit_expense_claim)
  RESOURCES  -- the CLIENT/application pulls them in, read-only, not
                something the model autonomously decides to fetch.
                (e.g. knowledge://docs/list, finance://claims/summary)
  PROMPTS    -- the USER explicitly selects them, like a slash-command
                in a chat UI. The server returns a ready-made prompt
                (or short message sequence); the user picks WHEN to use
                it, not the model. THIS FILE.

Why Prompts are useful here specifically: some tasks are common and
repetitive enough (onboarding a new hire, reviewing a pending expense
claim against policy) that typing free-text and hoping the LLM figures
out the right tool sequence is worse than just handing the user a
pre-built, parameterized starting point. A prompt template turns "I hope
the model figures out what I mean" into "the user picked exactly what
they meant."

IMPORTANT CAVEAT (worth knowing, not hiding): ollama_enterprise_client.py
as currently written only implements the TOOLS half of MCP -- its chat
loop discovers and calls tools, but has no code path for listing prompts
or letting a user select one (that would need a different UI affordance,
like a slash-command menu, not a free-text chat box). Registering this
server in servers.yaml makes its PROMPTS discoverable by any MCP client
that supports prompts (e.g. Claude Desktop, or a future version of this
client), but the current chat client will only see and use its prompts
if extended to support prompt selection. This file demonstrates the
primitive correctly; wiring a selection UI into the client is a
separate, not-yet-done piece of work.
"""

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("prompts_server")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[PROMPTS_SERVER] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False

mcp = FastMCP(name="prompts-server")


@mcp.prompt()
def onboard_new_hire(employee_name: str, department: str) -> str:
    """
    Guided checklist prompt for onboarding a new hire. This is a good
    example of a prompt template's real value: onboarding touches THREE
    different servers (hr, finance, knowledge) in a specific order, and
    a user picking this template gets that whole sequence framed
    correctly in one shot, rather than needing to know which tools exist
    across which servers.
    """
    logger.info(f"onboard_new_hire prompt requested: {employee_name} ({department})")
    return (
        f"Help me onboard a new hire: {employee_name}, joining the {department} "
        f"department.\n\n"
        f"Please walk through the following in order:\n"
        f"1. Look up whether an employee record already exists for {employee_name} "
        f"(they should not yet be in the system).\n"
        f"2. Summarize what the Employee Handbook's Onboarding Process section "
        f"says the first 90 days should look like, so I know what to schedule.\n"
        f"3. List anything from the IT and Data Security Policy that IT will need "
        f"to provision before day 1 (laptop, access, security training).\n"
        f"4. Note whether a home office setup allowance or equipment policy "
        f"applies, in case {employee_name} will be working remotely.\n\n"
        f"Summarize all of this as a single onboarding checklist I can hand to "
        f"the hiring manager."
    )


@mcp.prompt()
def review_expense_claim(claim_id: str) -> str:
    """
    Guided prompt for reviewing a specific pending expense claim against
    company policy before approving or rejecting it. Ties the finance
    domain (the claim itself) to the knowledge domain (the policy caps
    it should be checked against) in one structured request.
    """
    logger.info(f"review_expense_claim prompt requested: {claim_id}")
    return (
        f"I need to decide whether to approve or reject expense claim {claim_id}.\n\n"
        f"Please:\n"
        f"1. Look up the claim details (employee, category, amount, description).\n"
        f"2. Check the Expense and Reimbursement Policy section of the Employee "
        f"Handbook for the relevant category cap and any approval thresholds that "
        f"apply.\n"
        f"3. Tell me clearly whether the claimed amount is within policy limits, "
        f"and flag anything that looks unusual (e.g. missing description, "
        f"category/amount mismatch).\n"
        f"4. Give me a one-line recommendation: approve, reject, or escalate for "
        f"secondary review -- and why.\n\n"
        f"Do not change the claim's status yourself; this is a review only, I will "
        f"make the final call."
    )


@mcp.prompt()
def weekly_finance_digest() -> str:
    """
    No-argument prompt for a recurring weekly summary -- a good example
    of a prompt template that needs no parameters at all, just a
    consistent, repeatable framing for something a user runs often.
    """
    logger.info("weekly_finance_digest prompt requested")
    return (
        "Give me this week's finance digest:\n\n"
        "1. How many expense claims are currently pending, and what's the total "
        "amount pending approval?\n"
        "2. Are any pending claims above the secondary-approval threshold "
        "described in the Expense and Reimbursement Policy?\n"
        "3. Summarize anything that needs my attention versus anything routine "
        "that can wait."
    )


if __name__ == "__main__":
    logger.info("Starting Prompts MCP server (stdio transport)...")
    mcp.run()