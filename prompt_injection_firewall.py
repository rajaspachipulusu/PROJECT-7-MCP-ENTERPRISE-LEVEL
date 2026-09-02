"""
Prompt Injection Firewall
----------------------------
A lightweight, reusable content-scanning layer for text that will be
retrieved from external documents and fed into an LLM's context (RAG
chunks, tool outputs, etc.).

WHAT THIS IS: a heuristic tripwire. It pattern-matches on phrasing that
commonly appears in indirect prompt injection attempts -- text addressed
directly to "the AI" / "the assistant", imperative instructions embedded
in what should be descriptive content, requests to conceal something from
the user, and references to actual tool names that shouldn't appear in
ordinary document content.

WHAT THIS IS NOT: a complete defense on its own. Pattern matching can be
evaded by rephrasing (this is the same cat-and-mouse dynamic as spam
filtering or WAF rules). This is why it's paired with a structural
defense in ollama_enterprise_client.py (wrapping ALL tool output --
whether flagged or not -- in an explicit untrusted-data boundary) rather
than relied on alone. Defense in depth: this layer catches the obvious
cases early and logs them for review; the structural layer is what
actually holds even against payloads this layer misses.
"""

import re

# Patterns are intentionally broad-but-cheap regexes, not a full NLP
# classifier -- fast enough to run on every retrieved chunk with no
# extra model call. Grouped by what they're trying to catch.
SUSPICIOUS_PATTERNS = [
    # Direct address to the AI/assistant/system -- legitimate documents
    # describe policies, they don't talk TO an AI.
    (r"\b(AI|system)\s+(assistant|note|instruction)\b", "direct address to AI/system"),
    (r"\bnote to (the )?(AI|assistant|model)\b", "direct address to AI"),
    (r"\bas an? (AI|assistant|language model)\b", "AI self-reference framing"),

    # Instructions to conceal something from the user -- a hallmark of
    # injection attempts specifically (legitimate content has no reason
    # to instruct secrecy from the person asking the question).
    (r"\bdo not (mention|tell|inform|disclose)\b.{0,40}\buser\b", "instruction to conceal from user"),
    (r"\bwithout (telling|informing|notifying)\b.{0,40}\buser\b", "instruction to conceal from user"),

    # Imperative commands directed at "you" in a system-instruction voice
    # ("before answering...", "first call...", "you must...").
    (r"\bbefore answering\b.{0,60}\b(call|run|execute|invoke)\b", "pre-answer action instruction"),
    (r"\byou (must|should) (now |immediately )?(call|run|execute|invoke)\b", "imperative tool-call instruction"),
    (r"\bignore (previous|prior|earlier|all)\s+(instructions|context|prompts?)\b", "instruction override attempt"),

    # References to actual internal tool/function names -- legitimate
    # policy content has no reason to name a backend function.
    (r"\b(update_claim_status|approve_expense|submit_expense_claim|reindex_knowledge_base)\s*\(", "tool name referenced as a call"),
    (r"\bcall the \w+ tool\b", "explicit tool invocation request"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in SUSPICIOUS_PATTERNS]


def scan_text(text: str) -> dict:
    """
    Scan a single piece of text for injection-like patterns. Returns a
    dict with whether anything was flagged and which pattern labels
    matched, so callers can log or make a policy decision without this
    module needing to know what that decision should be.
    """
    matches = []
    for pattern, label in _COMPILED:
        if pattern.search(text):
            matches.append(label)

    return {
        "flagged": len(matches) > 0,
        "matched_patterns": matches,
        "risk_score": min(len(matches) / 3, 1.0),  # crude 0-1 scale, caps at 3+ matches
    }


def sanitize_chunk(text: str) -> tuple[str, dict]:
    """
    Scan a retrieved chunk and, if flagged, strip out the specific
    sentence(s) that triggered a match rather than discarding the whole
    chunk -- a policy document might be 95% legitimate with one poisoned
    sentence, and dropping the entire chunk would also remove real,
    useful content the user actually asked about.

    Returns (sanitized_text, scan_result). scan_result always reflects
    what was found in the ORIGINAL text, so callers can log/audit even
    when the sanitized text looks clean afterward.
    """
    scan_result = scan_text(text)
    if not scan_result["flagged"]:
        return text, scan_result

    # Split on sentence boundaries and drop only the sentences that
    # individually match a pattern, keeping legitimate surrounding text.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean_sentences = []
    for sentence in sentences:
        sentence_scan = scan_text(sentence)
        if sentence_scan["flagged"]:
            clean_sentences.append("[REDACTED: content removed by prompt injection firewall]")
        else:
            clean_sentences.append(sentence)

    return " ".join(clean_sentences), scan_result