"""
Live test questions for knowledge_server.py
----------------------------------------------
Companion to test_knowledge_server.py. Those tests check pipeline
MECHANICS with a fake embedding function (no Ollama needed). This file
checks retrieval QUALITY -- whether real nomic-embed-text embeddings
actually surface the right chunk for a given question.

This can't run in CI without a live Ollama instance, so for now it's a
runnable manual-check script: run it directly against a live
knowledge_server.py (Ollama must be running, nomic-embed-text pulled).

    python live_test_questions.py

For each question, it prints whether the top-ranked result's source_file
and page matched the expected answer -- a quick pass/fail signal you can
scan without reading every result by hand.

NOTE: page numbers below are pinned to the v2 (19-page) Northwind
Employee Handbook. If you add/reorder pages, update expected_page
accordingly -- run tests/test_knowledge_server.py's PDF-loading test
first if you're unsure the mapping is still correct.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import knowledge_server as ks

# (question, expected_source_file, expected_page, note)
HANDBOOK = "Northwind_Employee_Handbook.pdf"

LIVE_TEST_CASES = [
    # --- Direct / close-wording questions ---
    ("What is our leave policy?", HANDBOOK, 3, "direct"),
    ("How many days of annual leave do employees get?", HANDBOOK, 3, "direct"),
    ("What is the expense reimbursement policy?", HANDBOOK, 5, "direct"),
    ("What is the remote work policy?", HANDBOOK, 7, "direct"),
    ("What happens if I violate the code of conduct?", HANDBOOK, 8, "direct"),
    ("What are the password requirements for company systems?", HANDBOOK, 9, "direct"),
    ("What health insurance benefits are provided?", HANDBOOK, 10, "direct"),
    ("What happens during onboarding?", HANDBOOK, 12, "direct"),
    ("How often are performance reviews conducted?", HANDBOOK, 13, "direct"),
    ("What is the notice period when I resign?", HANDBOOK, 16, "direct"),
    ("How much is the employee referral bonus?", HANDBOOK, 17, "direct"),
    ("How long is client data retained?", HANDBOOK, 18, "direct"),

    # --- Paraphrased / semantic questions (no shared keywords with the PDF) ---
    ("How much time off do new hires get?", HANDBOOK, 3, "semantic"),
    ("Can I expense alcohol when out with a client?", HANDBOOK, 5, "semantic"),
    ("Am I allowed to work from a different city for a few weeks?", HANDBOOK, 7, "semantic"),
    ("What's the punishment for major misconduct?", HANDBOOK, 8, "semantic"),
    ("Do I need MFA to access client data?", HANDBOOK, 9, "semantic"),
    ("What happens if I get a low performance rating twice in a row?", HANDBOOK, 13, "semantic"),
    ("Is there a budget for courses or certifications?", HANDBOOK, 11, "semantic"),
    ("When am I eligible for a promotion?", HANDBOOK, 13, "semantic"),
    ("What flight class do I get as a senior consultant?", HANDBOOK, 6, "semantic"),
    ("Can I come back if I quit and change my mind later?", HANDBOOK, 16, "semantic"),

    # --- Specific-number questions (hallucination check -- verify exact figures manually) ---
    ("How many days can I carry forward unused leave?", HANDBOOK, 3, "number-check: expect 10 days"),
    ("What's the nightly hotel cap for domestic travel in metro cities?", HANDBOOK, 5, "number-check: expect Rs 6,000"),
    ("How many weeks of parental leave does the secondary caregiver get?", HANDBOOK, 3, "number-check: expect 2 weeks"),
    ("What's the minimum internet speed required for fully remote work?", HANDBOOK, 7, "number-check: expect 50 Mbps"),
    ("What's the notice period for senior employees at L5 and above?", HANDBOOK, 16, "number-check: expect 90 days"),
    ("How much is the referral bonus for a senior or niche role?", HANDBOOK, 17, "number-check: expect Rs 1,00,000"),
    ("How many days before payroll should I update my bank details?", HANDBOOK, 4, "number-check: expect 10 working days"),

    # --- Out-of-scope (should NOT confidently match anything) ---
    ("What is our dress code policy?", None, None, "out-of-scope: handbook has no dress code section"),
    ("Can I bring my dog to the office?", None, None, "out-of-scope: not covered anywhere"),
]


def run_live_tests(top_k: int = 1) -> None:
    print(f"Running {len(LIVE_TEST_CASES)} live retrieval checks against knowledge_server...\n")
    passed, failed = 0, 0

    for question, expected_file, expected_page, note in LIVE_TEST_CASES:
        results = ks.search_knowledge_base(question, top_k=top_k)

        if results and "error" in results[0]:
            print(f"[ERROR] {question!r} -> {results[0]['error']}")
            failed += 1
            continue

        top = results[0]
        top_score = top["similarity_score"]

        if expected_file is None:
            flag = "LOW SCORE (expected)" if top_score < 0.5 else "SCORE SEEMS HIGH -- verify manually"
            print(f"[{flag}] {question!r} (out-of-scope) -> "
                  f"top match: {top['source_file']} p{top['page']} (score={top_score})")
            continue

        match = (top["source_file"] == expected_file and top["page"] == expected_page)
        status = "PASS" if match else "FAIL"
        if match:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {question!r} [{note}]")
        print(f"       expected: {expected_file} p{expected_page}")
        print(f"       got:      {top['source_file']} p{top['page']} (score={top_score})")

    print(f"\n{passed} passed, {failed} failed (out of {passed + failed} scored checks; "
          f"out-of-scope checks are informational only).")


if __name__ == "__main__":
    ks._build_index()
    if ks._doc_matrix is None:
        print("Index is empty or failed to build -- check Ollama is running "
              "and nomic-embed-text is pulled, then try again.")
        sys.exit(1)
    run_live_tests()