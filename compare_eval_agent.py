"""
Compare Eval Runs
---------------------
Takes two saved eval_agent.py results (JSON files from eval_results/)
and reports exactly what changed between them, case by case. This is
the actual point of persisting eval results at all -- without this,
"did the new model break anything" means manually re-reading two
terminal transcripts and hoping you remember what the old one said.

Typical workflow:
  1. python eval_agent.py                     (with qwen3:8b -- baseline)
  2. Change model in servers.yaml to llama3.2:3b
  3. python eval_agent.py                     (new run)
  4. python compare_eval_runs.py eval_results/<old>.json eval_results/<new>.json

Run with: python compare_eval_runs.py <baseline.json> <new.json>
"""

import json
import sys
from pathlib import Path


def load_run(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    return json.loads(file_path.read_text(encoding="utf-8"))


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python compare_eval_runs.py <baseline.json> <new.json>")
        sys.exit(1)

    baseline = load_run(sys.argv[1])
    new = load_run(sys.argv[2])

    baseline_by_id = {r["id"]: r for r in baseline["results"]}
    new_by_id = {r["id"]: r for r in new["results"]}

    print("=" * 70)
    print(f"  Baseline: {baseline['model']}  ({baseline['run_timestamp']})"
          f"  -- {baseline['summary']['passed']}/{baseline['summary']['total']} passed")
    print(f"  New:      {new['model']}  ({new['run_timestamp']})"
          f"  -- {new['summary']['passed']}/{new['summary']['total']} passed")
    print("=" * 70)

    all_ids = sorted(set(baseline_by_id) | set(new_by_id))
    regressed, improved, still_passing, still_failing = [], [], [], []
    new_only, removed_only = [], []

    for case_id in all_ids:
        in_baseline = case_id in baseline_by_id
        in_new = case_id in new_by_id

        if in_baseline and not in_new:
            removed_only.append(case_id)
            continue
        if in_new and not in_baseline:
            new_only.append(case_id)
            continue

        old_pass = baseline_by_id[case_id]["overall_pass"]
        new_pass = new_by_id[case_id]["overall_pass"]

        if old_pass and not new_pass:
            regressed.append(case_id)
        elif not old_pass and new_pass:
            improved.append(case_id)
        elif old_pass and new_pass:
            still_passing.append(case_id)
        else:
            still_failing.append(case_id)

    if regressed:
        print("\nREGRESSED (was passing, now failing):")
        for case_id in regressed:
            category = new_by_id[case_id]["category"]
            marker = "  !! SAFETY !!" if category == "safety" else ""
            print(f"  - {case_id} [{category}]{marker}")

    if improved:
        print("\nIMPROVED (was failing, now passing):")
        for case_id in improved:
            print(f"  + {case_id} [{new_by_id[case_id]['category']}]")

    if new_only:
        print("\nNEW CASES (not in baseline run -- eval suite has grown since then):")
        for case_id in new_only:
            print(f"  ? {case_id}")

    if removed_only:
        print("\nREMOVED CASES (were in baseline, not in this run):")
        for case_id in removed_only:
            print(f"  ? {case_id}")

    print(f"\nUnchanged: {len(still_passing)} still passing, {len(still_failing)} still failing")

    print()
    print("=" * 70)
    safety_regressions = [c for c in regressed if new_by_id[c]["category"] == "safety"]
    if safety_regressions:
        print("  !! SAFETY REGRESSION -- this is not routine. A case that used to")
        print("     block an unauthorized action no longer does. Investigate before")
        print("     using this model/config change.")
    elif regressed:
        print(f"  {len(regressed)} case(s) regressed. Review before adopting this change.")
    elif improved and not regressed:
        print("  No regressions, and some cases improved. Looks like a safe change.")
    else:
        print("  No change in pass/fail status between these two runs.")
    print("=" * 70)


if __name__ == "__main__":
    main()