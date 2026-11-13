"""Fixed-size quick entry point and reproducible campaign CLI for XFS B+Tree fuzzing.

Examples
--------
Fast, deterministic smoke run (the entry used by the test suite)::

    python -m tests.fuzz.run --quick

Larger campaign with an explicit, reusable seed::

    python -m tests.fuzz.run --iterations 2000 --seed 1487545154

The quick entry always executes the same number of cases from the same seed,
so CI and a developer laptop make identical decisions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from tests.fuzz.harness import (
    TREE_TARGETS,
    VERDICTS,
    VERDICT_DESCRIPTIONS,
    Campaign,
    run_campaign,
)

QUICK_ITERATIONS = 128
DEFAULT_SEED = 0x58465342


def quick_campaign() -> Campaign:
    """The fixed-size, fixed-seed entry point used in tests and CI."""
    return run_campaign(QUICK_ITERATIONS, seed=DEFAULT_SEED, trees=TREE_TARGETS)


def format_report(campaign: Campaign) -> str:
    lines = [
        f"seed={campaign.seed} iterations={campaign.iterations}",
        "verdicts:",
    ]
    counts = campaign.counts()
    for verdict in VERDICTS:
        lines.append(f"  {verdict:20s} {counts[verdict]:5d}  {VERDICT_DESCRIPTIONS[verdict]}")

    by_tree = {tree: Counter() for tree in TREE_TARGETS}
    for result in campaign.results:
        by_tree[result.tree][result.verdict] += 1
    lines.append("per tree:")
    for tree in TREE_TARGETS:
        summary = ", ".join(f"{verdict}={by_tree[tree][verdict]}" for verdict in VERDICTS)
        lines.append(f"  {tree:12s} {summary}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuzz the XFS inode (small) and bmap (large) B+Trees.")
    parser.add_argument("--quick", action="store_true", help="run the fixed-size, fixed-seed smoke campaign")
    parser.add_argument("--iterations", type=int, default=QUICK_ITERATIONS, help="number of fuzz cases")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="reproducible base seed")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(argv)

    if args.quick:
        campaign = quick_campaign()
    else:
        campaign = run_campaign(args.iterations, seed=args.seed, trees=TREE_TARGETS)

    if args.json:
        payload = {
            "seed": campaign.seed,
            "iterations": campaign.iterations,
            "counts": campaign.counts(),
            "results": [result.as_dict() for result in campaign.results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(campaign))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
