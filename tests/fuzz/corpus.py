"""Generate a compact, reproducible regression corpus for the B+Tree fuzzer.

One representative case per non-``ok`` verdict is selected from a seeded
campaign, minimized and written to ``tests/_data/fuzz_btree`` together with a
JSON manifest. The manifest records the originating seed, target tree,
mutation log, verdict and size so a failure can be rebuilt and understood
without re-running the whole campaign.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.fuzz.harness import (
    DEFAULT_READ_BUDGET,
    TREE_TARGETS,
    VERDICT_BUDGET,
    VERDICT_OK,
    VERDICTS,
    run_campaign,
    run_case,
    replay,
)
from tests.fuzz.imagebuild import build_valid_image
from tests.fuzz.minimizer import minimize

CORPUS_DIR = Path(__file__).resolve().parent.parent / "_data" / "fuzz_btree"
CORPUS_SEED = 0x58465342
CORPUS_ITERATIONS = 2000
# Tight enough that a valid multi-block tree trips it, exercising the budget path.
BUDGET_CASE_BUDGET = 128


def _candidate_seeds() -> dict[tuple[str, str], int]:
    campaign = run_campaign(CORPUS_ITERATIONS, seed=CORPUS_SEED, trees=TREE_TARGETS)
    chosen: dict[tuple[str, str], int] = {}
    for result in campaign.results:
        key = (result.tree, result.verdict)
        if result.verdict != VERDICT_OK and key not in chosen:
            chosen[key] = result.seed
    return chosen


def build_corpus() -> list[dict[str, object]]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    seen_verdicts: set[str] = set()

    for (tree, verdict), seed in sorted(_candidate_seeds().items(), key=lambda item: item[1]):
        result, image = run_case(seed, tree, read_budget=DEFAULT_READ_BUDGET)
        minimized, report = minimize(image, tree, read_budget=DEFAULT_READ_BUDGET)
        name = f"{verdict}__{tree}__seed{seed}.bin"
        (CORPUS_DIR / name).write_bytes(minimized)
        entries.append(
            {
                "file": name,
                "seed": seed,
                "tree": tree,
                "verdict": verdict,
                "mutations": list(result.mutations),
                "original_size": report.original_size,
                "minimized_size": report.minimized_size,
                "reduction_pct": report.reduction_pct,
                "read_budget": DEFAULT_READ_BUDGET,
            }
        )
        seen_verdicts.add(verdict)

    # Explicit budget case: a valid image replayed under a deliberately tight budget.
    valid_image = bytes(build_valid_image())
    budget_result = replay(valid_image, TREE_TARGETS[1], read_budget=BUDGET_CASE_BUDGET)
    assert budget_result.verdict == VERDICT_BUDGET
    name = f"{VERDICT_BUDGET}__large_bmap__tight-budget.bin"
    (CORPUS_DIR / name).write_bytes(valid_image)
    entries.append(
        {
            "file": name,
            "seed": None,
            "tree": TREE_TARGETS[1],
            "verdict": VERDICT_BUDGET,
            "mutations": [],
            "original_size": len(valid_image),
            "minimized_size": len(valid_image),
            "reduction_pct": 0.0,
            "read_budget": BUDGET_CASE_BUDGET,
        }
    )
    seen_verdicts.add(VERDICT_BUDGET)

    manifest = {
        "seed_base": CORPUS_SEED,
        "campaign_iterations": CORPUS_ITERATIONS,
        "default_read_budget": DEFAULT_READ_BUDGET,
        "verdicts_covered": sorted(seen_verdicts),
        "all_verdicts": list(VERDICTS),
        "cases": entries,
    }
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return entries


if __name__ == "__main__":
    cases = build_corpus()
    print(f"Wrote {len(cases)} corpus cases to {CORPUS_DIR}")
