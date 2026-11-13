from __future__ import annotations

import pytest

from tests.fuzz.budget import BudgetedStream, ReadBudgetExceeded
from tests.fuzz.harness import (
    LARGE_TREE,
    SMALL_TREE,
    TREE_TARGETS,
    VERDICTS,
    VERDICT_BUDGET,
    VERDICT_OK,
    VERDICT_UNEXPECTED,
    run_campaign,
    run_case,
)
from tests.fuzz.imagebuild import build_valid_image
from tests.fuzz.minimizer import minimize
from tests.fuzz.run import quick_campaign


def test_quick_campaign_is_fixed_and_clean() -> None:
    first = quick_campaign()
    second = quick_campaign()

    assert first.seed == second.seed
    assert first.iterations == second.iterations
    assert [result.as_dict() for result in first.results] == [result.as_dict() for result in second.results]
    # The whole point of the hardening: nothing escapes the documented hierarchy.
    assert first.counts()[VERDICT_UNEXPECTED] == 0
    assert first.counts()[VERDICT_OK] > 0


def test_campaign_covers_both_trees_and_all_error_verdicts() -> None:
    campaign = run_campaign(400, seed=2024)
    counts = campaign.counts()
    observed = {verdict for verdict in VERDICTS if counts[verdict] > 0}

    assert {result.tree for result in campaign.results} == set(TREE_TARGETS)
    # Every rejection family must be reachable through seeded mutation.
    assert {
        "short_read",
        "cycle",
        "bad_address",
        "format_rejected",
    }.issubset(observed)


def test_seed_reproduces_identical_case() -> None:
    result_a, image_a = run_case(777, SMALL_TREE)
    result_b, image_b = run_case(777, SMALL_TREE)

    assert image_a == image_b
    assert result_a.verdict == result_b.verdict
    assert result_a.mutations == result_b.mutations


def test_valid_image_is_ok_for_both_trees() -> None:
    from tests.fuzz.harness import replay

    image = bytes(build_valid_image())
    assert replay(image, SMALL_TREE).verdict == VERDICT_OK
    assert replay(image, LARGE_TREE).verdict == VERDICT_OK


def test_read_budget_is_parser_independent() -> None:
    import io

    image = bytes(build_valid_image())
    stream = BudgetedStream(io.BytesIO(image), budget=128)
    with pytest.raises(ReadBudgetExceeded):
        stream.read(256)
    assert stream.bytes_read == 0


def test_tight_budget_trips_on_legit_multi_block_tree() -> None:
    from tests.fuzz.harness import replay

    image = bytes(build_valid_image())
    result = replay(image, LARGE_TREE, read_budget=128)
    assert result.verdict == VERDICT_BUDGET


def test_minimizer_preserves_verdict_and_shrinks() -> None:
    result, image = run_case(31337, LARGE_TREE)
    if result.verdict == VERDICT_OK:
        result, image = run_case(31338, LARGE_TREE)
    assert result.verdict != VERDICT_OK

    minimized, report = minimize(image, LARGE_TREE, read_budget=64 * 1024)
    from tests.fuzz.harness import replay

    replayed = replay(minimized, LARGE_TREE, read_budget=64 * 1024)

    assert replayed.verdict == report.verdict == result.verdict
    assert report.minimized_size <= report.original_size
