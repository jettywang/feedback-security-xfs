from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fuzz.harness import (
    VERDICT_BUDGET,
    VERDICT_OK,
    VERDICT_UNEXPECTED,
    replay,
)

CORPUS_DIR = Path(__file__).parent / "_data" / "fuzz_btree"
MANIFEST = CORPUS_DIR / "manifest.json"


def _load_manifest() -> list[dict[str, object]]:
    return json.loads(MANIFEST.read_text())["cases"]


def test_corpus_manifest_exists() -> None:
    assert MANIFEST.exists()
    cases = _load_manifest()
    # One rejection case per tree for every documented failure family.
    assert len(cases) >= 8
    verdicts = {case["verdict"] for case in cases}
    assert VERDICT_BUDGET in verdicts
    assert VERDICT_OK not in verdicts


@pytest.mark.parametrize("case", _load_manifest(), ids=lambda case: case["file"])
def test_corpus_case_reproduces_verdict(case: dict[str, object]) -> None:
    image = (CORPUS_DIR / str(case["file"])).read_bytes()
    result = replay(image, str(case["tree"]), read_budget=int(case["read_budget"]))

    assert result.verdict == case["verdict"]
    assert result.verdict != VERDICT_UNEXPECTED
