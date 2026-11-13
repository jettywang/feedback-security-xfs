"""Failure material minimization.

Given a fuzzed image that produces a non-``ok`` verdict, the minimizer shrinks
the bytes while preserving the verdict. It combines two passes:

1. truncation minimization (find the shortest prefix that still fails the same
   way), which is very effective for short-read and pointer-driven cases;
2. byte-level delta debugging that zeroes contiguous chunks, keeping a change
   only when the replay still yields the identical verdict.

The minimized image is guaranteed to reproduce the original verdict, keeping
the stored corpus small and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.fuzz.harness import (
    VERDICT_OK,
    VERDICT_SHORT_READ,
    FuzzResult,
    replay,
)
from tests.fuzz.imagebuild import BLOCK_SIZE, LAYOUT
from tests.fuzz.targets import LARGE_TREE, SMALL_TREE


@dataclass(slots=True)
class MinimizationReport:
    original_size: int
    minimized_size: int
    verdict: str
    iterations: int

    @property
    def reduction_pct(self) -> float:
        if self.original_size == 0:
            return 0.0
        return round(100.0 * (1 - self.minimized_size / self.original_size), 2)


def _minimize_truncation(image: bytes, tree: str, verdict: str, budget: int) -> tuple[bytes, int]:
    """Binary-search the shortest prefix that keeps the same verdict."""
    lo, hi = 1, len(image)
    iterations = 0
    best = image
    while lo < hi:
        mid = (lo + hi) // 2
        iterations += 1
        candidate = image[:mid]
        result = replay(candidate, tree, read_budget=budget)
        if result.verdict == verdict:
            hi = mid
            best = candidate
        else:
            lo = mid + 1
    return best, iterations


def _zero_chunks_keep_verdict(
    image: bytearray, tree: str, verdict: str, budget: int, chunk_size: int, iterations: int
) -> tuple[bytearray, int]:
    index = 0
    while index < len(image):
        end = min(index + chunk_size, len(image))
        original = image[index:end]
        candidate = bytearray(image)
        candidate[index:end] = b"\x00" * (end - index)
        iterations += 1
        result = replay(bytes(candidate), tree, read_budget=budget)
        if result.verdict == verdict:
            image = candidate
        index = end
    return image, iterations


def _tree_blocks(tree: str) -> tuple[int, ...]:
    if tree == SMALL_TREE:
        return (LAYOUT.small_root_block, *LAYOUT.small_leaf_blocks)
    if tree == LARGE_TREE:
        return (LAYOUT.large_root_block, *LAYOUT.large_leaf_blocks)
    return ()  # pragma: no cover - targets are a fixed tuple


def _minimize_short_read(image: bytes, tree: str, verdict: str, budget: int) -> tuple[bytes, int]:
    """Shrink a short-read case to the first tree block that still fails to fully read.

    A purely binary truncation would collapse these cases to a single byte, which
    is no longer useful as forensic material. Instead keep every whole block up
    to (and including) the first tree block whose contents are missing.
    """
    iterations = 0
    best = image
    for block in sorted(_tree_blocks(tree)):
        cut = LAYOUT.block_offset(block) + BLOCK_SIZE
        if cut >= len(image):
            continue
        iterations += 1
        candidate = image[:cut]
        if replay(candidate, tree, read_budget=budget).verdict == verdict:
            best = candidate
            break
    return best, iterations


def minimize(image: bytes, tree: str, *, read_budget: int) -> tuple[bytes, MinimizationReport]:
    baseline: FuzzResult = replay(image, tree, read_budget=read_budget)
    verdict = baseline.verdict
    if verdict == VERDICT_OK:
        return image, MinimizationReport(len(image), len(image), verdict, 0)

    iterations = 0
    if verdict == VERDICT_SHORT_READ:
        current, used = _minimize_short_read(image, tree, verdict, read_budget)
    else:
        current, used = _minimize_truncation(image, tree, verdict, read_budget)
    iterations += used

    working = bytearray(current)
    # Coarse-to-fine chunk zeroing; skip chunks larger than what remains.
    for chunk_size in (256, 64, 16, 4, 1):
        if chunk_size > len(working):
            continue
        working, iterations = _zero_chunks_keep_verdict(working, tree, verdict, read_budget, chunk_size, iterations)

    minimized = bytes(working)
    report = MinimizationReport(len(image), len(minimized), verdict, iterations)
    return minimized, report
