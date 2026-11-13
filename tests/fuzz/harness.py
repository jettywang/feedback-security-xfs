"""B+Tree fuzzing harness for the XFS parser.

The harness drives the two B+Tree entry points independently:

* ``walk_agi`` -> small, per-allocation-group inode btree,
* ``walk_extents`` -> large, filesystem-wide bmap btree.

Each run is wrapped in a parser-agnostic :class:`BudgetedStream` and its
outcome is reduced to one of the stable verdicts documented in
``VERDICT_DESCRIPTIONS``. The verdicts deliberately distinguish a truncated
image (short read), a cyclic tree, an invalid address translation, a normal
format rejection, a clean parse and a read-budget trip (the DoS condition the
fuzzer exists to prevent).
"""

from __future__ import annotations

import io
import random
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field

from dissect.xfs.exceptions import (
    BTreeCycleError,
    Error,
    InvalidBlockAddressError,
)
from dissect.xfs.xfs import XFS

from tests.fuzz.budget import BudgetedStream, ReadBudgetExceeded
from tests.fuzz.imagebuild import LAYOUT, build_valid_image
from tests.fuzz.mutators import mutate
from tests.fuzz.targets import LARGE_TREE, SMALL_TREE, TREE_TARGETS

VERDICT_SHORT_READ = "short_read"
VERDICT_CYCLE = "cycle"
VERDICT_BAD_ADDRESS = "bad_address"
VERDICT_FORMAT_REJECTED = "format_rejected"
VERDICT_OK = "ok"
VERDICT_BUDGET = "read_budget_exceeded"
VERDICT_UNEXPECTED = "unexpected_exception"

VERDICTS: tuple[str, ...] = (
    VERDICT_SHORT_READ,
    VERDICT_CYCLE,
    VERDICT_BAD_ADDRESS,
    VERDICT_FORMAT_REJECTED,
    VERDICT_OK,
    VERDICT_BUDGET,
    VERDICT_UNEXPECTED,
)

VERDICT_DESCRIPTIONS: dict[str, str] = {
    VERDICT_SHORT_READ: "Image truncated or record area short; parser stopped on an explicit short read.",
    VERDICT_CYCLE: "Back edge or excessive depth detected; traversal aborted before looping forever.",
    VERDICT_BAD_ADDRESS: "Child pointer did not translate to an existing filesystem/AG block.",
    VERDICT_FORMAT_REJECTED: "Well-bounded but invalid metadata (bad magic/numrecs) rejected cleanly.",
    VERDICT_OK: "Image parsed and the tree was walked without an exception.",
    VERDICT_BUDGET: "Parser-independent read budget tripped; the index forced excessive reads.",
    VERDICT_UNEXPECTED: "An exception outside the parser's documented error hierarchy escaped.",
}

# A well-formed image only reads a handful of blocks; the budget stays far above
# that while still being tiny relative to the 128 KiB image.
DEFAULT_READ_BUDGET = 64 * 1024
SEED_BASE = 0x58465342  # 'XFSB'


@dataclass(slots=True)
class FuzzResult:
    seed: int
    tree: str
    verdict: str
    mutations: tuple[str, ...]
    exception: str | None
    bytes_read: int
    image_size: int

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "tree": self.tree,
            "verdict": self.verdict,
            "mutations": list(self.mutations),
            "exception": self.exception,
            "bytes_read": self.bytes_read,
            "image_size": self.image_size,
        }


@dataclass(slots=True)
class Campaign:
    seed: int
    iterations: int
    results: list[FuzzResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        counts = {verdict: 0 for verdict in VERDICTS}
        for result in self.results:
            counts[result.verdict] += 1
        return counts


def _classify(exc: BaseException) -> str:
    if isinstance(exc, BTreeCycleError):
        return VERDICT_CYCLE
    if isinstance(exc, InvalidBlockAddressError):
        return VERDICT_BAD_ADDRESS
    if isinstance(exc, ReadBudgetExceeded):
        return VERDICT_BUDGET
    if isinstance(exc, EOFError):
        return VERDICT_SHORT_READ
    if isinstance(exc, Error):
        message = str(exc)
        if "Short read" in message:
            return VERDICT_SHORT_READ
        return VERDICT_FORMAT_REJECTED
    return VERDICT_UNEXPECTED


def _drive_tree(xfs: XFS, tree: str) -> None:
    if tree == SMALL_TREE:
        list(xfs.get_allocation_group(0).walk_agi())
    elif tree == LARGE_TREE:
        list(xfs.walk_extents(LAYOUT.large_root_block))
    else:  # pragma: no cover - defensive, targets are a fixed tuple
        raise ValueError(f"Unknown fuzz target: {tree}")


def run_case(
    seed: int,
    tree: str,
    *,
    read_budget: int = DEFAULT_READ_BUDGET,
    image: bytes | bytearray | None = None,
) -> tuple[FuzzResult, bytes]:
    """Run a single deterministic case and return the verdict plus the image bytes."""
    rng = random.Random(seed)
    buf = bytearray(image) if image is not None else build_valid_image()
    if image is None:
        mutations = tuple(mutate(buf, rng, tree))
    else:
        # A pre-built (e.g. minimized) image is replayed without further mutation.
        mutations = ()

    stream = BudgetedStream(io.BytesIO(bytes(buf)), read_budget)
    exception_name: str | None = None
    try:
        xfs = XFS(stream)
        _drive_tree(xfs, tree)
        verdict = VERDICT_OK
    except Exception as exc:
        verdict = _classify(exc)
        exception_name = f"{type(exc).__name__}: {exc}"
    finally:
        with suppress(Exception):
            stream.close()

    result = FuzzResult(
        seed=seed,
        tree=tree,
        verdict=verdict,
        mutations=mutations,
        exception=exception_name,
        bytes_read=stream.bytes_read,
        image_size=len(buf),
    )
    return result, bytes(buf)


def run_campaign(
    iterations: int,
    *,
    seed: int = SEED_BASE,
    trees: Sequence[str] = TREE_TARGETS,
    read_budget: int = DEFAULT_READ_BUDGET,
) -> Campaign:
    """Run a fixed-size, reproducible campaign interleaving both tree targets."""
    campaign = Campaign(seed=seed, iterations=iterations)
    for index in range(iterations):
        case_seed = seed + index
        tree = trees[index % len(trees)]
        result, _ = run_case(case_seed, tree, read_budget=read_budget)
        campaign.results.append(result)
    return campaign


def replay(image: bytes, tree: str, *, seed: int = 0, read_budget: int = DEFAULT_READ_BUDGET) -> FuzzResult:
    """Replay a persisted (possibly minimized) image without mutating it."""
    result, _ = run_case(seed, tree, read_budget=read_budget, image=image)
    return result


__all__ = [
    "Campaign",
    "DEFAULT_READ_BUDGET",
    "FuzzResult",
    "LARGE_TREE",
    "SMALL_TREE",
    "SEED_BASE",
    "TREE_TARGETS",
    "VERDICTS",
    "VERDICT_BAD_ADDRESS",
    "VERDICT_BUDGET",
    "VERDICT_CYCLE",
    "VERDICT_DESCRIPTIONS",
    "VERDICT_FORMAT_REJECTED",
    "VERDICT_OK",
    "VERDICT_SHORT_READ",
    "VERDICT_UNEXPECTED",
    "mutate",
    "replay",
    "run_campaign",
    "run_case",
]
