# XFS B+Tree fuzzing

Fuzzing harness for the two on-disk XFS B+Tree flavours, aimed at keeping the
forensic worker safe against malformed indices:

- `small_inobt` — the per-allocation-group inode btree driven through
  `AllocationGroup.walk_agi()` / `XFS.walk_small_tree()` (32-bit, AG-local
  pointers).
- `large_bmap` — the filesystem-wide extent btree driven through
  `XFS.walk_extents()` / `XFS.walk_large_tree()` (64-bit `agno|agbno`
  pointers, allowed to cross allocation groups).

## Layout

- `tests/fuzz/imagebuild.py` — builds a fixed 128 KiB, 4-AG synthetic image
  containing a two-level small tree and a two-level large tree. Every mutated
  field has a known offset exposed via `ImageLayout`.
- `tests/fuzz/budget.py` — `BudgetedStream`, a read budget wrapper that only
  relies on the standard `BinaryIO` protocol. It is independent of the parser;
  the parser is given no knowledge of the budget.
- `tests/fuzz/mutators.py` — seeded, target-scoped mutators for tree level,
  record count, child addresses, back edges, magic/header corruption and
  truncation.
- `tests/fuzz/harness.py` — drives a tree under the budget and reduces the
  outcome to a stable verdict.
- `tests/fuzz/minimizer.py` — truncation + coarse-to-fine delta debugging that
  shrinks a failing image while preserving its verdict.
- `tests/fuzz/run.py` — fixed-size quick entry and campaign CLI.
- `tests/fuzz/corpus.py` — regenerates the minimized regression corpus under
  `tests/_data/fuzz_btree/`.

## Reproducible seeds

All randomness comes from a seeded `random.Random`. A case is fully identified
by `(seed, tree)`; replaying the same pair rebuilds byte-identical input.

```bash
# Fixed-size, fixed-seed smoke entry (also used by the test suite)
python -m tests.fuzz.run --quick

# Larger, still reproducible campaign
python -m tests.fuzz.run --iterations 2000 --seed 1481003842
```

## Verdicts (decision guide)

| Verdict | Meaning | Typical trigger | Parser behaviour expected |
| --- | --- | --- | --- |
| `short_read` | The image ends before a node header or the claimed records are available. | truncation mutator | explicit `Error("Short read ...")`, no partial record |
| `cycle` | A node points back at a block already on the walk path, or the depth cap is hit. | back edge / self loop / huge level | `BTreeCycleError`, traversal aborts instead of looping |
| `bad_address` | A pointer cannot be translated to an existing filesystem/AG block. | AG escape (small), missing AG or block (large) | `InvalidBlockAddressError` before seeking |
| `format_rejected` | Bounded but invalid metadata. | bad magic, `bb_numrecs` larger than the block | clean `Error`, no out-of-bounds read |
| `ok` | The tree walked without an exception (includes harmless mutations and the legal cross-AG pointer). | — | records yielded normally |
| `read_budget_exceeded` | The parser-independent byte budget was exhausted. | tight budget on a multi-block tree | `ReadBudgetExceeded`; bounds worker I/O regardless of parser bugs |
| `unexpected_exception` | Anything outside the documented hierarchy escaped. | parser defect | must never occur; treated as a finding |

### Address rules under test

- Large (64-bit) bmap pointers encode `agno|agbno` and may legitimately target
  a different allocation group. Only non-existent AGs or out-of-range block
  numbers are rejected. The corpus includes a valid cross-AG child leaf.
- Small (32-bit) inobt pointers are relative to the owning AG. Any pointer at
  or beyond `sb_agblocks`, including the root block itself, is rejected as
  `bad_address` (or `cycle` when it points at an ancestor that is still in
  range).

## Read budget

The budget counts bytes returned by `read()`; seeks are free. A valid image
reads well under 1 KiB per tree walk, while the default campaign budget
(64 KiB) leaves generous headroom. A deliberately tight budget (128 bytes) is
used for the dedicated `read_budget_exceeded` corpus case to prove the wrapper
trips even for a structurally valid but block-heavy index — defending the
worker without changing parser code.

## Regenerating the corpus

```bash
python -m tests.fuzz.corpus
```

Each `tests/_data/fuzz_btree/*.bin` is a minimized failing image and
`manifest.json` records its seed, tree, mutation log, verdict and size.
`tests/test_btree_corpus.py` replays every entry and asserts the verdict.
