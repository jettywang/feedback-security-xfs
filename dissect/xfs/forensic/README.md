# XFS inode forensics checklist (path-independent)

When the directory tree may be damaged, allocation state must be established from
the per-AG inode allocation btrees (`inobt`, cross-checked against `finobt`) —
never from path traversal. This package reconstructs the allocated-inode
inventory directly from those indexes and treats path reconstruction as a
separate, best-effort annotation.

## Sources of truth (in order)

1. Primary superblock (`struct xfs_sb`): geometry — block/inode size, `inopblock`,
   `agblocks`, `agcount`, `agblklog`, sparse-inode feature, `sb_rootino`.
2. Per-AG headers packed into the sectors of AG block 0: superblock (sector 0),
   `AGF` (sector 1), `AGI` (sector 2), `AGFL` (sector 3). The `AGI` gives
   `agi_root` (inobt root), `agi_free_root` (finobt root) and counters; the AGF
   length is cross-checked.
3. `inobt` btree records (`xfs_inobt_rec`, 16 bytes each, immediately after the
   56-byte short-format CRC btree header):
   - `ir_startino` — first AG-relative inode of the 1024-inode record span.
   - `holemask` — 16 bits, one per 64-inode chunk; a set bit means the chunk's
     inode blocks are **not owned** by the filesystem (sparse inode block).
   - `ir_free` (freemask) — 64 bits per chunk; a set bit means the slot is free.
   - `ir_freecount` — popcount of the free mask.
4. `finobt` — same layout; used only as a cross-check. A missing finobt is
   tolerated. A disagreement is flagged `freemask_finobt_mismatch`; the `inobt`
   remains the primary allocation source.
5. The on-disk `xfs_dinode` at the computed byte offset is read for confirmation
   (magic, mode, links) but never overrides the bitmap classification on its own.

## Slot classification (the four states are never conflated)

| State | Meaning | Allocation conclusion |
| --- | --- | --- |
| `SPARSE_HOLE` | `holemask` bit set; no inode blocks exist | not an inode at all |
| `FREE_SLOT` | chunk present, freemask bit set | free space, never allocated |
| `ALLOCATED` | chunk present, freemask bit clear | allocated per the index |
| `UNREADABLE` | record/inode bytes could not be read, or a present chunk runs past the AG extent | **undetermined** — never called deleted |

`ALLOCATED` slots additionally carry a `dinode_state`:

- `OK` — `XFS_DINODE_MAGIC` present and core parsed.
- `ZEROED` — bitmap claims allocation but the slot is fully zeroed.
- `BAD_MAGIC` — readable bytes but no valid inode magic.
- `UNREADABLE` — the bytes themselves could not be obtained.

A zeroed or bad-magic slot stays `ALLOCATED` (the index still claims it) but is
not reported as a healthy live inode. **An unreachable inode is labelled
`NO_PATH`, never "deleted":** deletion would mean the freemask bit is set, which
is a distinct, separately reported condition.

## Reviewable evidence per result

Every emitted inode carries: global and AG-relative inode numbers, AG number,
byte offset, source btree block, record index, chunk index, slot, the raw
`holemask`/`freemask`/`inodemask` hex values, and the list of decision `reasons`.
Each btree record is also emitted in the per-AG `records` audit trail with its
filesystem block and byte offset, so any conclusion can be re-derived by hand.

## Bounds and corruption handling

- All btree block and record pointers are checked against the *actual* readable
  extent of the AG (`min(sb_agblocks, bytes_present / block_size)`), so a corrupt
  pointer cannot read into the next AG or off the end of the image.
- A present chunk whose start is beyond the AG capacity is emitted as
  `UNREADABLE` plus a `StructureError`, not read and not relabelled free.
- Read failures (EOF/short read/injected unreadable range) raise
  `ReadFailure`; well-formed-but-invalid data raises `StructureError`. The scan
  continues across all AGs and records — one bad region never aborts the rest.
- The last AG is reported with both its nominal and actual block counts.

## Path annotation

Shortform (`XFS_DINODE_FMT_LOCAL`) directories are indexed and walked from
`sb_rootino` using BFS. Both inode-number widths are handled (32-bit when
`i8count == 0`, otherwise 64-bit directory-wide). Reachable inodes (including
cross-AG children) get `RECOVERED` paths. Allocated inodes not reached get
`NO_PATH` with an explicit reason. Directories in extent/btree format are noted
as skipped; this pass never changes allocation state.

## CLI

```bash
python -m dissect.xfs.forensic.cli image.img                 # text checklist
python -m dissect.xfs.forensic.cli --json image.img          # machine-readable
python -m dissect.xfs.forensic.cli --unreadable 0x18000:0x19000 image.img
```

`--unreadable START:END` (repeatable, hex/decimal) injects an unreadable byte
range to exercise read-failure handling without modifying the image.

## Validation image

`image_builder.build_image()` assembles a deterministic two-AG, v5 (CRC),
sparse-inode-enabled image that contains: free slots, a full sparse inode block,
a stale inobt/finobt bit, cross-AG reachable inodes, and unreachable orphan
inodes. `tests/test_forensic.py` verifies bitmap interpretation, all four slot
states, cross-AG inode numbers, the partial last-AG boundary (truncated AGI and
truncated inobt leaf), leaf-magic and record-level index corruption, injected
read failures, and the rule that unreachable inodes are never reported deleted.

The same scanner is also cross-validated against the real `mkfs.xfs` images
shipped under `tests/_data/` (single-AG v5 images including sparse, bigtime and
symlink variants): it reproduces their inobt allocation with zero structure or
read errors and recovers shortform-directory paths that match the reference
`dissect.xfs` parser.
