# XFS directory scan integrity

This note documents the forensic error strategy used when reading XFS directory blocks, how callers
query scan completeness, and how the minimal regression images are maintained.

## Why this exists

A directory scan that returns an empty result used to be ambiguous: there was no way to tell whether
the directory genuinely had no entries or whether one or more directory blocks simply were not read.
The block scanner also treated every non-data magic as a "probably sparse" block and silently skipped
it, so a damaged block and a benign sparse/index block looked identical.

The scanner now separates those cases and records queryable integrity information while still
recovering every entry it can.

## Block classification

For block-format directories (`XFS_DINODE_FMT_EXTENTS` / `XFS_DINODE_FMT_BTREE`) each logical
directory block is read in full and classified by its first four bytes:

- **Data block** - one of `XFS_DIR2_BLOCK_MAGIC`, `XFS_DIR2_DATA_MAGIC`, `XFS_DIR3_BLOCK_MAGIC` or
  `XFS_DIR3_DATA_MAGIC`. Its entries are parsed.
- **Sparse block** - magic `0`. These are unallocated holes that read back as zeroes and are recorded
  in `sparse_blocks`; they are never corruption.
- **Index block** - free-space, leaf and b+tree node magics (`XFS_DIR2/3_FREE_MAGIC`,
  `XFS_DIR2/3_LEAF1/LEAFN_MAGIC`, `XFS_DA_NODE_MAGIC`, `XFS_DA3_NODE_MAGIC`). They manage the hash
  index and hold no entries, so they are recorded in `index_blocks` and never treated as damage.
- **Unknown magic** - anything else is recorded as a `DirectoryBlockMagicError` in
  `corrupt_blocks`; the block is skipped and the scan continues.

## Recoverable errors

Scanning never aborts on a single bad block. Each recoverable condition is a `DirectoryScanError`
subclass carrying the logical directory block number, and is appended to the status:

- `DirectoryBlockMagicError` - unrecognized block magic.
- `DirectoryRecordTruncatedError` - a record could not be parsed because its block ended prematurely
  (the underlying `EOFError` from `dissect.cstruct`).
- `DirectoryReadError` - the block could not be read in full (short read) or the underlying stream
  raised `OSError`.

Entries parsed before the error are still yielded, so partial recovery is the default. Every
recovered entry keeps its location on `dir_logical_block` (block relative to the directory inode) and
`dir_physical_block` (filesystem block, or `None` for a sparse hole).

## Querying completeness

- `INode.scan_directory()` eagerly returns a `DirectoryScan(entries, status)`.
- `INode.iterdir()` stays a lazy generator and stores its `DirectoryScanStatus` on
  `INode.dir_scan_status` while it runs.
- `INode.listdir()` (and the historical `dirlist` alias) keep their original `{name: inode}` return
  type and never raise on damaged blocks. Existing callers are unchanged; they can opt in by reading
  `dir_scan_status` after the call.

On `DirectoryScanStatus`:

- `state` is `COMPLETE` (all blocks read, no errors), `PARTIAL` (at least one recoverable error) or
  `EMPTY` (error-free scan with zero entries).
- `complete` / `is_complete()` distinguish a fully read directory from an incomplete one.
- `is_empty()` is only true for an error-free scan that produced no entries. This is what tells a
  genuinely empty directory apart from "the blocks were not all read".
- `data_blocks`, `sparse_blocks`, `index_blocks`, `corrupt_blocks`, `unreadable_blocks`,
  `truncated_blocks`, `affected_blocks`, `scanned_blocks` and `total_blocks` expose the per-block
  accounting, and `errors` holds the exception objects for detailed reporting.

## Regression images

The hand-crafted images live in `tests/_data/` and are rebuilt from `xfs.bin.gz` by
`tests/_data/tools/build_dir_scan_images.py`, so the suite does not depend on `mkfs.xfs`:

- `xfs_dir_scan_complete.bin.gz` - two valid data blocks plus a sparse block; a complete scan that
  recovers four entries.
- `xfs_dir_scan_badblock.bin.gz` - the second data block has an invalid magic; a partial scan that
  still recovers the first block's entries and reports `corrupt_blocks == [1]`.
- `xfs_dir_scan_truncated.bin.gz` - the second data block ends in a truncated record; a partial scan
  that keeps its intact entries and reports `truncated_blocks == [1]`.

Read interruption (`DirectoryReadError`) is exercised in `tests/test_directory_scan.py` with a stream
wrapper that short-reads a single block, avoiding the need for another image.

Rebuild after touching the layout helpers:

```sh
python tests/_data/tools/build_dir_scan_images.py
```

The script self-checks that the complete image reports `complete` and both damaged images report
`partial` before finishing.
