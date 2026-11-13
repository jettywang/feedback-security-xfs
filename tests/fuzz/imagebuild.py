"""Synthetic, fixed-size XFS image builder used by the B+Tree fuzzing harness.

The generated image deliberately contains both XFS B+Tree flavours:

* a *small* (per-AG, 32-bit pointers) inode btree rooted from the AGI header,
* a *large* (filesystem-wide, 64-bit ``agno|agbno`` pointers) bmap btree.

Every tree is two levels deep (an interior node pointing at leaves) so that
mutators can rewrite levels, record counts, child addresses and back edges at
known offsets. The image is fully self-contained: no real filesystem image is
required and the total byte size is a deterministic constant.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from dissect.xfs.c_xfs import XFS_NULL, c_xfs

BLOCK_SIZE = 512
AG_BLOCKS = 64
AG_COUNT = 4
TOTAL_BLOCKS = AG_BLOCKS * AG_COUNT
AGBLKLOG = 6

# Block layout (all absolute filesystem block numbers).
SMALL_ROOT_BLOCK = 4
SMALL_LEAF_BLOCKS = (5, 6)
LARGE_ROOT_BLOCK = 40
LARGE_LEAF_BLOCKS = (41, 2 * AG_BLOCKS + 7)  # second leaf deliberately lives in AG 1
INOBT_MAGIC = c_xfs.XFS_IBT_MAGIC
BMAP_MAGIC = c_xfs.XFS_BMAP_MAGIC

SMALL_HEADER_SIZE = len(c_xfs.xfs_btree_sblock)
LARGE_HEADER_SIZE = len(c_xfs.xfs_btree_lblock)
SMALL_MAXRECS = (BLOCK_SIZE - SMALL_HEADER_SIZE) // 8
LARGE_MAXRECS = (BLOCK_SIZE - LARGE_HEADER_SIZE) // 16
RECORD_SIZE = 16


@dataclass(frozen=True)
class ImageLayout:
    """Offsets and sizes the mutators need to target individual tree fields."""

    size: int
    small_root_block: int
    small_leaf_blocks: tuple[int, ...]
    large_root_block: int
    large_leaf_blocks: tuple[int, ...]

    def block_offset(self, block: int) -> int:
        return block * BLOCK_SIZE

    def small_pointer_offset(self, pointer_index: int) -> int:
        # Interior pointers start after the header and the (unused) key area.
        return self.block_offset(self.small_root_block) + SMALL_HEADER_SIZE + SMALL_MAXRECS * 4 + pointer_index * 4

    def large_pointer_offset(self, pointer_index: int) -> int:
        return self.block_offset(self.large_root_block) + LARGE_HEADER_SIZE + LARGE_MAXRECS * 8 + pointer_index * 8

    def header_offset(self, block: int, large: bool) -> int:
        return self.block_offset(block)


LAYOUT = ImageLayout(
    size=TOTAL_BLOCKS * BLOCK_SIZE,
    small_root_block=SMALL_ROOT_BLOCK,
    small_leaf_blocks=SMALL_LEAF_BLOCKS,
    large_root_block=LARGE_ROOT_BLOCK,
    large_leaf_blocks=LARGE_LEAF_BLOCKS,
)


def _pack_inobt_record(startino: int, freecount: int, free: int = 0xFFFF) -> bytes:
    return struct.pack(">IIQ", startino, freecount, free)


def _pack_bmap_record(logical_offset: int, block: int, count: int, flag: int = 0) -> bytes:
    word0 = (flag << 63) | ((logical_offset & ((1 << 54) - 1)) << 9) | (block >> 43)
    word1 = ((block & ((1 << 43) - 1)) << 21) | (count & ((1 << 21) - 1))
    return struct.pack(">QQ", word0, word1)


def _write(buf: bytearray, offset: int, data: bytes) -> None:
    buf[offset : offset + len(data)] = data


def _small_node(level: int, numrecs: int) -> bytes:
    node = c_xfs.xfs_btree_sblock()
    node.bb_magic = INOBT_MAGIC
    node.bb_level = level
    node.bb_numrecs = numrecs
    node.bb_leftsib = XFS_NULL & 0xFFFFFFFF
    node.bb_rightsib = XFS_NULL & 0xFFFFFFFF
    return node.dumps()


def _large_node(level: int, numrecs: int) -> bytes:
    node = c_xfs.xfs_btree_lblock()
    node.bb_magic = BMAP_MAGIC
    node.bb_level = level
    node.bb_numrecs = numrecs
    node.bb_leftsib = XFS_NULL
    node.bb_rightsib = XFS_NULL
    return node.dumps()


def build_valid_image() -> bytearray:
    """Return a bytearray holding a deterministic, parseable XFS image."""
    buf = bytearray(LAYOUT.size)

    sb = c_xfs.xfs_sb()
    sb.sb_magicnum = c_xfs.XFS_SB_MAGIC
    sb.sb_blocksize = BLOCK_SIZE
    sb.sb_dblocks = TOTAL_BLOCKS
    sb.sb_rblocks = 0
    sb.sb_rextents = 0
    sb.sb_logstart = 0
    sb.sb_rootino = 0
    sb.sb_agblocks = AG_BLOCKS
    sb.sb_agcount = AG_COUNT
    sb.sb_versionnum = c_xfs.XFS_SB_VERSION_4
    sb.sb_sectsize = BLOCK_SIZE
    sb.sb_inodesize = 256
    sb.sb_inopblock = 8
    sb.sb_blocklog = 9
    sb.sb_sectlog = 9
    sb.sb_inodelog = 8
    sb.sb_inopblog = 3
    sb.sb_agblklog = AGBLKLOG
    _write(buf, 0, sb.dumps())

    agi = c_xfs.xfs_agi()
    agi.agi_magicnum = c_xfs.XFS_AGI_MAGIC
    agi.agi_versionnum = 1
    agi.agi_seqno = 0
    agi.agi_length = AG_BLOCKS
    agi.agi_root = SMALL_ROOT_BLOCK
    agi.agi_level = 2
    _write(buf, 2 * BLOCK_SIZE, agi.dumps())

    # Small inode btree: root (level 1) -> two leaves inside AG 0.
    _write(buf, LAYOUT.block_offset(SMALL_ROOT_BLOCK), _small_node(level=1, numrecs=2))
    _write(buf, LAYOUT.small_pointer_offset(0), struct.pack(">I", SMALL_LEAF_BLOCKS[0]))
    _write(buf, LAYOUT.small_pointer_offset(1), struct.pack(">I", SMALL_LEAF_BLOCKS[1]))

    for leaf_index, leaf_block in enumerate(SMALL_LEAF_BLOCKS):
        _write(buf, LAYOUT.block_offset(leaf_block), _small_node(level=0, numrecs=2))
        base = LAYOUT.block_offset(leaf_block) + SMALL_HEADER_SIZE
        _write(buf, base, _pack_inobt_record(leaf_index * 16, 1))
        _write(buf, base + RECORD_SIZE, _pack_inobt_record(leaf_index * 16 + 8, 0))

    # Large bmap btree: root (level 1) -> two leaves, one of them in AG 1.
    _write(buf, LAYOUT.block_offset(LARGE_ROOT_BLOCK), _large_node(level=1, numrecs=2))
    _write(buf, LAYOUT.large_pointer_offset(0), struct.pack(">Q", LARGE_LEAF_BLOCKS[0]))
    _write(buf, LAYOUT.large_pointer_offset(1), struct.pack(">Q", LARGE_LEAF_BLOCKS[1]))

    for leaf_index, leaf_block in enumerate(LARGE_LEAF_BLOCKS):
        _write(buf, LAYOUT.block_offset(leaf_block), _large_node(level=0, numrecs=2))
        base = LAYOUT.block_offset(leaf_block) + LARGE_HEADER_SIZE
        _write(buf, base, _pack_bmap_record(leaf_index * 4, 20 + leaf_index, 2))
        _write(buf, base + RECORD_SIZE, _pack_bmap_record(leaf_index * 4 + 2, 30 + leaf_index, 1))

    return buf
