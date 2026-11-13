from __future__ import annotations

import io

import pytest

from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.exceptions import BTreeCycleError, Error, InvalidBlockAddressError
from dissect.xfs.xfs import BTREE_MAX_DEPTH, XFS

from tests.fuzz.imagebuild import (
    AG_BLOCKS,
    AG_COUNT,
    BLOCK_SIZE,
    LAYOUT,
    SMALL_HEADER_SIZE,
    SMALL_ROOT_BLOCK,
    build_valid_image,
)


def _xfs(image: bytearray | bytes) -> XFS:
    return XFS(io.BytesIO(bytes(image)))


def test_valid_image_walks_both_trees() -> None:
    xfs = _xfs(build_valid_image())

    records = list(xfs.get_allocation_group(0).walk_agi())
    assert len(records) == 4

    # Walking the large tree already dereferences a child pointer into AG 1;
    # reaching these records proves the legal cross-AG pointer was not rejected.
    extents = list(xfs.walk_extents(LAYOUT.large_root_block))
    assert len(extents) == 4
    assert {logical for logical, _, _, _ in extents} == {0, 2, 4, 6}


def test_large_tree_accepts_legit_cross_ag_pointer() -> None:
    image = build_valid_image()
    target_ag = AG_COUNT - 1
    cross_block = target_ag * AG_BLOCKS + 3
    image[LAYOUT.large_pointer_offset(0) : LAYOUT.large_pointer_offset(0) + 8] = cross_block.to_bytes(8, "big")
    # Provide a valid leaf at that otherwise-unused cross-AG block.
    leaf = c_xfs.xfs_btree_lblock()
    leaf.bb_magic = c_xfs.XFS_BMAP_MAGIC
    leaf.bb_level = 0
    leaf.bb_numrecs = 1
    image[cross_block * BLOCK_SIZE : cross_block * BLOCK_SIZE + len(leaf.dumps())] = leaf.dumps()

    xfs = _xfs(image)
    assert list(xfs.walk_extents(LAYOUT.large_root_block))


def test_small_tree_pointer_escaping_ag_is_rejected() -> None:
    image = build_valid_image()
    escaping = AG_BLOCKS  # one block past the owning AG
    image[LAYOUT.small_pointer_offset(0) : LAYOUT.small_pointer_offset(0) + 4] = escaping.to_bytes(4, "big")

    xfs = _xfs(image)
    with pytest.raises(InvalidBlockAddressError, match="escapes allocation group"):
        list(xfs.get_allocation_group(0).walk_agi())


def test_small_tree_root_escaping_ag_is_rejected() -> None:
    image = build_valid_image()
    xfs = _xfs(image)
    with pytest.raises(InvalidBlockAddressError):
        list(xfs.walk_small_tree(AG_BLOCKS, 0, 16, (c_xfs.XFS_IBT_MAGIC,)))


def test_large_tree_pointer_into_missing_ag_is_rejected() -> None:
    image = build_valid_image()
    missing_ag = (AG_COUNT << 6) | 1
    image[LAYOUT.large_pointer_offset(1) : LAYOUT.large_pointer_offset(1) + 8] = missing_ag.to_bytes(8, "big")

    xfs = _xfs(image)
    with pytest.raises(InvalidBlockAddressError, match="allocation group"):
        list(xfs.walk_extents(LAYOUT.large_root_block))


def test_small_tree_back_edge_is_a_cycle_not_a_hang() -> None:
    image = build_valid_image()
    # Point the root's first child back at the root itself.
    image[LAYOUT.small_pointer_offset(0) : LAYOUT.small_pointer_offset(0) + 4] = SMALL_ROOT_BLOCK.to_bytes(4, "big")

    xfs = _xfs(image)
    with pytest.raises(BTreeCycleError):
        list(xfs.get_allocation_group(0).walk_agi())


def test_large_tree_back_edge_is_a_cycle_not_a_hang() -> None:
    image = build_valid_image()
    root = LAYOUT.large_root_block
    image[LAYOUT.large_pointer_offset(0) : LAYOUT.large_pointer_offset(0) + 8] = root.to_bytes(8, "big")

    xfs = _xfs(image)
    with pytest.raises(BTreeCycleError):
        list(xfs.walk_extents(root))


def test_oversized_numrecs_is_rejected() -> None:
    image = build_valid_image()
    offset = LAYOUT.block_offset(SMALL_ROOT_BLOCK) + 6
    image[offset : offset + 2] = (0xFFFF).to_bytes(2, "big")

    xfs = _xfs(image)
    with pytest.raises(Error, match="at most"):
        list(xfs.get_allocation_group(0).walk_agi())


def test_bad_magic_is_format_rejection() -> None:
    image = build_valid_image()
    offset = LAYOUT.block_offset(SMALL_ROOT_BLOCK)
    image[offset : offset + 4] = (0xDEADBEEF).to_bytes(4, "big")

    xfs = _xfs(image)
    with pytest.raises(Error, match="invalid magic"):
        list(xfs.get_allocation_group(0).walk_agi())


def test_truncated_header_is_short_read() -> None:
    image = build_valid_image()
    cut = LAYOUT.block_offset(SMALL_ROOT_BLOCK) + 4
    xfs = XFS(io.BytesIO(bytes(image[:cut])))
    with pytest.raises(Error, match="Short read"):
        list(xfs.get_allocation_group(0).walk_agi())


def test_deep_non_repeating_chain_hits_depth_cap() -> None:
    image = build_valid_image()
    maxrecs = (BLOCK_SIZE - SMALL_HEADER_SIZE) // 8

    def put_interior(block: int, pointer: int) -> None:
        node = c_xfs.xfs_btree_sblock()
        node.bb_magic = c_xfs.XFS_IBT_MAGIC
        node.bb_level = 1
        node.bb_numrecs = 1
        offset = LAYOUT.block_offset(block)
        image[offset : offset + SMALL_HEADER_SIZE] = node.dumps()
        pointer_offset = offset + SMALL_HEADER_SIZE + maxrecs * 4
        image[pointer_offset : pointer_offset + 4] = pointer.to_bytes(4, "big")

    # 55 distinct interior blocks: deeper than the cap but every block stays inside AG 0.
    chain = [SMALL_ROOT_BLOCK, *range(8, 63)]
    assert len(chain) > BTREE_MAX_DEPTH and max(chain) < AG_BLOCKS
    for index, block in enumerate(chain[:-1]):
        put_interior(block, chain[index + 1])

    leaf = c_xfs.xfs_btree_sblock()
    leaf.bb_magic = c_xfs.XFS_IBT_MAGIC
    leaf.bb_level = 0
    leaf.bb_numrecs = 0
    leaf_offset = LAYOUT.block_offset(chain[-1])
    image[leaf_offset : leaf_offset + SMALL_HEADER_SIZE] = leaf.dumps()

    xfs = _xfs(image)
    with pytest.raises(BTreeCycleError, match="maximum depth"):
        list(xfs.get_allocation_group(0).walk_agi())
