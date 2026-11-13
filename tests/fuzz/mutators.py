"""Seeded, target-aware mutators for the synthetic XFS B+Tree image.

Every mutation is a pure, idempotent transformation on a ``bytearray`` copy.
Mutators deliberately target the four index integrity properties requested:

* tree level (``bb_level``),
* record/pointer count (``bb_numrecs``),
* child node addresses (32-bit small tree / 64-bit large tree pointers),
* back edges (a child pointing at its ancestor, or a self loop),

plus header corruption and image truncation so short reads are reachable.
Mutations are scoped to the fuzzed tree so a case driving the small inode
btree never has its mutation land on an unrelated bmap block. The
:class:`random.Random` instance is passed in, making every campaign fully
reproducible from its seed.
"""

from __future__ import annotations

import random
import struct
from collections.abc import Callable

from dissect.xfs.c_xfs import c_xfs

from tests.fuzz.imagebuild import (
    AGBLKLOG,
    AG_BLOCKS,
    AG_COUNT,
    BLOCK_SIZE,
    LAYOUT,
    LARGE_HEADER_SIZE,
    LARGE_LEAF_BLOCKS,
    LARGE_MAXRECS,
    LARGE_ROOT_BLOCK,
    SMALL_HEADER_SIZE,
    SMALL_LEAF_BLOCKS,
    SMALL_MAXRECS,
    SMALL_ROOT_BLOCK,
    TOTAL_BLOCKS,
)
from tests.fuzz.targets import LARGE_TREE, SMALL_TREE

# bb_magic / bb_level / bb_numrecs live at the start of every btree block.
LEVEL_OFFSET = 4
NUMRECS_OFFSET = 6
MAGIC_OFFSET = 0

Mutation = Callable[[bytearray, random.Random], str]


def _patch(buf: bytearray, offset: int, data: bytes) -> None:
    buf[offset : offset + len(data)] = data


def _small_blocks() -> tuple[int, ...]:
    return (SMALL_ROOT_BLOCK, *SMALL_LEAF_BLOCKS)


def _large_blocks() -> tuple[int, ...]:
    return (LARGE_ROOT_BLOCK, *LARGE_LEAF_BLOCKS)


def _blocks_for(tree: str) -> tuple[int, ...]:
    return _small_blocks() if tree == SMALL_TREE else _large_blocks()


def _is_large(tree: str) -> bool:
    return tree == LARGE_TREE


def make_level_mutator(tree: str) -> Mutation:
    def mutate_level(buf: bytearray, rng: random.Random) -> str:
        block = rng.choice(_blocks_for(tree))
        offset = LAYOUT.block_offset(block) + LEVEL_OFFSET
        # 0 (leaf) is the most dangerous flip for an interior node; also try deep levels.
        value = rng.choice((0, 1, 2, 0xFFFF))
        _patch(buf, offset, struct.pack(">H", value))
        return f"level:block={block}:value={value}"

    return mutate_level


def make_numrecs_mutator(tree: str) -> Mutation:
    def mutate_numrecs(buf: bytearray, rng: random.Random) -> str:
        block = rng.choice(_blocks_for(tree))
        maxrecs = LARGE_MAXRECS if _is_large(tree) else SMALL_MAXRECS
        offset = LAYOUT.block_offset(block) + NUMRECS_OFFSET
        value = rng.choice((0, 1, maxrecs, maxrecs + 1, 0xFFFF))
        _patch(buf, offset, struct.pack(">H", value))
        return f"numrecs:block={block}:value={value}"

    return mutate_numrecs


def make_small_pointer_mutator(tree: str) -> Mutation:
    def mutate_small_pointer(buf: bytearray, rng: random.Random) -> str:
        index = rng.randrange(2)
        value = rng.choice(
            (
                SMALL_ROOT_BLOCK,  # back edge to the root (cycle)
                SMALL_LEAF_BLOCKS[index],  # self loop
                AG_BLOCKS,  # first block just outside the owning AG
                AG_BLOCKS + rng.randrange(1, AG_BLOCKS),
                0xFFFFFFFF,
                rng.randrange(TOTAL_BLOCKS),
            )
        )
        offset = LAYOUT.small_pointer_offset(index)
        _patch(buf, offset, struct.pack(">I", value))
        return f"small_ptr:index={index}:value={value}"

    return mutate_small_pointer


def make_large_pointer_mutator(tree: str) -> Mutation:
    def mutate_large_pointer(buf: bytearray, rng: random.Random) -> str:
        index = rng.randrange(2)
        cross_ag = (AG_COUNT - 1) * AG_BLOCKS + rng.randrange(AG_BLOCKS)  # valid cross-AG pointer
        bad_ag = AG_COUNT << AGBLKLOG  # allocation group that does not exist
        value = rng.choice(
            (
                LARGE_ROOT_BLOCK,  # back edge (cycle)
                LARGE_LEAF_BLOCKS[index],  # self loop
                cross_ag,  # legal: another AG
                bad_ag,
                bad_ag | rng.randrange(AG_BLOCKS),
                0xFFFFFFFFFFFFFFFF,
                TOTAL_BLOCKS + rng.randrange(1, 32),
            )
        )
        offset = LAYOUT.large_pointer_offset(index)
        _patch(buf, offset, struct.pack(">Q", value))
        return f"large_ptr:index={index}:value={value}"

    return mutate_large_pointer


def make_magic_mutator(tree: str) -> Mutation:
    def mutate_magic(buf: bytearray, rng: random.Random) -> str:
        block = rng.choice(_blocks_for(tree))
        offset = LAYOUT.block_offset(block) + MAGIC_OFFSET
        value = rng.choice((0, 0xDEADBEEF, c_xfs.XFS_IBT_MAGIC, c_xfs.XFS_BMAP_MAGIC))
        _patch(buf, offset, struct.pack(">I", value))
        return f"magic:block={block}:value=0x{value:x}"

    return mutate_magic


def make_header_garbage_mutator(tree: str) -> Mutation:
    def mutate_header_garbage(buf: bytearray, rng: random.Random) -> str:
        block = rng.choice(_blocks_for(tree))
        header_size = LARGE_HEADER_SIZE if _is_large(tree) else SMALL_HEADER_SIZE
        offset = LAYOUT.block_offset(block) + rng.randrange(header_size)
        byte_count = rng.randrange(1, min(8, max(1, len(buf) - offset)) + 1)
        payload = bytes(rng.randrange(256) for _ in range(byte_count))
        _patch(buf, offset, payload)
        return f"header_garbage:block={block}:offset={offset}:len={byte_count}"

    return mutate_header_garbage


def make_truncate_mutator(tree: str) -> Mutation:
    def truncate_image(buf: bytearray, rng: random.Random) -> str:
        # Cut within or before a block of the targeted tree to force a short read.
        target = rng.choice(_blocks_for(tree))
        cut = LAYOUT.block_offset(target) + rng.randrange(0, BLOCK_SIZE)
        cut = max(1, min(cut, len(buf)))
        del buf[cut:]
        return f"truncate:cut={cut}"

    return truncate_image


def mutators_for(tree: str) -> tuple[Mutation, ...]:
    pointer_mutator = make_small_pointer_mutator(tree) if tree == SMALL_TREE else make_large_pointer_mutator(tree)
    return (
        make_level_mutator(tree),
        make_numrecs_mutator(tree),
        pointer_mutator,
        make_magic_mutator(tree),
        make_header_garbage_mutator(tree),
        make_truncate_mutator(tree),
    )


def mutate(buf: bytearray, rng: random.Random, tree: str) -> list[str]:
    """Apply one or more seeded, target-scoped mutations and return descriptions."""
    available = mutators_for(tree)
    count = rng.choice((1, 1, 1, 2, 3))
    chosen = rng.sample(available, k=min(count, len(available)))
    return [mutation(buf, rng) for mutation in chosen]
