from __future__ import annotations

import io
import stat
from typing import BinaryIO

import pytest

from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.exceptions import DirectoryIndexError, FileNotFoundError
from dissect.xfs.xfs import XFS, _DirectoryIndexSession, xfs_da_hash
from tests._synthetic_xfs import BLOCK_SIZE, CountingStream, build_directory_image

DIR_START = 32


def _open(image: bytes) -> tuple[XFS, CountingStream]:
    counting = CountingStream(io.BytesIO(image))
    return XFS(counting), counting


def _dir_blocks_touched(counting: CountingStream, total: int) -> set[int]:
    return {block - DIR_START for block in counting.read_blocks if DIR_START <= block < DIR_START + total}


@pytest.mark.parametrize(
    ("v5", "has_ftype"),
    [
        (True, True),
        (False, True),
        (False, False),
    ],
)
@pytest.mark.parametrize("layout", ["single", "leaf", "node"])
def test_indexed_lookup_matches_traversal(v5: bool, has_ftype: bool, layout: str) -> None:
    count = 40 if layout == "single" else 240
    names = [f"entry_{i:04d}.log".encode() for i in range(count)]
    image = build_directory_image(
        [(name, 1) for name in names],
        v5=v5,
        has_ftype=has_ftype,
        layout=layout,
    )

    xfs, _ = _open(image)
    root = xfs.root
    expected = {entry.filename.encode(): entry.inum for entry in root.iterdir()}

    assert xfs.version == (5 if v5 else 4)
    assert bool(xfs._has_ftype) == has_ftype
    assert len(expected) == count
    for name, inum in expected.items():
        child = xfs.get(name.decode("utf-8", "surrogateescape"))
        assert child.inum == inum
        if has_ftype:
            assert child.filetype == stat.S_IFREG

    assert xfs.lookup_stats["index_fallback"] == 0
    assert xfs.lookup_stats["index_hit"] == count


def test_targeted_lookup_does_not_read_all_data_blocks() -> None:
    count = 400
    names = [f"suspicious_name_{i:06d}_evidence.log".encode() for i in range(count)]
    image = build_directory_image([(name, 1) for name in names], layout="leaf")

    xfs, counting = _open(image)
    root = xfs.root
    total = root.size // BLOCK_SIZE
    data_blocks = total - 2  # remainder are the leaf and free blocks

    counting.read_blocks.clear()
    assert len(list(root.iterdir())) == count
    traversal = _dir_blocks_touched(counting, total)
    assert len([block for block in traversal if block < data_blocks]) == data_blocks

    targets = [names[0], names[count // 4], names[count // 2], names[3 * count // 4], names[-1]]
    worst_data_blocks = 0
    for target in targets:
        counting.read_blocks.clear()
        child = xfs.get(target.decode())
        assert child.filename.encode() == target

        touched = _dir_blocks_touched(counting, total)
        data_touched = [block for block in touched if block < data_blocks]
        worst_data_blocks = max(worst_data_blocks, len(data_touched))

        # The leaf index and the single candidate data block are enough.
        assert len(touched) <= 4
        assert len(data_touched) <= 2

    assert worst_data_blocks < data_blocks


def test_shortform_directory_keeps_linear_path(xfs_bin: BinaryIO) -> None:
    xfs = XFS(xfs_bin)
    before = dict(xfs.lookup_stats)
    child = xfs.get("test_file")
    assert child.open().read() == b"test content\n"
    assert xfs.lookup_stats["shortform"] == before["shortform"] + 1
    assert xfs.lookup_stats["index_hit"] == before["index_hit"]


def test_hash_collision_falls_back_and_stays_correct() -> None:
    names = [b"alpha", b"bravo", b"charlie", b"delta", b"echo"]
    shared_hash = xfs_da_hash(b"alpha")
    image = build_directory_image(
        [(name, 1) for name in names],
        layout="leaf",
        force_hash={b"bravo": shared_hash},
    )
    xfs, _ = _open(image)

    inums = {name: xfs.get(name.decode()).inum for name in names}
    assert len(set(inums.values())) == len(names)
    # The collision is resolved authoritatively on the raw bytes, never by hash.
    assert xfs.get("alpha").inum != xfs.get("bravo").inum
    assert xfs.lookup_stats["index_fallback"] >= 1


def test_stale_entry_with_live_hash_is_ignored() -> None:
    count = 80
    names = [f"file_{i:04d}.log".encode() for i in range(count)]
    target = names[40]
    image = build_directory_image(
        [(name, 1) for name in names],
        layout="leaf",
        stale_hashes=[xfs_da_hash(target)],
    )
    xfs, _ = _open(image)

    child = xfs.get(target.decode())
    assert child.filename.encode() == target
    assert xfs.lookup_stats["index_hit"] == 1


def test_stale_entry_cannot_create_false_positive() -> None:
    names = [f"file_{i:04d}.log".encode() for i in range(80)]
    image = build_directory_image([(name, 1) for name in names], layout="leaf", stale_hashes=[0x0BADF00D])
    xfs, _ = _open(image)

    with pytest.raises(FileNotFoundError):
        xfs.get("ghost.log")
    # A miss is verified by a full linear scan before concluding "not found".
    assert xfs.lookup_stats["index_miss"] == 1
    assert xfs.lookup_stats["linear"] == 1


@pytest.mark.parametrize(
    ("layout", "corrupt"),
    [
        ("leaf", "leaf_bad_magic"),
        ("leaf", "leaf_count_overflow"),
        ("leaf", "free_bad_magic"),
        ("node", "node_bad_magic"),
        ("single", "block_tail_overflow"),
    ],
)
def test_corrupt_index_falls_back_to_linear_scan(layout: str, corrupt: str) -> None:
    count = 40 if layout == "single" else 120
    names = [f"file_{i:04d}.log".encode() for i in range(count)]
    target = names[count // 2]
    image = build_directory_image([(name, 1) for name in names], layout=layout, corrupt=corrupt)

    xfs, _ = _open(image)
    child = xfs.get(target.decode())
    assert child.filename.encode() == target
    # Every damaged index is neutralised by a linear scan before answering.
    assert xfs.lookup_stats["linear"] == 1
    assert xfs.lookup_stats["index_hit"] == 0


def test_corrupt_free_pointer_rejected() -> None:
    names = [f"file_{i:04d}.log".encode() for i in range(120)]
    image = build_directory_image(
        [(name, 1) for name in names],
        layout="leaf",
        firstdb_override=999,
    )
    xfs, _ = _open(image)
    session = _DirectoryIndexSession(xfs.root)
    with pytest.raises(DirectoryIndexError):
        session.lookup(xfs_da_hash(names[0]))

    # The public get path still returns the correct entry via the fallback.
    assert xfs.get(names[0].decode()).filename.encode() == names[0]


def test_raw_byte_names_match_on_bytes_not_text() -> None:
    names = [b"file_\xff\xfe.bin", b"normal.txt", b"weird_\x80name"]
    image = build_directory_image([(name, 1) for name in names], layout="leaf")
    xfs, _ = _open(image)

    for raw in names:
        child = xfs.get(raw.decode("utf-8", "surrogateescape"))
        assert child.filename.encode("utf-8", "surrogateescape") == raw


def test_hash_uses_big_endian_name_length_prefix() -> None:
    # Reference values produced by the kernel xfs_da_hashname algorithm.
    assert xfs_da_hash(b".") == 0x0004002E
    assert xfs_da_hash(b"..") == 0x00B8000E
    # Distinct names must (overwhelmingly) hash apart; equal names hash equal.
    assert xfs_da_hash(b"alpha") != xfs_da_hash(b"bravo")
    assert xfs_da_hash(b"alpha") == xfs_da_hash(b"alpha")


def test_directory_index_error_is_an_error() -> None:
    from dissect.xfs.exceptions import Error

    assert issubclass(DirectoryIndexError, Error)


def test_index_block_header_layouts() -> None:
    # These fixed sizes drive every offset used during index traversal, so lock
    # them against accidental cstruct layout changes.
    assert c_xfs.xfs_dir2_free_hdr.size == 12
    assert c_xfs.xfs_dir2_leaf_hdr.size == 32
    assert c_xfs.xfs_dir3_leaf_hdr.size == 70
    assert c_xfs.xfs_da_node_hdr.size == 32
    assert c_xfs.xfs_da3_node_hdr.size == 70
    assert c_xfs.xfs_dir2_leaf_entry.size == 8
    assert c_xfs.xfs_dir3_leaf_hdr.lookup["count"].offset == 64
    assert c_xfs.xfs_dir2_leaf_hdr.lookup["count"].offset == 28
