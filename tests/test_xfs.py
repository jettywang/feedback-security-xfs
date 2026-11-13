from __future__ import annotations

import datetime
import io
import stat
from typing import BinaryIO

import pytest

from dissect.xfs.exceptions import SymlinkUnavailableException
from dissect.xfs.xfs import XFS
from tests.symlink_image import build_xfs_symlink_image

PAYLOAD = 512 - 56
SYMLINK_INODE = 44


def make_fragments(target: bytes) -> list[tuple[int, bytes]]:
    return [(offset, target[offset : offset + PAYLOAD]) for offset in range(0, len(target), PAYLOAD)]


def read_link(image: bytes) -> str:
    xfs = XFS(io.BytesIO(image))
    return xfs.get_inode(SYMLINK_INODE).link


def test_xfs(xfs_bin: BinaryIO) -> None:
    xfs = XFS(xfs_bin)

    assert xfs.version == 5
    assert xfs.block_size == 4096
    assert str(xfs.uuid) == "3fb8342e-e144-4f0c-8bd7-725e78966200"
    assert str(xfs.meta_uuid) == "00000000-0000-0000-0000-000000000000"

    root = xfs.root
    assert root.inum == 11072
    assert root.filetype == stat.S_IFDIR
    assert root.crtime == datetime.datetime(2022, 4, 22, 14, 24, 9, 264560, tzinfo=datetime.timezone.utc)
    assert root.crtime_ns == 1650637449264560000
    assert list(root.listdir().keys()) == [".", "..", "test_file", "test_dir", "test_link"]

    test_file = xfs.get("test_file")
    assert test_file.nblocks == 1
    assert test_file.open().read() == b"test content\n"

    test_link = xfs.get("test_link")
    assert test_link.nblocks == 0
    assert test_link.filetype == stat.S_IFLNK
    assert test_link.link == "test_dir/test_file"


def test_xfs_sparse(xfs_sparse_bin: BinaryIO) -> None:
    xfs = XFS(xfs_sparse_bin)

    sparse_start = xfs.get("sparse_start")
    assert sparse_start.size == 0x258000
    assert sparse_start.nblocks == 200
    assert sparse_start.dataruns == [(None, 400), (1392, 200)]

    sparse_hole = xfs.get("sparse_hole")
    assert sparse_hole.size == 0x258000
    assert sparse_hole.nblocks == 400
    assert sparse_hole.dataruns == [(1792, 200), (None, 200), (2192, 200)]

    sparse_end = xfs.get("sparse_end")
    assert sparse_end.size == 0x190000
    assert sparse_end.nblocks == 200
    assert sparse_end.dataruns == [(2392, 200), (None, 200)]

    sparse_all = xfs.get("sparse_all")
    assert sparse_all.size == 0x500000
    assert sparse_all.nblocks == 0
    assert sparse_all.dataruns == [(None, 1280)]


def test_xfs_bigtime(xfs_bigtime_bin: BinaryIO) -> None:
    xfs = XFS(xfs_bigtime_bin)

    assert xfs.version == 5

    test_file = xfs.get("file")
    assert test_file._has_bigtime()
    assert test_file.crtime == datetime.datetime(2023, 4, 7, 9, 15, 9, 223364, tzinfo=datetime.timezone.utc)
    assert test_file.crtime_ns == 1680858909223364005


def test_symlinks(xfs_symlink_bin: BinaryIO) -> None:
    path = "/path/to/dir/with/file.ext"

    xfs = XFS(xfs_symlink_bin)
    assert xfs.get(path).link == "../../../../other/path/source/to/my/file.ext"



def test_symlink_fragmented_512_byte_blocks() -> None:
    target = b"/very/long/" + b"a" * 500 + b"/" + b"b" * 300 + b"/target"
    fragments = make_fragments(target)
    assert len(fragments) == 2

    assert read_link(build_xfs_symlink_image(fragments, owner=SYMLINK_INODE)) == target.decode()


def test_symlink_fragmented_out_of_order() -> None:
    target = b"/fragments/" + b"a" * 600 + b"/" + b"b" * 500 + b"/target"
    fragments = make_fragments(target)
    assert len(fragments) == 3

    image = build_xfs_symlink_image(fragments, owner=SYMLINK_INODE, physical_order=[2, 0, 1])
    assert read_link(image) == target.decode()


def test_symlink_fragmented_missing_fragment() -> None:
    target = b"/missing/" + b"a" * 600 + b"/" + b"b" * 300 + b"/target"
    fragments = make_fragments(target)
    del fragments[1]

    xfs = XFS(io.BytesIO(build_xfs_symlink_image(fragments, owner=SYMLINK_INODE, size=len(target))))
    node = xfs.get_inode(SYMLINK_INODE)

    with pytest.raises(SymlinkUnavailableException) as exc_info:
        node.link

    error = exc_info.value
    assert error.missing == [(PAYLOAD, 2 * PAYLOAD)]
    assert "456-912" in str(error)
    assert error.partial is not None
    assert error.partial[:PAYLOAD] == target[:PAYLOAD]
    assert error.partial[2 * PAYLOAD :] == target[2 * PAYLOAD :]


def test_symlink_fragmented_duplicate_fragment() -> None:
    target = b"/duplicate/" + b"a" * 600 + b"/" + b"b" * 300 + b"/target"
    fragments = make_fragments(target)
    fragments.append(fragments[1])

    assert read_link(build_xfs_symlink_image(fragments, owner=SYMLINK_INODE)) == target.decode()


def test_symlink_fragmented_duplicate_fragment_conflict() -> None:
    target = b"/conflict/" + b"a" * 700 + b"/target"
    fragments = make_fragments(target)
    conflicting = bytearray(fragments[1][1])
    conflicting[0] ^= 0xFF
    fragments.append((PAYLOAD, bytes(conflicting)))

    xfs = XFS(io.BytesIO(build_xfs_symlink_image(fragments, owner=SYMLINK_INODE)))
    node = xfs.get_inode(SYMLINK_INODE)

    with pytest.raises(SymlinkUnavailableException) as exc_info:
        node.link

    error = exc_info.value
    assert error.duplicates == [PAYLOAD]
    assert error.conflicts == [(PAYLOAD, PAYLOAD + 1)]
    assert "456-457" in str(error)


def test_symlink_fragmented_length_conflict() -> None:
    target = b"/length/" + b"a" * 600 + b"/" + b"b" * 300 + b"/target"
    fragments = make_fragments(target)
    lengths = [len(data) for _, data in fragments]
    lengths[1] += 100

    xfs = XFS(io.BytesIO(build_xfs_symlink_image(fragments, owner=SYMLINK_INODE, lengths=lengths)))
    node = xfs.get_inode(SYMLINK_INODE)

    with pytest.raises(SymlinkUnavailableException, match="length conflict"):
        node.link


def test_symlink_fragmented_foreign_fragment() -> None:
    target = b"/foreign/" + b"a" * 600 + b"/" + b"b" * 300 + b"/target"
    fragments = make_fragments(target)
    owners = [SYMLINK_INODE, SYMLINK_INODE + 1, SYMLINK_INODE]

    xfs = XFS(io.BytesIO(build_xfs_symlink_image(fragments, owner=SYMLINK_INODE, owners=owners)))
    node = xfs.get_inode(SYMLINK_INODE)

    with pytest.raises(SymlinkUnavailableException) as exc_info:
        node.link

    error = exc_info.value
    assert error.missing == [(PAYLOAD, 2 * PAYLOAD)]
    assert error.ignored == [(PAYLOAD, len(fragments[1][1]), "owner")]
    assert error.partial[:PAYLOAD] == target[:PAYLOAD]
    assert error.partial[2 * PAYLOAD :] == target[2 * PAYLOAD :]
