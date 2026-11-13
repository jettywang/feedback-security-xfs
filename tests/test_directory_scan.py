from __future__ import annotations

import io
from typing import BinaryIO

from dissect.xfs import (
    DirectoryBlockMagicError,
    DirectoryReadError,
    DirectoryRecordTruncatedError,
)
from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.xfs import ScanState, XFS

DIR_INODE = 11079
BLOCK_SIZE = 4096
# Absolute byte offset of the directory inode used by the generated regression images.
DIR_INODE_OFFSET = 1384 * BLOCK_SIZE + (DIR_INODE - 11072) * 512
FIRST_DIR_BLOCK = 1380
SECOND_DIR_BLOCK = 1381


class _ShortReadStream(io.RawIOBase):
    """File-like wrapper that returns only ``readable`` bytes from one target block.

    Reads fully inside or outside the target block behave normally; a read that would cover the
    whole target block is truncated to simulate a read interruption without affecting the inode
    area or other directory blocks.
    """

    def __init__(self, fh: BinaryIO, block_num: int, readable: int) -> None:
        self._fh = fh
        self._block_start = block_num * BLOCK_SIZE
        self._block_end = self._block_start + readable

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        return self._fh.seek(pos, whence)

    def tell(self) -> int:
        return self._fh.tell()

    def readinto(self, b: bytearray) -> int:
        pos = self._fh.tell()
        count = len(b)
        if self._block_start <= pos < self._block_end:
            count = min(count, self._block_end - pos)
        data = self._fh.read(count)
        b[: len(data)] = data
        return len(data)


def test_scan_complete_directory(xfs_dir_scan_complete_bin: BinaryIO) -> None:
    node = XFS(xfs_dir_scan_complete_bin).get_inode(DIR_INODE)
    result = node.scan_directory()

    assert [entry.filename for entry in result.entries] == ["alpha", "bravo", "charlie", "delta"]

    status = result.status
    assert status.state == ScanState.COMPLETE
    assert status.complete is True
    assert status.is_complete() is True
    assert status.is_empty() is False
    assert status.total_blocks == 3
    assert status.scanned_blocks == 3
    assert status.data_blocks == [0, 1]
    assert status.sparse_blocks == [2]
    assert status.corrupt_blocks == []
    assert status.unreadable_blocks == []
    assert status.truncated_blocks == []
    assert status.errors == []

    # Recovered entries retain their directory block location.
    locations = {entry.filename: (entry.dir_logical_block, entry.dir_physical_block) for entry in result.entries}
    assert locations == {
        "alpha": (0, FIRST_DIR_BLOCK),
        "bravo": (0, FIRST_DIR_BLOCK),
        "charlie": (1, SECOND_DIR_BLOCK),
        "delta": (1, SECOND_DIR_BLOCK),
    }


def test_scan_bad_block_directory(xfs_dir_scan_badblock_bin: BinaryIO) -> None:
    node = XFS(xfs_dir_scan_badblock_bin).get_inode(DIR_INODE)
    result = node.scan_directory()

    # Entries from the intact first block are still recovered.
    assert [entry.filename for entry in result.entries] == ["alpha", "bravo"]

    status = result.status
    assert status.state == ScanState.PARTIAL
    assert status.complete is False
    assert status.is_empty() is False
    assert status.data_blocks == [0]
    assert status.sparse_blocks == [2]
    assert status.corrupt_blocks == [1]
    assert status.affected_blocks == [1]
    assert status.entries_recovered == 2

    assert len(status.errors) == 1
    error = status.errors[0]
    assert isinstance(error, DirectoryBlockMagicError)
    assert error.block_num == 1

    recovered = {entry.filename: entry for entry in result.entries}
    assert recovered["alpha"].dir_logical_block == 0
    assert recovered["alpha"].dir_physical_block == FIRST_DIR_BLOCK


def test_iterdir_populates_queryable_status(xfs_dir_scan_badblock_bin: BinaryIO) -> None:
    node = XFS(xfs_dir_scan_badblock_bin).get_inode(DIR_INODE)
    assert node.dir_scan_status is None

    names = [entry.filename for entry in node.iterdir()]
    assert names == ["alpha", "bravo"]

    status = node.dir_scan_status
    assert status is not None
    assert status.state == ScanState.PARTIAL
    assert status.corrupt_blocks == [1]


def test_listdir_stays_compatible(xfs_dir_scan_complete_bin: BinaryIO) -> None:
    node = XFS(xfs_dir_scan_complete_bin).get_inode(DIR_INODE)

    listing = node.listdir()
    assert list(listing.keys()) == ["alpha", "bravo", "charlie", "delta"]
    assert list(node.dirlist().keys()) == list(listing.keys())
    assert node.dir_scan_status is not None
    assert node.dir_scan_status.complete is True


def test_intact_directory_is_complete(xfs_bin: BinaryIO) -> None:
    # test_dir is an intact shortform directory; an error-free scan must be reported as complete and
    # never confused with a directory whose blocks could not be read.
    node = XFS(xfs_bin).get("test_dir")
    result = node.scan_directory()

    assert [entry.filename for entry in result.entries] == [".", "..", "test_file"]
    assert result.status.state == ScanState.COMPLETE
    assert result.status.complete is True
    assert result.status.is_empty() is False
    assert result.status.errors == []


def test_zero_sized_block_directory_is_empty(xfs_dir_scan_complete_bin: BinaryIO) -> None:
    image = bytearray(xfs_dir_scan_complete_bin.read())
    # Shrink the directory inode to zero logical blocks so a block-format scan finds nothing but
    # still completes without errors: a genuine empty directory.
    image[DIR_INODE_OFFSET + 56 : DIR_INODE_OFFSET + 64] = (0).to_bytes(8, "big")

    node = XFS(io.BytesIO(image)).get_inode(DIR_INODE)
    result = node.scan_directory()

    assert result.entries == []
    assert result.status.state == ScanState.EMPTY
    assert result.status.complete is True
    assert result.status.is_empty() is True
    assert result.status.errors == []


def test_truncated_record_is_reported(xfs_dir_scan_truncated_bin: BinaryIO) -> None:
    node = XFS(xfs_dir_scan_truncated_bin).get_inode(DIR_INODE)
    result = node.scan_directory()

    # The intact entries are still recovered; only the trailing truncated record is reported.
    assert [entry.filename for entry in result.entries] == ["alpha", "bravo", "charlie", "delta"]
    assert result.status.state == ScanState.PARTIAL
    assert result.status.truncated_blocks == [1]
    assert isinstance(result.status.errors[0], DirectoryRecordTruncatedError)
    assert result.status.errors[0].block_num == 1


def test_read_interruption_is_reported(xfs_dir_scan_complete_bin: BinaryIO) -> None:
    xfs_dir_scan_complete_bin.seek(0, io.SEEK_END)
    size = xfs_dir_scan_complete_bin.tell()
    xfs_dir_scan_complete_bin.seek(0)

    # Short-read only the second directory data block; the inode area and first block stay intact.
    truncated = _ShortReadStream(xfs_dir_scan_complete_bin, SECOND_DIR_BLOCK, 128)
    node = XFS(truncated).get_inode(DIR_INODE)
    result = node.scan_directory()

    assert [entry.filename for entry in result.entries] == ["alpha", "bravo"]
    assert result.status.state == ScanState.PARTIAL
    assert result.status.unreadable_blocks == [1]
    assert isinstance(result.status.errors[0], DirectoryReadError)
    assert result.status.errors[0].block_num == 1
    assert size > SECOND_DIR_BLOCK * BLOCK_SIZE


def test_sparse_block_is_not_corruption(xfs_dir_scan_complete_bin: BinaryIO) -> None:
    node = XFS(xfs_dir_scan_complete_bin).get_inode(DIR_INODE)
    status = node.scan_directory().status

    assert status.sparse_blocks == [2]
    assert status.corrupt_blocks == []
    assert status.complete is True


def test_index_block_magic_is_benign() -> None:
    # Leaf/node and free-space index blocks are valid directory blocks without entries and must be
    # classified as index blocks, not corruption.
    from dissect.xfs.xfs import _DIR_DATA_MAGICS, _DIR_INDEX_MAGICS

    assert _DIR_DATA_MAGICS.isdisjoint(_DIR_INDEX_MAGICS)
    for magic in (
        c_xfs.XFS_DIR2_FREE_MAGIC,
        c_xfs.XFS_DIR3_FREE_MAGIC,
        c_xfs.XFS_DIR2_LEAF1_MAGIC,
        c_xfs.XFS_DIR2_LEAFN_MAGIC,
        c_xfs.XFS_DIR3_LEAF1_MAGIC,
        c_xfs.XFS_DIR3_LEAFN_MAGIC,
        c_xfs.XFS_DA_NODE_MAGIC,
        c_xfs.XFS_DA3_NODE_MAGIC,
    ):
        assert magic in _DIR_INDEX_MAGICS
        assert magic not in _DIR_DATA_MAGICS
