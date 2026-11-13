from __future__ import annotations

import datetime
import hashlib
import io
import stat
import struct
from typing import BinaryIO

from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.xfs import XFS


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


def _extent_record(offset: int, block: int, count: int, flag: int) -> bytes:
    l0 = (flag << 63) | (offset << 9) | (block >> 43)
    l1 = ((block & ((1 << 43) - 1)) << 21) | count
    return struct.pack(">QQ", l0, l1)


def test_xfs_unwritten_extent_inline(xfs_bin: BinaryIO) -> None:
    image = bytearray(xfs_bin.read())

    xfs = XFS(io.BytesIO(image))
    test_file = xfs.get("test_file")
    assert test_file.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS
    [(offset, block, count, flag)] = test_file._iter_blocks()
    assert (offset, count, flag) == (0, 1, 0)

    # Flip the unwritten flag on the inline extent record, leaving the physical
    # block (and its residual bytes b"test content\n") untouched
    inode_offset = test_file.relative_inum * xfs.sb.sb_inodesize
    image[inode_offset + 0xB0] |= 0x80

    xfs = XFS(io.BytesIO(image))
    test_file = xfs.get("test_file")

    # The two views are distinguishable in their returned runs
    assert test_file.dataruns == [(None, 1)]
    assert test_file.raw_dataruns == [(block, 1)]

    # The default logical view reads the unwritten extent as zeroes
    logical = test_file.open().read()
    assert logical == b"\x00" * test_file.size
    assert hashlib.sha256(logical).digest() == hashlib.sha256(b"\x00" * test_file.size).digest()

    # The explicit raw view still exposes the physical residual bytes
    assert test_file.open_raw().read() == b"test content\n"

    # Partial reads
    fh = test_file.open()
    fh.seek(4)
    assert fh.read(6) == b"\x00" * 6
    fh = test_file.open_raw()
    fh.seek(4)
    assert fh.read(6) == b" conte"

    # End of file
    fh = test_file.open()
    fh.seek(test_file.size)
    assert fh.read() == b""
    fh = test_file.open_raw()
    fh.seek(test_file.size)
    assert fh.read() == b""


def test_xfs_unwritten_extent_btree(xfs_bin: BinaryIO) -> None:
    image = bytearray(xfs_bin.read())

    xfs = XFS(io.BytesIO(image))
    test_file = xfs.get("test_file")
    block_size = xfs.block_size
    inode_offset = test_file.relative_inum * xfs.sb.sb_inodesize

    # Free blocks at the end of the image, verified unused above
    leaf_block, data_block, unwritten_block = 4000, 4001, 4002

    written_data = bytes(range(256)) * (block_size // 256)
    stale_data = b"STALE" * (block_size // 5)
    image[data_block * block_size : (data_block + 1) * block_size] = written_data
    image[unwritten_block * block_size : (unwritten_block + 1) * block_size] = stale_data

    # B+tree leaf with one written and one unwritten extent
    leaf = struct.pack(">IHH", c_xfs.XFS_BMAP_CRC_MAGIC, 0, 2)
    leaf = leaf.ljust(len(c_xfs.xfs_btree_lblock_crc), b"\x00")
    leaf += _extent_record(0, data_block, 1, 0)
    leaf += _extent_record(1, unwritten_block, 1, 1)
    image[leaf_block * block_size : leaf_block * block_size + len(leaf)] = leaf

    # Convert test_file to a B+tree format inode pointing at the leaf
    datafork_offset = inode_offset + 0xB0
    datafork_size = test_file.inode.di_forkoff * 8
    maxrecs = (datafork_size - 4) // 16
    image[inode_offset + 5] = c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE.value
    struct.pack_into(">Q", image, inode_offset + 56, block_size + 100)  # di_size
    struct.pack_into(">HH", image, datafork_offset, 1, 1)  # bb_level, bb_numrecs
    struct.pack_into(">Q", image, datafork_offset + 4 + maxrecs * 8, leaf_block)  # bb_ptrs[0]

    xfs = XFS(io.BytesIO(image))
    test_file = xfs.get("test_file")
    assert test_file.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE
    assert test_file.size == block_size + 100

    # The two views are distinguishable in their returned runs
    assert test_file.dataruns == [(data_block, 1), (None, 1)]
    assert test_file.raw_dataruns == [(data_block, 1), (unwritten_block, 1)]

    # The default logical view reads the unwritten extent as zeroes
    logical = test_file.open().read()
    assert len(logical) == test_file.size
    assert logical[:block_size] == written_data
    assert logical[block_size:] == b"\x00" * 100
    assert hashlib.sha256(logical).digest() == hashlib.sha256(written_data + b"\x00" * 100).digest()

    # The explicit raw view still exposes the physical residual bytes
    raw = test_file.open_raw().read()
    assert raw[:block_size] == written_data
    assert raw[block_size:] == stale_data[:100]

    # Partial read crossing the extent boundary
    fh = test_file.open()
    fh.seek(block_size - 8)
    assert fh.read(16) == written_data[-8:] + b"\x00" * 8
    fh = test_file.open_raw()
    fh.seek(block_size - 8)
    assert fh.read(16) == written_data[-8:] + stale_data[:8]

    # End of file: reads stop at the logical size in both views
    for stream in (test_file.open(), test_file.open_raw()):
        fh = stream
        fh.seek(test_file.size)
        assert fh.read() == b""
        fh.seek(test_file.size - 4)
        assert len(fh.read(100)) == 4
