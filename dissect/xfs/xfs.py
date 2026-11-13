from __future__ import annotations

import io
import logging
import os
import stat
from functools import cached_property
from threading import RLock
from typing import TYPE_CHECKING, Any, BinaryIO
from uuid import UUID

from dissect.util import ts
from dissect.util.stream import RangeStream, RunlistStream

from dissect.xfs.c_xfs import FILETYPES, c_xfs
from dissect.xfs.exceptions import (
    Error,
    FileNotFoundError,
    NotADirectoryError,
    NotASymlinkError,
    UnsupportedDataforkException,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

log = logging.getLogger(__name__)
log.setLevel(os.getenv("DISSECT_LOG_XFS", "CRITICAL"))


class XFS:
    def __init__(self, fh: BinaryIO):
        self.fh = fh
        # Guards all cursor movement on the shared source handle. Every read path
        # must go through `_read_at`, which does an absolute seek/read while
        # holding this lock and hands the bytes back in a private buffer.
        self._lock = RLock()
        self.ag = {}
        # AG 0 is the bootstrap AG and is always read from the image handle.
        # Its superblock defines `_ag_size`, so other AGs can only be created
        # after this succeeds; they get private streams over the same image.
        self.ag[0] = AllocationGroup(self, fh, 0, 0)
        self.sb = self.ag[0].sb

        self.block_size = self.sb.sb_blocksize
        self.version = self.sb.sb_versionnum & c_xfs.XFS_SB_VERSION_NUMBITS
        self._has_ftype = (self.version == 5 and self.sb.sb_features_incompat & c_xfs.XFS_SB_FEAT_INCOMPAT_FTYPE) or (
            self.sb.sb_features2 & c_xfs.XFS_SB_VERSION2_FTYPE
        )
        self._has_crc = self.version == 5

        # This should be the same across all AG's
        self._inum_bits = self.sb.sb_agblklog + self.sb.sb_inopblog
        self._inum_mask = (1 << self._inum_bits) - 1
        self._inum_max = self.sb.sb_agblocks * self.sb.sb_inopblock
        # Not necessarily correct for the last AG, but doesn't harm
        self._ag_size = self.sb.sb_agblocks * self.block_size

        self._lblock_s = c_xfs.xfs_btree_lblock_crc if self._has_crc else c_xfs.xfs_btree_lblock
        self._sblock_s = c_xfs.xfs_btree_sblock_crc if self._has_crc else c_xfs.xfs_btree_sblock

        self.name = self.sb.sb_fname.split(b"\x00")[0].decode(errors="surrogateescape")
        self.uuid = UUID(bytes=self.sb.sb_uuid)
        self.meta_uuid = UUID(bytes=self.sb.sb_meta_uuid)

        self.root = self.get_inode(self.sb.sb_rootino)

    def _reader(self) -> SharedHandle:
        """Create an independent cursor over the shared image handle."""
        return SharedHandle(self)

    def _read_at(self, fh: BinaryIO, offset: int, size: int) -> bytes:
        """Read `size` bytes at an absolute `offset` from `fh` atomically.

        The underlying handle is shared by every scan, so no caller may depend on
        the cursor it leaves behind. Seeking and reading under one lock keeps a
        single read from being split by another scan's seek.
        """
        with self._lock:
            fh.seek(offset)
            return fh.read(size)

    def _size_of(self, fh: BinaryIO) -> int:
        """Return the total size of `fh` without disturbing reader state."""
        with self._lock:
            fh.seek(0, io.SEEK_END)
            return fh.tell()

    def _read_struct(self, fh: BinaryIO, offset: int, struct: Any) -> Any:
        """Parse a struct starting at an absolute `offset` of `fh`."""
        return struct(io.BytesIO(self._read_at(fh, offset, len(struct))))

    def get(self, path: int | str, node: INode | None = None) -> INode:
        if isinstance(path, int):
            return self.get_inode(path)

        node = node if node else self.root

        parts = path.split("/")
        for part in parts:
            if not part:
                continue

            # Each level gets a fresh iterator over a directory snapshot, so a
            # concurrently paused scan elsewhere cannot affect this lookup.
            directory = node.listdir()
            try:
                node = directory[part]
            except KeyError:
                raise FileNotFoundError(f"File not found: {path}") from None

        return node

    def get_allocation_group(self, agnum: int) -> AllocationGroup:
        if agnum not in self.ag:
            offset = agnum * self._ag_size
            self.ag[agnum] = AllocationGroup(self, self._ag_stream(agnum), agnum, offset)
        return self.ag[agnum]

    def _ag_stream(self, agnum: int) -> RangeStream:
        """A private stream over one allocation group.

        The stream gets its own cursor over the shared image handle, so cached
        AG streams can never be moved by a different scanner.
        """
        return RangeStream(self._reader(), agnum * self._ag_size, self._ag_size)

    def get_inode(self, absinum: int, *args, **kwargs) -> INode:
        return self.get_relative_inode(absinum >> self._inum_bits, absinum & self._inum_mask, *args, **kwargs)

    def get_relative_inode(self, agnum: int, inum: int, *args, **kwargs) -> INode:
        if agnum >= self.sb.sb_agcount:
            raise Error(f"Allocation group num exceeds number of allocation groups: {agnum} >= {self.sb.sb_agcount}")
        if inum >= self._inum_max:
            raise Error(f"inode number exceeds number of inodes per allocation group: {inum} >= {self._inum_max}")

        return self.get_allocation_group(agnum).get_inode(inum, *args, **kwargs)

    def walk_agi(self, block: int, agnum: int) -> Iterator[c_xfs.xfs_inobt_rec]:
        for record in self.walk_small_tree(block, agnum, 16, (c_xfs.XFS_IBT_MAGIC, c_xfs.XFS_IBT_CRC_MAGIC)):
            yield c_xfs.xfs_inobt_rec(record)

    def walk_extents(self, block: int) -> Iterator[tuple[int, int, int, int]]:
        for record in self.walk_large_tree(block, 16, (c_xfs.XFS_BMAP_MAGIC, c_xfs.XFS_BMAP_CRC_MAGIC)):
            yield parse_fsblock(record)

    def walk_large_tree(self, block: int, leaf_size: int, magic: list[int] | None = None) -> Iterator[bytes]:
        offset = block * self.block_size
        root = self._read_struct(self.fh, offset, self._lblock_s)

        yield from self._walk_large_tree(root, offset, leaf_size, magic)

    def walk_small_tree(
        self, block: int, agnum: int, leaf_size: int, magic: list[int] | None = None
    ) -> Iterator[bytes]:
        block = agnum * self.sb.sb_agblocks + block
        offset = block * self.block_size
        root = self._read_struct(self.fh, offset, self._sblock_s)

        yield from self._walk_small_tree(root, offset, leaf_size, agnum, magic)

    def _walk_small_tree(
        self,
        node: c_xfs.xfs_btree_sblock | c_xfs.xfs_btree_sblock_crc,
        node_offset: int,
        leaf_size: int,
        agnum: int,
        magic: list[int] | None = None,
    ) -> Iterator[bytes]:
        if magic and node.bb_magic not in magic:
            magic_values = ", ".join([f"0x{magic_value:x}" for magic_value in magic])
            raise Error(f"B+Tree node has invalid magic. Expected one of ({magic_values}), got 0x{node.bb_magic:x}")

        if node.bb_level == 0:
            buf = self._read_at(self.fh, node_offset + len(self._sblock_s), node.bb_numrecs * leaf_size)
            for rec_num in range(node.bb_numrecs):
                yield buf[rec_num * leaf_size : (rec_num + 1) * leaf_size]
        else:
            # Pointers start around halfway, we should already be at offset len(struct)
            maxrecs = (self.block_size - len(self._sblock_s)) // 8
            ptr_offset = node_offset + len(self._sblock_s) + maxrecs * 4
            ptrs = c_xfs.uint32[node.bb_numrecs](io.BytesIO(self._read_at(self.fh, ptr_offset, node.bb_numrecs * 4)))
            for ptr in ptrs:
                block = agnum * self.sb.sb_agblocks + ptr
                child_offset = block * self.block_size
                child = self._read_struct(self.fh, child_offset, self._sblock_s)

                yield from self._walk_small_tree(child, child_offset, leaf_size, agnum, magic)

    def _walk_large_tree(
        self,
        node: c_xfs.xfs_btree_lblock | c_xfs.xfs_btree_lblock_crc,
        node_offset: int,
        leaf_size: int,
        magic: list[int] | None = None,
    ) -> Iterator[bytes]:
        if magic and node.bb_magic not in magic:
            magic_values = ", ".join([f"0x{magic_value:x}" for magic_value in magic])
            raise Error(f"B+Tree node has invalid magic. Expected one of ({magic_values}), got 0x{node.bb_magic:x}")

        if node.bb_level == 0:
            buf = self._read_at(self.fh, node_offset + len(self._lblock_s), node.bb_numrecs * leaf_size)
            for rec_num in range(node.bb_numrecs):
                yield buf[rec_num * leaf_size : (rec_num + 1) * leaf_size]
        else:
            # Pointers start around halfway, we should already be at offset len(struct)
            maxrecs = (self.block_size - len(self._lblock_s)) // 16
            ptr_offset = node_offset + len(self._lblock_s) + maxrecs * 8
            ptrs = c_xfs.uint64[node.bb_numrecs](io.BytesIO(self._read_at(self.fh, ptr_offset, node.bb_numrecs * 8)))
            for ptr in ptrs:
                agnum, blknum = fsb_to_bb(ptr, self.sb.sb_agblklog)
                block = agnum * self.sb.sb_agblocks + blknum
                child_offset = block * self.block_size
                child = self._read_struct(self.fh, child_offset, self._lblock_s)

                yield from self._walk_large_tree(child, child_offset, leaf_size, magic)


class AllocationGroup:
    def __init__(self, xfs: XFS, fh: BinaryIO, num: int, offset: int | None = None):
        self.xfs = xfs
        self.fh = fh
        self.num = num
        # Absolute offset of this AG in the image. Metadata is read from the
        # image handle at this offset, never through the mutable `fh` cursor.
        self._offset = offset if offset is not None else num * xfs._ag_size

        sb = self.xfs._read_struct(self.xfs.fh, self._offset, c_xfs.xfs_sb)
        self.sb = sb

        if sb.sb_magicnum != c_xfs.XFS_SB_MAGIC:
            raise Error("Not a valid XFS filesystem (magic mismatch)")

        self.block_size = sb.sb_blocksize
        if self.block_size == 0 or self.block_size % 512:
            raise Error("Not a valid XFS filesystem (invalid block size)")

        # This seems to be all related to allocation
        # Do we really need to parse this?
        # self.agf = self.xfs._read_struct(self.xfs.fh, self._offset + sb.sb_sectsize * 1, c_xfs.xfs_agf)

        self.agi = self.xfs._read_struct(self.xfs.fh, self._offset + sb.sb_sectsize * 2, c_xfs.xfs_agi)
        if self.agi.agi_magicnum != c_xfs.XFS_AGI_MAGIC:
            raise Error("Not a valid XFS filesystem (AGI magic mismatch)")

        # This should be the same across all AG's
        self._inum_bits = sb.sb_agblklog + sb.sb_inopblog
        self._inum_mask = (1 << self._inum_bits) - 1
        self._inum_max = sb.sb_agblocks * sb.sb_inopblock

    def get_inode(
        self,
        inum: int,
        filename: str | None = None,
        filetype: int | None = None,
    ) -> INode:
        return INode(self, inum, filename, filetype)

    def inode_buf(self, inum: int) -> bytes:
        """Return the raw inode bytes for a relative inode number in this AG.

        Read from an absolute image offset instead of the AG's stream cursor, so
        the result does not depend on what any other scan has done to it.
        """
        offset = self._offset + inum * self.sb.sb_inodesize
        return self.xfs._read_at(self.xfs.fh, offset, self.sb.sb_inodesize)

    def walk_extents(self, fsb: int) -> Iterator[tuple[int, int, int, int]]:
        agnum, blknum = fsb_to_bb(fsb, self.sb.sb_agblklog)
        block = agnum * self.xfs.sb.sb_agblocks + blknum
        yield from self.xfs.walk_extents(block)

    def walk_agi(self) -> Iterator[c_xfs.xfs_inobt_rec]:
        yield from self.xfs.walk_agi(self.agi.agi_root, self.num)

    def walk_tree(self, fsb: int, magic: list[int] | None = None, small: bool = False) -> Iterator[bytes]:
        agnum, blknum = fsb_to_bb(fsb, self.sb.sb_agblklog)
        block = agnum * self.xfs.sb.sb_agblocks + blknum

        if small:
            yield from self.xfs.walk_small_tree(block, agnum, 16, magic)
        else:
            yield from self.xfs.walk_large_tree(block, 16, magic)


class SharedHandle(io.RawIOBase):
    """Independent cursor over an underlying file-like object shared by many readers.

    XFS parses metadata, directory blocks and file data from one image handle. Any
    code that seeks and then reads that handle depends on no other reader moving
    the cursor between the two calls. `SharedHandle` removes that dependency: it
    keeps its own logical position, performs every actual read with an absolute
    offset under the parser lock, and leaves the backing cursor wherever it was.

    It is a drop-in `BinaryIO` for structures that read sequentially, so
    ``c_xfs.bmdr_block(reader)``-style parsing can consume a private view.
    """

    def __init__(self, xfs: XFS, offset: int = 0, size: int | None = None, fh: BinaryIO | None = None):
        super().__init__()
        self._xfs = xfs
        self._fh = fh if fh is not None else xfs.fh
        self._offset = offset
        self._pos = 0
        self._size = size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    @property
    def size(self) -> int | None:
        return self._size

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = pos
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + pos
        elif whence == io.SEEK_END:
            if self._size is None:
                raise ValueError(f"invalid whence value for {self!r} with no size")
            new_pos = self._size + pos
        else:
            raise ValueError(f"invalid whence value: {whence}")

        if new_pos < 0:
            raise ValueError(f"negative seek position {new_pos}")

        self._pos = new_pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            if self._size is None:
                # Bound an unbounded read by the image size instead of trying to
                # materialize it. The end offset is captured through the same
                # absolute-read lock used by every other read.
                end = self._xfs._size_of(self._fh)
                n = max(0, end - self._offset - self._pos)
            else:
                n = self._size - self._pos

        if self._size is not None:
            n = min(n, max(0, self._size - self._pos))

        if n == 0:
            return b""

        data = self._xfs._read_at(self._fh, self._offset + self._pos, n)
        self._pos += len(data)
        return data

    def readinto(self, b: bytearray) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


class INode:
    def __init__(
        self,
        ag: AllocationGroup,
        inum: int,
        filename: str | None = None,
        filetype: int | None = None,
    ) -> None:
        self.ag = ag
        self.xfs = ag.xfs
        self.inum = inum + (ag.num << ag._inum_bits)
        self.relative_inum = inum
        self._buf = None

        self.filename = filename
        self._filetype = filetype

    def __repr__(self) -> str:
        return f"<inode {self.inum} ({self.ag.num}:{self.relative_inum})>"

    @cached_property
    def inode(self) -> c_xfs.xfs_dinode:
        self._buf = io.BytesIO(self.ag.inode_buf(self.relative_inum))
        inode = c_xfs.xfs_dinode(self._buf)

        if inode.di_magic != c_xfs.XFS_DINODE_MAGIC:
            raise Error(f"{self!r} has invalid inode magic")

        return inode

    def _has_bigtime(self) -> bool:
        return self.inode.di_version >= 3 and self.inode.di_flags2 & c_xfs.XFS_DIFLAG2_BIGTIME != 0

    def _has_large_extent_counts(self) -> bool:
        return self.inode.di_version >= 3 and self.inode.di_flags2 & c_xfs.XFS_DIFLAG2_NREXT64 != 0

    @property
    def size(self) -> int:
        return self.inode.di_size

    @property
    def nblocks(self) -> int:
        return self.inode.di_nblocks

    @property
    def data_extents(self) -> int:
        if self._has_large_extent_counts():
            return self.inode.di_big_nextents
        # Actually di_nextents, but we optimized the union away
        return self.inode.di_big_anextents

    @property
    def attr_extents(self) -> int:
        if self._has_large_extent_counts():
            return self.inode.di_big_anextents
        # Actually di_anextents, but we optimized the union away
        return self.inode.di_nrext64_pad

    @property
    def filetype(self) -> int:
        if not self._filetype:
            self._filetype = stat.S_IFMT(self.inode.di_mode)
        return self._filetype

    @cached_property
    def link(self) -> str:
        if self.filetype != stat.S_IFLNK:
            raise NotASymlinkError(f"{self!r} is not a symlink")

        if self.inode.di_format != c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL and self.xfs.version == 5:
            # Almost always, symlinks (max size of 1024) fit within a block. If the block size if 512, we might
            # need three blocks. These three blocks could theoretially be distributed over multiple extents.
            # Linux kernel handles this by using sl_offset to piece the symlink back together.
            # As this edge case of an edge case is very unlikely, it is unsupported until we observe it.
            # Ticket: https://github.com/fox-it/dissect.xfs/issues/36
            if len(self.dataruns) > 1:
                raise NotImplementedError(f"{self!r} has a symlink distributed over multiple extents")

            # We do not use open because for non-resident symlinks self.size does not include the symlink header
            symlink_size = len(c_xfs.xfs_dsymlink_hdr) + self.size
            fh = RunlistStream(self.xfs._reader(), self.dataruns, symlink_size, self.xfs.block_size)

            header = c_xfs.xfs_dsymlink_hdr(fh)
            if header.sl_magic != c_xfs.XFS_SYMLINK_MAGIC:
                raise NotASymlinkError(f"{self!r} has invalid symlink magic")

            return fh.read(header.sl_bytes).decode(errors="surrogateescape")

        return self.open().read().decode(errors="surrogateescape")

    @property
    def atime(self) -> datetime:
        return ts.from_unix_ns(self.atime_ns)

    @property
    def atime_ns(self) -> int:
        return _parse_ts(self.inode.di_atime, self._has_bigtime())

    @property
    def mtime(self) -> datetime:
        return ts.from_unix_ns(self.mtime_ns)

    @property
    def mtime_ns(self) -> int:
        return _parse_ts(self.inode.di_mtime, self._has_bigtime())

    @property
    def ctime(self) -> datetime:
        return ts.from_unix_ns(self.ctime_ns)

    @property
    def ctime_ns(self) -> int:
        return _parse_ts(self.inode.di_ctime, self._has_bigtime())

    @property
    def crtime(self) -> datetime:
        return ts.from_unix_ns(self.crtime_ns)

    @property
    def crtime_ns(self) -> int:
        return _parse_ts(self.inode.di_crtime, self._has_bigtime())

    def listdir(self) -> dict[str, INode]:
        return {node.filename: node for node in self.iterdir()}

    dirlist = listdir

    def iterdir(self) -> Iterator[INode]:
        if self.filetype != stat.S_IFDIR:
            raise NotADirectoryError(f"{self!r} is not a directory")

        buf = self.open()
        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL:
            header = c_xfs.xfs_dir2_sf_hdr(buf)
            # Parse all entries into local state before yielding, so a consumer
            # pausing this generator can never move the cursor under us.
            inum_s = c_xfs.uint64 if header.i8count else c_xfs.uint32

            if header.i8count:
                header.parent = (header.parent << 32) | c_xfs.uint32(buf)

            entries = [(b".", self.inum, self.filetype), (b"..", header.parent, stat.S_IFDIR)]
            for _ in range(header.count):
                entry = c_xfs.xfs_dir2_sf_entry(buf)
                fname = entry.name
                ftype = c_xfs.uint8(buf) if self.xfs._has_ftype else 0
                inum = inum_s(buf)

                entries.append((fname, inum, FILETYPES[ftype]))

            for fname, inum, ftype in entries:
                yield self.xfs.get_inode(inum, fname.decode(errors="surrogateescape"), ftype)
        elif self.inode.di_format in (
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS,
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE,
        ):
            for block_num in range((self.size + self.xfs.block_size - 1) // self.xfs.block_size):
                # One absolute read per block: no parser state depends on a
                # cursor that another scan can move.
                block_data = buf.readoffset(block_num * self.xfs.block_size, self.xfs.block_size)
                block = io.BytesIO(block_data)

                block.seek(-len(c_xfs.xfs_dir2_block_tail), io.SEEK_END)
                tail = c_xfs.xfs_dir2_block_tail(block)
                block.seek(0)

                if self.data_extents > 1:
                    entries_end = self.xfs.block_size
                else:
                    entries_end = self.xfs.block_size
                    entries_end -= len(c_xfs.xfs_dir2_block_tail)
                    entries_end -= len(c_xfs.xfs_dir2_leaf_entry) * tail.count

                if self.xfs.version == 5:
                    header = c_xfs.xfs_dir3_data_hdr(block)
                    if header.magic not in (c_xfs.XFS_DIR3_BLOCK_MAGIC, c_xfs.XFS_DIR3_DATA_MAGIC):
                        # Probably a sparse block
                        continue
                else:
                    header = c_xfs.xfs_dir2_data_hdr(block)
                    if header.magic not in (c_xfs.XFS_DIR2_BLOCK_MAGIC, c_xfs.XFS_DIR2_DATA_MAGIC):
                        # Probably a sparse block
                        continue

                data_entry = c_xfs.xfs_dir2_data_entry_ftype if self.xfs._has_ftype else c_xfs.xfs_dir2_data_entry
                entries = []

                while True:
                    if block.tell() >= entries_end:
                        break

                    try:
                        if block_data[block.tell() : block.tell() + 2] == b"\xff\xff":
                            unused = c_xfs.xfs_dir2_data_unused(block)

                            block.read(unused.length - 6)

                            misalign = block.tell() % 8
                            if misalign:
                                block.seek(8 - misalign, io.SEEK_CUR)

                            continue

                        entry = data_entry(block)

                        # Entries are 8 byte aligned
                        # uint64 inum | uint8 namelen | variable name | uint8 ftype | uint16 tag
                        misalign = block.tell() % 8
                        if misalign:
                            block.seek(8 - misalign, io.SEEK_CUR)
                    except EOFError:
                        break

                    inum = entry.inumber
                    if inum >> 48 == 0xFFFF:  # XFS_DIR2_DATA_FREE_TAG
                        break

                    ftype = FILETYPES[entry.ftype] if self.xfs._has_ftype else None
                    fname = entry.name.decode(errors="surrogateescape")

                    entries.append((fname, inum, ftype))

                for fname, inum, ftype in entries:
                    yield self.xfs.get_inode(inum, fname, ftype)
        else:
            raise Error(f"{self!r} has invalid inode format for dirlist")

    @property
    def fh(self) -> BinaryIO:
        """A cursor over this inode's bytes that no other scan can move.

        Fork streams seek their source, so handing out the shared inode buffer
        would let one reader move another's cursor. The inode buffer itself is
        immutable after parsing; each fork stream gets a new cursor over it.
        """
        self.inode
        return io.BytesIO(self._buf.getbuffer())

    def datafork(self) -> BinaryIO:
        offset = 0xB0 if self.inode.di_version == 0x3 else 0x64
        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL:
            size = self.size
        elif self.inode.di_forkoff:
            size = self.inode.di_forkoff * 8
        else:
            size = self.ag.sb.sb_inodesize - offset

        return RangeStream(self.fh, offset, size)

    def attrfork(self) -> BinaryIO:
        if self.inode.di_forkoff == 0:
            raise Error(f"{self!r} has no extended attributes")

        offset = (0xB0 if self.inode.di_version == 0x3 else 0x64) + self.inode.di_forkoff * 8
        size = self.ag.sb.sb_inodesize - offset

        return RangeStream(self.fh, offset, size)

    @cached_property
    def dataruns(self) -> list[tuple[int | None, int]]:
        runs = []
        run_offset = 0
        expected_runs = (self.size + self.xfs.block_size - 1) // self.xfs.block_size

        for offset, block, count, _ in self._iter_blocks():
            if offset != run_offset:
                gap = offset - run_offset
                runs.append((None, gap))
                run_offset += gap

            # Convert filesystem blocks to logical blocks
            agnum, blknum = fsb_to_bb(block, self.ag.sb.sb_agblklog)
            block = agnum * self.xfs.sb.sb_agblocks + blknum

            runs.append((block, count))
            run_offset += count

        if run_offset < expected_runs:
            runs.append((None, expected_runs - run_offset))

        return runs

    def _iter_blocks(self) -> Iterator[tuple[int, int, int, int]]:
        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS:
            buf = self.datafork().read(self.data_extents * 16)
            for extent_num in range(self.data_extents):
                yield parse_fsblock(buf[extent_num * 16 : (extent_num + 1) * 16])
        elif self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE:
            # B+tree extent lists are always large trees (64bit block numbers)
            df = self.datafork()
            root = c_xfs.xfs_bmdr_block(df)

            # Pointers start around halfway
            maxrecs = (df.size - 4) // 16
            df.seek(4 + maxrecs * 8)
            ptrs = c_xfs.uint64[root.bb_numrecs](df)

            for ptr in ptrs:
                yield from self.ag.walk_extents(ptr)

    def open(self) -> BinaryIO:
        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL:
            return self.datafork()
        if self.inode.di_format in (
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS,
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE,
        ):
            return RunlistStream(self.xfs._reader(), self.dataruns, self.size, self.xfs.block_size)
        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_DEV:
            raise UnsupportedDataforkException(f"{self!r} is a character/block device.")
        dinode_type = c_xfs.xfs_dinode_fmt(self.inode.di_format)
        raise UnsupportedDataforkException(f"{self!r} is an unsupported datafork: {dinode_type}")


def parse_fsblock(s: bytes) -> tuple[int, int, int, int]:
    # MSB -> LSB
    # flag = 1 bit
    # offset = 54 bits
    # block = 52 bits
    # count = 21 bits
    l0, l1 = c_xfs.uint64[2](s)

    flag = l0 >> 63
    offset = (l0 & ((1 << 63) - 1)) >> 9
    block = (l0 & ((1 << 9) - 1)) << 43 | (l1 >> 21)
    count = l1 & ((1 << 21) - 1)

    return offset, block, count, flag


def fsb_to_bb(block: int, agblklog: int) -> tuple[int, int]:
    agnum = block >> agblklog
    blknum = block & ((1 << agblklog) - 1)
    return agnum, blknum


def _parse_ts(ts: int, is_bigtime: bool) -> int:
    if is_bigtime:
        sec, nsec = divmod(ts, 1000000000)
        sec -= 1 << 31
    else:
        sec, nsec = ts >> 32, ts & 0xFFFFFFFF
    return (sec * 1000000000) + nsec
