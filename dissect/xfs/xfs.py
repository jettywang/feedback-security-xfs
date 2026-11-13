from __future__ import annotations

import io
import logging
import os
import stat
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID

from dissect.util import ts
from dissect.util.stream import RangeStream, RunlistStream

from dissect.xfs.c_xfs import FILETYPES, c_xfs
from dissect.xfs.exceptions import (
    DirectoryBlockMagicError,
    DirectoryReadError,
    DirectoryRecordTruncatedError,
    DirectoryScanError,
    Error,
    FileNotFoundError,
    NotADirectoryError,
    NotASymlinkError,
    UnsupportedDataforkException,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from datetime import datetime

log = logging.getLogger(__name__)
log.setLevel(os.getenv("DISSECT_LOG_XFS", "CRITICAL"))


class ScanState(str, Enum):
    """Outcome of scanning a single directory block or a whole directory."""

    COMPLETE = "complete"
    """Every block was read and parsed without errors."""

    PARTIAL = "partial"
    """At least one block was unreadable or malformed; recovered entries are still yielded."""

    EMPTY = "empty"
    """The scan completed without errors but the directory produced no entries."""


@dataclass
class DirectoryScanStatus:
    """Queryable integrity information for a directory scan.

    A status object is attached to every directory inode while it is being scanned through
    :meth:`INode.iterdir` or :meth:`INode.scan_directory`. Inspect :attr:`complete` (or the
    convenience :meth:`is_complete` / :meth:`is_empty` helpers) to distinguish a genuinely empty
    directory from one whose blocks were not fully read.

    Block numbers in the per-block tracking are logical block numbers relative to the start of the
    directory inode, matching the indexing used by :attr:`INode.dataruns`. Physical filesystem block
    numbers, when known, are stored on the recovered :class:`INode` entries and in :attr:`errors`.
    """

    total_blocks: int = 0
    """Number of directory blocks the inode size says the directory spans."""

    scanned_blocks: int = 0
    """Number of blocks that were read in full."""

    data_blocks: list[int] = field(default_factory=list)
    """Logical block numbers holding directory entry data."""

    sparse_blocks: list[int] = field(default_factory=list)
    """Logical block numbers that are unallocated sparse holes (all zeroes)."""

    index_blocks: list[int] = field(default_factory=list)
    """Logical block numbers holding leaf/node/free-space index blocks rather than entries."""

    corrupt_blocks: list[int] = field(default_factory=list)
    """Logical block numbers with an unrecognized (non-benign) magic number."""

    unreadable_blocks: list[int] = field(default_factory=list)
    """Logical block numbers that could not be read in full."""

    truncated_blocks: list[int] = field(default_factory=list)
    """Logical block numbers whose entry parsing hit an unexpected end of block."""

    entries_recovered: int = 0
    """Number of directory entries successfully parsed, including ``.`` and ``..``."""

    errors: list[DirectoryScanError] = field(default_factory=list)
    """Recoverable errors encountered during the scan."""

    def _record_error(self, error: DirectoryScanError) -> None:
        self.errors.append(error)
        block_num = error.block_num
        if isinstance(error, DirectoryReadError) and block_num not in self.unreadable_blocks:
            self.unreadable_blocks.append(block_num)
        elif isinstance(error, DirectoryBlockMagicError) and block_num not in self.corrupt_blocks:
            self.corrupt_blocks.append(block_num)
        elif isinstance(error, DirectoryRecordTruncatedError) and block_num not in self.truncated_blocks:
            self.truncated_blocks.append(block_num)

    @property
    def complete(self) -> bool:
        """Whether every directory block was read and parsed without errors."""
        return not self.errors and self.scanned_blocks == self.total_blocks

    @property
    def state(self) -> ScanState:
        """Coarse-grained scan outcome."""
        if self.errors:
            return ScanState.PARTIAL
        if self.entries_recovered == 0:
            return ScanState.EMPTY
        return ScanState.COMPLETE

    def is_complete(self) -> bool:
        """Return whether the scan covered every block without errors."""
        return self.complete

    def is_empty(self) -> bool:
        """Return whether an error-free scan yielded no entries."""
        return self.complete and self.entries_recovered == 0

    @property
    def affected_blocks(self) -> list[int]:
        """Logical block numbers involved in any recoverable error, sorted and de-duplicated."""
        return sorted({error.block_num for error in self.errors if error.block_num >= 0})


@dataclass
class DirectoryScan:
    """Result of a fully materialized directory scan.

    ``entries`` only contains entries that could be recovered; consult ``status`` to learn whether
    the scan was complete or blocks were skipped.
    """

    entries: list[INode]
    status: DirectoryScanStatus


class XFS:
    def __init__(self, fh: BinaryIO):
        self.fh = fh
        self.ag = {0: AllocationGroup(self, fh, 0)}
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

    def get(self, path: int | str, node: INode | None = None) -> INode:
        if isinstance(path, int):
            return self.get_inode(path)

        node = node if node else self.root

        parts = path.split("/")
        for part in parts:
            if not part:
                continue

            for entry in node.iterdir():
                if entry.filename == part:
                    node = entry
                    break
            else:
                raise FileNotFoundError(f"File not found: {path}")

        return node

    def get_allocation_group(self, agnum: int) -> AllocationGroup:
        if agnum not in self.ag:
            self.ag[agnum] = AllocationGroup(self, RangeStream(self.fh, agnum * self._ag_size, self._ag_size), agnum)
        return self.ag[agnum]

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
        self.fh.seek(block * self.block_size)
        root = self._lblock_s(self.fh)

        yield from self._walk_large_tree(root, leaf_size, magic)

    def walk_small_tree(
        self, block: int, agnum: int, leaf_size: int, magic: list[int] | None = None
    ) -> Iterator[bytes]:
        block = agnum * self.sb.sb_agblocks + block
        self.fh.seek(block * self.block_size)
        root = self._sblock_s(self.fh)

        yield from self._walk_small_tree(root, leaf_size, agnum, magic)

    def _walk_small_tree(
        self,
        node: c_xfs.xfs_btree_sblock | c_xfs.xfs_btree_sblock_crc,
        leaf_size: int,
        agnum: int,
        magic: list[int] | None = None,
    ) -> Iterator[bytes]:
        fh = self.fh
        if magic and node.bb_magic not in magic:
            magic_values = ", ".join([f"0x{magic_value:x}" for magic_value in magic])
            raise Error(f"B+Tree node has invalid magic. Expected one of ({magic_values}), got 0x{node.bb_magic:x}")

        if node.bb_level == 0:
            buf = fh.read(node.bb_numrecs * leaf_size)
            for rec_num in range(node.bb_numrecs):
                yield buf[rec_num * leaf_size : (rec_num + 1) * leaf_size]
        else:
            # Pointers start around halfway, we should already be at offset len(struct)
            maxrecs = (self.block_size - len(self._sblock_s)) // 8
            fh.seek(maxrecs * 4, io.SEEK_CUR)
            ptrs = c_xfs.uint32[node.bb_numrecs](fh)
            for ptr in ptrs:
                block = agnum * self.sb.sb_agblocks + ptr
                fh.seek(block * self.block_size)

                yield from self._walk_small_tree(self._sblock_s(fh), leaf_size, agnum, magic)

    def _walk_large_tree(
        self,
        node: c_xfs.xfs_btree_lblock | c_xfs.xfs_btree_lblock_crc,
        leaf_size: int,
        magic: list[int] | None = None,
    ) -> Iterator[bytes]:
        fh = self.fh
        if magic and node.bb_magic not in magic:
            magic_values = ", ".join([f"0x{magic_value:x}" for magic_value in magic])
            raise Error(f"B+Tree node has invalid magic. Expected one of ({magic_values}), got 0x{node.bb_magic:x}")

        if node.bb_level == 0:
            buf = fh.read(node.bb_numrecs * leaf_size)
            for rec_num in range(node.bb_numrecs):
                yield buf[rec_num * leaf_size : (rec_num + 1) * leaf_size]
        else:
            # Pointers start around halfway, we should already be at offset len(struct)
            maxrecs = (self.block_size - len(self._lblock_s)) // 16
            fh.seek(maxrecs * 8, io.SEEK_CUR)
            ptrs = c_xfs.uint64[node.bb_numrecs](fh)
            for ptr in ptrs:
                agnum, blknum = fsb_to_bb(ptr, self.sb.sb_agblklog)
                block = agnum * self.sb.sb_agblocks + blknum
                fh.seek(block * self.block_size)

                yield from self._walk_large_tree(self._lblock_s(fh), leaf_size, magic)


class AllocationGroup:
    def __init__(self, xfs: XFS, fh: BinaryIO, num: int):
        self.xfs = xfs
        self.fh = fh
        self.num = num

        fh.seek(0)

        self.sb = c_xfs.xfs_sb(fh)
        sb = self.sb

        if sb.sb_magicnum != c_xfs.XFS_SB_MAGIC:
            raise Error("Not a valid XFS filesystem (magic mismatch)")

        self.block_size = sb.sb_blocksize
        if self.block_size == 0 or self.block_size % 512:
            raise Error("Not a valid XFS filesystem (invalid block size)")

        # This seems to be all related to allocation
        # Do we really need to parse this?
        # fh.seek(sb.sb_sectsize * 1)
        # self.agf = c_xfs.xfs_agf(fh)

        fh.seek(sb.sb_sectsize * 2)
        self.agi = c_xfs.xfs_agi(fh)
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


class INode:
    def __init__(
        self,
        ag: AllocationGroup,
        inum: int,
        filename: str | None = None,
        filetype: int | None = None,
        dir_logical_block: int | None = None,
        dir_physical_block: int | None = None,
    ) -> None:
        self.ag = ag
        self.xfs = ag.xfs
        self.inum = inum + (ag.num << ag._inum_bits)
        self.relative_inum = inum
        self._buf = None

        self.filename = filename
        self._filetype = filetype

        # Location of the directory entry that produced this inode. ``dir_logical_block`` is the
        # directory block relative to the parent inode and ``dir_physical_block`` the filesystem
        # block it was stored in. Both are ``None`` for inodes not recovered from a directory scan.
        self.dir_logical_block = dir_logical_block
        self.dir_physical_block = dir_physical_block

        self._dir_scan_status: DirectoryScanStatus | None = None

    def __repr__(self) -> str:
        return f"<inode {self.inum} ({self.ag.num}:{self.relative_inum})>"

    @cached_property
    def inode(self) -> c_xfs.xfs_dinode:
        self.ag.fh.seek(self.relative_inum * self.ag.sb.sb_inodesize)
        self._buf = io.BytesIO(self.ag.fh.read(self.ag.sb.sb_inodesize))
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
            fh = RunlistStream(self.xfs.fh, self.dataruns, symlink_size, self.xfs.block_size)

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
        """Return the recoverable directory entries as a ``{name: inode}`` mapping.

        This is the backwards-compatible entry point: it never raises on damaged directory blocks.
        Entries that can still be parsed are returned even when the scan is incomplete. Use
        :attr:`dir_scan_status` after iteration to inspect whether the listing is complete.
        """
        return {node.filename: node for node in self.iterdir()}

    dirlist = listdir

    def scan_directory(self) -> DirectoryScan:
        """Materialize the directory scan and return both the entries and the integrity status.

        Unlike :meth:`iterdir` this consumes the whole scan eagerly so the returned
        :class:`DirectoryScanStatus` is final and can be queried directly.
        """
        status = DirectoryScanStatus()
        entries = list(self._iterdir(status))
        return DirectoryScan(entries, status)

    @property
    def dir_scan_status(self) -> DirectoryScanStatus | None:
        """The status of the most recent directory scan, or ``None`` before a scan ran."""
        return self._dir_scan_status

    def iterdir(self) -> Iterator[INode]:
        """Yield the recoverable entries of this directory.

        Damaged or unreadable blocks are recorded on :attr:`dir_scan_status` instead of aborting the
        whole scan, so a damaged directory still yields everything that could be recovered. Query the
        status afterwards to tell a genuinely empty directory apart from an incomplete scan.
        """
        status = DirectoryScanStatus()
        self._dir_scan_status = status
        yield from self._iterdir(status)

    def _iterdir(self, status: DirectoryScanStatus) -> Iterator[INode]:
        if self.filetype != stat.S_IFDIR:
            raise NotADirectoryError(f"{self!r} is not a directory")

        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL:
            yield from self._iterdir_shortform(status)
        elif self.inode.di_format in (
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS,
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE,
        ):
            yield from self._iterdir_blocks(status)
        else:
            raise Error(f"{self!r} has invalid inode format for dirlist")

    def _emit(self, status: DirectoryScanStatus, node: INode) -> INode:
        status.entries_recovered += 1
        return node

    def _iterdir_shortform(self, status: DirectoryScanStatus) -> Iterator[INode]:
        # A shortform directory lives entirely inside the inode, so there are no directory blocks to
        # read and a completed parse is always complete.
        status.total_blocks = 0
        status.scanned_blocks = 0

        buf = self.open()
        header = c_xfs.xfs_dir2_sf_hdr(buf)
        inum_s = c_xfs.uint64 if header.i8count else c_xfs.uint32

        if header.i8count:
            header.parent = (header.parent << 32) | c_xfs.uint32(buf)

        yield self._emit(status, self.xfs.get_inode(self.inum, filename="."))
        yield self._emit(status, self.xfs.get_inode(header.parent, filename=".."))

        for _ in range(header.count):
            entry = c_xfs.xfs_dir2_sf_entry(buf)
            fname = entry.name.decode(errors="surrogateescape")
            ftype = c_xfs.uint8(buf) if self.xfs._has_ftype else 0
            inum = inum_s(buf)

            ftype = FILETYPES[ftype]

            yield self._emit(status, self.xfs.get_inode(inum, fname, ftype))

    def _iterdir_blocks(self, status: DirectoryScanStatus) -> Iterator[INode]:
        buf = self.open()
        runlist = self.dataruns
        total_blocks = self.size // self.xfs.block_size
        status.total_blocks = total_blocks

        for block_num in range(total_blocks):
            buf.seek(block_num * self.xfs.block_size)

            try:
                block_data = buf.read(self.xfs.block_size)
            except OSError as exc:
                status._record_error(
                    DirectoryReadError(f"{self!r} directory block {block_num} could not be read: {exc}", block_num)
                )
                log.warning("Unable to read directory block %d of %r: %s", block_num, self, exc)
                continue

            if len(block_data) < self.xfs.block_size:
                status._record_error(
                    DirectoryReadError(
                        f"{self!r} directory block {block_num} short read: "
                        f"got {len(block_data)} of {self.xfs.block_size} bytes",
                        block_num,
                    )
                )
                log.warning(
                    "Short read on directory block %d of %r (%d/%d bytes)",
                    block_num,
                    self,
                    len(block_data),
                    self.xfs.block_size,
                )
                continue

            status.scanned_blocks += 1
            block = io.BytesIO(block_data)
            physical_block = self._physical_dir_block(runlist, block_num)
            magic = int.from_bytes(block_data[:4], "big")

            # Sparse directory blocks are unallocated holes and always read back as zeroes.
            if magic == 0:
                status.sparse_blocks.append(block_num)
                continue

            # Leaf/node and free-space index blocks manage the directory hash index and allocation;
            # they never contain directory entries and must not be reported as corruption.
            if magic in _DIR_INDEX_MAGICS:
                status.index_blocks.append(block_num)
                continue

            if magic not in _DIR_DATA_MAGICS:
                status._record_error(
                    DirectoryBlockMagicError(
                        f"{self!r} directory block {block_num} has unrecognized magic 0x{magic:x}", block_num
                    )
                )
                log.warning("Invalid magic 0x%x on directory block %d of %r", magic, block_num, self)
                continue

            status.data_blocks.append(block_num)
            yield from self._iterdir_data_block(block, block_data, block_num, physical_block, status)

    def _iterdir_data_block(
        self,
        block: io.BytesIO,
        block_data: bytes,
        block_num: int,
        physical_block: int | None,
        status: DirectoryScanStatus,
    ) -> Iterator[INode]:
        block.seek(-len(c_xfs.xfs_dir2_block_tail), io.SEEK_END)
        tail = c_xfs.xfs_dir2_block_tail(block)
        block.seek(0)

        # Skip past the data header; its magic was already validated against _DIR_DATA_MAGICS.
        if self.xfs.version == 5:
            c_xfs.xfs_dir3_data_hdr(block)
        else:
            c_xfs.xfs_dir2_data_hdr(block)

        if self.data_extents > 1:
            entries_end = self.xfs.block_size
        else:
            entries_end = self.xfs.block_size
            entries_end -= len(c_xfs.xfs_dir2_block_tail)
            entries_end -= len(c_xfs.xfs_dir2_leaf_entry) * tail.count

        data_entry = c_xfs.xfs_dir2_data_entry_ftype if self.xfs._has_ftype else c_xfs.xfs_dir2_data_entry

        while block.tell() < entries_end:
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
            except EOFError as exc:
                status._record_error(
                    DirectoryRecordTruncatedError(
                        f"{self!r} directory block {block_num} contains a truncated record", block_num
                    )
                )
                log.warning("Truncated directory record in block %d of %r: %s", block_num, self, exc)
                return

            inum = entry.inumber
            if inum >> 48 == 0xFFFF:  # XFS_DIR2_DATA_FREE_TAG
                return

            ftype = FILETYPES[entry.ftype] if self.xfs._has_ftype else None
            fname = entry.name.decode(errors="surrogateescape")

            node = self.xfs.get_inode(inum, fname, ftype)
            node.dir_logical_block = block_num
            node.dir_physical_block = physical_block
            yield self._emit(status, node)

    @staticmethod
    def _physical_dir_block(runlist: Sequence[tuple[int | None, int]], block_num: int) -> int | None:
        """Resolve a logical directory block to its physical filesystem block, if allocated."""
        offset = 0
        for run_block, run_count in runlist:
            if block_num < offset + run_count:
                if run_block is None:
                    return None
                return run_block + (block_num - offset)
            offset += run_count
        return None

    def datafork(self) -> BinaryIO:
        offset = 0xB0 if self.inode.di_version == 0x3 else 0x64
        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL:
            size = self.size
        elif self.inode.di_forkoff:
            size = self.inode.di_forkoff * 8
        else:
            size = self.ag.sb.sb_inodesize - offset

        return RangeStream(self._buf, offset, size)

    def attrfork(self) -> BinaryIO:
        if self.inode.di_forkoff == 0:
            raise Error(f"{self!r} has no extended attributes")

        offset = 0xB0 if self.inode.di_version == 0x3 else 0x64
        offset += self.inode.di_forkoff * 8
        size = self.ag.sb.sb_inodesize - offset

        return RangeStream(self._buf, offset, size)

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
            return RunlistStream(self.xfs.fh, self.dataruns, self.size, self.xfs.block_size)
        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_DEV:
            raise UnsupportedDataforkException(f"{self!r} is a character/block device.")
        dinode_type = c_xfs.xfs_dinode_fmt(self.inode.di_format)
        raise UnsupportedDataforkException(f"{self!r} is an unsupported datafork: {dinode_type}")


# Magic numbers of blocks that actually carry directory entries.
_DIR_DATA_MAGICS = frozenset(
    (
        c_xfs.XFS_DIR2_BLOCK_MAGIC,
        c_xfs.XFS_DIR2_DATA_MAGIC,
        c_xfs.XFS_DIR3_BLOCK_MAGIC,
        c_xfs.XFS_DIR3_DATA_MAGIC,
    )
)

# Benign directory blocks without entries: hash leaf/node blocks and free-space index blocks.
# These are valid parts of a larger directory and must never be flagged as corruption.
_DIR_INDEX_MAGICS = frozenset(
    (
        c_xfs.XFS_DIR2_FREE_MAGIC,
        c_xfs.XFS_DIR3_FREE_MAGIC,
        c_xfs.XFS_DIR2_LEAF1_MAGIC,
        c_xfs.XFS_DIR2_LEAFN_MAGIC,
        c_xfs.XFS_DIR3_LEAF1_MAGIC,
        c_xfs.XFS_DIR3_LEAFN_MAGIC,
        c_xfs.XFS_DA_NODE_MAGIC,
        c_xfs.XFS_DA3_NODE_MAGIC,
    )
)


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
