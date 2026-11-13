from __future__ import annotations

import io
import logging
import os
import stat
import struct
from functools import cached_property
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID

from dissect.util import ts
from dissect.util.stream import RangeStream, RunlistStream

from dissect.xfs.c_xfs import FILETYPES, c_xfs
from dissect.xfs.exceptions import (
    DirectoryIndexError,
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

        # Observability counters for name lookups. ``index_*`` only count
        # block-format directories; shortform directories use ``shortform``.
        self.lookup_stats = {
            "shortform": 0,
            "index_hit": 0,
            "index_miss": 0,
            "index_fallback": 0,
            "linear": 0,
        }

        self.root = self.get_inode(self.sb.sb_rootino)

    def get(self, path: int | str, node: INode | None = None) -> INode:
        if isinstance(path, int):
            return self.get_inode(path)

        node = node if node else self.root

        parts = path.split("/")
        for part in parts:
            if not part:
                continue

            node = node.get_child(part)
            if node is None:
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
        return {node.filename: node for node in self.iterdir()}

    dirlist = listdir

    def get_child(self, name: str) -> INode | None:
        """Look up a single directory entry by name.

        Shortform (inline) directories are small enough to be read in full, so
        they keep using the linear traversal. Block-format directories use the
        hash-indexed leaf/node btree to touch only the relevant data blocks.

        The index is treated strictly as a hint: any corruption, stale entry
        or unsupported layout triggers a linear fallback scan, so an index
        problem can never produce a false "file does not exist" conclusion.
        Matching is always performed on the raw name bytes, never on the hash.
        """
        if self.filetype != stat.S_IFDIR:
            raise NotADirectoryError(f"{self!r} is not a directory")

        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL:
            self.xfs.lookup_stats["shortform"] += 1
            for entry in self.iterdir():
                if entry.filename == name:
                    return entry
            return None

        if self.inode.di_format not in (
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS,
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE,
        ):
            raise Error(f"{self!r} has invalid inode format for dirlist")

        try:
            return self._indexed_lookup(name)
        except DirectoryIndexError as err:
            log.warning("Directory index lookup failed for %r, falling back to linear scan: %s", self, err)
            self.xfs.lookup_stats["index_fallback"] += 1
            return self._linear_lookup(name)

    def _linear_lookup(self, name: str) -> INode | None:
        self.xfs.lookup_stats["linear"] += 1
        for entry in self.iterdir():
            if entry.filename == name:
                return entry
        return None

    def _indexed_lookup(self, name: str) -> INode | None:
        # Only the common directory block size equal to the filesystem block
        # size is handled by the index; dirblklog != 0 degrades to a scan.
        if self.xfs.sb.sb_dirblklog != 0:
            raise DirectoryIndexError(f"non-zero sb_dirblklog ({self.xfs.sb.sb_dirblklog}) is not indexable")

        if self.size < self.xfs.block_size:
            raise DirectoryIndexError("directory smaller than one block")

        session = _DirectoryIndexSession(self)
        target = name.encode("utf-8", "surrogateescape")
        target_hash = xfs_da_hash(target)

        candidate = session.lookup(target_hash)
        if candidate is None:
            # No live leaf entry for this hash. Verify the index answer with a
            # full scan so a stale/corrupt index cannot cause a false negative.
            self.xfs.lookup_stats["index_miss"] += 1
            return self._verify_miss(name)

        inode_num, entry_name, filetype = candidate
        if entry_name != target:
            # Hash collision with a differently named entry. The leaf may hold
            # more colliding entries, but rather than trust its completeness we
            # let the linear scan make the authoritative decision.
            log.debug("Hash collision in directory %r for name %r, verifying with linear scan", self, name)
            self.xfs.lookup_stats["index_fallback"] += 1
            return self._linear_lookup(name)

        self.xfs.lookup_stats["index_hit"] += 1
        return self.xfs.get_inode(inode_num, name, filetype)

    def _verify_miss(self, name: str) -> INode | None:
        self.xfs.lookup_stats["linear"] += 1
        for entry in self.iterdir():
            if entry.filename == name:
                log.warning("Directory index missed existing entry %r in %r", name, self)
                self.xfs.lookup_stats["index_fallback"] += 1
                return entry
        return None

    def iterdir(self) -> Iterator[INode]:
        if self.filetype != stat.S_IFDIR:
            raise NotADirectoryError(f"{self!r} is not a directory")

        buf = self.open()
        if self.inode.di_format == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL:
            header = c_xfs.xfs_dir2_sf_hdr(buf)
            inum_s = c_xfs.uint64 if header.i8count else c_xfs.uint32

            if header.i8count:
                header.parent = (header.parent << 32) | c_xfs.uint32(buf)

            yield self.xfs.get_inode(self.inum, filename=".")
            yield self.xfs.get_inode(header.parent, filename="..")

            for _ in range(header.count):
                entry = c_xfs.xfs_dir2_sf_entry(buf)
                fname = entry.name.decode(errors="surrogateescape")
                ftype = c_xfs.uint8(buf) if self.xfs._has_ftype else 0
                inum = inum_s(buf)

                ftype = FILETYPES[ftype]

                yield self.xfs.get_inode(inum, fname, ftype)
        elif self.inode.di_format in (
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS,
            c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE,
        ):
            for block_num in range(self.size // self.xfs.block_size):
                buf.seek(block_num * self.xfs.block_size)

                block_data = buf.read(self.xfs.block_size)
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

                        # Records start on an 8 byte boundary. cstruct has just
                        # consumed through the 2 byte tag, so advance from the
                        # cursor to the next boundary (no-op when already there).
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

                    yield self.xfs.get_inode(inum, fname, ftype)
        else:
            raise Error(f"{self!r} has invalid inode format for dirlist")

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


class _DirectoryIndexSession:
    """One hash-indexed name lookup against a block-format directory.

    A session caches the logical directory blocks it reads (de-duplicating
    repeated access) and counts those reads, so callers can demonstrate that a
    targeted lookup touches far fewer blocks than a full traversal.
    """

    LEAF_MAGICS = (
        c_xfs.XFS_DIR2_LEAF1_MAGIC,
        c_xfs.XFS_DIR2_LEAFN_MAGIC,
        c_xfs.XFS_DIR3_LEAF1_MAGIC,
        c_xfs.XFS_DIR3_LEAFN_MAGIC,
    )
    NODE_MAGICS = (c_xfs.XFS_DIR2_NODE_MAGIC, c_xfs.XFS_DIR3_NODE_MAGIC)

    def __init__(self, inode: INode):
        self.inode = inode
        self.xfs = inode.xfs
        self.block_size = self.xfs.block_size
        self.block_count = inode.size // self.block_size
        if self.block_count == 0:
            raise DirectoryIndexError("directory has no data blocks")

        self._cache: dict[int, bytes] = {}
        # Number of logical directory blocks pulled from the underlying stream.
        self.blocks_read = 0

    def lookup(self, target_hash: int) -> tuple[int, bytes, int | None] | None:
        leaf_block, embedded = self._find_leaf_block(target_hash)
        entries = self._read_embedded_leaf() if embedded else self._read_leaf_entries(leaf_block)

        match = self._binary_search(entries, target_hash)
        if match is None:
            return None

        hashval, address = match
        if address == 0:
            raise DirectoryIndexError(f"leaf entry for hash 0x{hashval:x} is stale (address == 0)")
        return self._read_data_entry(address)

    def _find_leaf_block(self, target_hash: int) -> tuple[int, bool]:
        """Return (logical block number, embedded-in-data-block)."""
        first = self._read_block(0)
        magic = struct.unpack(">I", first[:4])[0]

        if magic in (c_xfs.XFS_DIR2_BLOCK_MAGIC, c_xfs.XFS_DIR3_BLOCK_MAGIC):
            # Single-block directory: the leaf tail lives inside the data block.
            return 0, True
        if magic not in (c_xfs.XFS_DIR2_DATA_MAGIC, c_xfs.XFS_DIR3_DATA_MAGIC):
            raise DirectoryIndexError(f"unexpected magic 0x{magic:x} in first directory block")

        # Multiblock directory. The last logical block is either the freeindex
        # block (leaf format) or a leaf/node block, which tells us how the index
        # metadata is laid out.
        last_block = self.block_count - 1
        last = self._read_block(last_block)
        last_magic16 = struct.unpack(">H", last[:2])[0]
        last_magic32 = struct.unpack(">I", last[:4])[0]

        if last_magic32 in (c_xfs.XFS_DIR2_FREE_MAGIC, c_xfs.XFS_DIR3_FREE_MAGIC):
            # The free hdr follows the data block header (16 bytes in v2,
            # 64 bytes for the CRC-enabled v3 header).
            free_hdr_off = c_xfs.xfs_dir3_data_hdr.size if last_magic32 == c_xfs.XFS_DIR3_FREE_MAGIC else 16
            free = c_xfs.xfs_dir2_free_hdr(io.BytesIO(last[free_hdr_off:]))
            # firstdb is the first block beyond the data region, i.e. the leaf.
            leaf_block = free.firstdb
            if not 0 < leaf_block < last_block:
                raise DirectoryIndexError(f"free block points at invalid leaf block {leaf_block}")
        elif last_magic16 in self.LEAF_MAGICS or last_magic32 in self.LEAF_MAGICS:
            # No separate free block: the last block is the leaf (or node root).
            leaf_block = last_block
        else:
            raise DirectoryIndexError(f"unexpected magic 0x{last_magic32:x} in last directory block")

        leaf = self._read_block(leaf_block)
        leaf_magic16 = struct.unpack(">H", leaf[:2])[0]
        leaf_magic32 = struct.unpack(">I", leaf[:4])[0]
        if leaf_magic32 in self.NODE_MAGICS or leaf_magic16 == c_xfs.XFS_DIR2_NODE_MAGIC:
            return self._descend_node_btree(leaf_block, target_hash), False

        # In node format the node root sits one block above the first leaf.
        node_block = leaf_block + 1
        if node_block < last_block:
            candidate = self._read_block(node_block)
            candidate_magic16 = struct.unpack(">H", candidate[:2])[0]
            candidate_magic32 = struct.unpack(">I", candidate[:4])[0]
            if candidate_magic32 in self.NODE_MAGICS or candidate_magic16 == c_xfs.XFS_DIR2_NODE_MAGIC:
                return self._descend_node_btree(node_block, target_hash), False
            if candidate_magic32 in self.LEAF_MAGICS or candidate_magic16 in (
                c_xfs.XFS_DIR2_LEAF1_MAGIC,
                c_xfs.XFS_DIR2_LEAFN_MAGIC,
            ):
                # A second leaf without a node root is valid for leaf format.
                leaf_block = node_block
                leaf_magic16, leaf_magic32 = candidate_magic16, candidate_magic32
            else:
                raise DirectoryIndexError(
                    f"expected node root in block {node_block}, found magic 0x{candidate_magic32:x}"
                )

        if leaf_magic32 in self.LEAF_MAGICS or leaf_magic16 in (
            c_xfs.XFS_DIR2_LEAF1_MAGIC,
            c_xfs.XFS_DIR2_LEAFN_MAGIC,
        ):
            return leaf_block, False
        raise DirectoryIndexError(f"block {leaf_block} is neither a leaf nor a node (magic 0x{leaf_magic32:x})")

    def _descend_node_btree(self, root_block: int, target_hash: int) -> int:
        block_num = root_block
        visited: set[int] = set()
        while True:
            if block_num in visited or not 0 < block_num < self.block_count:
                raise DirectoryIndexError(f"cycle or out-of-range node block {block_num}")
            visited.add(block_num)

            block = self._read_block(block_num)
            header, header_size = self._parse_node_header(block)
            if header.count == 0:
                raise DirectoryIndexError(f"node block {block_num} has no records")

            entries_start = header_size
            entries = [
                struct.unpack(">II", block[entries_start + i * 8 : entries_start + (i + 1) * 8])
                for i in range(header.count)
            ]
            if entries[-1][0] < target_hash:
                raise DirectoryIndexError(
                    f"target hash 0x{target_hash:x} exceeds node range ending at 0x{entries[-1][0]:x}"
                )

            child = None
            for hashval, before in entries:
                if hashval >= target_hash:
                    child = before
                    break
            if child is None or not 0 < child < self.block_count:
                raise DirectoryIndexError(f"node entry points at invalid block {child}")

            child_block = self._read_block(child)
            child_magic16 = struct.unpack(">H", child_block[:2])[0]
            child_magic32 = struct.unpack(">I", child_block[:4])[0]
            if child_magic32 in self.NODE_MAGICS or child_magic16 == c_xfs.XFS_DIR2_NODE_MAGIC:
                block_num = child
                continue
            if child_magic32 in self.LEAF_MAGICS or child_magic16 in (
                c_xfs.XFS_DIR2_LEAF1_MAGIC,
                c_xfs.XFS_DIR2_LEAFN_MAGIC,
            ):
                return child
            raise DirectoryIndexError(f"node child {child} has unexpected magic 0x{child_magic32:x}")

    def _parse_node_header(self, block: bytes) -> tuple[object, int]:
        magic32 = struct.unpack(">I", block[:4])[0]
        if magic32 == c_xfs.XFS_DIR3_NODE_MAGIC:
            header = c_xfs.xfs_da3_node_hdr(io.BytesIO(block))
            return header, c_xfs.xfs_da3_node_hdr.size
        magic16 = struct.unpack(">H", block[:2])[0]
        if magic16 == c_xfs.XFS_DIR2_NODE_MAGIC:
            header = c_xfs.xfs_da_node_hdr(io.BytesIO(block))
            return header, c_xfs.xfs_da_node_hdr.size
        raise DirectoryIndexError(f"invalid node magic 0x{magic32:x}")

    def _read_embedded_leaf(self) -> list[tuple[int, int]]:
        block = self._read_block(0)
        tail = c_xfs.xfs_dir2_block_tail(io.BytesIO(block[len(block) - len(c_xfs.xfs_dir2_block_tail) :]))
        if tail.count == 0:
            raise DirectoryIndexError("single-block directory leaf tail reports zero entries")
        entries_end = len(block) - len(c_xfs.xfs_dir2_block_tail)
        entries_start = entries_end - tail.count * len(c_xfs.xfs_dir2_leaf_entry)
        header_size = c_xfs.xfs_dir3_data_hdr.size if self.xfs._has_crc else c_xfs.xfs_dir2_data_hdr.size
        if entries_start < header_size:
            raise DirectoryIndexError("leaf entries overlap the data block header")
        return [
            struct.unpack(">II", block[entries_start + i * 8 : entries_start + (i + 1) * 8]) for i in range(tail.count)
        ]

    def _read_leaf_entries(self, block_num: int) -> list[tuple[int, int]]:
        block = self._read_block(block_num)
        magic32 = struct.unpack(">I", block[:4])[0]
        if magic32 in (c_xfs.XFS_DIR3_LEAF1_MAGIC, c_xfs.XFS_DIR3_LEAFN_MAGIC):
            header_size = c_xfs.xfs_dir3_leaf_hdr.size
            count = c_xfs.xfs_dir3_leaf_hdr(io.BytesIO(block)).count
        else:
            magic16 = struct.unpack(">H", block[:2])[0]
            if magic16 not in (c_xfs.XFS_DIR2_LEAF1_MAGIC, c_xfs.XFS_DIR2_LEAFN_MAGIC):
                raise DirectoryIndexError(f"invalid leaf magic 0x{magic32:x} in block {block_num}")
            header_size = c_xfs.xfs_dir2_leaf_hdr.size
            count = c_xfs.xfs_dir2_leaf_hdr(io.BytesIO(block)).count

        if count == 0:
            raise DirectoryIndexError(f"leaf block {block_num} reports zero entries")
        entries_end = header_size + count * len(c_xfs.xfs_dir2_leaf_entry)
        if entries_end > len(block):
            raise DirectoryIndexError(f"leaf entries in block {block_num} overflow the block")
        return [struct.unpack(">II", block[header_size + i * 8 : header_size + (i + 1) * 8]) for i in range(count)]

    def _binary_search(self, entries: list[tuple[int, int]], target_hash: int) -> tuple[int, int] | None:
        # Leaf entries are sorted by hashval on disk. Find the first entry whose
        # hash is >= the target (lower bound).
        lo, hi = 0, len(entries)
        while lo < hi:
            mid = (lo + hi) // 2
            if entries[mid][0] < target_hash:
                lo = mid + 1
            else:
                hi = mid
        if lo >= len(entries) or entries[lo][0] != target_hash:
            return None
        return entries[lo]

    def _read_data_entry(self, address: int) -> tuple[int, bytes, int | None]:
        # With sb_dirblklog == 0 a data pointer encodes (logical block, offset).
        block_num, offset = divmod(address, self.block_size)
        if block_num >= self.block_count or offset >= self.block_size:
            raise DirectoryIndexError(f"leaf address 0x{address:x} points outside the directory")

        block = self._read_block(block_num)
        magic32 = struct.unpack(">I", block[:4])[0]
        if magic32 not in (
            c_xfs.XFS_DIR2_DATA_MAGIC,
            c_xfs.XFS_DIR3_DATA_MAGIC,
            c_xfs.XFS_DIR2_BLOCK_MAGIC,
            c_xfs.XFS_DIR3_BLOCK_MAGIC,
        ):
            raise DirectoryIndexError(f"data block {block_num} has unexpected magic 0x{magic32:x}")

        entry_stream = io.BytesIO(block)
        entry_stream.seek(offset)
        try:
            if self.xfs._has_ftype:
                entry = c_xfs.xfs_dir2_data_entry_ftype(entry_stream)
                filetype = FILETYPES[entry.ftype]
            else:
                entry = c_xfs.xfs_dir2_data_entry(entry_stream)
                filetype = None
            name = entry.name
        except EOFError as err:
            raise DirectoryIndexError(f"unable to parse data entry at 0x{address:x}: {err}")

        if not name or len(name) != entry.namelen or entry.namelen > 255:
            raise DirectoryIndexError(f"data entry at 0x{address:x} has invalid name length")
        if entry.inumber >> 48 == 0xFFFF:
            raise DirectoryIndexError(f"leaf address 0x{address:x} points at a free (unused) record")
        return entry.inumber, name, filetype

    def _read_block(self, block_num: int) -> bytes:
        if block_num not in self._cache:
            if not 0 <= block_num < self.block_count:
                raise DirectoryIndexError(f"logical block {block_num} out of range (0..{self.block_count - 1})")
            # Open a fresh stream per block so the dissect aligned read buffer
            # cannot pull neighbouring blocks into this lookup's read accounting.
            stream = self.inode.open()
            stream.seek(block_num * self.block_size)
            data = stream.read(self.block_size)
            if len(data) != self.block_size:
                raise DirectoryIndexError(f"short read of directory block {block_num} ({len(data)} bytes)")
            self._cache[block_num] = data
            self.blocks_read += 1
        return self._cache[block_num]


def xfs_da_hash(name: bytes) -> int:
    """Compute the XFS directory hash (xfs_da_hashname) of raw name bytes.

    Mirrors the kernel implementation, which prefixes the name with its 32 bit
    big endian length and folds the bytes together with a rotate-right-xor.
    """
    return _xfs_da_roll_hash(0, struct.pack(">I", len(name)) + name)


def _xfs_da_roll_hash(hashval: int, data: bytes) -> int:
    # Equivalent to the kernel's xfs_da_roll_hash / xfs_da_rotr. The rotate
    # count is the byte value masked to 5 bits. Python's shifts stay defined
    # where the C version relies on the mask keeping the count below 32.
    for byte in data:
        count = byte & 31
        rotr = ((hashval >> count) | (hashval << (32 - count))) & 0xFFFFFFFF
        hashval = rotr ^ byte
    return hashval


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
