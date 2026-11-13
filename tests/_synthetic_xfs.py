from __future__ import annotations

import io
import struct
from typing import BinaryIO

from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.xfs import xfs_da_hash

BLOCK_SIZE = 4096
INODE_SIZE = 256
INOPBLOCK = BLOCK_SIZE // INODE_SIZE  # 16
AGBLOCKS = 256
ROOT_INUM = 128  # starts a fresh inode chunk -> byte offset 0x8000

UUID = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10"
DATA_HDR_V5 = c_xfs.xfs_dir3_data_hdr.size  # 64
DATA_HDR_V2 = c_xfs.xfs_dir2_data_hdr.size  # 16


class CountingStream:
    """Wrap an image and count which filesystem blocks the parser touches.

    Delegates actual positioning to the underlying binary stream so relative
    seeks (used by the dissect Range/Runlist streams) stay correct, while
    recording every 4 KiB block overlapped by a read.
    """

    def __init__(self, fh: BinaryIO):
        self.fh = fh
        self.read_blocks: set[int] = set()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self.fh.seek(offset, whence)

    def tell(self) -> int:
        return self.fh.tell()

    def read(self, size: int = -1) -> bytes:
        start = self.fh.tell()
        data = self.fh.read(size)
        self._record(start, len(data))
        return data

    def _record(self, start: int, length: int) -> None:
        if length == 0:
            return
        first = start // BLOCK_SIZE
        last = (start + length - 1) // BLOCK_SIZE
        for block in range(first, last + 1):
            self.read_blocks.add(block)

    def close(self) -> None:
        self.fh.close()


def _extent_record(start_block: int, fs_block: int, count: int) -> bytes:
    # flag(1) offset(54) block(52) count(21), two big endian uint64
    l0 = (0 << 63) | ((start_block & ((1 << 54) - 1)) << 9) | ((fs_block >> 43) & 0x1FF)
    l1 = ((fs_block & ((1 << 43) - 1)) << 21) | (count & ((1 << 21) - 1))
    return struct.pack(">QQ", l0, l1)


def _data_entry(inum: int, name: bytes, ftype: int, has_ftype: bool) -> tuple[bytes, int]:
    # Kernel geometry: entry_size = round_up(8 + namelen + (ftype?) + tag(2), 8).
    # cstruct reads inum(8) namelen(1) name(n) [ftype(1)] with no internal
    # padding, so write the body, zero the padding, and place the tag at end.
    body = struct.pack(">QB", inum, len(name)) + name
    if has_ftype:
        body += struct.pack(">B", ftype)
    entry_size = len(body) + 2  # include tag, then round the whole record up
    if entry_size % 8:
        entry_size += 8 - entry_size % 8
    return body, entry_size


def build_directory_image(
    entries: list[tuple[bytes, int]],
    *,
    v5: bool = True,
    has_ftype: bool = True,
    layout: str = "leaf",
    force_hash: dict[bytes, int] | None = None,
    stale_hashes: list[int] | None = None,
    corrupt: str | None = None,
    firstdb_override: int | None = None,
) -> bytes:
    """Build a minimal XFS image whose root inode is a block-format directory.

    ``entries`` is a list of (raw name, ftype) tuples. Names are packed into as
    many 4 KiB data blocks as needed, then a hash index (embedded leaf,
    standalone leaf, or node+leaf btree) is generated from the same names.

    Set ``has_ftype=False`` with ``v5=False`` for old dir2 images lacking the
    ftype feature. ``force_hash`` rewrites leaf hashvals (hash collisions),
    ``stale_hashes`` inserts address==0 entries, and ``corrupt`` selects an
    index damage scenario that must force a linear fallback.
    """
    force_hash = force_hash or {}
    stale_hashes = stale_hashes or []
    header_size = DATA_HDR_V5 if v5 else DATA_HDR_V2

    data_bytes: list[bytearray] = []
    block_ends: list[int] = []
    leaf_records: list[tuple[int, int]] = []  # (hash, data pointer)
    inode_counter = ROOT_INUM + 16

    def new_data_block() -> bytearray:
        buf = bytearray(BLOCK_SIZE)
        if v5:
            struct.pack_into(">I", buf, 0, c_xfs.XFS_DIR3_DATA_MAGIC)
            buf[16:32] = UUID
            struct.pack_into(">Q", buf, 32, ROOT_INUM)
        else:
            struct.pack_into(">I", buf, 0, c_xfs.XFS_DIR2_DATA_MAGIC)
        return buf

    buf = new_data_block()
    offset = header_size

    for name, ftype in entries:
        inum = inode_counter
        inode_counter += 1
        body, record_size = _data_entry(inum, name, ftype, has_ftype)
        # Keep 64 bytes free so a block never reaches the single-block tail.
        if offset + record_size > BLOCK_SIZE - 64:
            data_bytes.append(buf)
            block_ends.append(offset)
            buf = new_data_block()
            offset = header_size
        block_index = len(data_bytes)
        buf[offset : offset + record_size] = b"\x00" * record_size
        buf[offset : offset + len(body)] = body
        # tag (uint16) sits at the end of the 8-byte aligned record.
        struct.pack_into(">H", buf, offset + record_size - 2, offset)
        pointer = block_index * BLOCK_SIZE + offset
        leaf_records.append((force_hash.get(name, xfs_da_hash(name)), pointer))
        offset += record_size
    data_bytes.append(buf)
    block_ends.append(offset)

    # Mark the trailing free space of every data block as an unused record so
    # linear traversal stops instead of parsing zero padding as entries.
    for block, end in zip(data_bytes, block_ends):
        if end >= BLOCK_SIZE - 6:
            continue
        length = BLOCK_SIZE - end
        struct.pack_into(">HH", block, end, 0xFFFF, length)  # freetag, length
        struct.pack_into(">H", block, end + length - 2, end)  # back-pointer tag

    for hashval in stale_hashes:
        leaf_records.append((hashval, 0))
    leaf_records.sort(key=lambda rec: rec[0])

    num_data = len(data_bytes)
    if layout == "single":
        total_dir_blocks = 1
    elif layout == "leaf":
        total_dir_blocks = num_data + 2  # leaf + free
    elif layout == "node":
        total_dir_blocks = num_data + 3  # leaf + node + free
    else:
        raise ValueError(f"unknown layout {layout}")

    image = bytearray(BLOCK_SIZE * (AGBLOCKS + 16))
    dir_start = 32

    for i, block in enumerate(data_bytes):
        image[(dir_start + i) * BLOCK_SIZE : (dir_start + i + 1) * BLOCK_SIZE] = block

    _write_metadata(image, layout, dir_start, num_data, total_dir_blocks, leaf_records, v5, corrupt, firstdb_override)
    _write_superblock(image, v5, has_ftype)
    _write_root_inode(image, v5, dir_start, total_dir_blocks)
    return bytes(image)


def _write_metadata(
    image: bytearray,
    layout: str,
    dir_start: int,
    num_data: int,
    total_dir_blocks: int,
    leaf_records: list[tuple[int, int]],
    v5: bool,
    corrupt: str | None,
    firstdb_override: int | None,
) -> None:
    if layout == "single":
        base = dir_start * BLOCK_SIZE
        entries_start = BLOCK_SIZE - c_xfs.xfs_dir2_block_tail.size - len(leaf_records) * 8
        for i, (hashval, ptr) in enumerate(leaf_records):
            struct.pack_into(">II", image, base + entries_start + i * 8, hashval, ptr)
        tail_off = base + BLOCK_SIZE - c_xfs.xfs_dir2_block_tail.size
        struct.pack_into(">II", image, tail_off, len(leaf_records), 0)
        block_magic = c_xfs.XFS_DIR3_BLOCK_MAGIC if v5 else c_xfs.XFS_DIR2_BLOCK_MAGIC
        struct.pack_into(">I", image, base, block_magic)
        if corrupt == "block_tail_overflow":
            # Corrupt the leaf hash of the middle entry so the indexed lookup
            # misses it, while the data entries stay linear-scannable.
            victim = len(leaf_records) // 2
            struct.pack_into(">I", image, base + entries_start + victim * 8, 0x00000000)
        return

    leaf_logical = num_data
    free_logical = total_dir_blocks - 1
    leaf_phys = dir_start + leaf_logical
    free_phys = dir_start + free_logical

    if layout == "node":
        node_logical = total_dir_blocks - 2
        node_phys = dir_start + node_logical
        _write_node_block(image, node_phys, leaf_logical, leaf_records, v5, corrupt)
        leaf_magic16, leaf_magic32 = c_xfs.XFS_DIR2_LEAFN_MAGIC, c_xfs.XFS_DIR3_LEAFN_MAGIC
    else:
        leaf_magic16, leaf_magic32 = c_xfs.XFS_DIR2_LEAF1_MAGIC, c_xfs.XFS_DIR3_LEAF1_MAGIC

    _write_leaf_block(image, leaf_phys, leaf_records, v5, leaf_magic16, leaf_magic32, corrupt)
    firstdb = num_data if firstdb_override is None else firstdb_override
    _write_free_block(image, free_phys, firstdb, v5, corrupt)


def _write_leaf_block(
    image: bytearray,
    phys: int,
    records: list[tuple[int, int]],
    v5: bool,
    magic16: int,
    magic32: int,
    corrupt: str | None,
) -> None:
    base = phys * BLOCK_SIZE
    if v5:
        struct.pack_into(">I", image, base, magic32)
        image[base + 16 : base + 32] = UUID
        struct.pack_into(">Q", image, base + 32, ROOT_INUM)
        header_size = c_xfs.xfs_dir3_leaf_hdr.size
        count_off = base + c_xfs.xfs_dir3_leaf_hdr.lookup["count"].offset
    else:
        struct.pack_into(">H", image, base, magic16)
        header_size = c_xfs.xfs_dir2_leaf_hdr.size
        count_off = base + c_xfs.xfs_dir2_leaf_hdr.lookup["count"].offset
    stale = sum(1 for _, addr in records if addr == 0)
    struct.pack_into(">HH", image, count_off, len(records), stale)
    for i, (hashval, ptr) in enumerate(records):
        struct.pack_into(">II", image, base + header_size + i * 8, hashval, ptr)
    if corrupt == "leaf_count_overflow":
        struct.pack_into(">H", image, count_off, len(records) + 1000)
    if corrupt == "leaf_bad_magic":
        struct.pack_into(">I" if v5 else ">H", image, base, 0xDEAD)


def _write_node_block(
    image: bytearray,
    phys: int,
    leaf_logical: int,
    records: list[tuple[int, int]],
    v5: bool,
    corrupt: str | None,
) -> None:
    base = phys * BLOCK_SIZE
    if v5:
        struct.pack_into(">I", image, base, c_xfs.XFS_DIR3_NODE_MAGIC)
        image[base + 16 : base + 32] = UUID
        struct.pack_into(">Q", image, base + 32, ROOT_INUM)
        header_size = c_xfs.xfs_da3_node_hdr.size
        level_off = base + c_xfs.xfs_da3_node_hdr.lookup["level"].offset
    else:
        struct.pack_into(">H", image, base, c_xfs.XFS_DIR2_NODE_MAGIC)
        header_size = c_xfs.xfs_da_node_hdr.size
        level_off = base + c_xfs.xfs_da_node_hdr.lookup["level"].offset
    struct.pack_into(">HH", image, level_off, 1, 1)  # level=1, count=1
    struct.pack_into(">II", image, base + header_size, records[-1][0], leaf_logical)
    if corrupt == "node_bad_magic":
        struct.pack_into(">I" if v5 else ">H", image, base, 0xDEAD)


def _write_free_block(image: bytearray, phys: int, firstdb: int, v5: bool, corrupt: str | None) -> None:
    base = phys * BLOCK_SIZE
    magic = c_xfs.XFS_DIR3_FREE_MAGIC if v5 else c_xfs.XFS_DIR2_FREE_MAGIC
    struct.pack_into(">I", image, base, magic)
    if v5:
        image[base + 16 : base + 32] = UUID
        struct.pack_into(">Q", image, base + 32, ROOT_INUM)
    hdr_off = c_xfs.xfs_dir3_data_hdr.size if v5 else c_xfs.xfs_dir2_data_hdr.size
    struct.pack_into(">III", image, base + hdr_off, firstdb, 3, 0)
    if corrupt == "free_bad_magic":
        struct.pack_into(">I", image, base, 0xDEADBEEF)


def _write_superblock(image: bytearray, v5: bool, has_ftype: bool) -> None:
    sb = c_xfs.xfs_sb()
    sb.sb_magicnum = c_xfs.XFS_SB_MAGIC
    sb.sb_blocksize = BLOCK_SIZE
    sb.sb_dblocks = AGBLOCKS
    sb.sb_uuid = UUID
    sb.sb_meta_uuid = UUID
    sb.sb_rootino = ROOT_INUM
    sb.sb_agblocks = AGBLOCKS
    sb.sb_agcount = 1
    sb.sb_versionnum = 5 if v5 else 4
    sb.sb_sectsize = 512
    sb.sb_inodesize = INODE_SIZE
    sb.sb_inopblock = INOPBLOCK
    sb.sb_fname = b"testfs\x00" + b"\x00" * 5
    sb.sb_blocklog = 12
    sb.sb_sectlog = 9
    sb.sb_inodelog = 8
    sb.sb_inopblog = 4
    sb.sb_agblklog = 8
    sb.sb_dirblklog = 0
    if v5 and has_ftype:
        sb.sb_features_incompat = c_xfs.XFS_SB_FEAT_INCOMPAT_FTYPE
    elif has_ftype:
        sb.sb_features2 = c_xfs.XFS_SB_VERSION2_FTYPE
    data = sb.dumps()
    image[0 : len(data)] = data

    # AllocationGroup reads the AGI at sector 2; fill in a valid minimal one.
    agi = c_xfs.xfs_agi()
    agi.agi_magicnum = c_xfs.XFS_AGI_MAGIC
    agi.agi_versionnum = 1
    agi.agi_seqno = 0
    agi.agi_length = AGBLOCKS
    agi_data = agi.dumps()
    image[1024 : 1024 + len(agi_data)] = agi_data


def _write_root_inode(image: bytearray, v5: bool, dir_start: int, total_dir_blocks: int) -> None:
    off = ROOT_INUM * INODE_SIZE
    dinode = c_xfs.xfs_dinode()
    dinode.di_magic = c_xfs.XFS_DINODE_MAGIC
    dinode.di_mode = 0o40755
    dinode.di_version = 3 if v5 else 2
    dinode.di_format = c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS
    dinode.di_nlink = 2
    dinode.di_size = total_dir_blocks * BLOCK_SIZE
    dinode.di_nblocks = total_dir_blocks
    dinode.di_big_anextents = 1  # di_nextents when NREXT64 is not set
    dinode.di_forkoff = 0
    data = dinode.dumps()
    image[off : off + len(data)] = data
    fork_off = off + (0xB0 if v5 else 0x64)
    image[fork_off : fork_off + 16] = _extent_record(0, dir_start, total_dir_blocks)
