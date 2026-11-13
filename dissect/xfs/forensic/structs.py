"""On-disk XFS structure definitions used by the inode forensics scanner.

Everything here is parsed directly with :mod:`struct` (big-endian, which is the
on-disk byte order of XFS) so that the forensic tooling has no third-party
dependency and every byte offset can be audited against the layout documented
in the Linux kernel / ``xfsprogs`` headers.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Magic numbers and feature flags (values mirror xfs_format.h / libxfs).
# ---------------------------------------------------------------------------
XFS_SB_MAGIC = 0x58465342  # 'XFSB'
XFS_AGF_MAGIC = 0x58414746  # 'XAGF'
XFS_AGI_MAGIC = 0x58414749  # 'XAGI'
XFS_AGFL_MAGIC = 0x5841464C  # 'XAFL'
XFS_DINODE_MAGIC = 0x494E  # 'IN'

XFS_IBT_MAGIC = 0x49414254  # 'IABT'  (inobt, no CRC)
XFS_IBT_CRC_MAGIC = 0x49414233  # 'IAB3'  (inobt, CRC/v5)
XFS_FIBT_MAGIC = 0x46494254  # 'FIBT'  (finobt, no CRC)
XFS_FIBT_CRC_MAGIC = 0x46494233  # 'FIB3'  (finobt, CRC/v5)

INOBT_MAGICS = (XFS_IBT_MAGIC, XFS_IBT_CRC_MAGIC)
FINOBT_MAGICS = (XFS_FIBT_MAGIC, XFS_FIBT_CRC_MAGIC)

XFS_SB_FEAT_INCOMPAT_FTYPE = 1 << 0
XFS_SB_FEAT_INCOMPAT_SPINODES = 1 << 1

# Inode record (v2/sparse) layout masks.
XFS_INOBT_MASK_NR_MASK = 0x0000000F
XFS_INOBT_MASK_NR_SHIFT = 28
XFS_INOBT_HOLEMASK_FULL = 0xFFFF  # 16 chunk-64 hole bits
XFS_INOBT_FREEMASK_FULL = 0xFFFFFFFFFFFFFFFF

# dinode di_version
XFS_DINODE_VERSION_3 = 3

# di_format values we need for distinguishing directory data forks.
XFS_DINODE_FMT_DEV = 0
XFS_DINODE_FMT_LOCAL = 1
XFS_DINODE_FMT_EXTENTS = 2
XFS_DINODE_FMT_BTREE = 3
XFS_DINODE_FMT_UUID = 4
XFS_DINODE_FMT_RMAP = 5

# Directory entry file types (xfs_dir3_ftype / S_IFMT >> 12).
FTYPE_DIR = 2
FTYPE_REG = 1
FTYPE_LNK = 7

# Mode type bits used when constructing synthetic inodes.
S_IFMT = 0o170000
S_IFREG = 0o100000
S_IFDIR = 0o040000
S_IFLNK = 0o120000


class ReadFailure(Exception):
    """Raised when a required on-disk region cannot be fully read.

    This deliberately never subclasses :class:`ValueError` so callers can keep
    "unreadable" strictly separate from "well-formed but logically invalid".
    """


class StructureError(Exception):
    """Raised when a structure is fully readable but fails a sanity check."""


def _u16(buf: bytes, off: int) -> int:
    return struct.unpack_from(">H", buf, off)[0]


def _u32(buf: bytes, off: int) -> int:
    return struct.unpack_from(">I", buf, off)[0]


def _u64(buf: bytes, off: int) -> int:
    return struct.unpack_from(">Q", buf, off)[0]


@dataclass(frozen=True)
class Superblock:
    raw: bytes
    block_size: int
    dblocks: int
    rootino: int
    agblocks: int
    agcount: int
    versionnum: int
    sectsize: int
    inodesize: int
    inopblock: int
    inopblog: int
    agblklog: int
    icount: int
    ifree: int
    inoalignmt: int
    features_incompat: int
    spino_align: int

    @property
    def version(self) -> int:
        return self.versionnum & 0x000F

    @property
    def has_crc(self) -> bool:
        return self.version == 5

    @property
    def has_spinodes(self) -> bool:
        return bool(self.features_incompat & XFS_SB_FEAT_INCOMPAT_SPINODES)

    @property
    def has_ftype(self) -> bool:
        return bool(self.features_incompat & XFS_SB_FEAT_INCOMPAT_FTYPE)

    @property
    def inum_bits(self) -> int:
        return self.agblklog + self.inopblog

    @property
    def inodes_per_ag_max(self) -> int:
        return self.agblocks * self.inopblock


def parse_superblock(raw: bytes) -> Superblock:
    if len(raw) < 0x100:
        raise ReadFailure(f"superblock truncated: only {len(raw)} bytes")
    if _u32(raw, 0x00) != XFS_SB_MAGIC:
        raise StructureError(f"superblock magic mismatch: 0x{_u32(raw, 0x00):08x}")
    # Offsets follow the on-disk struct xfs_sb layout in xfs_format.h.
    return Superblock(
        raw=raw,
        block_size=_u32(raw, 0x04),
        dblocks=_u64(raw, 0x08),
        rootino=_u64(raw, 0x38),
        agblocks=_u32(raw, 0x54),
        agcount=_u32(raw, 0x58),
        versionnum=_u16(raw, 0x64),
        sectsize=_u16(raw, 0x66),
        inodesize=_u16(raw, 0x68),
        inopblock=_u16(raw, 0x6A),
        inopblog=raw[0x7B],
        agblklog=raw[0x7C],
        icount=_u64(raw, 0x80),
        ifree=_u64(raw, 0x88),
        inoalignmt=_u32(raw, 0xB4),
        features_incompat=_u32(raw, 0xD8),
        spino_align=_u32(raw, 0xE4),
    )


@dataclass(frozen=True)
class AGI:
    magic: int
    seqno: int
    length: int
    count: int
    root: int
    level: int
    freecount: int
    newino: int
    dirino: int
    unlinked: tuple[int, ...]
    free_root: int
    free_level: int
    iblocks: int
    fblocks: int


def parse_agi(raw: bytes) -> AGI:
    if len(raw) < 0x80:
        raise ReadFailure(f"AGI header truncated: only {len(raw)} bytes")
    magic = _u32(raw, 0x00)
    if magic != XFS_AGI_MAGIC:
        raise StructureError(f"AGI magic mismatch: 0x{magic:08x}")
    unlinked = tuple(_u32(raw, 0x34 + i * 4) for i in range(64))
    return AGI(
        magic=magic,
        seqno=_u32(raw, 0x08),
        length=_u32(raw, 0x0C),
        count=_u32(raw, 0x10),
        root=_u32(raw, 0x14),
        level=_u32(raw, 0x18),
        freecount=_u32(raw, 0x1C),
        newino=_u32(raw, 0x20),
        dirino=_u32(raw, 0x24),
        unlinked=unlinked,
        free_root=_u32(raw, 0x60),
        free_level=_u32(raw, 0x64),
        iblocks=_u32(raw, 0x68),
        fblocks=_u32(raw, 0x6C),
    )


@dataclass(frozen=True)
class AGF:
    magic: int
    seqno: int
    length: int
    freeblks: int


def parse_agf(raw: bytes) -> AGF:
    if len(raw) < 0x50:
        raise ReadFailure(f"AGF header truncated: only {len(raw)} bytes")
    magic = _u32(raw, 0x00)
    if magic != XFS_AGF_MAGIC:
        raise StructureError(f"AGF magic mismatch: 0x{magic:08x}")
    return AGF(magic=magic, seqno=_u32(raw, 0x08), length=_u32(raw, 0x0C), freeblks=_u32(raw, 0x44))


@dataclass(frozen=True)
class BtreeBlock:
    magic: int
    level: int
    numrecs: int
    leftsib: int
    rightsib: int
    header_size: int


def parse_btree_block(raw: bytes, has_crc: bool) -> BtreeBlock:
    if has_crc:
        # Short-format (per-AG) btree CRC header is 56 bytes; the 72-byte size
        # belongs to the long-format (filesystem-wide) btree header.
        header_size = 56
        if len(raw) < header_size:
            raise ReadFailure(f"CRC btree block truncated: only {len(raw)} bytes")
        return BtreeBlock(
            magic=_u32(raw, 0x00),
            level=_u16(raw, 0x04),
            numrecs=_u16(raw, 0x06),
            leftsib=_u32(raw, 0x08),
            rightsib=_u32(raw, 0x0C),
            header_size=header_size,
        )
    header_size = 16
    if len(raw) < header_size:
        raise ReadFailure(f"btree block truncated: only {len(raw)} bytes")
    return BtreeBlock(
        magic=_u32(raw, 0x00),
        level=_u16(raw, 0x04),
        numrecs=_u16(raw, 0x06),
        leftsib=_u32(raw, 0x08),
        rightsib=_u32(raw, 0x0C),
        header_size=header_size,
    )


@dataclass(frozen=True)
class InoRec:
    """Decoded 16-byte inode btree record (v2/sparse format)."""

    startino: int
    holemask: int
    inodemask: int
    freecount: int
    freemask: int

    @property
    def present_chunk64_count(self) -> int:
        return 16 - bin(self.holemask & XFS_INOBT_HOLEMASK_FULL).count("1")


def parse_inobt_rec(raw: bytes) -> InoRec:
    if len(raw) < 16:
        raise ReadFailure(f"inode record truncated: only {len(raw)} bytes")
    startino = _u32(raw, 0x00)
    word1 = _u32(raw, 0x04)
    freemask = _u64(raw, 0x08)
    holemask = (word1 >> 16) & 0xFFFF
    # inodemask lives in bits 8..19 of the second big-endian word.
    inodemask = (word1 >> 8) & 0x000FFF00
    freecount = word1 & 0x000000FF
    return InoRec(
        startino=startino,
        holemask=holemask,
        inodemask=inodemask,
        freecount=freecount,
        freemask=freemask,
    )


@dataclass(frozen=True)
class DinodeCore:
    magic: int
    mode: int
    version: int
    fmt: int
    nlink: int
    size: int
    nblocks: int
    gen: int
    next_unlinked: int

    @property
    def is_dir(self) -> bool:
        return (self.mode & S_IFMT) == S_IFDIR

    @property
    def is_reg(self) -> bool:
        return (self.mode & S_IFMT) == S_IFREG

    @property
    def is_link(self) -> bool:
        return (self.mode & S_IFMT) == S_IFLNK


def parse_dinode_core(raw: bytes) -> DinodeCore:
    if len(raw) < 0x40:
        raise ReadFailure(f"dinode truncated: only {len(raw)} bytes")
    return DinodeCore(
        magic=_u16(raw, 0x00),
        mode=_u16(raw, 0x02),
        version=raw[0x04],
        fmt=raw[0x05],
        nlink=_u32(raw, 0x10),
        size=_u64(raw, 0x30),
        nblocks=_u64(raw, 0x38),
        gen=_u32(raw, 0x54),
        next_unlinked=_u32(raw, 0x58),
    )


# Shortform directory helpers. Layout (v3/64-bit inode numbers):
# Shortform directory (xfs_dir2_sf). There is no magic: the header is
# count(1) + i8count(1) + parent(8 bytes in the i8count>0 form), followed by
# variable-length entries that each carry a 2-byte saved offset.
@dataclass(frozen=True)
class SFDentry:
    name: bytes
    ino: int
    ftype: int


def parse_sf_dir(raw: bytes, has_ftype: bool) -> tuple[int, tuple[SFDentry, ...]]:
    """Return (parent inode, entries) for a shortform directory.

    Layout per xfs_dir2_sf.h. The header is count(1) i8count(1) followed by the
    parent inode. The inode-number width is directory-wide: when i8count is
    non-zero the parent and every entry inumber are uint64, otherwise uint32.
    Each entry is namelen(1) offset(2) name[namelen] [ftype(1)] inumber.
    """
    if len(raw) < 4:
        raise StructureError("shortform dir header truncated")
    count = raw[0x00]
    wide = raw[0x01] > 0
    width = 8 if wide else 4
    if wide:
        if len(raw) < 10:
            raise StructureError("shortform dir header truncated")
        parent = _u64(raw, 0x02)
        off = 0x0A
    else:
        parent = _u32(raw, 0x02)
        off = 0x06
    entries: list[SFDentry] = []
    for _ in range(count):
        if off + 3 > len(raw):
            raise StructureError("shortform entry header truncated")
        namelen = raw[off]
        off += 3  # namelen (1) + saved offset (2)
        name = raw[off : off + namelen]
        if len(name) != namelen:
            raise StructureError("shortform entry name truncated")
        off += namelen
        ftype = raw[off] if has_ftype else 0
        if has_ftype:
            off += 1
        if off + width > len(raw):
            raise StructureError("shortform entry inumber truncated")
        ino = _u64(raw, off) if wide else _u32(raw, off)
        off += width
        entries.append(SFDentry(name=name, ino=ino, ftype=ftype))
    return parent, tuple(entries)
