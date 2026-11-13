"""Build a minimal, fully controlled two-AG XFS v5 image for forensics tests.

Real ``mkfs.xfs`` is unavailable on the host, so this module hand-assembles just
the on-disk structures the scanner reasons about: a v5 superblock, per-AG AGF/
AGI headers, single-leaf ``inobt`` and ``finobt`` trees, and real ``v3`` dinodes
(including shortform directories). The layout deliberately spans two allocation
groups so cross-AG inode numbers, the partial last AG, sparse inode blocks, free
slots, orphan inodes and injected corruption can all be exercised deterministically.

The geometry (all block/inode numbers, masks and byte offsets) is exposed on the
returned :class:`BuiltImage`, giving tests an independent ground-truth oracle.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from dissect.xfs.forensic.structs import (
    FTYPE_DIR,
    FTYPE_LNK,
    FTYPE_REG,
    S_IFDIR,
    S_IFLNK,
    S_IFREG,
    XFS_AGF_MAGIC,
    XFS_AGI_MAGIC,
    XFS_FIBT_CRC_MAGIC,
    XFS_IBT_CRC_MAGIC,
    XFS_SB_FEAT_INCOMPAT_FTYPE,
    XFS_SB_FEAT_INCOMPAT_SPINODES,
    XFS_SB_MAGIC,
)

BLOCK_SIZE = 4096
SECT_SIZE = 512
INODE_SIZE = 512
INOPB = 8
AGBLOCKS = 128
AGCOUNT = 2
INOPB_LOG = 3
AGBLK_LOG = 7  # log2(128)
INUM_BITS = AGBLK_LOG + INOPB_LOG  # 10
INODES_PER_CHUNK = 64
NULLAGBLOCK = 0xFFFFFFFF
CRC_SBLOCK_HDR = 56  # short-format CRC btree header
UUID = b"\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff\x00"

# Block 0 of every AG holds the superblock (sector 0), AGF (sector 1), AGI
# (sector 2) and AGFL (sector 3) packed into its first sectors. The btree roots
# then occupy standalone blocks.
BLK_SB = 0
BLK_INOBT = 3
BLK_FINOBT = 4
# First inode chunk is aligned past the metadata region (mkfs-style inode
# alignment). agino 192 maps to block 24, clear of AG blocks 0-4.
FIRST_INODE_AGINO = 192
INODE_ALIGN_BLOCK = FIRST_INODE_AGINO // INOPB  # 24
INODE_CHUNK_BLOCKS = 8  # 64 inodes * 512B = 8 blocks
# Backing block for each 64-inode chunk (agino -> agino // INOPB).
CHUNK0_BLOCK = FIRST_INODE_AGINO // INOPB                  # agino 192 -> block 24
CHUNK1_BLOCK = (FIRST_INODE_AGINO + 64) // INOPB           # agino 256 -> block 32
CHUNK2_BLOCK = (FIRST_INODE_AGINO + 128) // INOPB          # agino 320 -> block 40


def inode_byte_offset_within_ag(agino: int) -> int:
    inode_block = agino // INOPB
    return inode_block * BLOCK_SIZE + (agino % INOPB) * INODE_SIZE


def be16(v: int) -> bytes:
    return struct.pack(">H", v)


def be32(v: int) -> bytes:
    return struct.pack(">I", v)


def be64(v: int) -> bytes:
    return struct.pack(">Q", v)


@dataclass
class ChunkSpec:
    index: int                       # chunk-64 index inside the 1024-inode record span
    start_agino: int
    block: int
    hole: bool
    allocated_slots: list[int] = field(default_factory=list)  # slots marked allocated (freemask bit 0)
    inobt_free: list[int] = field(default_factory=list)       # slots free per inobt
    finobt_free: list[int] | None = None                      # default: mirror inobt_free

    def __post_init__(self) -> None:
        if self.finobt_free is None:
            self.finobt_free = list(self.inobt_free)


@dataclass
class InodeSpec:
    ag: int
    agino: int
    mode: int
    nlink: int
    fmt: int
    size: int
    name: str
    sf_entries: tuple[tuple[str, int, int], ...] = ()  # (name, global ino, ftype)
    symlink_target: bytes | None = None
    parent_agino: int | None = None


@dataclass
class BuiltImage:
    data: bytearray
    chunks: dict[int, list[ChunkSpec]]
    inodes: dict[int, InodeSpec]
    sb_fields: dict
    layout: dict
    root_ino: int

    def write(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(self.data)


def encode_inobt_rec(startino: int, holemask: int, inodemask: int, freecount: int, freemask: int) -> bytes:
    word1 = (((holemask & 0xFFFF) << 16) | (inodemask & 0x000FFF00) | (freecount & 0xFF)) & 0xFFFFFFFF
    return be32(startino) + be32(word1) + be64(freemask & 0xFFFFFFFFFFFFFFFF)


def mask_from_free(free_slots: list[int]) -> int:
    mask = 0
    for slot in free_slots:
        mask |= 1 << slot
    return mask


def build_superblock() -> bytes:
    sb = bytearray(SECT_SIZE)
    struct.pack_into(">I", sb, 0x00, XFS_SB_MAGIC)
    struct.pack_into(">I", sb, 0x04, BLOCK_SIZE)
    struct.pack_into(">Q", sb, 0x08, AGBLOCKS * AGCOUNT)   # sb_dblocks
    sb[16:32] = UUID                                       # sb_uuid
    struct.pack_into(">Q", sb, 0x18, 1)                    # sb_logstart (dummy internal log)
    root_ino = FIRST_INODE_AGINO
    struct.pack_into(">Q", sb, 0x38, root_ino)             # sb_rootino
    struct.pack_into(">I", sb, 0x54, AGBLOCKS)             # sb_agblocks
    struct.pack_into(">I", sb, 0x58, AGCOUNT)              # sb_agcount
    struct.pack_into(">H", sb, 0x64, 5)                    # sb_versionnum (v5)
    struct.pack_into(">H", sb, 0x66, SECT_SIZE)            # sb_sectsize
    struct.pack_into(">H", sb, 0x68, INODE_SIZE)           # sb_inodesize
    struct.pack_into(">H", sb, 0x6A, INOPB)                # sb_inopblock
    sb[0x7B] = INOPB_LOG                                   # sb_inopblog
    sb[0x7C] = AGBLK_LOG                                   # sb_agblklog
    struct.pack_into(">I", sb, 0xB4, INODE_CHUNK_BLOCKS)   # sb_inoalignmt
    struct.pack_into(">I", sb, 0xD8, XFS_SB_FEAT_INCOMPAT_FTYPE | XFS_SB_FEAT_INCOMPAT_SPINODES)
    struct.pack_into(">I", sb, 0xE4, INODE_CHUNK_BLOCKS)   # sb_spino_align
    sb[0xF8:0x108] = UUID                                 # sb_meta_uuid
    return bytes(sb)


def build_agf(ag: int) -> bytes:
    buf = bytearray(SECT_SIZE)
    struct.pack_into(">I", buf, 0x00, XFS_AGF_MAGIC)
    struct.pack_into(">I", buf, 0x04, 1)                   # version
    struct.pack_into(">I", buf, 0x08, ag)                  # seqno
    struct.pack_into(">I", buf, 0x0C, AGBLOCKS)            # length
    return bytes(buf)


def build_agi(ag: int, count: int, freecount: int) -> bytes:
    buf = bytearray(SECT_SIZE)
    struct.pack_into(">I", buf, 0x00, XFS_AGI_MAGIC)
    struct.pack_into(">I", buf, 0x04, 1)
    struct.pack_into(">I", buf, 0x08, ag)
    struct.pack_into(">I", buf, 0x0C, AGBLOCKS)
    struct.pack_into(">I", buf, 0x10, count)               # agi_count (allocated)
    struct.pack_into(">I", buf, 0x14, BLK_INOBT)           # root
    struct.pack_into(">I", buf, 0x18, 1)                   # level
    struct.pack_into(">I", buf, 0x1C, freecount)
    for i in range(64):
        struct.pack_into(">I", buf, 0x34 + i * 4, NULLAGBLOCK)
    buf[0x44:0x54] = UUID
    struct.pack_into(">I", buf, 0x60, BLK_FINOBT)          # free_root
    struct.pack_into(">I", buf, 0x64, 1)                   # free_level
    struct.pack_into(">I", buf, 0x68, 1)                   # iblocks
    struct.pack_into(">I", buf, 0x6C, 1)                   # fblocks
    return bytes(buf)


def build_leaf(ag: int, ag_block: int, magic: int, records: list[bytes]) -> bytes:
    block = bytearray(BLOCK_SIZE)
    struct.pack_into(">I", block, 0x00, magic)
    struct.pack_into(">H", block, 0x04, 0)                 # leaf level
    struct.pack_into(">H", block, 0x06, len(records))
    struct.pack_into(">I", block, 0x08, NULLAGBLOCK)       # leftsib
    struct.pack_into(">I", block, 0x0C, NULLAGBLOCK)       # rightsib
    fs_block = ag * AGBLOCKS + ag_block
    struct.pack_into(">Q", block, 0x14, fs_block)          # bb_blkno
    block[0x24:0x34] = UUID                                # bb_uuid
    struct.pack_into(">I", block, 0x38, ag)                # bb_owner
    off = CRC_SBLOCK_HDR
    for rec in records:
        block[off : off + 16] = rec
        off += 16
    return bytes(block)


def build_dinode(spec: InodeSpec, global_ino: int) -> bytes:
    buf = bytearray(INODE_SIZE)
    struct.pack_into(">H", buf, 0x00, 0x494E)              # 'IN'
    struct.pack_into(">H", buf, 0x02, spec.mode)
    buf[0x04] = 3                                          # v3
    buf[0x05] = spec.fmt
    struct.pack_into(">I", buf, 0x10, spec.nlink)
    struct.pack_into(">Q", buf, 0x30, spec.size)
    struct.pack_into(">I", buf, 0x54, 0x1234)              # di_gen
    struct.pack_into(">Q", buf, 0x88, global_ino)          # di_ino
    buf[0x90:0xA0] = UUID
    if spec.mode & 0o170000 == S_IFLNK and spec.symlink_target is not None:
        buf[0xB0 : 0xB0 + len(spec.symlink_target)] = spec.symlink_target
    elif (spec.mode & 0o170000) == S_IFDIR:
        # Every directory, including an empty orphan, carries a shortform header.
        sf = build_sf_dir(spec)
        if 0xB0 + len(sf) > INODE_SIZE:
            raise ValueError("shortform directory does not fit in the inode")
        buf[0xB0 : 0xB0 + len(sf)] = sf
    return bytes(buf)


def build_sf_dir(spec: InodeSpec) -> bytes:
    # Real xfs_dir2_sf layout (no magic): count(1) i8count(1) parent(8), then
    # per entry namelen(1) offset(2) name [ftype(1)] inumber(8).
    out = bytearray()
    out += bytes([len(spec.sf_entries), 1])                # count, i8count=1
    assert spec.parent_agino is not None
    parent_global = (spec.ag << INUM_BITS) | spec.parent_agino
    out += be64(parent_global)
    # Saved offsets are opaque tag values; use a monotonically increasing,
    # 8-byte-aligned sequence as mkfs does for fresh shortform directories.
    tag = 0x10
    for name, child_global, ftype in spec.sf_entries:
        name_b = name.encode()
        out += bytes([len(name_b)])
        out += be16(tag)
        tag += 0x10
        out += name_b
        out += bytes([ftype])
        out += be64(child_global)
    return bytes(out)


def _chunk_records(chunks: list[ChunkSpec]) -> tuple[list[bytes], list[bytes], int, int]:
    inobt: list[bytes] = []
    finobt: list[bytes] = []
    allocated = 0
    free = 0
    by_start: dict[int, list[ChunkSpec]] = {}
    for chunk in chunks:
        by_start.setdefault(chunk.start_agino - chunk.index * INODES_PER_CHUNK, []).append(chunk)
    for record_start, group in sorted(by_start.items()):
        provided = {chunk.index for chunk in group}
        # A record spans 16 chunk-64s. Any index not backed by a present chunk
        # (either omitted or explicitly a hole) must be set in holemask, else the
        # scanner would treat those absent inode blocks as allocated.
        holemask = 0
        for idx in range(16):
            if idx not in provided:
                holemask |= 1 << idx
        # Start with no free bits; only present (non-hole) chunks participate.
        inobt_freemask = 0
        finobt_freemask = 0
        present = 0
        for chunk in group:
            bit_base = chunk.index * INODES_PER_CHUNK
            if chunk.hole:
                holemask |= 1 << chunk.index
                continue
            present += 1
            for slot in chunk.inobt_free:
                inobt_freemask |= 1 << (bit_base + slot)
            for slot in (chunk.finobt_free or []):
                finobt_freemask |= 1 << (bit_base + slot)
        inobt_freecount = bin(inobt_freemask).count("1")
        finobt_freecount = bin(finobt_freemask).count("1")
        inobt.append(encode_inobt_rec(record_start, holemask, 0, inobt_freecount, inobt_freemask))
        finobt.append(encode_inobt_rec(record_start, holemask, 0, finobt_freecount, finobt_freemask))
        # Allocated slots = present chunk slots whose inobt free bit is clear.
        allocated += present * INODES_PER_CHUNK - inobt_freecount
        free += inobt_freecount
    return inobt, finobt, allocated, free


def default_chunks() -> dict[int, list[ChunkSpec]]:
    # Each record covers a 1024-inode span (16 chunk-64s); only these chunks
    # are present. Chunk starts are absolute AG-relative inode numbers.
    c0, c1, c2 = FIRST_INODE_AGINO, FIRST_INODE_AGINO + 64, FIRST_INODE_AGINO + 128
    # AG 0: three present chunks; chunk 2 holds an orphan directory.
    ag0 = [
        ChunkSpec(0, c0, CHUNK0_BLOCK, hole=False, inobt_free=[8, 9, 10, 11, 12, 13, 14, 15]),
        ChunkSpec(1, c1, CHUNK1_BLOCK, hole=False, inobt_free=list(range(56, 64))),
        ChunkSpec(2, c2, CHUNK2_BLOCK, hole=False, inobt_free=[0]),
    ]
    # AG 1: chunk 1 is a full sparse hole; chunk 2 an orphan file; chunk 0 has a
    # stale free slot (inobt vs finobt disagree).
    ag1 = [
        ChunkSpec(0, c0, CHUNK0_BLOCK, hole=False, inobt_free=[2, 3, 4], finobt_free=[2, 3, 4, 5]),
        ChunkSpec(1, c1, CHUNK1_BLOCK, hole=True),
        ChunkSpec(2, c2, CHUNK2_BLOCK, hole=False, inobt_free=[1], finobt_free=[1]),
    ]
    return {0: ag0, 1: ag1}


def default_inodes(chunks: dict[int, list[ChunkSpec]]) -> dict[int, InodeSpec]:
    g = lambda ag, agino: (ag << INUM_BITS) | agino
    root = FIRST_INODE_AGINO
    specs: dict[int, InodeSpec] = {}

    # ---- AG 0 reachable tree (root lives at the first aligned inode) ----
    specs[g(0, root)] = InodeSpec(
        0, root, S_IFDIR | 0o755, 2, 1, 0, "root",
        sf_entries=(
            ("dir_a", g(0, root + 1), FTYPE_DIR),
            ("file_a", g(0, root + 2), FTYPE_REG),
            ("link_a", g(0, root + 3), FTYPE_LNK),
            ("cross_ag_dir", g(1, root), FTYPE_DIR),
        ),
        parent_agino=root,
    )
    specs[g(0, root + 1)] = InodeSpec(
        0, root + 1, S_IFDIR | 0o755, 2, 1, 0, "dir_a",
        sf_entries=(("file_b", g(0, root + 4), FTYPE_REG),),
        parent_agino=root,
    )
    specs[g(0, root + 2)] = InodeSpec(
        0, root + 2, S_IFREG | 0o644, 1, 1, 11, "file_a", parent_agino=root
    )
    specs[g(0, root + 3)] = InodeSpec(
        0, root + 3, S_IFLNK | 0o777, 1, 1, 6, "link_a",
        symlink_target=b"file_a", parent_agino=root,
    )
    specs[g(0, root + 4)] = InodeSpec(
        0, root + 4, S_IFREG | 0o644, 1, 1, 12, "file_b", parent_agino=root + 1
    )
    # Orphan directory in AG0 chunk2: allocated, no incoming dirent anywhere.
    orphan_dir = FIRST_INODE_AGINO + 128
    specs[g(0, orphan_dir)] = InodeSpec(
        0, orphan_dir, S_IFDIR | 0o700, 0, 1, 0, "orphan_dir_ag0",
        sf_entries=(), parent_agino=orphan_dir,
    )

    # ---- AG 1: cross-AG reachable child plus orphans ----
    specs[g(1, root)] = InodeSpec(
        1, root, S_IFDIR | 0o755, 2, 1, 0, "cross_ag_dir",
        sf_entries=(("cross_file", g(1, root + 1), FTYPE_REG),),
        parent_agino=root,
    )
    specs[g(1, root + 1)] = InodeSpec(
        1, root + 1, S_IFREG | 0o644, 1, 1, 9, "cross_file", parent_agino=root
    )
    # Orphan regular file in AG1 chunk2.
    orphan_file = FIRST_INODE_AGINO + 128
    specs[g(1, orphan_file)] = InodeSpec(
        1, orphan_file, S_IFREG | 0o600, 0, 1, 7, "orphan_file_ag1", parent_agino=orphan_file
    )
    return specs


def build_image(
    chunks: dict[int, list[ChunkSpec]] | None = None,
    inodes: dict[int, InodeSpec] | None = None,
) -> BuiltImage:
    chunks = chunks or default_chunks()
    total_blocks = AGBLOCKS * AGCOUNT
    data = bytearray(total_blocks * BLOCK_SIZE)
    data[0:SECT_SIZE] = build_superblock()

    for ag in range(AGCOUNT):
        ag_start = ag * AGBLOCKS * BLOCK_SIZE
        # Block 0 packs SB (sector 0), AGF (sector 1), AGI (sector 2), AGFL (3).
        data[ag_start : ag_start + SECT_SIZE] = build_superblock()
        ag_chunks = chunks[ag]
        inobt_recs, finobt_recs, allocated, free = _chunk_records(ag_chunks)

        data[ag_start + SECT_SIZE : ag_start + 2 * SECT_SIZE] = build_agf(ag)
        data[ag_start + 2 * SECT_SIZE : ag_start + 3 * SECT_SIZE] = build_agi(ag, allocated, free)
        data[ag_start + BLK_INOBT * BLOCK_SIZE : ag_start + (BLK_INOBT + 1) * BLOCK_SIZE] = build_leaf(
            ag, BLK_INOBT, XFS_IBT_CRC_MAGIC, inobt_recs
        )
        data[ag_start + BLK_FINOBT * BLOCK_SIZE : ag_start + (BLK_FINOBT + 1) * BLOCK_SIZE] = build_leaf(
            ag, BLK_FINOBT, XFS_FIBT_CRC_MAGIC, finobt_recs
        )

        inode_specs = inodes or default_inodes(chunks)
        # Place every dinode at its computed byte offset based on agino.
        for global_ino, spec in inode_specs.items():
            if spec.ag != ag:
                continue
            offset = ag_start + inode_byte_offset_within_ag(spec.agino)
            data[offset : offset + INODE_SIZE] = build_dinode(spec, global_ino)

    layout = {
        "block_size": BLOCK_SIZE,
        "agblocks": AGBLOCKS,
        "agcount": AGCOUNT,
        "chunks": {
            ag: [
                {
                    "block": c.block,
                    "start_agino": c.start_agino,
                    "hole": c.hole,
                    "inobt_free": c.inobt_free,
                    "finobt_free": c.finobt_free,
                }
                for c in ag_chunks
            ]
            for ag, ag_chunks in chunks.items()
        },
    }
    return BuiltImage(
        data=data,
        chunks=chunks,
        inodes=inodes or default_inodes(chunks),
        sb_fields={"rootino": FIRST_INODE_AGINO, "inum_bits": INUM_BITS},
        layout=layout,
        root_ino=FIRST_INODE_AGINO,
    )


def corrupt_inobt_magic(image: BuiltImage, ag: int = 1) -> int:
    """Corrupt the inobt leaf magic of ``ag``; return its byte offset."""
    offset = ag * AGBLOCKS * BLOCK_SIZE + BLK_INOBT * BLOCK_SIZE
    image.data[offset : offset + 4] = b"BAD!"
    return offset


def truncate_to_blocks(image: BuiltImage, blocks: int) -> int:
    """Shrink the image buffer to ``blocks`` filesystem blocks."""
    del image.data[blocks * BLOCK_SIZE :]
    return blocks * BLOCK_SIZE


# ---------------------------------------------------------------------------
# Partial inobt record corruption. A single leaf holds multiple 16-byte
# records; corrupting one record's startino so it points outside the AG proves
# the scanner skips the bad record while every sibling record still resolves.
# ---------------------------------------------------------------------------
def corrupt_inobt_record_startino(image: BuiltImage, ag: int, record_index: int) -> int:
    """Overwrite one inobt leaf record's startino with an out-of-range value.

    Returns the byte offset of the corrupted record.
    """
    leaf_offset = ag * AGBLOCKS * BLOCK_SIZE + BLK_INOBT * BLOCK_SIZE
    rec_offset = leaf_offset + CRC_SBLOCK_HDR + record_index * 16
    # Larger than the AG inode capacity triggers the bounds check.
    image.data[rec_offset : rec_offset + 4] = be32(AGBLOCKS * INOPB + 1024)
    return rec_offset
