from __future__ import annotations

import enum
import io
import stat
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID

from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.xfs import parse_fsblock

if TYPE_CHECKING:
    from collections.abc import Iterator

# Filesystem blocks whose header stores a block number express it as a disk
# address in 512 byte sectors (the historical XFS daddr unit).
SECTOR_SIZE = 512

# Null block pointers as stored on disk for the short (32 bit) and long
# (64 bit) btree formats.
NULL_AGBLOCK = 0xFFFFFFFF
NULL_FSBLOCK = 0xFFFFFFFFFFFFFFFF

# Per-AG btrees that use the short (32 bit sibling/owner) block format. Their
# owner field carries the allocation group number.
SHORT_BTREE_MAGICS = frozenset(
    {
        c_xfs.XFS_ABTB_CRC_MAGIC,
        c_xfs.XFS_ABTC_CRC_MAGIC,
        c_xfs.XFS_IBT_CRC_MAGIC,
        c_xfs.XFS_FIBT_CRC_MAGIC,
    }
)

# Inode owned btrees (bmap/attr) use the long format and store the owning
# inode number in the owner field.
INO_BTREE_MAGICS = frozenset({c_xfs.XFS_BMAP_CRC_MAGIC})


class BlockType(enum.Enum):
    """Metadata block types that carry a v5 self-identifying header."""

    SUPERBLOCK = "superblock"
    AGF = "agf"
    AGFL = "agfl"
    AGI = "agi"
    SHORT_BTREE = "short_btree"
    LONG_BTREE = "long_btree"
    INODE_BTREE = "inode_btree"
    DINODE = "dinode"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


@dataclass(frozen=True)
class PhysicalLocation:
    """Where a piece of metadata was physically read from the image.

    All offsets are absolute byte positions inside the source image, so a
    relocated/foreign block can still be traced back to its physical location.
    """

    agno: int
    fsblock: int
    byte_offset: int
    # Offset within the block for sub-block objects (e.g. an inode inside an
    # inode chunk). Zero for whole-block headers.
    block_offset: int = 0

    def __str__(self) -> str:
        base = f"AG{self.agno} fsblock={self.fsblock} byte=0x{self.byte_offset:x}"
        if self.block_offset:
            base += f" block_offset=0x{self.block_offset:x}"
        return base


@dataclass(frozen=True)
class Mismatch:
    """A single failed source-consistency comparison."""

    field: str
    expected: object
    actual: object
    detail: str


@dataclass
class AuditRecord:
    """The audit verdict for one metadata object.

    A record is always produced for a block that could be located and parsed,
    even in forensic lenient mode. When ``consistent`` is ``False`` the
    ``mismatches`` (and optionally ``evidence``) carry the proof of the source
    inconsistency, while ``readable`` indicates whether the payload remains
    parseable.
    """

    block_type: BlockType
    location: PhysicalLocation
    magic: int
    consistent: bool
    readable: bool = True
    mismatches: list[Mismatch] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)

    @property
    def has_evidence(self) -> bool:
        return not self.consistent


def _uuid(raw: bytes) -> UUID | None:
    if len(raw) != 16:
        return None
    return UUID(bytes=bytes(raw))


def _is_null_uuid(value: UUID | None) -> bool:
    return value is None or value.int == 0


class XFSSourceAuditor:
    """Read-only source-consistency auditor for v5 (CRC enabled) XFS images.

    The auditor never mutates the image and only reads the blocks it needs. It
    cross-checks the self-identifying fields embedded in every v5 metadata
    header against the context the block was actually read from:

    - the header UUID against the filesystem's metadata UUID (honouring the
      ``metadata_uuid`` incompat feature, so headers are not blindly compared
      against the user-visible UUID),
    - the owner field against the object that is expected to own the block,
      which differs per block type (AG number, AG start block, or inode number),
    - the header block number against the physical location the block was read
      from,
    - the AGF/AGI sequence number against the allocation group number.

    In strict mode the first unparseable/foreign structure stops traversal of
    that subtree but the audit continues elsewhere. In forensic lenient mode
    (``lenient=True``) every readable object is still reported, but inconsistent
    records keep their mismatches and raw evidence, allowing a triage of mixed
    volumes while retaining proof of the contamination.
    """

    def __init__(self, fh: BinaryIO, lenient: bool = False):
        self.fh = fh
        self.lenient = lenient

        fh.seek(0)
        self.sb = c_xfs.xfs_sb(fh)
        if self.sb.sb_magicnum != c_xfs.XFS_SB_MAGIC:
            raise ValueError("Not a valid XFS filesystem (magic mismatch)")

        self.version = self.sb.sb_versionnum & c_xfs.XFS_SB_VERSION_NUMBITS
        if self.version != c_xfs.XFS_SB_VERSION_5:
            raise ValueError(
                f"Source consistency auditing requires a v5 (CRC) filesystem, got v{self.version}"
            )

        self.block_size = self.sb.sb_blocksize
        self.agblklog = self.sb.sb_agblklog
        self.agblocks = self.sb.sb_agblocks
        self.agcount = self.sb.sb_agcount
        self.inopblock = self.sb.sb_inopblock
        self.inodesize = self.sb.sb_inodesize
        self.inodes_per_block = self.block_size // self.inodesize

        # The user-visible UUID and the UUID stamped into on-disk metadata are
        # only the same when the metadata_uuid feature is disabled.
        self.fs_uuid = _uuid(self.sb.sb_uuid)
        self.meta_uuid = _uuid(self.sb.sb_meta_uuid)
        self.has_metadata_uuid = bool(
            self.sb.sb_features_incompat & c_xfs.XFS_SB_FEAT_INCOMPAT_META_UUID
        )
        # With metadata_uuid, sb_meta_uuid is the value written into headers.
        # Without it the feature field is unused and headers carry sb_uuid.
        self.expected_metadata_uuid = self.meta_uuid if self.has_metadata_uuid else self.fs_uuid

        self._seen: set[tuple[BlockType, int, int]] = set()

    # ------------------------------------------------------------------
    # Low level helpers
    # ------------------------------------------------------------------
    def _read_block(self, fsblock: int) -> bytes:
        self.fh.seek(fsblock * self.block_size)
        return self.fh.read(self.block_size)

    def _location(self, agno: int, fsblock: int, block_offset: int = 0) -> PhysicalLocation:
        return PhysicalLocation(
            agno=agno,
            fsblock=fsblock,
            byte_offset=fsblock * self.block_size + block_offset,
            block_offset=block_offset,
        )

    def _expected_daddr(self, fsblock: int) -> int:
        return fsblock * (self.block_size // SECTOR_SIZE)

    def _ag_start_block(self, agno: int) -> int:
        return agno << self.agblklog

    def _check_uuid(self, actual_raw: bytes) -> Mismatch | None:
        actual = _uuid(actual_raw)
        expected = self.expected_metadata_uuid
        if actual != expected:
            return Mismatch(
                field="uuid",
                expected=str(expected),
                actual=str(actual),
                detail=(
                    "header UUID does not match the filesystem metadata UUID"
                    + (" (metadata_uuid feature active)" if self.has_metadata_uuid else "")
                ),
            )
        return None

    @staticmethod
    def _mismatch(field: str, expected: object, actual: object, detail: str) -> Mismatch:
        return Mismatch(field=field, expected=expected, actual=actual, detail=detail)

    def _record(
        self,
        block_type: BlockType,
        location: PhysicalLocation,
        magic: int,
        mismatches: list[Mismatch],
        readable: bool = True,
        evidence: dict[str, object] | None = None,
    ) -> AuditRecord:
        return AuditRecord(
            block_type=block_type,
            location=location,
            magic=magic,
            consistent=not mismatches,
            readable=readable,
            mismatches=mismatches,
            evidence=evidence or {},
        )

    # ------------------------------------------------------------------
    # Per block type checks
    # ------------------------------------------------------------------
    def _audit_superblock(self, agno: int) -> AuditRecord:
        # Each AG starts with a (possibly redundant) superblock.
        fsblock = self._ag_start_block(agno)
        location = self._location(agno, fsblock)
        data = self._read_block(fsblock)
        mismatches: list[Mismatch] = []
        evidence: dict[str, object] = {}
        try:
            sb = c_xfs.xfs_sb(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001 - keep auditing in lenient mode
            if not self.lenient:
                raise
            return self._record(
                BlockType.SUPERBLOCK, location, 0,
                [self._mismatch("parse", "xfs_sb", str(exc), "unparseable superblock")],
                readable=False,
            )

        if sb.sb_magicnum != c_xfs.XFS_SB_MAGIC:
            mismatches.append(
                self._mismatch("magic", hex(c_xfs.XFS_SB_MAGIC), hex(sb.sb_magicnum), "bad superblock magic")
            )

        sb_uuid = _uuid(sb.sb_uuid)
        if sb_uuid != self.fs_uuid:
            mismatches.append(
                self._mismatch("uuid", str(self.fs_uuid), str(sb_uuid), "superblock sb_uuid differs from primary")
            )
        sb_meta = _uuid(sb.sb_meta_uuid)
        # A redundant superblock must advertise the same metadata UUID as AG 0.
        if self.has_metadata_uuid and sb_meta != self.meta_uuid:
            mismatches.append(
                self._mismatch(
                    "meta_uuid", str(self.meta_uuid), str(sb_meta),
                    "superblock sb_meta_uuid differs from primary",
                )
            )
        evidence = {"sb_uuid": str(sb_uuid), "sb_meta_uuid": str(sb_meta)}
        return self._record(BlockType.SUPERBLOCK, location, sb.sb_magicnum, mismatches, evidence=evidence)

    def _audit_agf(self, agno: int) -> AuditRecord:
        fsblock = self._ag_start_block(agno)
        location = self._location(agno, fsblock, block_offset=self.sb.sb_sectsize)
        self.fh.seek(location.byte_offset)
        mismatches: list[Mismatch] = []
        evidence: dict[str, object] = {}
        try:
            agf = c_xfs.xfs_agf(self.fh)
        except Exception as exc:  # noqa: BLE001
            if not self.lenient:
                raise
            return self._record(
                BlockType.AGF, location, 0,
                [self._mismatch("parse", "xfs_agf", str(exc), "unparseable AGF")], readable=False,
            )

        if agf.agf_magicnum != c_xfs.XFS_AGF_MAGIC:
            mismatches.append(
                self._mismatch("magic", hex(c_xfs.XFS_AGF_MAGIC), hex(agf.agf_magicnum), "bad AGF magic")
            )
        if agf.agf_seqno != agno:
            mismatches.append(
                self._mismatch("seqno", agno, agf.agf_seqno, "AGF sequence number does not match its AG")
            )
        uuid_mm = self._check_uuid(agf.agf_uuid)
        if uuid_mm:
            mismatches.append(uuid_mm)
        evidence = {"seqno": agf.agf_seqno, "uuid": str(_uuid(agf.agf_uuid))}
        return self._record(BlockType.AGF, location, agf.agf_magicnum, mismatches, evidence=evidence)

    def _audit_agi(self, agno: int) -> AuditRecord:
        fsblock = self._ag_start_block(agno)
        location = self._location(agno, fsblock, block_offset=self.sb.sb_sectsize * 2)
        self.fh.seek(location.byte_offset)
        mismatches: list[Mismatch] = []
        evidence: dict[str, object] = {}
        try:
            agi = c_xfs.xfs_agi(self.fh)
        except Exception as exc:  # noqa: BLE001
            if not self.lenient:
                raise
            return self._record(
                BlockType.AGI, location, 0,
                [self._mismatch("parse", "xfs_agi", str(exc), "unparseable AGI")], readable=False,
            ), None

        if agi.agi_magicnum != c_xfs.XFS_AGI_MAGIC:
            mismatches.append(
                self._mismatch("magic", hex(c_xfs.XFS_AGI_MAGIC), hex(agi.agi_magicnum), "bad AGI magic")
            )
        if agi.agi_seqno != agno:
            mismatches.append(
                self._mismatch("seqno", agno, agi.agi_seqno, "AGI sequence number does not match its AG")
            )
        uuid_mm = self._check_uuid(agi.agi_uuid)
        if uuid_mm:
            mismatches.append(uuid_mm)
        evidence = {"seqno": agi.agi_seqno, "uuid": str(_uuid(agi.agi_uuid)), "agi_root": agi.agi_root}
        return self._record(BlockType.AGI, location, agi.agi_magicnum, mismatches, evidence=evidence), agi

    def _audit_agfl(self, agno: int) -> AuditRecord:
        fsblock = self._ag_start_block(agno)
        location = self._location(agno, fsblock, block_offset=self.sb.sb_sectsize * 3)
        self.fh.seek(location.byte_offset)
        mismatches: list[Mismatch] = []
        evidence: dict[str, object] = {}
        try:
            agfl = c_xfs.xfs_agfl(self.fh)
        except Exception as exc:  # noqa: BLE001
            if not self.lenient:
                raise
            return self._record(
                BlockType.AGFL, location, 0,
                [self._mismatch("parse", "xfs_agfl", str(exc), "unparseable AGFL")], readable=False,
            )

        if agfl.agfl_magicnum != c_xfs.XFS_AGFL_MAGIC:
            mismatches.append(
                self._mismatch("magic", hex(c_xfs.XFS_AGFL_MAGIC), hex(agfl.agfl_magicnum), "bad AGFL magic")
            )
        if agfl.agfl_seqno != agno:
            mismatches.append(
                self._mismatch("seqno", agno, agfl.agfl_seqno, "AGFL sequence number does not match its AG")
            )
        uuid_mm = self._check_uuid(agfl.agfl_uuid)
        if uuid_mm:
            mismatches.append(uuid_mm)
        evidence = {"seqno": agfl.agfl_seqno, "uuid": str(_uuid(agfl.agfl_uuid))}
        return self._record(BlockType.AGFL, location, agfl.agfl_magicnum, mismatches, evidence=evidence)

    def _audit_btree_block(
        self, agno: int, agbno: int, expected_owner: int, block_kind: BlockType, magic_set: frozenset
    ) -> Iterator[AuditRecord]:
        """Audit a btree block and recurse into its children.

        ``agbno`` is an AG-relative block number; ``expected_owner`` is the
        owner value the header must carry, which depends on ``block_kind``.
        """
        fsblock = self._ag_start_block(agno) + agbno
        key = (block_kind, fsblock, 0)
        if key in self._seen:
            return
        self._seen.add(key)

        location = self._location(agno, fsblock)
        data = self._read_block(fsblock)
        # Wrap the whole block: the header is read first, but child pointers
        # live in the upper half of the same block and require full-block seeks.
        head = io.BytesIO(data)

        is_short = block_kind == BlockType.SHORT_BTREE
        try:
            node = c_xfs.xfs_btree_sblock_crc(head) if is_short else c_xfs.xfs_btree_lblock_crc(head)
        except Exception as exc:  # noqa: BLE001
            if not self.lenient:
                raise
            yield self._record(
                block_kind, location, 0,
                [self._mismatch("parse", "btree header", str(exc), "unparseable btree block")],
                readable=False,
            )
            return

        mismatches: list[Mismatch] = []
        if node.bb_magic not in magic_set:
            names = ", ".join(f"0x{m:x}" for m in sorted(magic_set))
            mismatches.append(
                self._mismatch("magic", names, hex(node.bb_magic), "btree magic does not match expected tree")
            )

        uuid_mm = self._check_uuid(node.bb_uuid)
        if uuid_mm:
            mismatches.append(uuid_mm)

        # bb_blkno is a 512 byte sector disk address; compare against the
        # physical location this block was actually read from. The reference
        # count btree root is initialised with bb_blkno == 0 by on-disk format
        # convention, so a zero value is legal there even though it does not
        # match the physical address; any other value must still line up.
        expected_daddr = self._expected_daddr(fsblock)
        refcount_zero_root = node.bb_magic == c_xfs.XFS_REFC_CRC_MAGIC and node.bb_blkno == 0
        if node.bb_blkno != expected_daddr and not refcount_zero_root:
            mismatches.append(
                self._mismatch(
                    "blkno", expected_daddr, node.bb_blkno,
                    "header block number (daddr) does not match physical location",
                )
            )

        if node.bb_owner != expected_owner:
            mismatches.append(
                self._mismatch(
                    "owner", expected_owner, node.bb_owner,
                    self._owner_detail(block_kind),
                )
            )

        evidence = {
            "uuid": str(_uuid(node.bb_uuid)),
            "blkno": node.bb_blkno,
            "expected_blkno": expected_daddr,
            "owner": node.bb_owner,
            "expected_owner": expected_owner,
            "level": node.bb_level,
        }
        yield self._record(block_kind, location, node.bb_magic, mismatches, evidence=evidence)

        # A foreign/mis-parsed node cannot be trusted for descent; stop this
        # subtree but keep auditing the rest of the filesystem.
        if mismatches and not self.lenient:
            return

        if node.bb_level == 0:
            return

        # Descend into child pointers. Pointers live past the header and the
        # unused record area in the upper half of the block.
        try:
            if is_short:
                header_len = len(c_xfs.xfs_btree_sblock_crc)
                maxrecs = (self.block_size - header_len) // 8
                head.seek(header_len + maxrecs * 4)
                pointers = c_xfs.uint32[node.bb_numrecs](head)
            else:
                header_len = len(c_xfs.xfs_btree_lblock_crc)
                maxrecs = (self.block_size - header_len) // 16
                head.seek(header_len + maxrecs * 8)
                pointers = c_xfs.uint64[node.bb_numrecs](head)
        except Exception as exc:  # noqa: BLE001 - corrupted pointer table
            if not self.lenient:
                raise
            yield self._record(
                block_kind, location, node.bb_magic,
                [self._mismatch("parse", "btree pointers", str(exc), "unparseable btree pointer table")],
                readable=False,
            )
            return

        if is_short:
            for child_agbno in pointers:
                if child_agbno == NULL_AGBLOCK:
                    continue
                yield from self._audit_btree_block(
                    agno, child_agbno, expected_owner, block_kind, magic_set
                )
        else:
            for ptr in pointers:
                if ptr == NULL_FSBLOCK:
                    continue
                child_agno, child_agbno = self._fsb_to_bb(ptr)
                if child_agno != agno:
                    # Per-AG long btrees must stay inside their own AG.
                    yield self._record(
                        block_kind,
                        self._location(child_agno, self._ag_start_block(child_agno) + child_agbno),
                        node.bb_magic,
                        [self._mismatch("pointer", agno, child_agno, "btree child escapes its allocation group")],
                    )
                    continue
                yield from self._audit_btree_block(
                    child_agno, child_agbno, expected_owner, block_kind, magic_set
                )

    @staticmethod
    def _owner_detail(block_kind: BlockType) -> str:
        if block_kind == BlockType.SHORT_BTREE:
            return "short per-AG btree owner must be its allocation group number"
        if block_kind == BlockType.LONG_BTREE:
            return "long per-AG btree owner must be the filesystem block of AG block 0"
        return "inode owned btree owner must be the owning inode number"

    def _fsb_to_bb(self, fsblock: int) -> tuple[int, int]:
        agno = fsblock >> self.agblklog
        agbno = fsblock & ((1 << self.agblklog) - 1)
        return agno, agbno

    # ------------------------------------------------------------------
    # Inodes and their owned blocks
    # ------------------------------------------------------------------
    def _audit_inode_chunk(self, agno: int, startino: int) -> Iterator[AuditRecord]:
        # An inobt record describes a chunk aligned to inopblock. With the
        # sparse inode feature the chunk spans inopblock * spino_align inodes
        # and therefore multiple filesystem blocks; unallocated hole blocks are
        # simply zero filled and skipped. Each present block holds inodes_per_block
        # inodes (one filesystem block / inode size).
        chunk_inode_count = self.inopblock * max(self.sb.sb_spino_align, 1)
        chunk_block_count = chunk_inode_count // self.inodes_per_block
        chunk_index = startino // self.inopblock

        for chunk_block in range(chunk_block_count):
            fsblock = self._ag_start_block(agno) + chunk_index + chunk_block
            data = self._read_block(fsblock)

            # A sparse hole: no inodes were ever allocated in this block.
            if not any(data):
                continue

            block_base_ino = startino + chunk_block * self.inodes_per_block
            yield from self._audit_inode_block(agno, fsblock, block_base_ino, data)

    def _audit_inode_block(
        self, agno: int, fsblock: int, block_base_ino: int, data: bytes
    ) -> Iterator[AuditRecord]:
        for slot in range(self.inodes_per_block):
            offset = slot * self.inodesize
            location = self._location(agno, fsblock, block_offset=offset)
            raw = data[offset : offset + self.inodesize]
            ino_number = block_base_ino + slot
            try:
                di = c_xfs.xfs_dinode(io.BytesIO(raw))
            except Exception as exc:  # noqa: BLE001
                if not self.lenient:
                    raise
                yield self._record(
                    BlockType.DINODE, location, 0,
                    [self._mismatch("parse", "xfs_dinode", str(exc), "unparseable inode")],
                    readable=False,
                )
                continue

            # Unused slots are zero filled; they carry no header to audit.
            if di.di_magic == 0 and not any(raw):
                continue

            mismatches: list[Mismatch] = []
            if di.di_magic != c_xfs.XFS_DINODE_MAGIC:
                mismatches.append(
                    self._mismatch(
                        "magic", hex(c_xfs.XFS_DINODE_MAGIC), hex(di.di_magic), "bad inode magic"
                    )
                )
                if not self.lenient:
                    yield self._record(BlockType.DINODE, location, di.di_magic, mismatches, readable=False)
                    continue

            uuid_mm = self._check_uuid(di.di_uuid)
            if uuid_mm:
                mismatches.append(uuid_mm)
            if di.di_ino != ino_number:
                mismatches.append(
                    self._mismatch(
                        "owner", ino_number, di.di_ino,
                        "inode di_ino does not match the inode number at its physical location",
                    )
                )

            evidence = {
                "inode": ino_number,
                "di_ino": di.di_ino,
                "uuid": str(_uuid(di.di_uuid)),
                "di_format": di.di_format,
                "di_mode": di.di_mode,
            }
            yield self._record(BlockType.DINODE, location, c_xfs.XFS_DINODE_MAGIC, mismatches, evidence=evidence)

            if mismatches and not self.lenient:
                # A foreign inode's fork cannot be trusted; skip its blocks.
                continue

            yield from self._audit_inode_owned_blocks(agno, ino_number, di, raw)
            yield from self._audit_inode_data_blocks(agno, ino_number, di, raw)

    def _audit_inode_owned_blocks(
        self, agno: int, ino_number: int, di: c_xfs.xfs_dinode, raw_inode: bytes
    ) -> Iterator[AuditRecord]:
        # A btree format data fork roots an inode-owned (bmap) btree. The root
        # itself lives inside the inode; its child blocks are filesystem blocks
        # whose owner is the inode number.
        if di.di_format != c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE:
            return

        offset = 0xB0 if di.di_version == 3 else 0x64
        fork = io.BytesIO(raw_inode[offset:])
        try:
            root = c_xfs.xfs_bmdr_block(fork)
            maxrecs = (len(raw_inode) - offset - 4) // 16
            fork.seek(4 + maxrecs * 8)
            pointers = c_xfs.uint64[root.bb_numrecs](fork)
        except Exception as exc:  # noqa: BLE001
            if not self.lenient:
                raise
            yield self._record(
                BlockType.INODE_BTREE, self._location(agno, self._ag_start_block(agno)), 0,
                [self._mismatch("parse", "xfs_bmdr_block", str(exc), "unparseable inode btree root")],
                readable=False,
            )
            return

        for ptr in pointers:
            if ptr == NULL_FSBLOCK:
                continue
            child_agno, child_agbno = self._fsb_to_bb(ptr)
            yield from self._audit_btree_block(
                child_agno, child_agbno, ino_number, BlockType.INODE_BTREE, INO_BTREE_MAGICS
            )

    def _iter_inode_data_fsblocks(
        self, di: c_xfs.xfs_dinode, raw_inode: bytes
    ) -> Iterator[int]:
        """Yield filesystem blocks backing an inode's data fork.

        Handles both the inline extent list and the btree-rooted extent map.
        Sparse holes yield ``None`` runs which are skipped here because an
        unallocated hole carries no metadata header to verify.
        """
        fmt = di.di_format
        offset = 0xB0 if di.di_version == 3 else 0x64

        if fmt == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS:
            # Mirrors INode.data_extents: di_nextents lives in di_big_anextents
            # unless the NREXT64 feature promotes it to di_big_nextents.
            nextents = di.di_big_nextents if self._has_nrext64(di) else di.di_big_anextents
            fork = raw_inode[offset : offset + nextents * 16]
            for index in range(nextents):
                _, fsblock, count, _ = parse_fsblock(fork[index * 16 : (index + 1) * 16])
                agno, agbno = self._fsb_to_bb(fsblock)
                base = self._ag_start_block(agno) + agbno
                for sub in range(count):
                    yield base + sub
        elif fmt == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_BTREE:
            fork = io.BytesIO(raw_inode[offset:])
            root = c_xfs.xfs_bmdr_block(fork)
            maxrecs = (len(raw_inode) - offset - 4) // 16
            fork.seek(4 + maxrecs * 8)
            pointers = c_xfs.uint64[root.bb_numrecs](fork)
            for ptr in pointers:
                if ptr == NULL_FSBLOCK:
                    continue
                yield from self._walk_bmap_leaves(ptr)

    @staticmethod
    def _has_nrext64(di: c_xfs.xfs_dinode) -> bool:
        return di.di_version >= 3 and di.di_flags2 & c_xfs.XFS_DIFLAG2_NREXT64 != 0

    def _walk_bmap_leaves(self, fsblock_ptr: int) -> Iterator[int]:
        # Walk a bmap (long) btree down to its leaf records.
        stack = [fsblock_ptr]
        while stack:
            ptr = stack.pop()
            agno, agbno = self._fsb_to_bb(ptr)
            fsblock = self._ag_start_block(agno) + agbno
            data = self._read_block(fsblock)
            node = c_xfs.xfs_btree_lblock_crc(io.BytesIO(data))
            if node.bb_level != 0:
                header_len = len(c_xfs.xfs_btree_lblock_crc)
                maxrecs = (self.block_size - header_len) // 16
                head = io.BytesIO(data)
                head.seek(header_len + maxrecs * 8)
                child_ptrs = c_xfs.uint64[node.bb_numrecs](head)
                stack.extend(child_ptrs)
                continue
            head = io.BytesIO(data)
            head.seek(len(c_xfs.xfs_btree_lblock_crc))
            for _ in range(node.bb_numrecs):
                rec = head.read(16)
                _, ext_fsblock, count, _ = parse_fsblock(rec)
                ext_agno, ext_agbno = self._fsb_to_bb(ext_fsblock)
                base = self._ag_start_block(ext_agno) + ext_agbno
                for sub in range(count):
                    yield base + sub

    def _audit_inode_data_blocks(
        self, agno: int, ino_number: int, di: c_xfs.xfs_dinode, raw_inode: bytes
    ) -> Iterator[AuditRecord]:
        is_symlink = stat.S_IFMT(di.di_mode) == stat.S_IFLNK
        is_dir = stat.S_IFMT(di.di_mode) == stat.S_IFDIR
        if not (is_symlink or is_dir):
            return

        for fsblock in self._iter_inode_data_fsblocks(di, raw_inode):
            block_agno, _ = self._fsb_to_bb(fsblock)
            location = self._location(block_agno, fsblock)
            data = self._read_block(fsblock)
            magic = int.from_bytes(data[:4], "big")

            if is_symlink and magic == c_xfs.XFS_SYMLINK_MAGIC:
                yield from self._audit_symlink_block(location, data, ino_number)
            elif is_dir and magic in (c_xfs.XFS_DIR3_BLOCK_MAGIC, c_xfs.XFS_DIR3_DATA_MAGIC):
                yield from self._audit_directory_block(location, data, ino_number)
            # Sparse/zeroed directory blocks and inline data have no v5 header;
            # they are simply not self-identifying metadata.

    def _audit_directory_block(
        self, location: PhysicalLocation, data: bytes, owner_inode: int
    ) -> Iterator[AuditRecord]:
        try:
            hdr = c_xfs.xfs_dir3_data_hdr(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            if not self.lenient:
                raise
            yield self._record(
                BlockType.DIRECTORY, location, 0,
                [self._mismatch("parse", "xfs_dir3_data_hdr", str(exc), "unparseable directory block")],
                readable=False,
            )
            return

        mismatches: list[Mismatch] = []
        expected_daddr = self._expected_daddr(location.fsblock)
        if hdr.blkno != expected_daddr:
            mismatches.append(
                self._mismatch("blkno", expected_daddr, hdr.blkno,
                               "directory header block number does not match physical location")
            )
        if hdr.owner != owner_inode:
            mismatches.append(
                self._mismatch("owner", owner_inode, hdr.owner,
                               "directory block owner must be its parent inode number")
            )
        uuid_mm = self._check_uuid(hdr.uuid)
        if uuid_mm:
            mismatches.append(uuid_mm)

        evidence = {
            "uuid": str(_uuid(hdr.uuid)),
            "blkno": hdr.blkno,
            "expected_blkno": expected_daddr,
            "owner": hdr.owner,
            "expected_owner": owner_inode,
        }
        yield self._record(BlockType.DIRECTORY, location, hdr.magic, mismatches, evidence=evidence)

    def _audit_symlink_block(
        self, location: PhysicalLocation, data: bytes, owner_inode: int
    ) -> Iterator[AuditRecord]:
        try:
            hdr = c_xfs.xfs_dsymlink_hdr(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            if not self.lenient:
                raise
            yield self._record(
                BlockType.SYMLINK, location, 0,
                [self._mismatch("parse", "xfs_dsymlink_hdr", str(exc), "unparseable symlink block")],
                readable=False,
            )
            return

        mismatches: list[Mismatch] = []
        expected_daddr = self._expected_daddr(location.fsblock)
        if hdr.sl_blkno != expected_daddr:
            mismatches.append(
                self._mismatch("blkno", expected_daddr, hdr.sl_blkno,
                               "symlink header block number does not match physical location")
            )
        if hdr.sl_owner != owner_inode:
            mismatches.append(
                self._mismatch("owner", owner_inode, hdr.sl_owner,
                               "symlink block owner must be its owning inode number")
            )
        uuid_mm = self._check_uuid(hdr.sl_uuid)
        if uuid_mm:
            mismatches.append(uuid_mm)

        # Forensic readability: retain the target even if the source is foreign.
        target = bytes(data[len(c_xfs.xfs_dsymlink_hdr) : len(c_xfs.xfs_dsymlink_hdr) + hdr.sl_bytes])
        evidence = {
            "uuid": str(_uuid(hdr.sl_uuid)),
            "blkno": hdr.sl_blkno,
            "expected_blkno": expected_daddr,
            "owner": hdr.sl_owner,
            "expected_owner": owner_inode,
            "symlink_target": target.decode(errors="surrogateescape"),
        }
        yield self._record(BlockType.SYMLINK, location, hdr.sl_magic, mismatches, evidence=evidence)

    # ------------------------------------------------------------------
    # Top level orchestration
    # ------------------------------------------------------------------
    def _read_agf(self, agno: int) -> c_xfs.xfs_agf:
        self.fh.seek(self._ag_start_block(agno) * self.block_size + self.sb.sb_sectsize)
        return c_xfs.xfs_agf(self.fh)

    def _read_agi(self, agno: int) -> c_xfs.xfs_agi:
        self.fh.seek(self._ag_start_block(agno) * self.block_size + self.sb.sb_sectsize * 2)
        return c_xfs.xfs_agi(self.fh)

    def _audit_per_ag_btrees(self, agno: int) -> Iterator[AuditRecord]:
        agf = self._read_agf(agno)
        ag_start = self._ag_start_block(agno)

        # Short format, AG-number owned freespace/inode btrees. The root array
        # order is bno, cnt, rmap; rmap is long and handled separately.
        short_roots = (
            (agf.agf_roots[0], SHORT_BTREE_MAGICS & frozenset({c_xfs.XFS_ABTB_CRC_MAGIC})),
            (agf.agf_roots[1], SHORT_BTREE_MAGICS & frozenset({c_xfs.XFS_ABTC_CRC_MAGIC})),
        )
        for root_agbno, magic_set in short_roots:
            if root_agbno:
                yield from self._audit_btree_block(
                    agno, root_agbno, agno, BlockType.SHORT_BTREE, magic_set
                )

        # rmap btree is long format and owned by the AG start filesystem block.
        # An absent tree has both root 0 and level 0.
        rmap_root = agf.agf_roots[2]
        if rmap_root:
            yield from self._audit_btree_block(
                agno, rmap_root, ag_start, BlockType.LONG_BTREE,
                frozenset({c_xfs.XFS_RMAP_CRC_MAGIC}),
            )

        # refcount btree is long format and owned by the AG start filesystem block.
        refcount_root = agf.agf_refcount_root
        if refcount_root:
            yield from self._audit_btree_block(
                agno, refcount_root, ag_start, BlockType.LONG_BTREE,
                frozenset({c_xfs.XFS_REFC_CRC_MAGIC}),
            )

        agi = self._read_agi(agno)
        # inobt / finobt are short format, AG-number owned. The finobt is
        # optional; an absent tree has root 0 and level 0.
        for root_agbno, magic in (
            (agi.agi_root, c_xfs.XFS_IBT_CRC_MAGIC),
            (agi.agi_free_root, c_xfs.XFS_FIBT_CRC_MAGIC),
        ):
            if root_agbno:
                yield from self._audit_btree_block(
                    agno, root_agbno, agno, BlockType.SHORT_BTREE, frozenset({magic})
                )

    def _audit_inode_tree(self, agno: int) -> Iterator[AuditRecord]:
        # Enumerate inode chunks through the inobt leaves so that inodes are
        # visited at their physically recorded locations rather than assumed.
        agi = self._read_agi(agno)
        if not agi.agi_root:
            return

        ag_start = self._ag_start_block(agno)
        root_fsblock = ag_start + agi.agi_root
        stack = [root_fsblock]
        visited: set[int] = set()
        while stack:
            fsblock = stack.pop()
            if fsblock in visited:
                continue
            visited.add(fsblock)
            data = self._read_block(fsblock)
            try:
                node = c_xfs.xfs_btree_sblock_crc(io.BytesIO(data))
            except Exception as exc:  # noqa: BLE001
                if not self.lenient:
                    raise
                yield self._record(
                    BlockType.SHORT_BTREE, self._location(agno, fsblock), 0,
                    [self._mismatch("parse", "xfs_btree_sblock_crc", str(exc), "unparseable inobt node")],
                    readable=False,
                )
                continue
            if node.bb_magic != c_xfs.XFS_IBT_CRC_MAGIC:
                if self.lenient:
                    yield self._record(
                        BlockType.SHORT_BTREE, self._location(agno, fsblock), node.bb_magic,
                        [self._mismatch("magic", hex(c_xfs.XFS_IBT_CRC_MAGIC), hex(node.bb_magic),
                                        "inobt node has unexpected magic")],
                        readable=False,
                    )
                continue

            if node.bb_level != 0:
                header_len = len(c_xfs.xfs_btree_sblock_crc)
                maxrecs = (self.block_size - header_len) // 8
                head = io.BytesIO(data)
                head.seek(header_len + maxrecs * 4)
                try:
                    pointers = c_xfs.uint32[node.bb_numrecs](head)
                except Exception as exc:  # noqa: BLE001
                    if not self.lenient:
                        raise
                    yield self._record(
                        BlockType.SHORT_BTREE, self._location(agno, fsblock), node.bb_magic,
                        [self._mismatch("parse", "inobt pointers", str(exc),
                                        "unparseable inobt pointer table")],
                        readable=False,
                    )
                    continue
                stack.extend(ag_start + ptr for ptr in pointers if ptr != NULL_AGBLOCK)
                continue

            head = io.BytesIO(data)
            head.seek(len(c_xfs.xfs_btree_sblock_crc))
            for _ in range(node.bb_numrecs):
                rec = c_xfs.xfs_inobt_rec(head)
                yield from self._audit_inode_chunk(agno, rec.ir_startino)

    def audit(self) -> Iterator[AuditRecord]:
        """Yield an :class:`AuditRecord` for every self-identifying v5 object."""
        for agno in range(self.agcount):
            yield self._audit_superblock(agno)
            yield self._audit_agf(agno)
            yield self._audit_agfl(agno)
            agi_record, _ = self._audit_agi(agno)
            yield agi_record

            yield from self._audit_per_ag_btrees(agno)
            yield from self._audit_inode_tree(agno)

    def audit_report(self) -> AuditReport:
        """Collect the full audit into a queryable report."""
        return AuditReport(list(self.audit()))


@dataclass
class AuditReport:
    """Convenience container over a complete audit run."""

    records: list[AuditRecord]

    @property
    def inconsistent(self) -> list[AuditRecord]:
        return [record for record in self.records if not record.consistent]

    @property
    def is_consistent(self) -> bool:
        return all(record.consistent for record in self.records)

    def unreadable(self) -> list[AuditRecord]:
        return [record for record in self.records if not record.readable]
