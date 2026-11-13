from __future__ import annotations

import io
from uuid import UUID

import pytest

from dissect.xfs.audit import (
    INO_BTREE_MAGICS,
    AuditRecord,
    BlockType,
    Mismatch,
    PhysicalLocation,
    XFSSourceAuditor,
)
from dissect.xfs.c_xfs import c_xfs

BLOCK_SIZE = 4096
SECTORS_PER_BLOCK = BLOCK_SIZE // 512
INODE_SIZE = 512
INOPBLOCK = 8
AGBLOCKS = 4096
AGBLKLOG = 12

VOLUME_A_UUID = UUID("11111111-1111-1111-1111-111111111111")
VOLUME_B_UUID = UUID("22222222-2222-2222-2222-222222222222")
META_UUID = UUID("33333333-3333-3333-3333-333333333333")

# Fixed layout of the small synthetic filesystem (one AG).
AGF_BLOCK = 0
AGI_BLOCK = 0
BNO_ROOT = 1
CNT_ROOT = 2
INOBT_ROOT = 3
FIBT_ROOT = 4
REFC_ROOT = 5
INODE_CHUNK = 1384
DATA_BLOCK = 2000
ROOT_INO = 11072
BMAP_OWNER_INO = 11073


def _superblock(fs_uuid: UUID, meta_uuid: UUID | None = None, metadata_uuid: bool = False) -> bytes:
    sb = c_xfs.xfs_sb()
    sb.sb_magicnum = c_xfs.XFS_SB_MAGIC
    sb.sb_blocksize = BLOCK_SIZE
    sb.sb_sectsize = 512
    sb.sb_sectlog = 9
    sb.sb_blocklog = 12
    sb.sb_inodesize = INODE_SIZE
    sb.sb_inodelog = 9
    sb.sb_inopblock = INOPBLOCK
    sb.sb_inopblog = 3
    sb.sb_agblklog = AGBLKLOG
    sb.sb_agblocks = AGBLOCKS
    sb.sb_agcount = 1
    sb.sb_versionnum = c_xfs.XFS_SB_VERSION_5
    sb.sb_rootino = ROOT_INO
    sb.sb_uuid = fs_uuid.bytes
    sb.sb_meta_uuid = (meta_uuid or UUID(int=0)).bytes
    sb.sb_features_incompat = c_xfs.XFS_SB_FEAT_INCOMPAT_FTYPE
    if metadata_uuid:
        sb.sb_features_incompat |= c_xfs.XFS_SB_FEAT_INCOMPAT_META_UUID
    return sb.dumps()


def _agf(fs_uuid: UUID) -> bytes:
    agf = c_xfs.xfs_agf()
    agf.agf_magicnum = c_xfs.XFS_AGF_MAGIC
    agf.agf_versionnum = c_xfs.XFS_AGF_VERSION
    agf.agf_seqno = 0
    agf.agf_length = AGBLOCKS
    agf.agf_roots = [BNO_ROOT, CNT_ROOT, 0]
    agf.agf_levels = [1, 1, 0]
    agf.agf_refcount_root = REFC_ROOT
    agf.agf_refcount_level = 1
    agf.agf_uuid = fs_uuid.bytes
    return agf.dumps()


def _agi(fs_uuid: UUID) -> bytes:
    agi = c_xfs.xfs_agi()
    agi.agi_magicnum = c_xfs.XFS_AGI_MAGIC
    agi.agi_versionnum = c_xfs.XFS_AGI_VERSION
    agi.agi_seqno = 0
    agi.agi_length = AGBLOCKS
    agi.agi_root = INOBT_ROOT
    agi.agi_level = 1
    agi.agi_count = INOPBLOCK
    agi.agi_free_root = FIBT_ROOT
    agi.agi_free_level = 1
    agi.agi_uuid = fs_uuid.bytes
    return agi.dumps()


def _agfl(fs_uuid: UUID) -> bytes:
    agfl = c_xfs.xfs_agfl()
    agfl.agfl_magicnum = c_xfs.XFS_AGFL_MAGIC
    agfl.agfl_seqno = 0
    agfl.agfl_uuid = fs_uuid.bytes
    return agfl.dumps()


def _short_btree(magic: int, fsblock: int, fs_uuid: UUID, owner: int) -> bytes:
    node = c_xfs.xfs_btree_sblock_crc()
    node.bb_magic = magic
    node.bb_level = 0
    node.bb_numrecs = 0
    node.bb_leftsib = 0xFFFFFFFF
    node.bb_rightsib = 0xFFFFFFFF
    node.bb_blkno = fsblock * SECTORS_PER_BLOCK
    node.bb_lsn = 0
    node.bb_uuid = fs_uuid.bytes
    node.bb_owner = owner
    node.bb_crc = 0
    return node.dumps()


def _long_btree(magic: int, fsblock: int, fs_uuid: UUID, owner: int, blkno: int | None = None) -> bytes:
    node = c_xfs.xfs_btree_lblock_crc()
    node.bb_magic = magic
    node.bb_level = 0
    node.bb_numrecs = 0
    node.bb_leftsib = 0xFFFFFFFFFFFFFFFF
    node.bb_rightsib = 0xFFFFFFFFFFFFFFFF
    node.bb_blkno = fsblock * SECTORS_PER_BLOCK if blkno is None else blkno
    node.bb_uuid = fs_uuid.bytes
    node.bb_lsn = 0
    node.bb_owner = owner
    node.bb_crc = 0
    node.bb_pad = 0
    return node.dumps()


def _inobt_leaf(startino: int) -> bytes:
    node = c_xfs.xfs_btree_sblock_crc()
    node.bb_magic = c_xfs.XFS_IBT_CRC_MAGIC
    node.bb_level = 0
    node.bb_numrecs = 1
    node.bb_leftsib = 0xFFFFFFFF
    node.bb_rightsib = 0xFFFFFFFF
    node.bb_blkno = INOBT_ROOT * SECTORS_PER_BLOCK
    node.bb_lsn = 0
    header = bytearray(node.dumps())
    rec = c_xfs.xfs_inobt_rec()
    rec.ir_startino = startino
    rec.ir_freecount = 0
    rec.ir_free = 0
    header[len(c_xfs.xfs_btree_sblock_crc) : len(c_xfs.xfs_btree_sblock_crc) + len(rec.dumps())] = rec.dumps()
    return bytes(header)


def _dinode(ino: int, fs_uuid: UUID, *, fmt: int = 1, mode: int = 0o100644) -> bytes:
    di = c_xfs.xfs_dinode()
    di.di_magic = c_xfs.XFS_DINODE_MAGIC
    di.di_mode = mode
    di.di_version = 3
    di.di_format = fmt
    di.di_nlink = 1
    di.di_ino = ino
    di.di_uuid = fs_uuid.bytes
    return di.dumps()


def _dir3_block(fsblock: int, fs_uuid: UUID, owner: int) -> bytes:
    hdr = c_xfs.xfs_dir3_data_hdr()
    hdr.magic = c_xfs.XFS_DIR3_DATA_MAGIC
    hdr.crc = 0
    hdr.blkno = fsblock * SECTORS_PER_BLOCK
    hdr.lsn = 0
    hdr.uuid = fs_uuid.bytes
    hdr.owner = owner
    hdr.bestfree = [0, 0, 0]
    hdr.pad = 0
    return hdr.dumps()


def _symlink_block(fsblock: int, fs_uuid: UUID, owner: int, target: bytes) -> bytes:
    hdr = c_xfs.xfs_dsymlink_hdr()
    hdr.sl_magic = c_xfs.XFS_SYMLINK_MAGIC
    hdr.sl_offset = 0
    hdr.sl_bytes = len(target)
    hdr.sl_crc = 0
    hdr.sl_uuid = fs_uuid.bytes
    hdr.sl_owner = owner
    hdr.sl_blkno = fsblock * SECTORS_PER_BLOCK
    hdr.sl_lsn = 0
    return hdr.dumps() + target


def build_image(fs_uuid: UUID, *, meta_uuid: UUID | None = None, metadata_uuid: bool = False) -> bytearray:
    """Build a minimal, internally consistent v5 XFS image in memory."""
    image = bytearray(BLOCK_SIZE * (DATA_BLOCK + 1))

    def put(fsblock: int, payload: bytes, offset: int = 0) -> None:
        image[fsblock * BLOCK_SIZE + offset : fsblock * BLOCK_SIZE + offset + len(payload)] = payload

    header_uuid = meta_uuid if metadata_uuid else fs_uuid

    put(0, _superblock(fs_uuid, meta_uuid, metadata_uuid))
    put(0, _agf(header_uuid), offset=512)
    put(0, _agi(header_uuid), offset=1024)
    put(0, _agfl(header_uuid), offset=1536)

    put(BNO_ROOT, _short_btree(c_xfs.XFS_ABTB_CRC_MAGIC, BNO_ROOT, header_uuid, owner=0))
    put(CNT_ROOT, _short_btree(c_xfs.XFS_ABTC_CRC_MAGIC, CNT_ROOT, header_uuid, owner=0))
    put(INOBT_ROOT, _inobt_leaf(ROOT_INO))
    # Stamp the inobt leaf's self identifying header with the metadata UUID.
    inobt = c_xfs.xfs_btree_sblock_crc(io.BytesIO(bytes(image[INOBT_ROOT * BLOCK_SIZE :])))
    inobt.bb_uuid = header_uuid.bytes
    put(INOBT_ROOT, inobt.dumps())
    put(FIBT_ROOT, _short_btree(c_xfs.XFS_FIBT_CRC_MAGIC, FIBT_ROOT, header_uuid, owner=0))
    put(REFC_ROOT, _long_btree(c_xfs.XFS_REFC_CRC_MAGIC, REFC_ROOT, header_uuid, owner=0, blkno=0))

    # One inode chunk block; slot 0 is the root inode, slot 1 an inode that
    # owns a remote symlink data block through a data fork extent.
    chunk = bytearray(BLOCK_SIZE)
    chunk[0 * INODE_SIZE : 1 * INODE_SIZE] = _dinode(ROOT_INO, header_uuid, mode=0o040755)

    symlink_inode = bytearray(_dinode(BMAP_OWNER_INO, header_uuid, fmt=2, mode=0o120777))
    target = b"target/path"
    # v3 data fork starts at 0xB0; one extent record pointing at DATA_BLOCK.
    symlink_inode[0xB0 : 0xB0 + 16] = _extent_record(0, DATA_BLOCK, 1)
    # di_big_anextents (di_nextents without NREXT64) lives at offset 0x4C.
    symlink_inode[0x4C : 0x50] = (1).to_bytes(4, "big")
    chunk[1 * INODE_SIZE : 2 * INODE_SIZE] = bytes(symlink_inode)
    put(INODE_CHUNK, bytes(chunk))

    put(DATA_BLOCK, _symlink_block(DATA_BLOCK, header_uuid, BMAP_OWNER_INO, target))
    return image


def _extent_record(logical_offset: int, fsblock: int, count: int) -> bytes:
    # xfs_bmbt_rec: flag(1) | offset(54) | block(52) | count(21), big endian.
    l0 = (logical_offset << 9) | (fsblock >> 43)
    l1 = ((fsblock & ((1 << 43) - 1)) << 21) | count
    return l0.to_bytes(8, "big") + l1.to_bytes(8, "big")


def audit(image: bytearray, *, lenient: bool = False) -> list[AuditRecord]:
    return list(XFSSourceAuditor(io.BytesIO(bytes(image)), lenient=lenient).audit())


def fields(record: AuditRecord) -> set[str]:
    return {mismatch.field for mismatch in record.mismatches}


def find(records: list[AuditRecord], block_type: BlockType, fsblock: int | None = None) -> AuditRecord:
    for record in records:
        if record.block_type == block_type and (fsblock is None or record.location.fsblock == fsblock):
            return record
    raise AssertionError(f"no audit record for {block_type} @ {fsblock}")


# ---------------------------------------------------------------------------
# Legitimate layouts are accepted
# ---------------------------------------------------------------------------
def test_clean_image_is_consistent() -> None:
    records = audit(build_image(VOLUME_A_UUID))

    assert records
    assert all(record.consistent for record in records)
    report = XFSSourceAuditor(io.BytesIO(bytes(build_image(VOLUME_A_UUID)))).audit_report()
    assert report.is_consistent
    assert report.inconsistent == []


def test_metadata_uuid_headers_are_accepted() -> None:
    # With metadata_uuid, on-disk headers carry sb_meta_uuid rather than the
    # user visible sb_uuid; the auditor must not hard-compare against sb_uuid.
    image = build_image(VOLUME_A_UUID, meta_uuid=META_UUID, metadata_uuid=True)
    records = audit(image)

    assert all(record.consistent for record in records)
    auditor = XFSSourceAuditor(io.BytesIO(bytes(image)))
    assert auditor.has_metadata_uuid
    assert auditor.expected_metadata_uuid == META_UUID
    assert auditor.fs_uuid == VOLUME_A_UUID


def test_user_uuid_without_metadata_feature_is_accepted() -> None:
    # Without the feature the meta UUID field stays zero but headers still
    # carry the regular fs UUID.
    image = build_image(VOLUME_A_UUID)
    auditor = XFSSourceAuditor(io.BytesIO(bytes(image)))
    assert not auditor.has_metadata_uuid
    assert auditor.expected_metadata_uuid == VOLUME_A_UUID
    assert all(record.consistent for record in auditor.audit())


# ---------------------------------------------------------------------------
# Cross-volume contamination is detected while retaining physical evidence
# ---------------------------------------------------------------------------
def test_foreign_uuid_in_short_btree_is_detected() -> None:
    image = build_image(VOLUME_A_UUID)
    # Splice a volume B inobt block (same magic, wrong UUID/owner context)
    # into volume A's inobt location.
    foreign = _short_btree(c_xfs.XFS_IBT_CRC_MAGIC, INOBT_ROOT, VOLUME_B_UUID, owner=7)
    image[INOBT_ROOT * BLOCK_SIZE : INOBT_ROOT * BLOCK_SIZE + len(foreign)] = foreign

    records = audit(image)
    record = find(records, BlockType.SHORT_BTREE, INOBT_ROOT)

    assert not record.consistent
    assert "uuid" in fields(record)
    assert "owner" in fields(record)
    uuid_mismatch = next(m for m in record.mismatches if m.field == "uuid")
    assert str(VOLUME_B_UUID) == uuid_mismatch.actual
    assert str(VOLUME_A_UUID) == uuid_mismatch.expected
    # The physical location where the foreign block was found is preserved.
    assert isinstance(record.location, PhysicalLocation)
    assert record.location.fsblock == INOBT_ROOT
    assert record.location.byte_offset == INOBT_ROOT * BLOCK_SIZE


def test_foreign_inode_uuid_and_owner_are_detected() -> None:
    image = build_image(VOLUME_A_UUID)
    # A volume B inode stamped with a different di_ino placed in volume A's
    # inode chunk at the slot that must hold ROOT_INO.
    foreign = _dinode(ROOT_INO + 999, VOLUME_B_UUID, mode=0o040755)
    image[INODE_CHUNK * BLOCK_SIZE : INODE_CHUNK * BLOCK_SIZE + INODE_SIZE] = foreign

    records = audit(image)
    record = find(records, BlockType.DINODE, INODE_CHUNK)

    assert not record.consistent
    assert "uuid" in fields(record)
    assert "owner" in fields(record)
    owner_mismatch = next(m for m in record.mismatches if m.field == "owner")
    assert owner_mismatch.expected == ROOT_INO
    assert owner_mismatch.actual == ROOT_INO + 999
    # Sub-block physical position inside the inode chunk is retained.
    assert record.location.block_offset == 0


def test_foreign_symlink_block_is_detected_but_target_remains_readable() -> None:
    image = build_image(VOLUME_A_UUID)
    target = b"foreign/volume/target"
    # Correct owner/block number but a foreign UUID: magic alone would pass.
    foreign = _symlink_block(DATA_BLOCK, VOLUME_B_UUID, BMAP_OWNER_INO, target)
    image[DATA_BLOCK * BLOCK_SIZE : DATA_BLOCK * BLOCK_SIZE + len(foreign)] = foreign

    records = audit(image)
    record = find(records, BlockType.SYMLINK, DATA_BLOCK)

    assert not record.consistent
    assert "uuid" in fields(record)
    # Forensic leniency: the payload is still readable and evidence retained.
    assert record.readable
    assert record.evidence["symlink_target"] == target.decode()
    assert record.location.byte_offset == DATA_BLOCK * BLOCK_SIZE


def test_wrong_blkno_is_detected() -> None:
    image = build_image(VOLUME_A_UUID)
    # A block claiming to live at a different disk address than it was read.
    relocated = _short_btree(c_xfs.XFS_ABTB_CRC_MAGIC, BNO_ROOT + 50, VOLUME_A_UUID, owner=0)
    image[BNO_ROOT * BLOCK_SIZE : BNO_ROOT * BLOCK_SIZE + len(relocated)] = relocated

    records = audit(image)
    record = find(records, BlockType.SHORT_BTREE, BNO_ROOT)

    assert not record.consistent
    blkno_mismatch = next(m for m in record.mismatches if m.field == "blkno")
    assert blkno_mismatch.expected == BNO_ROOT * SECTORS_PER_BLOCK
    assert blkno_mismatch.actual == (BNO_ROOT + 50) * SECTORS_PER_BLOCK


def test_long_btree_owner_uses_ag_start_not_ag_number() -> None:
    image = build_image(VOLUME_A_UUID)
    # The refcount (long) btree owner is the AG start filesystem block. In a
    # single-AG filesystem this happens to be 0, so craft an AG-number owner to
    # prove the long semantics differ from the short btree owner check.
    wrong_owner = _long_btree(c_xfs.XFS_REFC_CRC_MAGIC, REFC_ROOT, VOLUME_A_UUID, owner=42, blkno=0)
    image[REFC_ROOT * BLOCK_SIZE : REFC_ROOT * BLOCK_SIZE + len(wrong_owner)] = wrong_owner

    records = audit(image)
    record = find(records, BlockType.LONG_BTREE, REFC_ROOT)

    assert not record.consistent
    owner_mismatch = next(m for m in record.mismatches if m.field == "owner")
    assert owner_mismatch.expected == 0  # AG start filesystem block for AG 0
    assert owner_mismatch.actual == 42


def test_agf_seqno_mismatch_is_detected() -> None:
    image = build_image(VOLUME_A_UUID)
    agf = c_xfs.xfs_agf(io.BytesIO(bytes(image[512 : 512 + len(c_xfs.xfs_agf)])))
    agf.agf_seqno = 9
    image[512 : 512 + len(agf.dumps())] = agf.dumps()

    records = audit(image)
    record = find(records, BlockType.AGF, 0)
    assert not record.consistent
    assert "seqno" in fields(record)


# ---------------------------------------------------------------------------
# Forensic lenient mode keeps readable content plus inconsistency evidence
# ---------------------------------------------------------------------------
def test_lenient_mode_keeps_readable_records_with_evidence() -> None:
    image = build_image(VOLUME_A_UUID)
    foreign = _symlink_block(DATA_BLOCK, VOLUME_B_UUID, BMAP_OWNER_INO, b"still/readable")
    image[DATA_BLOCK * BLOCK_SIZE : DATA_BLOCK * BLOCK_SIZE + len(foreign)] = foreign

    strict = audit(image, lenient=False)
    lenient = audit(image, lenient=True)

    strict_bad = [r for r in strict if not r.consistent]
    lenient_bad = [r for r in lenient if not r.consistent]
    assert strict_bad and lenient_bad
    # The inconsistent verdict and its evidence survive in lenient mode.
    assert all(isinstance(m, Mismatch) for record in lenient_bad for m in record.mismatches)
    symlink_record = find(lenient_bad, BlockType.SYMLINK, DATA_BLOCK)
    assert symlink_record.has_evidence
    assert symlink_record.evidence["symlink_target"] == "still/readable"
    # Readable content is still enumerated alongside the contamination proof.
    assert any(r.block_type == BlockType.SYMLINK and r.readable for r in lenient)


def test_rejects_non_v5_filesystem() -> None:
    image = build_image(VOLUME_A_UUID)
    sb = c_xfs.xfs_sb(io.BytesIO(bytes(image[: len(c_xfs.xfs_sb)])))
    sb.sb_versionnum = c_xfs.XFS_SB_VERSION_4
    image[0 : len(sb.dumps())] = sb.dumps()

    with pytest.raises(ValueError, match="v5"):
        XFSSourceAuditor(io.BytesIO(bytes(image)))


def test_directory_block_owner_is_inode_number() -> None:
    image = build_image(VOLUME_A_UUID)
    auditor = XFSSourceAuditor(io.BytesIO(bytes(image)))
    location = auditor._location(0, DATA_BLOCK)

    good = _dir3_block(DATA_BLOCK, VOLUME_A_UUID, BMAP_OWNER_INO)
    assert next(auditor._audit_directory_block(location, good, BMAP_OWNER_INO)).consistent

    foreign = _dir3_block(DATA_BLOCK, VOLUME_B_UUID, 999)
    record = next(auditor._audit_directory_block(location, foreign, BMAP_OWNER_INO))
    assert not record.consistent
    assert {"uuid", "owner"} <= fields(record)


def test_inode_owned_bmap_owner_semantics() -> None:
    bmap_block = 1999

    def audit_bmap(payload: bytes) -> AuditRecord:
        image = build_image(VOLUME_A_UUID)
        image[bmap_block * BLOCK_SIZE : bmap_block * BLOCK_SIZE + len(payload)] = payload
        auditor = XFSSourceAuditor(io.BytesIO(bytes(image)))
        records = list(
            auditor._audit_btree_block(
                0, bmap_block, BMAP_OWNER_INO, BlockType.INODE_BTREE, INO_BTREE_MAGICS
            )
        )
        return records[0]

    good = _long_btree(c_xfs.XFS_BMAP_CRC_MAGIC, bmap_block, VOLUME_A_UUID, BMAP_OWNER_INO)
    assert audit_bmap(good).consistent

    # Correct inode owner but a foreign UUID must flag *only* the UUID, proving
    # headers are never hard-compared against a single fixed UUID value.
    foreign_uuid = _long_btree(c_xfs.XFS_BMAP_CRC_MAGIC, bmap_block, VOLUME_B_UUID, BMAP_OWNER_INO)
    foreign_record = audit_bmap(foreign_uuid)
    assert fields(foreign_record) == {"uuid"}

    wrong_owner = _long_btree(c_xfs.XFS_BMAP_CRC_MAGIC, bmap_block, VOLUME_A_UUID, 999)
    assert "owner" in fields(audit_bmap(wrong_owner))


def test_btree_recursion_detects_foreign_child_block() -> None:
    header_len = len(c_xfs.xfs_btree_sblock_crc)
    maxrecs = (BLOCK_SIZE - header_len) // 8
    pointer_offset = header_len + maxrecs * 4
    child_block = 10

    def build_with_internal_node(child: bytes) -> bytearray:
        image = build_image(VOLUME_A_UUID)
        node = c_xfs.xfs_btree_sblock_crc()
        node.bb_magic = c_xfs.XFS_ABTB_CRC_MAGIC
        node.bb_level = 1
        node.bb_numrecs = 1
        node.bb_leftsib = 0xFFFFFFFF
        node.bb_rightsib = 0xFFFFFFFF
        node.bb_blkno = BNO_ROOT * SECTORS_PER_BLOCK
        node.bb_lsn = 0
        node.bb_uuid = VOLUME_A_UUID.bytes
        node.bb_owner = 0
        block = bytearray(BLOCK_SIZE)
        block[:header_len] = node.dumps()
        block[pointer_offset : pointer_offset + 4] = child_block.to_bytes(4, "big")
        image[BNO_ROOT * BLOCK_SIZE : BNO_ROOT * BLOCK_SIZE + BLOCK_SIZE] = bytes(block)
        image[child_block * BLOCK_SIZE : child_block * BLOCK_SIZE + len(child)] = child
        return image

    good_child = _short_btree(c_xfs.XFS_ABTB_CRC_MAGIC, child_block, VOLUME_A_UUID, owner=0)
    good_records = audit(build_with_internal_node(good_child))
    assert any(
        r.block_type == BlockType.SHORT_BTREE
        and r.location.fsblock == child_block
        and r.consistent
        for r in good_records
    )

    foreign_child = _short_btree(c_xfs.XFS_ABTB_CRC_MAGIC, child_block, VOLUME_B_UUID, owner=0)
    foreign_records = audit(build_with_internal_node(foreign_child))
    child_record = find(foreign_records, BlockType.SHORT_BTREE, child_block)
    assert not child_record.consistent
    assert "uuid" in fields(child_record)
    assert child_record.location.byte_offset == child_block * BLOCK_SIZE


def test_lenient_mode_survives_corrupt_btree_pointer_table() -> None:
    header_len = len(c_xfs.xfs_btree_sblock_crc)
    maxrecs = (BLOCK_SIZE - header_len) // 8
    pointer_offset = header_len + maxrecs * 4

    image = build_image(VOLUME_A_UUID)
    node = c_xfs.xfs_btree_sblock_crc()
    node.bb_magic = c_xfs.XFS_ABTB_CRC_MAGIC
    node.bb_level = 1
    node.bb_numrecs = 4000  # overruns the block when reading pointers
    node.bb_leftsib = 0xFFFFFFFF
    node.bb_rightsib = 0xFFFFFFFF
    node.bb_blkno = BNO_ROOT * SECTORS_PER_BLOCK
    node.bb_lsn = 0
    node.bb_uuid = VOLUME_A_UUID.bytes
    node.bb_owner = 0
    image[BNO_ROOT * BLOCK_SIZE : BNO_ROOT * BLOCK_SIZE + header_len] = node.dumps()
    assert pointer_offset < BLOCK_SIZE  # sanity for the layout above

    with pytest.raises(EOFError):
        audit(image, lenient=False)

    lenient_records = audit(image, lenient=True)
    assert any(not record.readable for record in lenient_records)
    # Other metadata remains enumerated despite the corrupted subtree.
    assert any(record.block_type == BlockType.AGF for record in lenient_records)
