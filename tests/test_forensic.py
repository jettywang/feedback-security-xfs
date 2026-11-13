from __future__ import annotations

import io

import pytest

from dissect.xfs.forensic.disk import ByteRange, DiskImage
from dissect.xfs.forensic.image_builder import (
    AGBLOCKS,
    BLOCK_SIZE,
    CHUNK0_BLOCK,
    FIRST_INODE_AGINO,
    INUM_BITS,
    INOPB,
    BuiltImage,
    build_image,
    corrupt_inobt_magic,
    corrupt_inobt_record_startino,
)
from dissect.xfs.forensic.scanner import (
    STATE_ALLOCATED,
    STATE_FREE_SLOT,
    STATE_SPARSE_HOLE,
    STATE_UNREADABLE,
    ForensicScanner,
)


def _scanner(image_bytes: bytes, unreadable: list[ByteRange] | None = None) -> ForensicScanner:
    return ForensicScanner(
        io.BytesIO(image_bytes),
        disk=DiskImage(io.BytesIO(image_bytes), unreadable=unreadable),
    )


@pytest.fixture
def image_bytes() -> bytes:
    return bytes(build_image().data)


@pytest.fixture
def report(image_bytes: bytes) -> dict:
    return _scanner(image_bytes).scan()


def g(ag: int, agino: int) -> int:
    return (ag << INUM_BITS) | agino


def find_entry(report: dict, ino: int) -> dict:
    for ag in report["allocation_groups"]:
        for inode in ag["inodes"]:
            if inode["ino"] == ino:
                return inode
    raise AssertionError(f"inode {ino} not present in scan")


# --------------------------------------------------------------- geometry / bitmap
def test_superblock_geometry(report: dict) -> None:
    fs = report["filesystem"]
    assert fs["agcount"] == 2
    assert fs["agblocks_nominal"] == AGBLOCKS
    assert fs["block_size"] == BLOCK_SIZE
    assert fs["inopblock"] == INOPB
    assert fs["has_crc"] is True
    assert fs["has_spinodes"] is True
    assert fs["rootino"] == FIRST_INODE_AGINO


def test_inode_offset_is_block_aligned_and_auditable(report: dict) -> None:
    root = find_entry(report, g(0, FIRST_INODE_AGINO))
    # agino 192 -> block 24 -> byte 24 * 4096 within AG 0.
    assert root["byte_offset"] == CHUNK0_BLOCK * BLOCK_SIZE
    assert root["record_index"] == 0
    assert root["chunk_index"] == 0
    assert root["slot"] == 0
    assert "freemask_bit_clear" in root["reasons"]
    assert "dinode_magic_ok" in root["reasons"]


def test_freemask_distinguishes_free_and_allocated_slots(report: dict) -> None:
    # AG0 chunk0 declares slots 8..15 free; slot 0 (root) is allocated.
    free_slot = find_entry(report, g(0, FIRST_INODE_AGINO + 8))
    assert free_slot["state"] == STATE_FREE_SLOT
    assert "freemask_bit_set" in free_slot["reasons"]
    assert free_slot["dinode_state"] is None
    root = find_entry(report, g(0, FIRST_INODE_AGINO))
    assert root["state"] == STATE_ALLOCATED
    assert root["dinode_state"] == "OK"


def test_holemask_marks_sparse_inode_blocks(report: dict) -> None:
    # AG1 chunk1 (agino 256..319) is a full sparse hole.
    hole_agino = FIRST_INODE_AGINO + 64
    entry = find_entry(report, g(1, hole_agino))
    assert entry["state"] == STATE_SPARSE_HOLE
    assert entry["byte_offset"] is None
    assert "holemask_bit" in entry["reasons"]
    # The 13 other absent chunk-64s in the record span are also holes.
    ag1 = report["allocation_groups"][1]
    assert sum(1 for i in ag1["inodes"] if i["state"] == STATE_SPARSE_HOLE) == 14


def test_stale_finobt_bit_is_flagged_not_silently_trusted(report: dict) -> None:
    # AG1 chunk0: inobt says slot 5 allocated, finobt says free -> mismatch.
    stale = find_entry(report, g(1, FIRST_INODE_AGINO + 5))
    assert stale["state"] == STATE_ALLOCATED  # primary source stays the inobt
    assert stale["finobt_warning"] == "freemask_finobt_mismatch"
    assert stale["finobt_free"] is True
    assert report["allocation_groups"][1]["summary"]["stale_bitmap_warnings"] >= 1


# --------------------------------------------------------------- cross-AG + paths
def test_global_inode_numbers_span_two_ags(report: dict) -> None:
    # inum_bits = 10, so AG 1 inodes start at global number 1024.
    assert g(1, 0) == 1024
    cross_dir = find_entry(report, g(1, FIRST_INODE_AGINO))
    assert cross_dir["ino"] == 1024 + FIRST_INODE_AGINO
    assert cross_dir["ag"] == 1


def test_reachable_paths_are_reconstructed(report: dict) -> None:
    expected = {
        g(0, FIRST_INODE_AGINO): "/",
        g(0, FIRST_INODE_AGINO + 1): "/dir_a",
        g(0, FIRST_INODE_AGINO + 2): "/file_a",
        g(0, FIRST_INODE_AGINO + 3): "/link_a",
        g(0, FIRST_INODE_AGINO + 4): "/dir_a/file_b",
        g(1, FIRST_INODE_AGINO): "/cross_ag_dir",
        g(1, FIRST_INODE_AGINO + 1): "/cross_ag_dir/cross_file",
    }
    for ino, path in expected.items():
        entry = find_entry(report, ino)
        assert entry["path_status"] == "RECOVERED"
        assert entry["path"] == path


def test_orphans_are_allocated_but_have_no_path_not_deleted(report: dict) -> None:
    orphan_dir = find_entry(report, g(0, FIRST_INODE_AGINO + 128))
    orphan_file = find_entry(report, g(1, FIRST_INODE_AGINO + 128))
    for orphan in (orphan_dir, orphan_file):
        assert orphan["state"] == STATE_ALLOCATED
        assert orphan["dinode_state"] == "OK"
        assert orphan["path_status"] == "NO_PATH"
        assert orphan["path"] is None
        # The critical anti-confabulation guarantee.
        assert "not referenced by any readable shortform directory" in orphan["path_reason"]
    totals = report["totals"]
    assert totals["allocated_inodes_dinode_ok"] == 9
    assert totals["allocated_without_path"] >= 2
    # Orphans must never be counted as free slots or sparse holes.
    assert orphan_dir["state"] != STATE_FREE_SLOT
    assert orphan_file["state"] != STATE_SPARSE_HOLE


# --------------------------------------------------------------- last AG boundary
def test_last_ag_boundary_on_full_image(report: dict) -> None:
    ag1 = report["allocation_groups"][1]
    assert ag1["is_last_ag"] is True
    assert ag1["actual_blocks"] == AGBLOCKS
    assert ag1["ag_start_byte"] == AGBLOCKS * BLOCK_SIZE
    assert report["totals"]["unreadable"] == 0
    assert report["totals"]["tree_read_failures"] == 0


def test_partial_last_ag_truncated_agi_is_unreadable_not_deleted(image_bytes: bytes) -> None:
    # The AGI sits in sector 2 of the last AG's block 0. Truncate inside that
    # sector so the SB (sector 0) and AGF (sector 1) remain but the AGI read
    # hits EOF.
    sector = 512
    ag1_start = AGBLOCKS * BLOCK_SIZE
    truncated = bytearray(image_bytes)
    new_size = ag1_start + 2 * sector + 100
    del truncated[new_size:]
    report = _scanner(bytes(truncated)).scan()
    ag1 = report["allocation_groups"][1]
    assert ag1["is_last_ag"] is True
    assert ag1["agi"] is None
    # The header read failure is surfaced rather than the missing inodes being
    # relabelled as free/deleted.
    assert any(e["tree"] == "agi" and e["kind"] == "ReadFailure" for e in ag1["errors"])
    assert ag1["summary"]["header_read_failures"] >= 1
    # AG0 remains fully interpretable.
    assert find_entry(report, g(0, FIRST_INODE_AGINO))["path_status"] == "RECOVERED"


def test_partial_last_ag_truncated_inobt_leaf_is_isolated(image_bytes: bytes) -> None:
    # Keep the whole AGI (block 0 sectors) but cut off before the inobt leaf
    # (AG block 3), so the tree root references an out-of-bounds block.
    truncated = bytearray(image_bytes)
    truncate_to_blocks_bytes(truncated, AGBLOCKS + 3)
    report = _scanner(bytes(truncated)).scan()
    ag1 = report["allocation_groups"][1]
    assert ag1["actual_blocks"] == 3
    assert ag1["agi"] is not None
    kinds = {(e["tree"], e["kind"]) for e in ag1["errors"]}
    assert any(tree == "inobt" and kind in ("ReadFailure", "StructureError") for tree, kind in kinds)
    # The failure must not be reinterpreted as free slots or sparse holes.
    assert ag1["summary"]["slots_allocated"] == 0
    # AG0 is untouched and still recovered.
    assert find_entry(report, g(0, FIRST_INODE_AGINO))["path_status"] == "RECOVERED"


def truncate_to_blocks_bytes(buf: bytearray, blocks: int) -> int:
    del buf[blocks * BLOCK_SIZE :]
    return blocks * BLOCK_SIZE


# --------------------------------------------------------------- index corruption
def test_corrupt_inobt_magic_is_reported_and_isolated(image_bytes: bytes) -> None:
    image = BuiltImage(data=bytearray(image_bytes), chunks={}, inodes={}, sb_fields={}, layout={}, root_ino=0)
    offset = corrupt_inobt_magic(image, ag=1)
    report = _scanner(bytes(image.data)).scan()
    ag1 = report["allocation_groups"][1]
    bad = [e for e in ag1["errors"] if e["tree"] == "inobt" and e["kind"] == "StructureError"]
    assert bad, "corrupt inobt magic must produce a structure error"
    assert bad[0]["byte_offset"] == offset
    # AG0 is untouched and still fully scanned.
    assert report["allocation_groups"][0]["summary"]["structure_errors"] == 0
    assert find_entry(report, g(0, FIRST_INODE_AGINO))["state"] == STATE_ALLOCATED


def test_corrupt_inobt_record_is_skipped_while_other_ag_survives(image_bytes: bytes) -> None:
    image = BuiltImage(data=bytearray(image_bytes), chunks={}, inodes={}, sb_fields={}, layout={}, root_ino=0)
    offset = corrupt_inobt_record_startino(image, ag=1, record_index=0)
    report = _scanner(bytes(image.data)).scan()
    ag1 = report["allocation_groups"][1]
    bad = [e for e in ag1["errors"] if e["tree"] == "inobt" and e["kind"] == "StructureError"]
    assert bad and "beyond AG capacity" in bad[0]["message"]
    # The corrupted record carries a reviewable byte location.
    assert bad[0]["byte_offset"] is not None
    assert offset == bad[0]["byte_offset"] + 56  # record 0 sits right after the 56-byte leaf header
    # The bad record yields no slots, but is never turned into free slots.
    assert ag1["summary"]["slots_allocated"] == 0
    assert ag1["summary"]["slots_free_slot"] == 0
    # The other AG's index is fully intact.
    ag0 = report["allocation_groups"][0]
    assert ag0["summary"]["structure_errors"] == 0
    assert find_entry(report, g(0, FIRST_INODE_AGINO))["path_status"] == "RECOVERED"


def test_injected_unreadable_inode_region_is_classified_unreadable(image_bytes: bytes) -> None:
    # Make the root inode's backing block unreadable without modifying bytes.
    root_block_byte = CHUNK0_BLOCK * BLOCK_SIZE
    region = ByteRange(root_block_byte, root_block_byte + BLOCK_SIZE)
    report = _scanner(image_bytes, unreadable=[region]).scan()
    root = find_entry(report, g(0, FIRST_INODE_AGINO))
    assert root["state"] == STATE_UNREADABLE
    assert root["dinode_state"] == "UNREADABLE"
    assert "read_failure" in root["reasons"]
    assert report["totals"]["unreadable"] >= 1
    # A read failure is never reinterpreted as a free slot or as deleted.
    assert root["state"] != STATE_FREE_SLOT


def test_injected_unreadable_ag_header_does_not_abort_scan(image_bytes: bytes) -> None:
    # The AGI lives in sector 2 of AG 1's block 0 (byte offset 2 * sector size).
    sector = 512
    ag1_agi_byte = AGBLOCKS * BLOCK_SIZE + 2 * sector
    region = ByteRange(ag1_agi_byte, ag1_agi_byte + sector)
    report = _scanner(image_bytes, unreadable=[region]).scan()
    ag1 = report["allocation_groups"][1]
    assert ag1["agi"] is None
    assert any(e["tree"] == "agi" and e["kind"] == "ReadFailure" for e in ag1["errors"])
    # AG0 still scanned successfully.
    assert find_entry(report, g(0, FIRST_INODE_AGINO))["state"] == STATE_ALLOCATED


# --------------------------------------------------------------- evidence completeness
def test_every_inode_carries_reviewable_location_and_basis(report: dict) -> None:
    for ag in report["allocation_groups"]:
        for inode in ag["inodes"]:
            assert isinstance(inode["reasons"], list) and inode["reasons"]
            assert inode["inobt_block"] is not None
            assert inode["record_index"] is not None
            if inode["state"] in (STATE_ALLOCATED, STATE_FREE_SLOT):
                assert inode["byte_offset"] is not None
                assert inode["freemask_hex"].startswith("0x")
                assert inode["holemask_hex"].startswith("0x")


def test_record_audit_trail_present_for_both_trees(report: dict) -> None:
    for ag in report["allocation_groups"]:
        trees = {r["tree"] for r in ag["records"]}
        assert trees == {"inobt", "finobt"}
        for rec in ag["records"]:
            assert rec["byte_offset"] == rec["fs_block"] * BLOCK_SIZE
            assert rec["freemask_hex"].startswith("0x")
            assert rec["holemask_hex"].startswith("0x")
