"""Human-readable rendering of :class:`ForensicScanner` results."""

from __future__ import annotations

from dissect.xfs.forensic.scanner import (
    STATE_ALLOCATED,
    STATE_FREE_SLOT,
    STATE_SPARSE_HOLE,
    STATE_UNREADABLE,
)

_STATE_LABEL = {
    STATE_ALLOCATED: "ALLOCATED ",
    STATE_FREE_SLOT: "FREE-SLOT ",
    STATE_SPARSE_HOLE: "SPARSE    ",
    STATE_UNREADABLE: "UNREADABLE",
}


def render_text(report: dict) -> str:
    lines: list[str] = []
    fs = report["filesystem"]
    lines.append("XFS INODE FORENSIC INVENTORY (path-independent)")
    lines.append("=" * 64)
    lines.append(
        f"block_size={fs['block_size']} inode_size={fs['inode_size']} "
        f"inopblock={fs['inopblock']} agcount={fs['agcount']} "
        f"agblocks={fs['agblocks_nominal']} version=v{fs['version']} "
        f"sparse_inodes={fs['has_spinodes']} rootino={fs['rootino']}"
    )
    lines.append(f"image_size={fs['image_size_bytes']} bytes")
    totals = report["totals"]
    lines.append(
        "TOTALS allocated_bitmap={allocated_inodes_bitmap} allocated_dinode_ok={allocated_inodes_dinode_ok} "
        "no_path={allocated_without_path} free={free_slots} sparse_holes={sparse_holes} "
        "unreadable={unreadable} zeroed={dinode_zeroed} bad_magic={dinode_bad_magic} "
        "read_failures={tree_read_failures} structure_errors={tree_structure_errors}".format(**totals)
    )
    lines.append("")

    for ag in report["allocation_groups"]:
        lines.extend(_render_ag(ag))
    return "\n".join(lines) + "\n"


def _render_ag(ag: dict) -> list[str]:
    lines: list[str] = []
    boundary = "LAST-AG" if ag["is_last_ag"] else "ag"
    lines.append(
        f"-- AG {ag['ag']} [{boundary}] start_byte=0x{ag['ag_start_byte']:x} "
        f"nominal_blocks={ag['nominal_blocks']} actual_blocks={ag['actual_blocks']}"
    )
    if ag["agi"]:
        agi = ag["agi"]
        lines.append(
            f"   AGI count={agi['count']} freecount={agi['freecount']} "
            f"inobt_root={agi['root']} finobt_root={agi['free_root']} "
            f"iblocks={agi['iblocks']} fblocks={agi['fblocks']}"
        )
    s = ag["summary"]
    lines.append(
        f"   slots={s['slots_total']} allocated={s['slots_allocated']} free={s['slots_free_slot']} "
        f"sparse={s['slots_sparse_hole']} unreadable={s['slots_unreadable']} "
        f"stale_bitmap={s['stale_bitmap_warnings']}"
    )
    for err in ag["errors"]:
        loc = "?" if err["ag_block"] is None else err["ag_block"]
        lines.append(f"   ! [{err['kind']}] {err['tree']} block {loc}: {err['message']}")

    lines.append("   inobt/finobt records (audit trail):")
    for rec in ag["records"]:
        lines.append(
            f"     {rec['tree']:6s} agblk={rec['ag_block']:>4} byte=0x{rec['byte_offset']:x} "
            f"rec#{rec['record_index']} startino={rec['startino']:>6} hole={rec['holemask_hex']} "
            f"inodemask={rec['inodemask_hex']} freecount={rec['freecount']:>3} free={rec['freemask_hex']}"
        )

    lines.append("   inode slots:")
    for inode in ag["inodes"]:
        lines.append("   " + _render_inode(inode))
    lines.append("")
    return lines


def _render_inode(inode: dict) -> str:
    label = _STATE_LABEL[inode["state"]]
    loc = "-" if inode["byte_offset"] is None else f"0x{inode['byte_offset']:x}"
    base = (
        f"{label} ino={inode['ino']:>8} (AG{inode['ag']}:{inode['ag_ino']:<6}) "
        f"byte={loc:<10} rec#{inode['record_index']} chunk={inode['chunk_index']} slot={inode['slot']}"
    )
    if inode["state"] == STATE_ALLOCATED:
        mode = "-" if inode["mode"] is None else f"0o{inode['mode']:06o}"
        base += f" mode={mode} nlink={inode['nlink']} magic_ok={inode['dinode_ok']}"
        if inode.get("finobt_warning"):
            base += " FINOBT-MISMATCH"
    if inode["state"] == STATE_UNREADABLE:
        base += f" error={inode['error']}"
    if inode["path_status"] in ("RECOVERED", "NO_PATH", "ERROR"):
        path = inode["path"] if inode["path"] is not None else "<unreachable>"
        base += f" path[{inode['path_status']}]={path}"
    base += f" evidence={','.join(inode['reasons'])}"
    return base
