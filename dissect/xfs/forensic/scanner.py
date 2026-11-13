"""Path-independent XFS inode forensics scanner.

The scanner reconstructs the allocated inode inventory from the per-AG inode
allocation btrees (``inobt`` and, when present, ``finobt``). It deliberately
never walks the directory hierarchy to decide whether an inode is allocated:
allocation state comes from the btree bitmaps, and path reconstruction is a
separate, best-effort annotation layer.

Slot classification (never conflated):

* ``SPARSE_HOLE``      - holemask marks the chunk-64 as absent; no inode blocks
* ``FREE_SLOT``        - present chunk, freemask bit set; the slot is free space
* ``ALLOCATED``        - present chunk, freemask bit clear; an inode exists
* ``UNREADABLE``       - the inode record/block could not be read; no allocation
                         conclusion is drawn
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import BinaryIO

from dissect.xfs.forensic.disk import DiskImage
from dissect.xfs.forensic.structs import (
    FINOBT_MAGICS,
    INOBT_MAGICS,
    XFS_DINODE_MAGIC,
    AGF,
    AGI,
    BtreeBlock,
    InoRec,
    ReadFailure,
    StructureError,
    parse_agf,
    parse_agi,
    parse_btree_block,
    parse_dinode_core,
    parse_inobt_rec,
    parse_superblock,
    parse_sf_dir,
)

STATE_SPARSE_HOLE = "SPARSE_HOLE"
STATE_FREE_SLOT = "FREE_SLOT"
STATE_ALLOCATED = "ALLOCATED"
STATE_UNREADABLE = "UNREADABLE"

# Reason codes (machine stable).
R_RECORD = "inobt_record"
R_HOLE = "holemask_bit"
R_FREE = "freemask_bit_set"
R_ALLOC = "freemask_bit_clear"
R_READFAIL = "read_failure"
R_DINODE = "dinode_magic_ok"
R_DINODE_ZEROED = "dinode_zeroed"
R_DINODE_BAD_MAGIC = "dinode_bad_magic"
R_FINOBT = "finobt_crosscheck"
R_STALE = "freemask_finobt_mismatch"
R_NULL_FREE = "finobt_root_null"

INODES_PER_CHUNK = 64
INODE_CHUNK_SLOTS = 64
CHUNK64_PER_RECORD = 16
NULLAGBLOCK = 0xFFFFFFFF


@dataclass
class InodeEntry:
    ag: int
    ag_ino: int
    ino: int
    state: str
    reasons: list[str]
    byte_offset: int | None
    inobt_block: int | None
    record_index: int | None
    chunk_index: int | None
    slot: int | None
    holemask_hex: str | None = None
    freemask_hex: str | None = None
    inodemask_hex: str | None = None
    mode: int | None = None
    nlink: int | None = None
    size: int | None = None
    gen: int | None = None
    fmt: int | None = None
    is_dir: bool | None = None
    dinode_ok: bool | None = None
    dinode_state: str | None = None  # OK / ZEROED / BAD_MAGIC / UNREADABLE
    finobt_free: bool | None = None
    finobt_present: bool | None = None
    finobt_warning: str | None = None
    path: str | None = None
    path_status: str | None = None  # RECOVERED / NO_PATH / NOT_ATTEMPTED / ERROR
    path_reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecordEvidence:
    tree: str
    ag: int
    ag_block: int
    fs_block: int
    byte_offset: int
    level: int
    record_index: int
    startino: int
    holemask_hex: str
    inodemask_hex: str
    freecount: int
    freemask_hex: str
    present_chunk64: int


@dataclass
class TreeError:
    ag: int
    tree: str
    ag_block: int | None
    fs_block: int | None
    byte_offset: int | None
    kind: str  # READ_FAILURE / STRUCTURE_ERROR
    message: str


@dataclass
class AGResult:
    ag: int
    ag_start_byte: int
    nominal_blocks: int
    actual_blocks: int
    is_last_ag: bool
    agi: dict | None
    agf: dict | None
    inobt_root: int | None
    finobt_root: int | None
    has_finobt: bool
    records: list[RecordEvidence] = field(default_factory=list)
    errors: list[TreeError] = field(default_factory=list)
    inodes: list[InodeEntry] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class ForensicScanner:
    def __init__(self, fh: BinaryIO, disk: DiskImage | None = None) -> None:
        self.disk = disk or DiskImage(fh)
        self._errors: list[TreeError] = []
        sb_raw = self.disk.read_exact(0, 512, "primary superblock")
        self.sb = parse_superblock(sb_raw)
        self._validate_geometry()

    # ------------------------------------------------------------------ utils
    def _validate_geometry(self) -> None:
        sb = self.sb
        if sb.block_size == 0 or sb.block_size % 512:
            raise StructureError(f"invalid block size {sb.block_size}")
        if sb.inopblock not in (8, 16, 32, 64):
            raise StructureError(f"unexpected inodes per block {sb.inopblock}")
        if sb.inodesize not in (256, 512, 1024, 2048):
            raise StructureError(f"unexpected inode size {sb.inodesize}")
        if sb.agcount == 0:
            raise StructureError("agcount is zero")

    @property
    def inum_bits(self) -> int:
        return self.sb.inum_bits

    def ag_start_byte(self, ag: int) -> int:
        return ag * self.sb.agblocks * self.sb.block_size

    def ag_actual_blocks(self, ag: int) -> int:
        start = self.ag_start_byte(ag)
        remaining = self.disk.size - start
        if remaining <= 0:
            return 0
        return min(self.sb.agblocks, remaining // self.sb.block_size)

    def inode_byte_offset(self, ag: int, ag_ino: int) -> int:
        # ag_ino is the AG-relative inode number (units of inopblock per fs
        # block). Convert through blocks rather than multiplying by inode bytes.
        inode_block = ag_ino // self.sb.inopblock
        inode_index = ag_ino % self.sb.inopblock
        within_ag = inode_block * self.sb.block_size + inode_index * self.sb.inodesize
        return self.ag_start_byte(ag) + within_ag

    def global_ino(self, ag: int, ag_ino: int) -> int:
        return (ag << self.inum_bits) | ag_ino

    # ------------------------------------------------------------- btree walk
    def _walk_tree(
        self,
        ag: int,
        root_ag_block: int,
        magics: tuple[int, ...],
        tree_name: str,
    ) -> tuple[list[tuple[InoRec, int, int]], list[TreeError]]:
        """Return ``[(record, leaf_ag_block, record_index)]`` plus tree errors.

        Bounds are checked against the *actual* on-disk extent of the AG, so a
        corrupt root/child pointer cannot cause an out-of-range read and every
        failure is captured rather than aborting the whole scan.
        """
        out: list[tuple[InoRec, int, int]] = []
        errors: list[TreeError] = []
        actual = self.ag_actual_blocks(ag)
        block_size = self.sb.block_size
        visited: set[int] = set()
        stack: list[int] = [root_ag_block]

        while stack:
            ag_block = stack.pop()
            if ag_block in visited:
                errors.append(self._terr(ag, tree_name, ag_block, "StructureError", "btree cycle detected"))
                continue
            visited.add(ag_block)

            if ag_block >= actual:
                errors.append(
                    self._terr(
                        ag,
                        tree_name,
                        ag_block,
                        "StructureError",
                        f"block {ag_block} outside AG bounds (actual AG blocks={actual})",
                    )
                )
                continue

            fs_block = ag * self.sb.agblocks + ag_block
            byte_off = fs_block * block_size
            try:
                raw = self.disk.read_block(fs_block, block_size, f"AG {ag} {tree_name} block {ag_block}")
                node = parse_btree_block(raw, self.sb.has_crc)
            except (ReadFailure, StructureError) as exc:
                errors.append(self._terr(ag, tree_name, ag_block, type(exc).__name__, str(exc)))
                continue

            if node.magic not in magics:
                errors.append(
                    self._terr(
                        ag,
                        tree_name,
                        ag_block,
                        "StructureError",
                        f"bad magic 0x{node.magic:x}, expected one of {[hex(m) for m in magics]}",
                    )
                )
                continue

            if node.level == 0:
                recs, leaf_errors = self._read_leaf_records(ag, ag_block, raw, node, tree_name)
                out.extend(recs)
                errors.extend(leaf_errors)
            else:
                pointers = self._read_pointers(ag, ag_block, raw, node, tree_name, errors)
                stack.extend(reversed(pointers))

        return out, errors

    def _read_leaf_records(
        self, ag: int, ag_block: int, raw: bytes, node: BtreeBlock, tree_name: str
    ) -> tuple[list[tuple[InoRec, int, int]], list[TreeError]]:
        out: list[tuple[InoRec, int, int]] = []
        errors: list[TreeError] = []
        body = raw[node.header_size :]
        actual = self.ag_actual_blocks(ag)
        for idx in range(node.numrecs):
            chunk = body[idx * 16 : (idx + 1) * 16]
            if len(chunk) != 16:
                errors.append(
                    self._terr(
                        ag, tree_name, ag_block, "ReadFailure",
                        f"record {idx} truncated inside leaf block",
                    )
                )
                continue
            try:
                rec = parse_inobt_rec(chunk)
            except (ReadFailure, StructureError) as exc:
                errors.append(self._terr(ag, tree_name, ag_block, type(exc).__name__, str(exc)))
                continue
            if rec.startino >= actual * self.sb.inopblock:
                errors.append(
                    self._terr(
                        ag, tree_name, ag_block, "StructureError",
                        f"record {idx} startino {rec.startino} beyond AG capacity "
                        f"({actual * self.sb.inopblock})",
                    )
                )
                continue
            out.append((rec, ag_block, idx))
        return out, errors

    def _read_pointers(
        self,
        ag: int,
        ag_block: int,
        raw: bytes,
        node: BtreeBlock,
        tree_name: str,
        errors: list[TreeError],
    ) -> list[int]:
        # Pointers are uint32 AG-relative blocks packed from the end of the block.
        maxrecs = (self.sb.block_size - node.header_size) // 8
        ptr_end = self.sb.block_size
        ptr_start = ptr_end - node.numrecs * 4
        if ptr_start < node.header_size + maxrecs * 4 or ptr_end > len(raw):
            errors.append(
                self._terr(ag, tree_name, ag_block, "StructureError", "pointer array out of block bounds")
            )
            return []
        return [int.from_bytes(raw[ptr_start + i * 4 : ptr_start + (i + 1) * 4], "big") for i in range(node.numrecs)]

    def _terr(self, ag: int, tree: str, ag_block: int | None, kind: str, message: str) -> TreeError:
        fs_block = None if ag_block is None else ag * self.sb.agblocks + ag_block
        byte_off = None if fs_block is None else fs_block * self.sb.block_size
        return TreeError(
            ag=ag, tree=tree, ag_block=ag_block, fs_block=fs_block, byte_offset=byte_off, kind=kind, message=message
        )

    # ------------------------------------------------------------- per-AG scan
    def scan_ag(self, ag: int) -> AGResult:
        if ag >= self.sb.agcount:
            raise StructureError(f"AG {ag} exceeds agcount {self.sb.agcount}")
        start = self.ag_start_byte(ag)
        actual = self.ag_actual_blocks(ag)
        is_last = ag == self.sb.agcount - 1
        result = AGResult(
            ag=ag,
            ag_start_byte=start,
            nominal_blocks=self.sb.agblocks,
            actual_blocks=actual,
            is_last_ag=is_last,
            agi=None,
            agf=None,
            inobt_root=None,
            finobt_root=None,
            has_finobt=False,
        )

        agi = self._read_agi(ag, result)
        agf = self._read_agf(ag, result)
        if agf is not None and agf.length != self.sb.agblocks:
            result.errors.append(
                self._terr(ag, "agf", None, "StructureError",
                           f"AGF length {agf.length} != sb agblocks {self.sb.agblocks}")
            )

        if agi is None:
            result.summary = self._summarize(result)
            return result

        result.inobt_root = agi.root
        result.finobt_root = agi.free_root
        result.has_finobt = agi.free_root != NULLAGBLOCK

        inobt_recs, inobt_errors = self._walk_tree(ag, agi.root, INOBT_MAGICS, "inobt")
        result.errors.extend(inobt_errors)

        finobt_index: dict[int, InoRec] = {}
        if agi.free_root != NULLAGBLOCK:
            fin_recs, fin_errors = self._walk_tree(ag, agi.free_root, FINOBT_MAGICS, "finobt")
            result.errors.extend(fin_errors)
            for rec, blk, idx in fin_recs:
                result.records.append(self._evidence("finobt", ag, blk, idx, rec))
                finobt_index.setdefault(rec.startino, rec)
        for rec, blk, idx in inobt_recs:
            result.records.append(self._evidence("inobt", ag, blk, idx, rec))

        self._classify_inodes(ag, inobt_recs, finobt_index, result)
        result.summary = self._summarize(result)
        return result

    def _read_agi(self, ag: int, result: AGResult) -> AGI | None:
        try:
            raw = self.disk.read_exact(
                self.ag_start_byte(ag) + self.sb.sectsize * 2,
                self.sb.sectsize,
                f"AG {ag} AGI header (block 0 sector 2)",
            )
            agi = parse_agi(raw)
        except (ReadFailure, StructureError) as exc:
            result.errors.append(self._terr(ag, "agi", None, type(exc).__name__, str(exc)))
            return None
        result.agi = {
            "seqno": agi.seqno,
            "length": agi.length,
            "count": agi.count,
            "root": agi.root,
            "level": agi.level,
            "freecount": agi.freecount,
            "free_root": agi.free_root,
            "free_level": agi.free_level,
            "iblocks": agi.iblocks,
            "fblocks": agi.fblocks,
        }
        if agi.seqno != ag:
            result.errors.append(
                self._terr(ag, "agi", None, "StructureError", f"AGI seqno {agi.seqno} != AG index {ag}")
            )
        return agi

    def _read_agf(self, ag: int, result: AGResult) -> AGF | None:
        try:
            raw = self.disk.read_exact(
                self.ag_start_byte(ag) + self.sb.sectsize,
                self.sb.sectsize,
                f"AG {ag} AGF header (block 0 sector 1)",
            )
            agf = parse_agf(raw)
        except (ReadFailure, StructureError) as exc:
            result.errors.append(self._terr(ag, "agf", None, type(exc).__name__, str(exc)))
            return None
        result.agf = {"seqno": agf.seqno, "length": agf.length, "freeblks": agf.freeblks}
        return agf

    def _evidence(self, tree: str, ag: int, ag_block: int, idx: int, rec: InoRec) -> RecordEvidence:
        fs_block = ag * self.sb.agblocks + ag_block
        return RecordEvidence(
            tree=tree,
            ag=ag,
            ag_block=ag_block,
            fs_block=fs_block,
            byte_offset=fs_block * self.sb.block_size,
            level=0,
            record_index=idx,
            startino=rec.startino,
            holemask_hex=f"0x{rec.holemask:04x}",
            inodemask_hex=f"0x{rec.inodemask:08x}",
            freecount=rec.freecount,
            freemask_hex=f"0x{rec.freemask:016x}",
            present_chunk64=rec.present_chunk64_count,
        )

    # ------------------------------------------------------- slot classification
    def _classify_inodes(
        self,
        ag: int,
        inobt_recs: list[tuple[InoRec, int, int]],
        finobt_index: dict[int, InoRec],
        result: AGResult,
    ) -> None:
        capacity = self.ag_actual_blocks(ag) * self.sb.inopblock
        for rec, leaf_block, rec_idx in inobt_recs:
            fin = finobt_index.get(rec.startino)
            for chunk64 in range(CHUNK64_PER_RECORD):
                chunk_start = rec.startino + chunk64 * INODES_PER_CHUNK
                hole_bit = (rec.holemask >> chunk64) & 1
                if hole_bit:
                    result.inodes.append(
                        self._hole_entry(ag, rec, leaf_block, rec_idx, chunk64, chunk_start)
                    )
                    continue
                if chunk_start >= capacity:
                    # A present (non-hole) chunk claims inode blocks beyond the
                    # readable extent of this AG. Record it as unreadable rather
                    # than reading into the next AG or off the end of the image.
                    result.errors.append(
                        self._terr(
                            ag,
                            "inobt",
                            leaf_block,
                            "StructureError",
                            f"record {rec_idx} chunk {chunk64} starts at agino {chunk_start} "
                            f"but AG capacity is only {capacity} inodes",
                        )
                    )
                    result.inodes.append(
                        self._out_of_range_entry(ag, rec, leaf_block, rec_idx, chunk64, chunk_start)
                    )
                    continue
                for slot in range(INODE_CHUNK_SLOTS):
                    bit = chunk64 * INODE_CHUNK_SLOTS + slot
                    ag_ino = chunk_start + slot
                    free = bool((rec.freemask >> bit) & 1)
                    entry = self._present_entry(
                        ag, rec, leaf_block, rec_idx, chunk64, slot, bit, ag_ino, free
                    )
                    self._apply_finobt(entry, fin, bit)
                    result.inodes.append(entry)

    def _hole_entry(
        self, ag: int, rec: InoRec, leaf_block: int, rec_idx: int, chunk64: int, chunk_start: int
    ) -> InodeEntry:
        return InodeEntry(
            ag=ag,
            ag_ino=chunk_start,
            ino=self.global_ino(ag, chunk_start),
            state=STATE_SPARSE_HOLE,
            reasons=[R_RECORD, R_HOLE],
            byte_offset=None,
            inobt_block=leaf_block,
            record_index=rec_idx,
            chunk_index=chunk64,
            slot=None,
            holemask_hex=f"0x{rec.holemask:04x}",
            freemask_hex=f"0x{rec.freemask:016x}",
            inodemask_hex=f"0x{rec.inodemask:08x}",
            path_status="NOT_ATTEMPTED",
            path_reason="sparse hole: no inode blocks owned by this record",
        )

    def _out_of_range_entry(
        self, ag: int, rec: InoRec, leaf_block: int, rec_idx: int, chunk64: int, chunk_start: int
    ) -> InodeEntry:
        return InodeEntry(
            ag=ag,
            ag_ino=chunk_start,
            ino=self.global_ino(ag, chunk_start),
            state=STATE_UNREADABLE,
            reasons=[R_RECORD, R_READFAIL],
            byte_offset=None,
            inobt_block=leaf_block,
            record_index=rec_idx,
            chunk_index=chunk64,
            slot=None,
            holemask_hex=f"0x{rec.holemask:04x}",
            freemask_hex=f"0x{rec.freemask:016x}",
            inodemask_hex=f"0x{rec.inodemask:08x}",
            dinode_state="UNREADABLE",
            path_status="ERROR",
            path_reason="claimed chunk lies outside the readable extent of the AG",
            error="present inobt chunk beyond AG capacity",
        )

    def _present_entry(
        self,
        ag: int,
        rec: InoRec,
        leaf_block: int,
        rec_idx: int,
        chunk64: int,
        slot: int,
        bit: int,
        ag_ino: int,
        free: bool,
    ) -> InodeEntry:
        byte_off = self.inode_byte_offset(ag, ag_ino)
        if free:
            return InodeEntry(
                ag=ag,
                ag_ino=ag_ino,
                ino=self.global_ino(ag, ag_ino),
                state=STATE_FREE_SLOT,
                reasons=[R_RECORD, R_FREE],
                byte_offset=byte_off,
                inobt_block=leaf_block,
                record_index=rec_idx,
                chunk_index=chunk64,
                slot=slot,
                holemask_hex=f"0x{rec.holemask:04x}",
                freemask_hex=f"0x{rec.freemask:016x}",
                inodemask_hex=f"0x{rec.inodemask:08x}",
                path_status="NOT_ATTEMPTED",
                path_reason="free slot: never allocated per freemask",
            )

        entry = InodeEntry(
            ag=ag,
            ag_ino=ag_ino,
            ino=self.global_ino(ag, ag_ino),
            state=STATE_ALLOCATED,
            reasons=[R_RECORD, R_ALLOC],
            byte_offset=byte_off,
            inobt_block=leaf_block,
            record_index=rec_idx,
            chunk_index=chunk64,
            slot=slot,
            holemask_hex=f"0x{rec.holemask:04x}",
            freemask_hex=f"0x{rec.freemask:016x}",
            inodemask_hex=f"0x{rec.inodemask:08x}",
            path_status="NOT_ATTEMPTED",
        )
        self._read_dinode(entry)
        return entry

    def _read_dinode(self, entry: InodeEntry) -> None:
        try:
            raw = self.disk.read_exact(
                entry.byte_offset,
                self.sb.inodesize,
                f"dinode AG{entry.ag}:{entry.ag_ino}",
            )
            core = parse_dinode_core(raw)
        except ReadFailure as exc:
            entry.state = STATE_UNREADABLE
            entry.dinode_state = "UNREADABLE"
            entry.error = str(exc)
            entry.reasons = [R_RECORD, R_READFAIL]
            entry.path_status = "ERROR"
            entry.path_reason = "inode bytes unreadable; allocation left undetermined beyond bitmap"
            return
        except StructureError as exc:
            entry.dinode_ok = False
            entry.dinode_state = "BAD_MAGIC"
            entry.error = str(exc)
            return

        if raw[: self.sb.inodesize] == b"\x00" * self.sb.inodesize:
            # Bitmap claims allocation but the inode slot is fully zeroed. Keep the
            # bitmap-based ALLOCATED state but record that there is no dinode to
            # confirm it; never silently treat it as a healthy live inode.
            entry.dinode_ok = False
            entry.dinode_state = "ZEROED"
            entry.reasons.append(R_DINODE_ZEROED)
            entry.path_status = "NO_PATH"
            entry.path_reason = "bitmap says allocated but dinode slot is zeroed"
            return

        if core.magic != XFS_DINODE_MAGIC:
            entry.dinode_ok = False
            entry.dinode_state = "BAD_MAGIC"
            entry.error = f"dinode magic 0x{core.magic:04x} != 0x{XFS_DINODE_MAGIC:04x}"
            entry.reasons.append(R_DINODE_BAD_MAGIC)
            entry.path_status = "NO_PATH"
            entry.path_reason = "bitmap says allocated but dinode magic invalid"
            return

        entry.dinode_ok = True
        entry.dinode_state = "OK"
        entry.reasons.append(R_DINODE)
        entry.mode = core.mode
        entry.nlink = core.nlink
        entry.size = core.size
        entry.gen = core.gen
        entry.fmt = core.fmt
        entry.is_dir = core.is_dir

    def _apply_finobt(self, entry: InodeEntry, fin: InoRec | None, bit: int) -> None:
        if fin is None:
            entry.finobt_present = False
            return
        entry.finobt_present = True
        fin_free = bool((fin.freemask >> bit) & 1)
        entry.finobt_free = fin_free
        inobt_free = entry.state == STATE_FREE_SLOT
        if fin_free != inobt_free:
            entry.finobt_warning = R_STALE
            entry.reasons.append(R_STALE)

    # ------------------------------------------------------------- summarizing
    def _summarize(self, result: AGResult) -> dict:
        counts = {STATE_SPARSE_HOLE: 0, STATE_FREE_SLOT: 0, STATE_ALLOCATED: 0, STATE_UNREADABLE: 0}
        dinode_states = {"OK": 0, "ZEROED": 0, "BAD_MAGIC": 0, "UNREADABLE": 0}
        for inode in result.inodes:
            counts[inode.state] += 1
            if inode.state == STATE_ALLOCATED and inode.dinode_state in dinode_states:
                dinode_states[inode.dinode_state] += 1
        read_failures = sum(1 for e in result.errors if e.kind == "ReadFailure")
        structure_errors = sum(1 for e in result.errors if e.kind == "StructureError")
        return {
            "slots_total": len(result.inodes),
            **{f"slots_{k.lower()}": v for k, v in counts.items()},
            "allocated_dinode_ok": dinode_states["OK"],
            "allocated_dinode_zeroed": dinode_states["ZEROED"],
            "allocated_dinode_bad_magic": dinode_states["BAD_MAGIC"],
            "inobt_records": sum(1 for r in result.records if r.tree == "inobt"),
            "finobt_records": sum(1 for r in result.records if r.tree == "finobt"),
            "read_failures": read_failures,
            "structure_errors": structure_errors,
            "header_read_failures": sum(
                1 for e in result.errors if e.kind == "ReadFailure" and e.tree in ("agi", "agf")
            ),
            "stale_bitmap_warnings": sum(1 for i in result.inodes if i.finobt_warning),
        }

    # ------------------------------------------------------------ full filesystem
    def scan(self) -> dict:
        ag_results = [self.scan_ag(ag) for ag in range(self.sb.agcount)]
        self._reconstruct_paths(ag_results)
        allocated = [i for ag in ag_results for i in ag.inodes if i.state == STATE_ALLOCATED]
        allocated_ok = [i for i in allocated if i.dinode_state == "OK"]
        no_path = [i for i in allocated if i.path_status == "NO_PATH"]
        return {
            "filesystem": {
                "block_size": self.sb.block_size,
                "inode_size": self.sb.inodesize,
                "inopblock": self.sb.inopblock,
                "agcount": self.sb.agcount,
                "agblocks_nominal": self.sb.agblocks,
                "dblocks": self.sb.dblocks,
                "version": self.sb.version,
                "has_crc": self.sb.has_crc,
                "has_spinodes": self.sb.has_spinodes,
                "rootino": self.sb.rootino,
                "image_size_bytes": self.disk.size,
            },
            "allocation_groups": [self._ag_dict(ag) for ag in ag_results],
            "totals": {
                "allocated_inodes_bitmap": len(allocated),
                "allocated_inodes_dinode_ok": len(allocated_ok),
                "allocated_without_path": len(no_path),
                "free_slots": sum(a.summary.get("slots_free_slot", 0) for a in ag_results),
                "sparse_holes": sum(a.summary.get("slots_sparse_hole", 0) for a in ag_results),
                "unreadable": sum(a.summary.get("slots_unreadable", 0) for a in ag_results),
                "dinode_zeroed": sum(a.summary.get("allocated_dinode_zeroed", 0) for a in ag_results),
                "dinode_bad_magic": sum(a.summary.get("allocated_dinode_bad_magic", 0) for a in ag_results),
                "tree_read_failures": sum(a.summary.get("read_failures", 0) for a in ag_results),
                "tree_structure_errors": sum(a.summary.get("structure_errors", 0) for a in ag_results),
            },
        }

    def _ag_dict(self, ag: AGResult) -> dict:
        return {
            "ag": ag.ag,
            "ag_start_byte": ag.ag_start_byte,
            "nominal_blocks": ag.nominal_blocks,
            "actual_blocks": ag.actual_blocks,
            "is_last_ag": ag.is_last_ag,
            "agi": ag.agi,
            "agf": ag.agf,
            "inobt_root": ag.inobt_root,
            "finobt_root": ag.finobt_root,
            "has_finobt": ag.has_finobt,
            "summary": ag.summary,
            "errors": [asdict(e) for e in ag.errors],
            "records": [asdict(r) for r in ag.records],
            "inodes": [i.to_dict() for i in ag.inodes],
        }

    # ------------------------------------------------------------ path annotation
    def _reconstruct_paths(self, ag_results: list[AGResult]) -> None:
        """Best-effort path annotation from shortform directories only.

        Allocation decisions never depend on this pass. We index every readable
        shortform directory, then walk from ``sb_rootino`` breadth-first so that
        any reachable inode (directories, regular files, symlinks, across AGs)
        receives a path. An allocated inode not reached from the root is flagged
        ``NO_PATH`` and keeps its ALLOCATED state; it is never called deleted.
        """
        entries = {i.ino: i for ag in ag_results for i in ag.inodes}
        children: dict[int, list[tuple[int, bytes]]] = {}

        for entry in entries.values():
            if entry.state != STATE_ALLOCATED or not entry.is_dir or entry.fmt is None:
                continue
            self._index_sf_dir(entry, children)

        root_ino = self.sb.rootino
        resolved: dict[int, str] = {}
        if root_ino in entries and entries[root_ino].state == STATE_ALLOCATED:
            resolved[root_ino] = "/"
            queue: list[tuple[int, str]] = [(root_ino, "")]
            seen: set[int] = {root_ino}
            while queue:
                parent_ino, parent_path = queue.pop(0)
                for child_ino, name_b in children.get(parent_ino, []):
                    if child_ino in seen or child_ino not in entries:
                        continue
                    seen.add(child_ino)
                    name = name_b.decode(errors="surrogateescape")
                    child_path = f"{parent_path}/{name}" if parent_path else f"/{name}"
                    resolved[child_ino] = child_path
                    child_entry = entries[child_ino]
                    if child_entry.is_dir:
                        queue.append((child_ino, child_path))

        for ino, entry in entries.items():
            if entry.state != STATE_ALLOCATED:
                continue
            if entry.path_status in ("ERROR",):
                continue
            if ino in resolved:
                entry.path = resolved[ino]
                entry.path_status = "RECOVERED"
                entry.path_reason = "reachable via shortform directory entries from sb_rootino"
            else:
                entry.path = None
                entry.path_status = "NO_PATH"
                entry.path_reason = (
                    "allocated per inobt but not referenced by any readable shortform directory "
                    "reachable from sb_rootino"
                )

    def _index_sf_dir(
        self,
        entry: InodeEntry,
        children: dict[int, list[tuple[int, bytes]]],
    ) -> None:
        try:
            raw = self.disk.read_exact(entry.byte_offset, self.sb.inodesize, f"dir inode {entry.ino}")
            core = parse_dinode_core(raw)
        except (ReadFailure, StructureError):
            entry.path_status = "ERROR"
            entry.path_reason = "directory inode unreadable during path annotation"
            return
        if core.fmt != 1:  # XFS_DINODE_FMT_LOCAL (shortform)
            entry.path_reason = "directory uses extents/btree format; shortform-only reconstruction skipped"
            return
        data_off = 0xB0 if core.version >= 3 else 0x64
        try:
            _parent, dentries = parse_sf_dir(raw[data_off:], self.sb.has_ftype)
        except StructureError as exc:
            entry.path_status = "ERROR"
            entry.path_reason = f"shortform dir parse failed: {exc}"
            return
        bucket = children.setdefault(entry.ino, [])
        for de in dentries:
            bucket.append((de.ino, de.name))
