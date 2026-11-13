"""Path-independent XFS inode forensics tooling."""

from __future__ import annotations

from dissect.xfs.forensic.disk import ByteRange, DiskImage
from dissect.xfs.forensic.scanner import (
    STATE_ALLOCATED,
    STATE_FREE_SLOT,
    STATE_SPARSE_HOLE,
    STATE_UNREADABLE,
    ForensicScanner,
)
from dissect.xfs.forensic.structs import ReadFailure, StructureError

__all__ = [
    "ByteRange",
    "DiskImage",
    "ForensicScanner",
    "ReadFailure",
    "STATE_ALLOCATED",
    "STATE_FREE_SLOT",
    "STATE_SPARSE_HOLE",
    "STATE_UNREADABLE",
    "StructureError",
]
