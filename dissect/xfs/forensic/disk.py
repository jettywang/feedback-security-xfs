"""Guarded byte-level access to an XFS image.

The reader never raises a generic ``IOError`` that callers might confuse with a
parse problem. Two outcomes are modelled explicitly:

* :class:`~dissect.xfs.forensic.structs.ReadFailure` - the bytes themselves could
  not be obtained (truncated image / EOF / injected unreadable region);
* successful return of exactly the requested number of bytes.

Unreadable ranges can be injected with ``unreadable`` so that damaged media can
be simulated without physically corrupting the file.
"""

from __future__ import annotations

from dataclasses import dataclass

from dissect.xfs.forensic.structs import ReadFailure


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int  # exclusive

    def overlaps(self, offset: int, length: int) -> bool:
        return self.start < offset + length and offset < self.end


class DiskImage:
    def __init__(self, fh, unreadable: list[ByteRange] | None = None) -> None:
        self.fh = fh
        self._unreadable = list(unreadable or [])
        self.fh.seek(0, 2)
        self.size = self.fh.tell()

    def add_unreadable(self, start: int, end: int) -> None:
        self._unreadable.append(ByteRange(start, end))

    def _blocked(self, offset: int, length: int) -> ByteRange | None:
        for region in self._unreadable:
            if region.overlaps(offset, length):
                return region
        return None

    def read_exact(self, offset: int, length: int, what: str) -> bytes:
        """Read exactly ``length`` bytes or raise :class:`ReadFailure`.

        ``what`` is a short label (e.g. ``"AG 1 inobt block 4"``) embedded in
        the failure so every error can be traced back to a concrete location.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        blocked = self._blocked(offset, length)
        if blocked is not None:
            raise ReadFailure(
                f"{what}: unreadable region 0x{blocked.start:x}-0x{blocked.end:x} "
                f"covers request at byte 0x{offset:x}+{length}"
            )
        if offset >= self.size:
            raise ReadFailure(
                f"{what}: offset 0x{offset:x} is beyond end of image (size 0x{self.size:x})"
            )
        self.fh.seek(offset)
        data = self.fh.read(length)
        if len(data) != length:
            short = length - len(data)
            raise ReadFailure(
                f"{what}: short read at byte 0x{offset:x}, "
                f"wanted {length} bytes got {len(data)} (missing {short}, image ends at 0x{self.size:x})"
            )
        return data

    def read_block(self, fs_block: int, block_size: int, what: str) -> bytes:
        return self.read_exact(fs_block * block_size, block_size, what)
