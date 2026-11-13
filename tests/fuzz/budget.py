"""Read budget wrapper, independent of the XFS parser implementation.

The forensic worker must never be held hostage by a malformed index that points
at a huge number of blocks. :class:`BudgetedStream` wraps any binary stream and
counts the *actual* bytes handed to the parser through ``read`` (seeks are
free). Once the cumulative read volume exceeds the budget a
:class:`ReadBudgetExceeded` is raised. Because it only relies on the standard
``BinaryIO`` protocol, the parser is not modified to enforce it: the same
wrapper protects any future consumer of the image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from types import TracebackType


class ReadBudgetExceeded(Exception):
    """Raised when a stream is read past its configured byte budget."""

    def __init__(self, budget: int, requested: int):
        super().__init__(f"Read budget of {budget} bytes exceeded while serving {requested} bytes")
        self.budget = budget
        self.requested = requested


class BudgetedStream:
    def __init__(self, fh: BinaryIO, budget: int):
        self._fh = fh
        self._budget = budget
        self.bytes_read = 0

    @property
    def budget(self) -> int:
        return self._budget

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            # Never allow an unbounded read to bypass the budget.
            size = max(self._budget - self.bytes_read, 0)
        if self.bytes_read + size > self._budget:
            raise ReadBudgetExceeded(self._budget, size)
        data = self._fh.read(size)
        self.bytes_read += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._fh.seek(offset, whence)

    def tell(self) -> int:
        return self._fh.tell()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> BudgetedStream:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
