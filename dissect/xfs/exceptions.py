from __future__ import annotations


class Error(Exception):
    pass


class UnsupportedDataforkException(Error):
    pass


class FileNotFoundError(Error, FileNotFoundError):
    pass


class IsADirectoryError(Error, IsADirectoryError):
    pass


class NotADirectoryError(Error, NotADirectoryError):
    pass


class NotASymlinkError(Error):
    pass


class SymlinkUnavailableException(Error):
    pass


class DirectoryScanError(Error):
    """Base class for errors recovered while scanning a directory.

    Every error carries the logical directory block (``block_num``) it occurred in so the
    scan report can point at the exact block that could not be parsed fully. A block number
    of ``-1`` is used for problems that are not tied to a specific directory block.
    """

    def __init__(self, message: str, block_num: int = -1) -> None:
        super().__init__(message)
        self.block_num = block_num


class DirectoryBlockMagicError(DirectoryScanError):
    """A directory block has a magic number that is neither a data block nor a known benign block."""


class DirectoryRecordTruncatedError(DirectoryScanError):
    """A directory record could not be parsed because its block ended prematurely."""


class DirectoryReadError(DirectoryScanError):
    """A directory block could not be read completely (short read or underlying I/O failure)."""
