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


class DirectoryIndexError(Error):
    """Raised when a directory index (leaf/node) block cannot be trusted.

    Signals that the hash-indexed lookup could not produce an authoritative
    answer. Callers must fall back to a linear scan of the directory data
    blocks instead of concluding that a name is absent.
    """
