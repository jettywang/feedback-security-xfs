from __future__ import annotations


class Error(Exception):
    pass


class UnsupportedDataforkException(Error):
    pass


class BTreeCycleError(Error):
    """Raised when a B+Tree node references a block already visited on its walk path."""


class InvalidBlockAddressError(Error):
    """Raised when a B+Tree pointer does not translate to a valid filesystem block."""


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
