"""Shared fuzz target identifiers."""

from __future__ import annotations

SMALL_TREE = "small_inobt"
LARGE_TREE = "large_bmap"
TREE_TARGETS: tuple[str, ...] = (SMALL_TREE, LARGE_TREE)
