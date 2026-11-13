"""Command line entry point for the XFS inode forensics scanner."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from dissect.xfs.forensic.disk import ByteRange, DiskImage
from dissect.xfs.forensic.report import render_text
from dissect.xfs.forensic.scanner import ForensicScanner
from dissect.xfs.forensic.structs import ReadFailure, StructureError


def _parse_ranges(values: Sequence[str]) -> list[ByteRange]:
    ranges: list[ByteRange] = []
    for value in values:
        start_s, _, end_s = value.partition(":")
        ranges.append(ByteRange(start=int(start_s, 0), end=int(end_s, 0)))
    return ranges


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xfs-inode-forensics",
        description="Reconstruct the allocated XFS inode inventory from per-AG inode btrees "
        "without path traversal.",
    )
    parser.add_argument("image", help="path to the XFS image (regular file or block device dump)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument(
        "--unreadable",
        action="append",
        default=[],
        metavar="START:END",
        help="inject an unreadable byte range (hex or decimal); repeatable. Used to validate "
        "read-failure handling without damaging the image.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with open(args.image, "rb") as fh:
            disk = DiskImage(fh, unreadable=_parse_ranges(args.unreadable))
            scanner = ForensicScanner(fh, disk=disk)
            report = scanner.scan()
    except (ReadFailure, StructureError) as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Kept available for programmatic callers that want raw dataclasses.
def scan_image(path: str, unreadable: Sequence[str] = ()) -> dict:
    with open(path, "rb") as fh:
        disk = DiskImage(fh, unreadable=_parse_ranges(unreadable))
        return ForensicScanner(fh, disk=disk).scan()


__all__ = ["build_parser", "main", "scan_image"]
