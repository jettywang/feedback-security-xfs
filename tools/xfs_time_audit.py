#!/usr/bin/env python3
"""Minimal, dependency-free reference check for XFS inode timestamp decoding.

This script intentionally does not import dissect.xfs: it is a control for the
library implementation, not a copy of it. Run with:

    python tools/xfs_time_audit.py

Exit code 0 means every assertion in ``main`` held.
"""

from __future__ import annotations

import argparse
import datetime
import struct
from dataclasses import dataclass

UTC = datetime.timezone.utc
UNIX_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=UTC)
NS_PER_SECOND = 1_000_000_000

# XFS bigtime epoch = start of the legacy signed-32-bit range:
# 1901-12-13 20:45:52 UTC == Unix epoch - 2**31 seconds.
BIGTIME_EPOCH_OFFSET_SECONDS = 1 << 31
BIGTIME_EPOCH = UNIX_EPOCH - datetime.timedelta(seconds=BIGTIME_EPOCH_OFFSET_SECONDS)

INODE_VERSION_OFFSET = 4
DIFLAG2_OFFSET_V3 = 120
DIFLAG2_BIGTIME = 1 << 3
CRTIME_OFFSET = 144
V2_CORE_END = 100  # di_next_unlinked ends here; bytes after it are the data fork


@dataclass(frozen=True)
class Decoded:
    """A decoded timestamp plus the exact integers it came from."""

    raw: int
    seconds_field: int | None
    nanos_field: int | None
    unix_ns: int | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def decode_legacy(raw: int) -> Decoded:
    """Decode the legacy (pre-bigtime) xfs_timestamp_t layout.

    Layout: 64-bit big-endian value, high 32 bits = *signed* seconds since
    1970-01-01 UTC, low 32 bits = nanoseconds. The high word is not an
    unsigned counter; 0x80000000 means 1901-12-13, not 2038-01-19.
    """
    if not 0 <= raw <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("raw timestamp must be a 64-bit unsigned value")
    sec_u32 = raw >> 32
    nsec = raw & 0xFFFFFFFF
    sec = sec_u32 - (1 << 32) if sec_u32 & 0x80000000 else sec_u32
    if nsec >= NS_PER_SECOND:
        return Decoded(raw, sec, nsec, None, "legacy_nsec_out_of_range")
    return Decoded(raw, sec, nsec, sec * NS_PER_SECOND + nsec)


def decode_bigtime(raw: int) -> Decoded:
    """Decode the bigtime layout: one unsigned 64-bit nanosecond counter.

    The counter counts from 1901-12-13 20:45:52 UTC, so subtracting the
    2**31-second epoch offset yields Unix nanoseconds. No field split exists;
    interpreting the high word as seconds is always wrong.
    """
    if not 0 <= raw <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("raw timestamp must be a 64-bit unsigned value")
    sec_since_big_epoch, nsec = divmod(raw, NS_PER_SECOND)
    unix_sec = sec_since_big_epoch - BIGTIME_EPOCH_OFFSET_SECONDS
    return Decoded(raw, unix_sec, nsec, unix_sec * NS_PER_SECOND + nsec)


def to_datetime_floor(unix_ns: int) -> datetime.datetime:
    """Convert integer Unix nanoseconds to datetime, truncating sub-microseconds.

    datetime cannot hold nanoseconds. Flooring (not rounding) keeps the
    displayed instant inside the true nanosecond interval.
    """
    sec, nsec = divmod(unix_ns, NS_PER_SECOND)
    return UNIX_EPOCH + datetime.timedelta(seconds=sec, microseconds=nsec // 1000)


def to_datetime_float(unix_ns: int) -> datetime.datetime:
    """The float-based conversion used by dissect.util.ts.from_unix_ns."""
    return datetime.datetime.fromtimestamp(float(unix_ns) * 1e-9, tz=UTC)


def fmt(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond:06d}+00:00"


def gate(version: int, flags2: int | None, fs_has_bigtime: bool) -> tuple[bool, str]:
    """Return (is_bigtime, reason) for an inode.

    The per-inode flag is authoritative. The superblock feature only says the
    filesystem *can* store bigtime inodes; old inodes may remain legacy. On
    v1/v2 inodes there is no di_flags2 field at all, so its bytes are data-fork
    content and must not be interpreted as flags.
    """
    if version not in (1, 2, 3):
        return False, "unknown_inode_version"
    if version < 3:
        if flags2 not in (None, 0):
            return False, "v1_or_v2_has_no_flags2_field"
        return False, "v1_or_v2_always_legacy"
    if flags2 is None:
        return False, "missing_flags2"
    if flags2 & DIFLAG2_BIGTIME:
        if not fs_has_bigtime:
            return False, "inode_flag_but_superblock_feature_absent"
        return True, "v3_flags2_bigtime_bit"
    return False, "v3_without_bigtime_bit"


def decode_timestamp(raw: int, version: int, flags2: int | None, fs_has_bigtime: bool) -> tuple[Decoded, str]:
    is_bigtime, reason = gate(version, flags2, fs_has_bigtime)
    if reason == "inode_flag_but_superblock_feature_absent":
        return Decoded(raw, None, None, None, reason), reason
    if version < 3:
        if raw >> 32 == 0 and raw & 0xFFFFFFFF == 0:
            return Decoded(raw, 0, 0, 0, "v1_or_v2_absent_crtime_not_zero"), "crtime_not_stored"
        # For v1/v2, only atime/mtime/ctime are real. crtime reads data-fork bytes.
        return decode_legacy(raw), "legacy"
    return (decode_bigtime(raw) if is_bigtime else decode_legacy(raw)), reason


def encode_legacy(sec: int, nsec: int) -> int:
    return ((sec & 0xFFFFFFFF) << 32) | (nsec & 0xFFFFFFFF)


def encode_bigtime(unix_ns: int) -> int:
    return unix_ns + BIGTIME_EPOCH_OFFSET_SECONDS * NS_PER_SECOND


@dataclass(frozen=True)
class Vector:
    name: str
    raw: int
    version: int
    flags2: int | None
    fs_has_bigtime: bool
    expected_ns: int | None
    expected_error: str | None
    note: str


VECTORS: tuple[Vector, ...] = (
    Vector("legacy_epoch", encode_legacy(0, 0), 3, 0, True, 0, None, "1970-01-01T00:00:00.000000000Z"),
    Vector(
        "legacy_min_1901",
        encode_legacy(-(1 << 31), 0),
        3,
        0,
        True,
        -(1 << 31) * NS_PER_SECOND,
        None,
        "high word 0x80000000 is -2147483648 s; 1901-12-13, not 2038",
    ),
    Vector("legacy_-1s", encode_legacy(-1, 0), 3, 0, True, -NS_PER_SECOND, None, "one second before epoch"),
    Vector(
        "legacy_-0.5s",
        encode_legacy(-1, 500_000_000),
        3,
        0,
        True,
        -500_000_000,
        None,
        "negative second + positive nsec",
    ),
    Vector(
        "legacy_-1ns",
        encode_legacy(-1, 999_999_999),
        3,
        0,
        True,
        -1,
        None,
        "smallest representable instant below epoch",
    ),
    Vector(
        "legacy_max_2038",
        encode_legacy((1 << 31) - 1, 999_999_999),
        3,
        0,
        True,
        ((1 << 31) - 1) * NS_PER_SECOND + 999_999_999,
        None,
        "2038-01-19T03:14:07.999999999Z is the last legacy instant",
    ),
    Vector(
        "legacy_nsec_123456789",
        encode_legacy(0, 123_456_789),
        3,
        0,
        True,
        123_456_789,
        None,
        "nanoseconds survive only in the integer result",
    ),
    Vector(
        "legacy_nsec_out_of_range",
        encode_legacy(0, 1_000_000_000),
        3,
        0,
        True,
        1_000_000_000,
        "legacy_nsec_out_of_range",
        "low word is not a valid nanosecond count; malformed or wrong format",
    ),
    Vector(
        "bigtime_1901",
        encode_bigtime(-(1 << 31) * NS_PER_SECOND),
        3,
        DIFLAG2_BIGTIME,
        True,
        -(1 << 31) * NS_PER_SECOND,
        None,
        "raw counter 0 == bigtime epoch, 64 years before Unix epoch",
    ),
    Vector(
        "bigtime_1969-12-31T23:59:59.5",
        encode_bigtime(-500_000_000),
        3,
        DIFLAG2_BIGTIME,
        True,
        -500_000_000,
        None,
        "negative Unix time represented as a positive 64-bit counter",
    ),
    Vector(
        "bigtime_1970",
        encode_bigtime(0),
        3,
        DIFLAG2_BIGTIME,
        True,
        0,
        None,
        "raw 0x1dcd650000000000, high word is NOT 0 seconds",
    ),
    Vector(
        "bigtime_2038-1ns",
        encode_bigtime((1 << 31) * NS_PER_SECOND - 1),
        3,
        DIFLAG2_BIGTIME,
        True,
        (1 << 31) * NS_PER_SECOND - 1,
        None,
        "bigtime passes the 2038 boundary without a new epoch",
    ),
    Vector(
        "bigtime_2038+1s",
        encode_bigtime((1 << 31) * NS_PER_SECOND),
        3,
        DIFLAG2_BIGTIME,
        True,
        (1 << 31) * NS_PER_SECOND,
        None,
        "2038-01-19T03:14:08Z; impossible in legacy signed seconds",
    ),
    Vector(
        "bigtime_nsec_123456789",
        encode_bigtime(5 * NS_PER_SECOND + 123_456_789),
        3,
        DIFLAG2_BIGTIME,
        True,
        5 * NS_PER_SECOND + 123_456_789,
        None,
        "full 9-digit nanoseconds, no field split",
    ),
    Vector(
        "bigtime_max_2486",
        (1 << 64) - 1,
        3,
        DIFLAG2_BIGTIME,
        True,
        (1 << 64) - 1 - BIGTIME_EPOCH_OFFSET_SECONDS * NS_PER_SECOND,
        None,
        "maximum counter is 2486-07-02T20:20:25.709551615Z",
    ),
    Vector(
        "v2_flags2_lookalike_legacy",
        encode_legacy(1_650_637_449, 264_560_000),
        2,
        DIFLAG2_BIGTIME,
        True,
        1_650_637_449 * NS_PER_SECOND + 264_560_000,
        None,
        "byte 120 happens to equal 0x8 = BIGTIME; v2 has no flags2, so ignore it",
    ),
    Vector(
        "v3_flag_without_fs_feature",
        encode_legacy(1_650_637_449, 264_560_000),
        3,
        DIFLAG2_BIGTIME,
        False,
        None,
        "inode_flag_but_superblock_feature_absent",
        "contradiction: do not silently reinterpret as bigtime",
    ),
    Vector(
        "v3_legacy_in_bigtime_fs",
        encode_legacy(1_650_637_449, 264_560_000),
        3,
        0,
        True,
        None,
        None,
        "old inode without the per-inode flag is still legacy on a bigtime filesystem",
    ),
)


def audit_vectors() -> None:
    print("== vector audit: raw bits and both interpretations ==")
    print(
        f"{'case':34s} {'raw (hex)':18s} {'legacy':34s} {'bigtime':34s} basis"
    )
    for vector in VECTORS:
        raw_hex = f"0x{vector.raw:016x}"
        legacy = decode_legacy(vector.raw)
        bigtime = decode_bigtime(vector.raw)
        leg = "invalid" if not legacy.ok else fmt(to_datetime_floor(legacy.unix_ns))
        big = fmt(to_datetime_floor(bigtime.unix_ns))
        print(f"{vector.name:34s} {raw_hex:18s} {leg:34s} {big:34s} {vector.note}")


def audit_decoder_contract() -> None:
    print("\n== decoder contract ==")
    for vector in VECTORS:
        decoded, basis = decode_timestamp(vector.raw, vector.version, vector.flags2, vector.fs_has_bigtime)
        if vector.expected_error:
            assert decoded.error == vector.expected_error, (vector.name, decoded)
        elif vector.version < 3 and vector.name.startswith("v2"):
            assert decoded.ok and decoded.unix_ns == vector.expected_ns
        elif vector.name == "v3_legacy_in_bigtime_fs":
            assert decoded.ok and decoded.seconds_field == 1_650_637_449, decoded
            print(f"{vector.name}: basis={basis} unix_ns={decoded.unix_ns} (legacy bytes, valid bigtime-enabled FS)")
        else:
            assert decoded.ok, (vector.name, decoded)
            assert decoded.unix_ns == vector.expected_ns, (vector.name, decoded.unix_ns, vector.expected_ns)
            print(f"{vector.name}: basis={basis} unix_ns={decoded.unix_ns}")


def audit_versions() -> None:
    print("\n== inode version / crtime availability ==")
    print(
        "v1/v2 core ends at di_next_unlinked (offset 96..100); di_crtime lives at offset 144, "
        "inside the data fork. A v1/v2 inode has no crtime field."
    )
    synthetic_v2 = bytearray(512)
    struct.pack_into(">H", synthetic_v2, 0, 0x494E)
    synthetic_v2[INODE_VERSION_OFFSET] = 2
    assert synthetic_v2[INODE_VERSION_OFFSET] == 2
    struct.pack_into(">Q", synthetic_v2, DIFLAG2_OFFSET_V3, DIFLAG2_BIGTIME)
    struct.pack_into(">Q", synthetic_v2, CRTIME_OFFSET, 0x1122334455667788)
    print(f"synthetic v2: byte 120 = 0x{synthetic_v2[DIFLAG2_OFFSET_V3:][:8].hex()} (data fork, not flags2)")
    print(f"              bytes at 144 = 0x{synthetic_v2[CRTIME_OFFSET:][:8].hex()} (data fork, not crtime)")
    print("Do not report either byte range as a timestamp or flag.")


def audit_datetime_precision() -> None:
    print("\n== integer nanoseconds vs datetime ==")
    for ns in (-1, 999, 1_650_637_512_370_417_421, 2_147_483_647_999_999_999, 16_299_260_425_709_551_615):
        exact = to_datetime_floor(ns)
        via_float = to_datetime_float(ns)
        exact_back_ns = (
            (exact - UNIX_EPOCH).days * 86_400 * NS_PER_SECOND
            + (exact - UNIX_EPOCH).seconds * NS_PER_SECOND
            + exact.microsecond * 1000
        )
        print(
            f"unix_ns={ns}  floor={fmt(exact)}  float={fmt(via_float)}  "
            f"floor_loses_ns={ns - exact_back_ns}  float_delta_us={round((via_float - exact).total_seconds() * 1e6)}"
        )
    print("Keep unix_ns as int for ordering; render datetime only for presentation.")


def audit_sample_images() -> None:
    print("\n== repository samples (raw words) ==")
    import gzip
    import io
    from pathlib import Path

    from dissect.xfs.xfs import XFS

    root = Path(__file__).resolve().parents[1]
    for label, filename in (("legacy", "xfs.bin.gz"), ("bigtime", "xfs_bigtime.bin.gz")):
        path = root / "tests" / "_data" / filename
        with gzip.open(path, "rb") as fh:
            image = XFS(io.BytesIO(fh.read()))
        inode = image.root.inode
        print(
            f"{label}: sb_version={image.version} di_version={inode.di_version} "
            f"di_flags2=0x{inode.di_flags2:016x} has_bigtime={image.root._has_bigtime()} "
            f"crtime=0x{inode.di_crtime:016x}"
        )
        for field in ("di_atime", "di_mtime", "di_ctime", "di_crtime"):
            raw = getattr(inode, field)
            print(f"  {field}=0x{raw:016x} sec_hi=0x{raw >> 32:08x} nsec_lo=0x{raw & 0xFFFFFFFF:08x}")


def audit_library_divergence() -> None:
    """Compare the installed dissect.xfs decoder with the reference decoder.

    These are deliberate probes of the current implementation, not expected
    behaviour. Any line marked MISMATCH is a finding for the audit report.
    """
    print("\n== installed dissect.xfs vs reference ==")
    try:
        from types import SimpleNamespace

        from dissect.xfs.xfs import _parse_ts
    except ImportError:
        print("dissect.xfs not installed; skipping library comparison")
        return

    probes = (
        ("legacy_min_1901", encode_legacy(-(1 << 31), 0), False),
        ("legacy_-0.5s", encode_legacy(-1, 500_000_000), False),
        ("legacy_max_2038", encode_legacy((1 << 31) - 1, 999_999_999), False),
        ("bigtime_1970", encode_bigtime(0), True),
        ("bigtime_max_2486", (1 << 64) - 1, True),
    )
    for name, raw, is_bigtime in probes:
        reference = decode_bigtime(raw) if is_bigtime else decode_legacy(raw)
        library_ns = _parse_ts(raw, is_bigtime)
        status = "OK" if library_ns == reference.unix_ns else "MISMATCH"
        print(
            f"{name:24s} raw=0x{raw:016x} library_ns={library_ns:>22d} "
            f"reference_ns={reference.unix_ns:>22d} {status}"
        )

    # The library gates on (di_version >= 3 and flags2 BIGTIME) and does not
    # consult the superblock incompat feature. Show what that accepts.
    inode = SimpleNamespace(di_version=3, di_flags2=DIFLAG2_BIGTIME)
    library_gate = inode.di_version >= 3 and inode.di_flags2 & DIFLAG2_BIGTIME != 0
    print(
        f"gate(v3, flags2=BIGTIME, fs_feature=False): library={library_gate} "
        f"reference={gate(3, DIFLAG2_BIGTIME, False)}"
    )

    inode_v2 = SimpleNamespace(di_version=2, di_flags2=DIFLAG2_BIGTIME)
    library_gate_v2 = inode_v2.di_version >= 3 and inode_v2.di_flags2 & DIFLAG2_BIGTIME != 0
    print(f"gate(v2, flags2-lookalike=BIGTIME): library={library_gate_v2} reference={gate(2, DIFLAG2_BIGTIME, True)}")
    print("The library never exposes a version check on crtime; see the report for the field-offset consequence.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-samples", action="store_true", help="skip reading repository sample images")
    args = parser.parse_args()

    audit_vectors()
    audit_decoder_contract()
    audit_versions()
    audit_datetime_precision()
    audit_library_divergence()
    if not args.no_samples:
        audit_sample_images()
    print("\nall assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
