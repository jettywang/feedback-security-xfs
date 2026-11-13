"""Synthetic XFS images for exercising the extended attribute storage forms.

The shipped images only contain shortform (inline) attributes. This helper
copies a real image and rewrites a single inode plus a few unused filesystem
blocks so the same attribute can be parsed from every supported storage
location: shortform, leaf (inline value), leaf (remote value) and the node
B+tree. It also produces deliberately invalid attributes used to verify that
incomplete updates and truncated values are never exported as labels.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.xfs import XFS

DATA = Path(__file__).parent / "_data" / "xfs.bin.gz"

SELINUX_NAME = b"selinux"
SELINUX_VALUE = b"unconfined_u:object_r:unlabeled_t:s0\x00"
BIG_VALUE = bytes((offset % 251) + 1 for offset in range(9000))

# Byte offsets inside struct xfs_dinode. For non-NREXT inodes the attr extent
# count is di_anextents, the high 16 bits of the packed di_big_anextents union.
ANEXTENTS_OFF = 80
FORKOFF_OFF = 82
AFORMAT_OFF = 83
FORK_DATA_OFF = 0xB0
FMT_EXTENTS = c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS
FMT_LOCAL = c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL


def base_image() -> tuple[bytearray, XFS]:
    with gzip.open(DATA, "rb") as fh:
        image = bytearray(fh.read())
    return image, XFS(io.BytesIO(image))


def inode_start(xfs: XFS, inode: object) -> int:
    return inode.relative_inum * xfs.sb.sb_inodesize


def set_attr_fork(image: bytearray, xfs: XFS, inode: object, extents: list[int]) -> None:
    start = inode_start(xfs, inode)
    image[start + AFORMAT_OFF] = FMT_EXTENTS
    image[start + ANEXTENTS_OFF : start + ANEXTENTS_OFF + 2] = len(extents).to_bytes(2, "big")

    forkoff = image[start + FORKOFF_OFF]
    fork_abs = start + FORK_DATA_OFF + forkoff * 8
    records = b"".join(pack_extent(block, 1) for block in extents)
    image[fork_abs : fork_abs + len(records)] = records


def set_incomplete_shortform(image: bytearray, xfs: XFS, inode: object) -> None:
    start = inode_start(xfs, inode)
    forkoff = image[start + FORKOFF_OFF]
    fork_abs = start + FORK_DATA_OFF + forkoff * 8
    # Shortform: 4 byte header, then a 3 byte entry (namelen, valuelen, flags).
    image[fork_abs + 6] = c_xfs.XFS_ATTR_SECURE | c_xfs.XFS_ATTR_INCOMPLETE


def pack_extent(block: int, count: int) -> bytes:
    l0 = block >> 43
    l1 = ((block & ((1 << 43) - 1)) << 21) | (count & ((1 << 21) - 1))
    return b"".join(value.to_bytes(8, "big") for value in (l0, l1))


def blkinfo(magic: int, block: int, owner: int) -> bytes:
    return (
        magic.to_bytes(4, "big")
        + b"\x00" * 4  # crc
        + block.to_bytes(8, "big")  # blkno
        + b"\x00" * 8  # lsn
        + b"\x00" * 16  # uuid
        + owner.to_bytes(8, "big")  # owner
        + b"\x00" * 4  # pad
    )


def leaf_block(xfs: XFS, block: int, owner: int, entries: list[tuple[int, int, bytes]]) -> bytes:
    """Build a v3 attribute leaf block.

    Each entry is ``(flags, hashval, nameblob)`` where ``nameblob`` already
    contains the local or remote name/value header plus bytes.
    """
    buf = bytearray(xfs.block_size)
    info_size = len(c_xfs.xfs_da3_blkinfo)
    leaf_hdr_size = len(c_xfs.xfs_attr_leaf_hdr)
    entry_size = len(c_xfs.xfs_attr_leaf_entry)

    name_start = info_size + leaf_hdr_size + len(entries) * entry_size
    cursor = name_start
    leaf_entries = []
    used = 0

    for flags, hashval, blob in entries:
        leaf_entries.append((hashval, cursor, flags))
        buf[cursor : cursor + len(blob)] = blob
        cursor += len(blob)
        used += len(blob)

    buf[0:info_size] = blkinfo(c_xfs.XFS_ATTR3_LEAF_MAGIC, block, owner)

    # Leaf header layout: freemap[3] (12), count, usedbytes, firstused, holes, pad1, padx, pad2.
    hdr = bytearray(leaf_hdr_size)
    pos = 12
    hdr[pos : pos + 2] = len(entries).to_bytes(2, "big")
    pos += 2
    hdr[pos : pos + 2] = used.to_bytes(2, "big")
    pos += 2
    hdr[pos : pos + 2] = name_start.to_bytes(2, "big")
    buf[info_size : info_size + leaf_hdr_size] = hdr

    for idx, (hashval, nameidx, flags) in enumerate(leaf_entries):
        off = info_size + leaf_hdr_size + idx * entry_size
        buf[off : off + 4] = hashval.to_bytes(4, "big")
        buf[off + 4 : off + 6] = nameidx.to_bytes(2, "big")
        buf[off + 6] = flags
        buf[off + 7] = 0

    return bytes(buf)


def node_block(xfs: XFS, block: int, owner: int, children: list[tuple[int, int]]) -> bytes:
    """Build a v3 node block referencing ``(hashval, child_agblock)`` leaves."""
    buf = bytearray(xfs.block_size)
    info_size = len(c_xfs.xfs_da3_blkinfo)
    buf[0:info_size] = blkinfo(c_xfs.XFS_DA3_NODE_MAGIC, block, owner)

    hdr = len(children).to_bytes(2, "big") + b"\x00\x01"
    buf[info_size : info_size + len(c_xfs.xfs_da_node_hdr)] = hdr

    for idx, (hashval, child) in enumerate(children):
        off = info_size + len(c_xfs.xfs_da_node_hdr) + idx * len(c_xfs.xfs_da_node_entry)
        buf[off : off + 4] = hashval.to_bytes(4, "big")
        buf[off + 4 : off + 8] = child.to_bytes(4, "big")

    return bytes(buf)


def remote_block(xfs: XFS, block: int, owner: int, offset: int, data: bytes) -> bytes:
    buf = bytearray(xfs.block_size)
    header = (
        c_xfs.XFS_ATTR3_RMT_MAGIC.to_bytes(4, "big")
        + offset.to_bytes(4, "big")
        + len(data).to_bytes(4, "big")  # rm_bytes
        + b"\x00" * 4  # crc
        + b"\x00" * 16  # uuid
        + owner.to_bytes(8, "big")
        + block.to_bytes(8, "big")
        + b"\x00" * 8
    )
    start = len(c_xfs.xfs_attr3_rmt_hdr)
    buf[0 : len(header)] = header
    buf[start : start + len(data)] = data
    return bytes(buf)


def local_blob(name: bytes, value: bytes, flags: int) -> tuple[int, int, bytes]:
    header = len(value).to_bytes(2, "big") + len(name).to_bytes(1, "big")
    return flags, 0, header + name + value


def remote_blob(name: bytes, value_length: int, value_block: int, flags: int) -> tuple[int, int, bytes]:
    header = value_block.to_bytes(4, "big") + value_length.to_bytes(2, "big") + len(name).to_bytes(1, "big")
    return flags, 0, header + name


def install_block(image: bytearray, xfs: XFS, block: int, data: bytes) -> None:
    start = block * xfs.block_size
    image[start : start + len(data)] = data


def make_leaf_local() -> XFS:
    image, xfs = base_image()
    inode = xfs.get("test_file")
    leaf_agblock = 3000

    blob = local_blob(SELINUX_NAME, SELINUX_VALUE, c_xfs.XFS_ATTR_LOCAL | c_xfs.XFS_ATTR_SECURE)
    install_block(image, xfs, leaf_agblock, leaf_block(xfs, leaf_agblock, inode.inum, [blob]))
    set_attr_fork(image, xfs, inode, [leaf_agblock])
    return XFS(io.BytesIO(image))


def make_leaf_remote(truncated: bool = False) -> XFS:
    image, xfs = base_image()
    inode = xfs.get("test_file")

    leaf_agblock = 3100
    remote_agblock = 3200
    value = BIG_VALUE
    value_length = len(value)
    blocks = (value_length + xfs.block_size - 1) // xfs.block_size
    payload_capacity = xfs.block_size - len(c_xfs.xfs_attr3_rmt_hdr)

    for idx in range(blocks):
        start = idx * payload_capacity
        chunk = value[start : start + payload_capacity]
        install_block(
            image,
            xfs,
            remote_agblock + idx,
            remote_block(xfs, remote_agblock + idx, inode.inum, idx * xfs.block_size, chunk),
        )

    if truncated:
        last = remote_agblock + blocks - 1
        corrupt = bytearray(remote_block(xfs, last, inode.inum, (blocks - 1) * xfs.block_size, b""))
        corrupt[8:12] = (0).to_bytes(4, "big")  # rm_bytes = 0
        install_block(image, xfs, last, bytes(corrupt))

    blob = remote_blob(SELINUX_NAME, value_length, remote_agblock, c_xfs.XFS_ATTR_SECURE)
    install_block(image, xfs, leaf_agblock, leaf_block(xfs, leaf_agblock, inode.inum, [blob]))
    set_attr_fork(image, xfs, inode, [leaf_agblock])
    return XFS(io.BytesIO(image))


def make_node_remote() -> XFS:
    image, xfs = base_image()
    inode = xfs.get("test_file")

    node_agblock = 3300
    leaf_agblock = 3301
    remote_agblock = 3400

    install_block(image, xfs, remote_agblock, remote_block(xfs, remote_agblock, inode.inum, 0, SELINUX_VALUE))
    blob = remote_blob(SELINUX_NAME, len(SELINUX_VALUE), remote_agblock, c_xfs.XFS_ATTR_SECURE)
    install_block(image, xfs, leaf_agblock, leaf_block(xfs, leaf_agblock, inode.inum, [blob]))
    install_block(image, xfs, node_agblock, node_block(xfs, node_agblock, inode.inum, [(0x12345678, leaf_agblock)]))
    set_attr_fork(image, xfs, inode, [node_agblock])
    return XFS(io.BytesIO(image))


def make_incomplete_shortform() -> XFS:
    image, xfs = base_image()
    inode = xfs.get("test_file")
    set_incomplete_shortform(image, xfs, inode)
    return XFS(io.BytesIO(image))


def make_incomplete_leaf() -> XFS:
    image, xfs = base_image()
    inode = xfs.get("test_file")
    leaf_agblock = 3500

    flags = c_xfs.XFS_ATTR_LOCAL | c_xfs.XFS_ATTR_SECURE | c_xfs.XFS_ATTR_INCOMPLETE
    blob = local_blob(SELINUX_NAME, SELINUX_VALUE, flags)
    install_block(image, xfs, leaf_agblock, leaf_block(xfs, leaf_agblock, inode.inum, [blob]))
    set_attr_fork(image, xfs, inode, [leaf_agblock])
    return XFS(io.BytesIO(image))
