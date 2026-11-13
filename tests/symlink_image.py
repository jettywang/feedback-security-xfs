from __future__ import annotations

import io
import stat

from dissect.xfs.c_xfs import c_xfs

DEFAULT_BLOCK_SIZE = 512
SYMLINK_HEADER_SIZE = len(c_xfs.xfs_dsymlink_hdr)


def build_xfs_symlink_image(
    fragments: list[tuple[int, bytes]],
    owner: int,
    size: int | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    fs_blocks: int = 128,
    ag_blocks: int = 128,
    symlink_inode: int = 44,
    physical_order: list[int] | None = None,
    owners: list[int] | None = None,
    lengths: list[int] | None = None,
    uuid: bytes = bytes(range(16)),
) -> bytes:
    """Build a minimal XFS v5 image containing a remote symlink with one block per fragment.

    Args:
        fragments: ``(logical byte offset, payload)`` tuples, one per on-disk ``xfs_dsymlink_hdr``.
        owner: Default ``sl_owner`` (inode number) written to every fragment header.
        size: Symlink size stored in the inode; defaults to the furthest fragment end.
        block_size: Filesystem block size.
        fs_blocks: Number of blocks in the image.
        ag_blocks: Number of blocks per allocation group.
        symlink_inode: Inode number of the symlink.
        physical_order: Optional permutation for placing fragments in physical blocks out of logical order.
        owners: Optional per-fragment ``sl_owner`` overrides, for foreign-fragment samples.
        lengths: Optional per-fragment ``sl_bytes`` overrides, for length-conflict samples.
        uuid: Filesystem UUID and default metadata UUID used in fragment headers.
    """
    payload_size = block_size - SYMLINK_HEADER_SIZE
    if size is None:
        size = max((offset + len(data) for offset, data in fragments), default=0)
    if physical_order is None:
        physical_order = list(range(len(fragments)))
    if owners is None:
        owners = [owner] * len(fragments)
    if lengths is None:
        lengths = [len(data) for _, data in fragments]

    image = bytearray(fs_blocks * block_size)

    def write_block(index: int, data: bytes) -> None:
        image[index * block_size : index * block_size + len(data)] = data

    sb = c_xfs.xfs_sb()
    sb.sb_magicnum = c_xfs.XFS_SB_MAGIC
    sb.sb_blocksize = block_size
    sb.sb_dblocks = fs_blocks
    sb.sb_uuid = uuid
    sb.sb_logstart = 2
    sb.sb_rootino = 4
    sb.sb_agblocks = ag_blocks
    sb.sb_agcount = fs_blocks // ag_blocks
    sb.sb_versionnum = 0xB4B5
    sb.sb_sectsize = 512
    sb.sb_inodesize = 512
    sb.sb_inopblock = block_size // 512
    sb.sb_blocklog = block_size.bit_length() - 1
    sb.sb_sectlog = 9
    sb.sb_inodelog = 9
    sb.sb_inopblog = sb.sb_inopblock.bit_length() - 1
    sb.sb_agblklog = ag_blocks.bit_length() - 1
    sb.sb_features2 = 0x18A
    sb.sb_features_incompat = c_xfs.XFS_SB_FEAT_INCOMPAT_FTYPE
    sb.sb_inoalignmt = 1
    sb.sb_icount = 8
    sb_buf = io.BytesIO()
    c_xfs.xfs_sb.write(sb_buf, sb)
    write_block(0, sb_buf.getvalue())

    agi = c_xfs.xfs_agi()
    agi.agi_magicnum = c_xfs.XFS_AGI_MAGIC
    agi.agi_versionnum = 1
    agi.agi_length = fs_blocks
    agi.agi_count = 8
    agi.agi_root = 2
    agi.agi_level = 1
    agi_buf = io.BytesIO()
    c_xfs.xfs_agi.write(agi_buf, agi)
    write_block(2, agi_buf.getvalue())

    def write_inode(inum: int, data: bytes) -> None:
        byte_offset = inum * sb.sb_inodesize
        write_block(byte_offset // block_size, b"\x00" * (byte_offset % block_size) + data)

    physical_blocks = []
    for index, (logical_offset, data) in enumerate(fragments):
        if len(data) > payload_size:
            raise ValueError("fragment payload does not fit in a single block")
        physical_blocks.append((logical_offset, 8 + physical_order[index], data))

    records = []
    for extent_index, (_, block, _) in enumerate(sorted(physical_blocks, key=lambda item: item[0])):
        l0 = (extent_index << 9) | (block >> 43)
        l1 = ((block & ((1 << 43) - 1)) << 21) | 1
        records.append(l0.to_bytes(8, "big") + l1.to_bytes(8, "big"))

    dinode = c_xfs.xfs_dinode()
    dinode.di_magic = c_xfs.XFS_DINODE_MAGIC
    dinode.di_mode = stat.S_IFLNK | 0o777
    dinode.di_version = 3
    dinode.di_format = c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS
    dinode.di_size = size
    dinode.di_nblocks = len(fragments)
    dinode.di_big_anextents = len(fragments)
    dinode.di_ino = symlink_inode
    dinode_buf = io.BytesIO()
    c_xfs.xfs_dinode.write(dinode_buf, dinode)
    dinode_buf.seek(0xB0)
    dinode_buf.write(b"".join(records))
    write_inode(symlink_inode, dinode_buf.getvalue())

    for index, (logical_offset, block, data) in enumerate(physical_blocks):
        header = c_xfs.xfs_dsymlink_hdr()
        header.sl_magic = c_xfs.XFS_SYMLINK_MAGIC
        header.sl_offset = logical_offset
        header.sl_bytes = lengths[index]
        header.sl_uuid = uuid
        header.sl_owner = owners[index]
        header.sl_blkno = block * (block_size // 512)
        header_buf = io.BytesIO()
        c_xfs.xfs_dsymlink_hdr.write(header_buf, header)
        write_block(block, header_buf.getvalue() + data)

    return bytes(image)
