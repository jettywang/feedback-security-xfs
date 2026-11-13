from __future__ import annotations

import datetime
import io
import stat
import struct
from typing import TYPE_CHECKING, BinaryIO

from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.xfs import XFS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def test_xfs(xfs_bin: BinaryIO) -> None:
    xfs = XFS(xfs_bin)

    assert xfs.version == 5
    assert xfs.block_size == 4096
    assert str(xfs.uuid) == "3fb8342e-e144-4f0c-8bd7-725e78966200"
    assert str(xfs.meta_uuid) == "00000000-0000-0000-0000-000000000000"

    root = xfs.root
    assert root.inum == 11072
    assert root.filetype == stat.S_IFDIR
    assert root.crtime == datetime.datetime(2022, 4, 22, 14, 24, 9, 264560, tzinfo=datetime.timezone.utc)
    assert root.crtime_ns == 1650637449264560000
    assert list(root.listdir().keys()) == [".", "..", "test_file", "test_dir", "test_link"]

    test_file = xfs.get("test_file")
    assert test_file.nblocks == 1
    assert test_file.open().read() == b"test content\n"

    test_link = xfs.get("test_link")
    assert test_link.nblocks == 0
    assert test_link.filetype == stat.S_IFLNK
    assert test_link.link == "test_dir/test_file"


def test_xfs_sparse(xfs_sparse_bin: BinaryIO) -> None:
    xfs = XFS(xfs_sparse_bin)

    sparse_start = xfs.get("sparse_start")
    assert sparse_start.size == 0x258000
    assert sparse_start.nblocks == 200
    assert sparse_start.dataruns == [(None, 400), (1392, 200)]

    sparse_hole = xfs.get("sparse_hole")
    assert sparse_hole.size == 0x258000
    assert sparse_hole.nblocks == 400
    assert sparse_hole.dataruns == [(1792, 200), (None, 200), (2192, 200)]

    sparse_end = xfs.get("sparse_end")
    assert sparse_end.size == 0x190000
    assert sparse_end.nblocks == 200
    assert sparse_end.dataruns == [(2392, 200), (None, 200)]

    sparse_all = xfs.get("sparse_all")
    assert sparse_all.size == 0x500000
    assert sparse_all.nblocks == 0
    assert sparse_all.dataruns == [(None, 1280)]


def test_xfs_bigtime(xfs_bigtime_bin: BinaryIO) -> None:
    xfs = XFS(xfs_bigtime_bin)

    assert xfs.version == 5

    test_file = xfs.get("file")
    assert test_file._has_bigtime()
    assert test_file.crtime == datetime.datetime(2023, 4, 7, 9, 15, 9, 223364, tzinfo=datetime.timezone.utc)
    assert test_file.crtime_ns == 1680858909223364005


def test_symlinks(xfs_symlink_bin: BinaryIO) -> None:
    path = "/path/to/dir/with/file.ext"

    xfs = XFS(xfs_symlink_bin)
    assert xfs.get(path).link == "../../../../other/path/source/to/my/file.ext"


def _walk_directory(xfs: XFS) -> Iterator[tuple[str, int, str, bytes | None]]:
    """Enumerate a tree as evidence tuples: (path, inode, kind, extracted bytes).

    Only the public API is used; each item is produced exactly where a generator
    consumer would pause, so tests can inject unrelated reads at those points.
    """
    seen = set()
    stack = [(xfs.root, "/")]
    while stack:
        node, path = stack.pop()
        if node.inum in seen:
            continue
        seen.add(node.inum)
        yield (path, node.inum, "dir", None)

        children = list(node.iterdir())
        for child in children:
            child_path = path.rstrip("/") + "/" + child.filename
            if child.filetype == stat.S_IFDIR:
                stack.append((child, child_path))
            elif child.filetype == stat.S_IFLNK:
                yield (child_path, child.inum, "link", child.link.encode())
            elif child.filetype == stat.S_IFREG:
                yield (child_path, child.inum, "file", child.open().read())


def _walk_inodes(xfs: XFS) -> Iterator[tuple[int, int, str, bytes | None]]:
    """Scan every allocated inode through the AG inode btree.

    Mirrors the second scanner: inode number, dataruns and extracted file bytes.
    """
    for agnum in range(xfs.sb.sb_agcount):
        ag = xfs.get_allocation_group(agnum)
        for record in ag.walk_agi():
            for offset in range(64):
                if record.ir_free >> offset & 1:
                    continue

                node = ag.get_inode(record.ir_startino + offset)
                yield (node.inum, node.filetype, "meta", None)

                if node.filetype == stat.S_IFREG:
                    yield (node.inum, node.filetype, "file", node.open().read())
                elif node.filetype == stat.S_IFLNK:
                    yield (node.inum, node.filetype, "link", node.link.encode())


def _interleave(scanners: list[Iterator], on_pause: Callable[[], None] | None = None) -> list[list]:
    """Round-robin several generators, running `on_pause` between every yield."""
    results = [[] for _ in scanners]
    active = [True] * len(scanners)

    while any(active):
        for idx, scanner in enumerate(scanners):
            if not active[idx]:
                continue

            try:
                results[idx].append(next(scanner))
            except StopIteration:
                active[idx] = False

            if on_pause:
                on_pause()

    return results


def test_interleaved_scanners_match_independent_scans(xfs_bin: BinaryIO, xfs_symlink_long_bin: BinaryIO) -> None:
    for fh in (xfs_bin, xfs_symlink_long_bin):
        # Each scanner gets its own independent XFS instance for the baseline.
        directory_evidence = list(_walk_directory(XFS(fh)))
        inode_evidence = list(_walk_inodes(XFS(fh)))

        # A single shared image and parser instance for the interleaved run.
        fh.seek(0)
        shared = XFS(fh)
        interleaved_directory, interleaved_inodes = _interleave([_walk_directory(shared), _walk_inodes(shared)])

        assert interleaved_directory == directory_evidence
        assert interleaved_inodes == inode_evidence

        # Extracted bytes must be identical, not merely the counts.
        assert {item[0]: item[3] for item in interleaved_directory if item[2] == "file"} == {
            item[0]: item[3] for item in directory_evidence if item[2] == "file"
        }


def test_interleaved_scanners_survive_full_nested_scan(xfs_bin: BinaryIO, xfs_sparse_bin: BinaryIO) -> None:
    """Insert a complete independent scan at every pause point of both generators."""
    for fh in (xfs_bin, xfs_sparse_bin):
        directory_evidence = list(_walk_directory(XFS(fh)))
        inode_evidence = list(_walk_inodes(XFS(fh)))

        fh.seek(0)
        shared = XFS(fh)

        def nested_scan() -> None:
            assert list(_walk_directory(shared)) == directory_evidence
            assert list(_walk_inodes(shared)) == inode_evidence

        directory, inodes = _interleave([_walk_directory(shared), _walk_inodes(shared)], on_pause=nested_scan)

        assert directory == directory_evidence
        assert inodes == inode_evidence


def test_closing_one_interleaved_iterator_does_not_stop_the_other(xfs_bin: BinaryIO) -> None:
    fh = xfs_bin
    expected_directory = list(_walk_directory(XFS(fh)))
    expected_inodes = list(_walk_inodes(XFS(fh)))

    fh.seek(0)
    xfs = XFS(fh)
    directory = _walk_directory(xfs)
    inodes = _walk_inodes(xfs)

    # Advance both, then abandon the directory scan after its second entry.
    first = [next(directory), next(directory)]
    inodes_first = next(inodes)

    directory.close()
    assert first == expected_directory[:2]

    assert inodes_first == expected_inodes[0]
    assert list(inodes) == expected_inodes[1:]


def test_closing_a_file_stream_does_not_affect_directory_iteration(xfs_bin: BinaryIO) -> None:
    xfs = XFS(xfs_bin)
    expected = [(entry.filename, entry.inum) for entry in xfs.root.iterdir()]

    stream = xfs.get("test_file").open()
    assert stream.read(4) == b"test"

    directory = xfs.root.iterdir()
    collected = [(entry.filename, entry.inum) for entry in directory]

    stream.close()
    assert collected == expected
    assert [(entry.filename, entry.inum) for entry in xfs.root.iterdir()] == expected


def test_shared_handle_position_is_not_used_as_state(xfs_bin: BinaryIO) -> None:
    """Raw reads through the same handle must not disturb an open iterator."""
    xfs = XFS(xfs_bin)
    expected = list(xfs.root.iterdir())

    xfs_bin.seek(0)
    iterator = xfs.root.iterdir()
    collected = [next(iterator), next(iterator)]

    # Another scanner reads unrelated blocks on the shared handle, including the
    # next inode chunk and a directory data block.
    xfs_bin.seek(0)
    xfs_bin.read(4096)
    xfs_bin.seek(11072 * 512)
    xfs_bin.read(512)

    collected.extend(iterator)

    assert [(entry.filename, entry.inum, entry.filetype) for entry in collected] == [
        (entry.filename, entry.inum, entry.filetype) for entry in expected
    ]


def test_two_level_inode_btree_interleaves_with_directory_scan() -> None:
    """B+tree traversal must not rely on the shared cursor between records."""
    xfs = XFS(_btree_image())
    expected_inodes = [(record.ir_startino, record.ir_free) for record in xfs.get_allocation_group(0).walk_agi()]
    expected_entries = [(entry.filename, entry.inum) for entry in xfs.root.iterdir()]

    xfs = XFS(_btree_image())

    def scan_tree() -> Iterator[tuple[int, int]]:
        for record in xfs.get_allocation_group(0).walk_agi():
            yield (record.ir_startino, record.ir_free)

    def scan_dir() -> Iterator[tuple[str, int]]:
        for entry in xfs.root.iterdir():
            yield (entry.filename, entry.inum)

    scanner_a, scanner_b = scan_tree(), scan_dir()
    tree_results, dir_results = [], []
    active = [True, True]

    while any(active):
        for idx, scanner in enumerate((scanner_a, scanner_b)):
            if not active[idx]:
                continue
            try:
                item = next(scanner)
            except StopIteration:
                active[idx] = False
                continue

            # Every pause point performs a full competing walk on the same parser.
            list(scan_tree())
            list(scan_dir())
            (tree_results if idx == 0 else dir_results).append(item)

    assert tree_results == expected_inodes
    assert dir_results == expected_entries


def _btree_image() -> BinaryIO:
    """Build a minimal XFS image whose AGI inode tree has a real root node.

    The shipped fixtures only contain a single-leaf inode tree, so they cannot
    catch a traversal that resumes from the shared handle's cursor. This image
    has a level-1 root at block 3 pointing at two level-0 leaves (blocks 4-5)
    that describe 128 inodes. The root's key area is filled with a decoy so a
    misread key/pointer split yields bogus records instead of passing silently.
    """
    block_size = 4096
    ag_blocks = 32
    image = bytearray(block_size * ag_blocks)

    sb = c_xfs.xfs_sb(bytes(512))
    sb.sb_magicnum = c_xfs.XFS_SB_MAGIC
    sb.sb_blocksize = block_size
    sb.sb_dblocks = ag_blocks
    sb.sb_agblocks = ag_blocks
    sb.sb_agcount = 1
    sb.sb_versionnum = 5
    sb.sb_sectsize = 512
    sb.sb_inodesize = 512
    sb.sb_inopblock = 8
    sb.sb_blocklog = 12
    sb.sb_sectlog = 9
    sb.sb_inodelog = 9
    sb.sb_inopblog = 3
    sb.sb_agblklog = 5
    sb.sb_rootino = 1
    sb.sb_features2 = c_xfs.XFS_SB_VERSION2_FTYPE
    sb.sb_features_incompat = c_xfs.XFS_SB_FEAT_INCOMPAT_FTYPE
    image[0 : len(sb.dumps())] = sb.dumps()

    agi = c_xfs.xfs_agi(bytes(512))
    agi.agi_magicnum = c_xfs.XFS_AGI_MAGIC
    agi.agi_versionnum = c_xfs.XFS_AGI_VERSION
    agi.agi_length = ag_blocks
    agi.agi_count = 128
    agi.agi_root = 3
    agi.agi_level = 1
    image[1024 : 1024 + len(agi.dumps())] = agi.dumps()

    def write_sblock(block: int, level: int, numrecs: int) -> None:
        header = bytearray(56)
        struct.pack_into(">IHH", header, 0, c_xfs.XFS_IBT_CRC_MAGIC, level, numrecs)
        struct.pack_into(">II", header, 8, 0xFFFFFFFF, 0xFFFFFFFF)
        image[block * block_size : block * block_size + 56] = header

    maxrecs = (block_size - 56) // 8
    write_sblock(3, 1, 2)
    image[3 * block_size + 56 : 3 * block_size + 64] = b"\xde\xad\xbe\xef" * 2
    struct.pack_into(">II", image, 3 * block_size + 56 + maxrecs * 4, 4, 5)

    # `ir_free` bit 0 is the first inode in the record, and a record covers 64
    # inodes. Only inodes 1 (root), 48 and 49 (files) are allocated.
    allocated = {1, 48, 49}
    for block, start in ((4, 0), (5, 64)):
        write_sblock(block, 0, 1)
        record = c_xfs.xfs_inobt_rec(bytes(16))
        record.ir_startino = start
        free = 0
        for bit in range(64):
            if start + bit not in allocated:
                free |= 1 << bit
        record.ir_free = free
        record.ir_freecount = bin(free).count("1")
        image[block * block_size + 56 : block * block_size + 72] = record.dumps()

    # Root directory is shortform, holding two regular files.
    names = [(48, b"one"), (49, b"two")]
    shortform = bytearray(6)
    shortform[0] = len(names)
    struct.pack_into(">I", shortform, 2, 1)
    for ino, name in names:
        shortform.append(len(name))
        shortform.extend(b"\x00\x00")
        shortform.extend(name)
        shortform.append(1)  # FILETYPES[1] == S_IFREG
        shortform.extend(struct.pack(">I", ino))

    def write_inode(inum: int, data: bytes, *, mode: int, fmt: int) -> None:
        raw = bytearray(512)
        inode = c_xfs.xfs_dinode(bytes(512))
        inode.di_magic = c_xfs.XFS_DINODE_MAGIC
        inode.di_version = 3
        inode.di_mode = mode
        if fmt == c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL:
            inode.di_format = fmt
            inode.di_size = len(data)
            raw[: len(inode.dumps())] = inode.dumps()
            raw[176 : 176 + len(data)] = data
        else:
            block = 20 + inum - 48
            image[block * block_size : block * block_size + len(data)] = data
            inode.di_format = fmt
            inode.di_size = len(data)
            inode.di_nblocks = 1
            inode.di_big_anextents = 1
            raw[: len(inode.dumps())] = inode.dumps()
            # fsb extent: 1 bit flag | 54 bit offset | 52 bit block | 21 bit count
            l0 = (block >> 43) & 0x1FF
            l1 = ((block & ((1 << 43) - 1)) << 21) | 1
            struct.pack_into(">QQ", raw, 176, l0, l1)
        image[inum * 512 : inum * 512 + 512] = raw

    write_inode(1, bytes(shortform), mode=stat.S_IFDIR | 0o755, fmt=c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_LOCAL)
    write_inode(48, b"payload-one\n", mode=stat.S_IFREG | 0o644, fmt=c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS)
    write_inode(49, b"payload-two\n", mode=stat.S_IFREG | 0o644, fmt=c_xfs.xfs_dinode_fmt.XFS_DINODE_FMT_EXTENTS)

    return io.BytesIO(bytes(image))


class _InterleavingHandle:
    """Image wrapper that imitates another scanner using the same handle.

    After every read it moves the shared cursor somewhere else. A parser that
    assumes its seek is still valid when it reads gets different bytes; one that
    reads at absolute offsets is unaffected.
    """

    def __init__(self, fh: BinaryIO) -> None:
        self._fh = fh

    def seek(self, *args) -> int:
        return self._fh.seek(*args)

    def tell(self) -> int:
        return self._fh.tell()

    def read(self, size: int = -1) -> bytes:
        data = self._fh.read(size)
        # Another scan reads from the start of the image between our reads.
        self._fh.seek(0)
        return data

    def __getattr__(self, name: str):
        return getattr(self._fh, name)


def _evidence(xfs: XFS) -> tuple[list, list, dict[str, bytes]]:
    directories = [(node.filename, node.inum) for node in xfs.root.iterdir()]
    inodes = [record.ir_startino for record in xfs.get_allocation_group(0).walk_agi()]
    files = {
        node.filename: node.open().read()
        for node in xfs.root.iterdir()
        if node.filetype == stat.S_IFREG
    }
    return directories, inodes, files


def test_reads_are_absolute_not_cursor_relative(xfs_bin: BinaryIO, xfs_sparse_bin: BinaryIO) -> None:
    for fh in (xfs_bin, xfs_sparse_bin):
        expected = _evidence(XFS(fh))

        fh.seek(0)
        actual = _evidence(XFS(_InterleavingHandle(fh)))

        assert actual == expected


def test_metadata_btree_is_not_read_through_a_shared_cursor() -> None:
    expected = _evidence(XFS(_btree_image()))
    actual = _evidence(XFS(_InterleavingHandle(_btree_image())))
    assert actual == expected
