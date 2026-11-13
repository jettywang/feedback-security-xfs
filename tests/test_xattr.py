from __future__ import annotations

import gzip
import io
from collections.abc import Callable

import pytest

from dissect.xfs.c_xfs import c_xfs
from dissect.xfs.xfs import XFS, XAttrSource
from tests import attr_fixtures as af


def test_shortform_xattr(xfs_bin: io.BytesIO) -> None:
    xfs = XFS(xfs_bin)
    inode = xfs.get("test_file")

    assert len(inode.xattr) == 1
    attr = inode.xattr[0]

    assert attr.namespace == "security"
    assert attr.full_name == "security.selinux"
    assert attr.name == b"selinux"
    assert attr.value == b"unconfined_u:object_r:unlabeled_t:s0\x00"
    assert attr.source == XAttrSource.INODE
    assert attr.flags & c_xfs.XFS_ATTR_SECURE
    assert inode.attributes == inode.xattr


def test_inode_without_xattr(xfs_bin: io.BytesIO) -> None:
    xfs = XFS(xfs_bin)
    assert xfs.root.xattr == []
    assert list(xfs.root.iter_xattr()) == []


@pytest.mark.parametrize(
    ("builder", "source", "value"),
    [
        (af.make_leaf_local, XAttrSource.LEAF, af.SELINUX_VALUE),
        (af.make_leaf_remote, XAttrSource.REMOTE, af.BIG_VALUE),
        (af.make_node_remote, XAttrSource.REMOTE, af.SELINUX_VALUE),
    ],
)
def test_block_backed_xattr(builder: Callable[[], XFS], source: XAttrSource, value: bytes) -> None:
    xfs = builder()
    attr = xfs.get("test_file").xattr[0]

    assert attr.source == source
    assert attr.namespace == "security"
    assert attr.name == af.SELINUX_NAME
    assert attr.full_name == "security.selinux"
    assert attr.value == value


def test_same_attribute_consistent_across_storage_forms() -> None:
    image, base = af.base_image()
    shortform = base.get("test_file").xattr[0]
    leaf_local = af.make_leaf_local().get("test_file").xattr[0]
    node_remote = af.make_node_remote().get("test_file").xattr[0]

    for attr in (shortform, leaf_local, node_remote):
        assert attr.namespace == "security"
        assert attr.name == af.SELINUX_NAME
        assert attr.full_name == "security.selinux"
        assert attr.value == af.SELINUX_VALUE


def test_remote_value_spans_multiple_blocks() -> None:
    xfs = af.make_leaf_remote()
    attr = xfs.get("test_file").xattr[0]

    assert attr.source == XAttrSource.REMOTE
    assert len(attr.value) == len(af.BIG_VALUE)
    assert attr.value == af.BIG_VALUE


def test_incomplete_shortform_attribute_is_skipped() -> None:
    xfs = af.make_incomplete_shortform()
    assert xfs.get("test_file").xattr == []


def test_incomplete_leaf_attribute_is_skipped() -> None:
    xfs = af.make_incomplete_leaf()
    assert xfs.get("test_file").xattr == []


def test_truncated_remote_value_is_skipped() -> None:
    xfs = af.make_leaf_remote(truncated=True)
    assert xfs.get("test_file").xattr == []


def test_attr_fork_does_not_read_data_fork(xfs_bin: io.BytesIO) -> None:
    xfs = XFS(xfs_bin)
    inode = xfs.get("test_file")
    _ = inode.inode  # populate the inode buffer
    raw = inode._buf.getvalue()
    fork_offset = 0xB0 + inode.inode.di_forkoff * 8

    # The data fork ends exactly where the attribute fork starts.
    assert 0xB0 + inode.datafork().size == fork_offset

    with gzip.open(af.DATA, "rb") as fh:
        image = bytearray(fh.read())

    inode_start = inode.relative_inum * xfs.sb.sb_inodesize

    # Poison every byte of the data fork with a marker; the label must survive.
    image[inode_start + 0xB0 : inode_start + fork_offset] = b"Z" * (fork_offset - 0xB0)
    poisoned = XFS(io.BytesIO(image)).get("test_file")

    assert poisoned.xattr[0].value == b"unconfined_u:object_r:unlabeled_t:s0\x00"
    assert poisoned.xattr[0].source == XAttrSource.INODE
    assert b"Z" not in poisoned.xattr[0].value

    # The raw attribute fork itself is untouched and still parses identically.
    assert raw[fork_offset:] == poisoned._buf.getvalue()[fork_offset:]


def test_user_namespace_attribute() -> None:
    # The leaf builder defaults to the security namespace; flip the flags off to
    # exercise the user namespace and verify the raw name is returned unprefixed.
    image, base = af.base_image()
    inode = base.get("test_file")
    blob = af.local_blob(b"comment", b"hello", c_xfs.XFS_ATTR_LOCAL)
    af.install_block(image, base, 3000, af.leaf_block(base, 3000, inode.inum, [blob]))
    af.set_attr_fork(image, base, inode, [3000])

    parsed = XFS(io.BytesIO(image)).get("test_file").xattr[0]
    assert parsed.namespace == "user"
    assert parsed.full_name == "comment"
    assert parsed.name == b"comment"
    assert parsed.value == b"hello"


def test_trusted_namespace_and_multiple_entries() -> None:
    image, base = af.base_image()
    inode = base.get("test_file")
    trusted = af.local_blob(b"gateway", b"locked", c_xfs.XFS_ATTR_LOCAL | c_xfs.XFS_ATTR_ROOT)
    security = af.local_blob(af.SELINUX_NAME, af.SELINUX_VALUE, c_xfs.XFS_ATTR_LOCAL | c_xfs.XFS_ATTR_SECURE)
    af.install_block(image, base, 3000, af.leaf_block(base, 3000, inode.inum, [trusted, security]))
    af.set_attr_fork(image, base, inode, [3000])

    attrs = XFS(io.BytesIO(image)).get("test_file").xattr
    assert [(attr.namespace, attr.full_name) for attr in attrs] == [
        ("trusted", "trusted.gateway"),
        ("security", "security.selinux"),
    ]
    assert [attr.source for attr in attrs] == [XAttrSource.LEAF, XAttrSource.LEAF]
