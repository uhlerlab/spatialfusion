from spatialfusion.embed.embed import (
    DEFAULT_GCN_CKPT_RELPATH,
    DEFAULT_HE_ONLY_GCN_CKPT_RELPATH,
    DEFAULT_RNA_ONLY_GCN_CKPT_RELPATH,
    _default_gcn_ckpt_relpath,
)
from spatialfusion.utils.pkg_ckpt import resolve_pkg_ckpt


def test_default_gcn_checkpoint_matches_combine_mode():
    assert _default_gcn_ckpt_relpath("z1") == DEFAULT_HE_ONLY_GCN_CKPT_RELPATH
    assert _default_gcn_ckpt_relpath("z2") == DEFAULT_RNA_ONLY_GCN_CKPT_RELPATH

    for mode in ("average", "concat", "gated"):
        assert _default_gcn_ckpt_relpath(mode) == DEFAULT_GCN_CKPT_RELPATH


def test_default_gcn_checkpoints_are_packaged():
    for checkpoint in (
        DEFAULT_GCN_CKPT_RELPATH,
        DEFAULT_HE_ONLY_GCN_CKPT_RELPATH,
        DEFAULT_RNA_ONLY_GCN_CKPT_RELPATH,
    ):
        assert resolve_pkg_ckpt(f"checkpoint_dir_gcn/{checkpoint}").is_file()
