import pytest
import torch
import numpy as np
import pandas as pd
import dgl
from finetune_models_modular import (
    build_ae_dataset,
    finetune_autoencoder,
    build_graphs,
    finetune_gcn,
    get_device
)


@pytest.fixture
def dummy_data():
    np.random.seed(0)
    feat1 = np.random.randn(100, 16).astype(np.float32)
    feat2 = np.random.randn(100, 16).astype(np.float32)
    return [(feat1, feat2)]


def test_build_ae_dataset(dummy_data):
    loader, d1, d2 = build_ae_dataset(
        samples=["S1"], preloaded_data=dummy_data)
    batch = next(iter(loader))
    assert d1 == 16 and d2 == 16
    assert isinstance(batch, list)
    assert batch[0].shape[1] == 16


def test_finetune_autoencoder(tmp_path, dummy_data):
    device = get_device()
    loader, d1, d2 = build_ae_dataset(
        samples=["S1"], preloaded_data=dummy_data)
    # Create a dummy AE checkpoint for testing
    from spatialfusion.models.multi_ae import PairedAE
    model = PairedAE(d1, d2, 8, [8], [8])
    ckpt_path = tmp_path / "dummy.pt"
    torch.save(model.state_dict(), ckpt_path)

    trained = finetune_autoencoder(
        loader, d1, d2, str(ckpt_path), tmp_path, device, epochs=1
    )
    assert isinstance(trained, torch.nn.Module)
    assert (tmp_path / "paired_model_finetuned.pt").exists()


def test_build_graphs(tmp_path):
    import scanpy as sc
    from spatialfusion.utils.embed_ae_utils import safe_standardize

    # Fake data
    adata = sc.AnnData(
        X=np.random.randn(50, 5),
        obsm={"spatial": np.random.randn(50, 2)},
    )
    adatas = {"S1": adata}
    z_joint_df = pd.DataFrame(
        np.random.randn(50, 8), index=[f"cell_{i}" for i in range(50)]
    )

    graphs = build_graphs(["S1"], z_joint_df, adatas=adatas)
    assert len(graphs) > 0
    assert isinstance(graphs[0], dgl.DGLGraph)
    assert "feat" in graphs[0].ndata


def test_finetune_gcn(tmp_path):
    import torch.nn as nn
    # Create a dummy DGL graph
    g = dgl.rand_graph(10, 30)
    g.ndata["feat"] = torch.randn(10, 8)
    g.ndata["label"] = torch.randn(10, 3)
    graphs = [g]

    # Create dummy pretrained GCN checkpoint
    from spatialfusion.models.gcn import GCNAutoencoder
    model = GCNAutoencoder(8, 4, 8, 0.9, 2, 3)
    ckpt_path = tmp_path / "gcn.pt"
    torch.save(model.state_dict(), ckpt_path)

    device = get_device()
    trained = finetune_gcn(
        graphs, str(ckpt_path), tmp_path, device, epochs=1, batch_size=1
    )
    assert isinstance(trained, torch.nn.Module)
    assert (tmp_path / "model.pt").exists()
