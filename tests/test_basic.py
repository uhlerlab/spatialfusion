import pandas as pd
import numpy as np
import torch
from spatialfusion.models.multi_ae import PairedDataset, PairedAE
from spatialfusion.models.gcn import GCNAutoencoder
from spatialfusion.utils.embed_ae_utils import safe_standardize

def test_paired_dataset():
    df1 = pd.DataFrame(np.random.rand(10, 5), index=[f"cell{i}" for i in range(10)])
    df2 = pd.DataFrame(np.random.rand(10, 5), index=[f"cell{i}" for i in range(10)])
    dataset = PairedDataset(df1, df2)
    assert len(dataset) == 10
    x1, x2 = dataset[0]
    assert x1.shape == (5,)
    assert x2.shape == (5,)

def test_safe_standardize():
    df = pd.DataFrame(np.random.rand(10, 5))
    std_df = safe_standardize(df)
    assert std_df.shape == df.shape
    assert np.isfinite(std_df.values).all()

def test_gcn_autoencoder_init():
    model = GCNAutoencoder(in_dim=5, hidden_dim=4, out_dim=3)
    assert isinstance(model, torch.nn.Module)
