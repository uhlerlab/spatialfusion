import argparse
import logging
import os
import pathlib as pl
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import timm
import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_SCFOUNDATION_DIR = pl.Path("/opt/zero-shot-scfoundation")


def load_UNI_model(model_path: str, device: str = "cuda"):
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }

    model = timm.create_model(pretrained=False, **timm_kwargs)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return model, transform


def embed_UNI(
    wsi, # whole-slide image as a NumPy array
    adata, # AnnData object; obs_names align with he_coords
    he_coords, # typically adata.obsm['spatial'], shape (n_spots, 2) with (x,y) pixel coords
    output_dir: str, # where to write embeddings, UNI.parquet
    model_path: str, # path to UNI weights
    output_name: str = "UNI.parquet",
    batch_size: int = 128, # number of patches per forward pass
    device:str = "cuda", # "cuda" or "cpu"
):

    print('Load UNI model')
    model, transform = load_UNI_model(model_path, device)

    os.makedirs(output_dir, exist_ok=True)

    embeddings = []
    cell_ids = []
    
    batch_imgs = []
    batch_ids = []
    
    print(f"Embedding {len(he_coords)} image patches in batches of {batch_size}...")
    for cid, (x, y) in tqdm(zip(adata.obs_names, he_coords), total=len(adata)):
        x, y = int(x), int(y)
        x0, x1 = x - 128, x + 128
        y0, y1 = y - 128, y + 128
    
        pad_x0 = max(0, -x0)
        pad_x1 = max(0, x1 - wsi.shape[1])
        pad_y0 = max(0, -y0)
        pad_y1 = max(0, y1 - wsi.shape[0])
    
        patch = np.pad(
            wsi[max(0, y0):min(wsi.shape[0], y1), max(0, x0):min(wsi.shape[1], x1)],
            ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)),
            mode="constant"
        )
    
        if patch.shape[:2] != (256, 256):
            continue
    
        tensor_img = transform(Image.fromarray(patch))
        batch_imgs.append(tensor_img)
        batch_ids.append(cid)
    
        if len(batch_imgs) == batch_size:
            img_tensor = torch.stack(batch_imgs).to(device)
    
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
                batch_embs = model(img_tensor).to(torch.float16).cpu().numpy()
    
            embeddings.extend(batch_embs)
            cell_ids.extend(batch_ids)
            batch_imgs.clear()
            batch_ids.clear()
    
    # Final batch (if any)
    if batch_imgs:
        img_tensor = torch.stack(batch_imgs).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
            batch_embs = model(img_tensor).to(torch.float16).cpu().numpy()
        embeddings.extend(batch_embs)
        cell_ids.extend(batch_ids)

    # Save embedding matrix
    df = pd.DataFrame(embeddings, index=cell_ids)
    uni_out = pl.Path(output_dir) / output_name
    df.to_parquet(uni_out)
    print(f"Saved {len(df)} embeddings to {uni_out}") 


def run_embed_scGPT(
    dataset_path: str,
    model_dir: str,  # path to the pre-trained model, 3 files are expected: model_weights (best_model.pt), model args (args.json), and model vocab (vocab.json)
    output_dir: str,  # output_dir is the path to which the results should be saved
    n_hvg: int,
    gene_col: str = "index",  # in which column in adata.obs are gene names stored? if they are in index, the index will be copied to a column with this name
    layer_key: str = "X",  # where the raw counts are stored?
    log_norm: bool = False,  # are the values log_norm already?
    seed: int = 42,
    max_seq_len: int = 1200,  # maximum sequence of the input is controlled by max_seq_len, here I'm using the pretrained default
    batch_size: int = 32,  # batch_size depends on available GPU memory; should be a multiple of 8
    input_bins: int = 51,
    model_run: str = "pretrained",
    num_workers: int = 0,  # if you can use multithreading specify num_workers
    output_name: str = "scGPT.parquet",
    scfoundation_dir: str | None = None,
) -> None:
    if scfoundation_dir:
        sys.path.append(scfoundation_dir)
    else:
        sys.path.append(str(DEFAULT_SCFOUNDATION_DIR))

    from sc_foundation_evals import scgpt_forward, data
    from sc_foundation_evals.helpers.custom_logging import log

    log.setLevel(logging.INFO)


    # create the model
    scgpt_model = scgpt_forward.scGPT_instance(
        saved_model_path=model_dir,
        model_run=model_run,
        batch_size=batch_size,
        save_dir=output_dir,
        num_workers=num_workers,
        explicit_save_dir=True,
    )

    # create config
    scgpt_model.create_configs(seed=seed, max_seq_len=max_seq_len, n_bins=input_bins)

    scgpt_model.load_pretrained_model()

    # This is to keep the model running if the amount of genes is small
    input_data = data.InputData(adata_dataset_path=dataset_path)
    vocab_list = scgpt_model.vocab.get_stoi().keys()

    adata = input_data.adata
    genes_in_vocab = adata.var_names.intersection(vocab_list)
    if len(genes_in_vocab) / len(adata.var_names) < 0.5:
        log.warning("Fewer than 50% of genes are found in the model vocab — continuing anyway.")
    
    adata._inplace_subset_var(genes_in_vocab)
    input_data.adata = adata

    input_data.preprocess_data(
        gene_vocab=vocab_list,
        model_type="scGPT",
        gene_col=gene_col,
        data_is_raw=not log_norm,
        counts_layer=layer_key,
        n_bins=input_bins,
        n_hvg=n_hvg,
    )

    scgpt_model.tokenize_data(
        data=input_data, input_layer_key="X_binned", include_zero_genes=False
    )

    scgpt_model.extract_embeddings(data=input_data)

    pd.DataFrame(
        input_data.adata.obsm["X_scGPT"],
        index=input_data.adata.obs["cell_id"] if "cell_id" in input_data.adata.obs.columns else input_data.adata.obs.index
    ).to_parquet(pl.Path(output_dir) / output_name)



def parse_args() -> argparse.Namespace:
    script_dir = pl.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run unimodal embedding extraction with scGPT and/or UNI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    shared = parser.add_argument_group("Shared Inputs")
    shared.add_argument(
        "--mode",
        choices=["scgpt", "uni", "both"],
        default="both",
        help="Which embedding workflow to run.",
    )
    shared.add_argument(
        "--adata",
        type=pl.Path,
        default=script_dir / "../../../tutorials/data/tutadata_subset.h5ad",
        help="Input AnnData (.h5ad). Used by both scGPT and UNI flows.",
    )
    shared.add_argument(
        "--wsi",
        type=pl.Path,
        default=script_dir / "../../../tutorials/data/wsi_crop.ome.tif",
        help="Input H&E/WSI TIFF file. Used by UNI flow.",
    )
    shared.add_argument(
        "--output-dir",
        type=pl.Path,
        default=script_dir / "example-results",
        help="Directory where output parquet files are written.",
    )
    shared.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output parquet files if present.",
    )

    scgpt = parser.add_argument_group("scGPT Parameters")
    scgpt.add_argument(
        "--scgpt-weights",
        type=pl.Path,
        default=script_dir / "../weights/scgpt",
        help="Path to scGPT model directory (expects args/vocab/weights files).",
    )
    scgpt.add_argument(
        "--scfoundation-dir",
        type=pl.Path,
        default=DEFAULT_SCFOUNDATION_DIR,
        help="Path added to PYTHONPATH so `sc_foundation_evals` can be imported.",
    )
    scgpt.add_argument(
        "--n-hvg",
        type=int,
        default=1200,
        help="Number of highly variable genes used during preprocessing.",
    )
    scgpt.add_argument(
        "--scgpt-batch-size",
        type=int,
        default=16,
        help="Batch size passed to scGPT inference.",
    )
    scgpt.add_argument(
        "--gene-col",
        type=str,
        default="index",
        help="Column in `adata.var` containing gene names (or `index`).",
    )
    scgpt.add_argument(
        "--layer-key",
        type=str,
        default="X",
        help="AnnData layer containing counts used by scGPT preprocessing.",
    )
    scgpt.add_argument(
        "--log-norm",
        action="store_true",
        help="Set if the selected layer is already log-normalized.",
    )
    scgpt.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to scGPT config creation.",
    )
    scgpt.add_argument(
        "--max-seq-len",
        type=int,
        default=1200,
        help="Max sequence length passed to scGPT config.",
    )
    scgpt.add_argument(
        "--input-bins",
        type=int,
        default=51,
        help="Number of bins used when discretizing expression values.",
    )
    scgpt.add_argument(
        "--model-run",
        type=str,
        default="pretrained",
        help="Value forwarded to `scGPT_instance(model_run=...)`; valid values depend on scFoundation.",
    )
    scgpt.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for scGPT dataloading/tokenization.",
    )
    scgpt.add_argument(
        "--scgpt-output-name",
        type=str,
        default="scGPT.parquet",
        help="Output filename for scGPT embeddings.",
    )

    uni = parser.add_argument_group("UNI Parameters")
    uni.add_argument(
        "--uni-weights",
        type=pl.Path,
        default=script_dir / "../weights/uni2/pytorch_model.bin",
        help="Path to UNI model weights file.",
    )
    uni.add_argument(
        "--uni-batch-size",
        type=int,
        default=512,
        help="Batch size for UNI patch inference.",
    )
    uni.add_argument(
        "--spatial-key",
        type=str,
        default="spatial",
        help="Key in `adata.obsm` containing spot pixel coordinates for UNI.",
    )
    uni.add_argument(
        "--uni-output-name",
        type=str,
        default="UNI.parquet",
        help="Output filename for UNI embeddings.",
    )
    uni.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for UNI model inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode in {"scgpt", "both"}:
        # Keep notebook-like defaults but source paths from CLI args.
        sample_name = "SUBSET_Xenium_Ovarian-5k"
        adata_path = args.adata
        model_dir_GPT = args.scgpt_weights
        print(f"Starting for {sample_name}")
        scgpt_out = args.output_dir / args.scgpt_output_name
        if scgpt_out.exists() and not args.overwrite:
            raise FileExistsError(f"{scgpt_out} exists. Use --overwrite to replace it.")
        run_embed_scGPT(
            dataset_path=str(adata_path),
            model_dir=str(model_dir_GPT),
            output_dir=str(args.output_dir),
            n_hvg=args.n_hvg,
            gene_col=args.gene_col,
            layer_key=args.layer_key,
            log_norm=args.log_norm,
            seed=args.seed,
            max_seq_len=args.max_seq_len,
            batch_size=args.scgpt_batch_size,
            input_bins=args.input_bins,
            model_run=args.model_run,
            num_workers=args.num_workers,
            output_name=args.scgpt_output_name,
            scfoundation_dir=str(args.scfoundation_dir),
        )

    if args.mode in {"uni", "both"}:
        adata = sc.read_h5ad(args.adata)
        with tifffile.TiffFile(args.wsi) as tif:
            wsi = tif.series[0].asarray()

        output_dir = str(args.output_dir)
        device = args.device
        model_path = str(args.uni_weights)
        he_coords = adata.obsm[args.spatial_key]
        uni_out = args.output_dir / args.uni_output_name
        if uni_out.exists() and not args.overwrite:
            raise FileExistsError(f"{uni_out} exists. Use --overwrite to replace it.")

        embed_UNI(
            wsi,
            adata,
            he_coords,
            output_dir,
            model_path,
            output_name=args.uni_output_name,
            batch_size=args.uni_batch_size,
            device=device,
        )


if __name__ == "__main__":
    main()
